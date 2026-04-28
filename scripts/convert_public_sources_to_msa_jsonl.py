#!/usr/bin/env python3
"""Convert public retrieval-style parquet sources into minimal MSA JSONL.

Conservative mode only uses rows with an explicit query, positive document,
and, by default, an explicit negative document. The output follows the minimal
MSA JSONL data contract documented in docs/training_data_spec.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit('pyarrow is required. Install it with: python -m pip install pyarrow') from exc


PAPER_SOURCE_MAP = {
    'WikiAnswers': ('WikiAnswers', 'high'),
    'GooAQ': ('gooaq_pairs', 'high'),
    'PAQ_pairs': ('PAQ_pairs', 'high'),
    'triviaqa': ('TriviaQA_pairs', 'medium'),
    'wikipedia-nq': ('NQ_train_pairs', 'medium'),
    'squad_v2': ('squad_pairs', 'medium'),
    'SearchQA': ('searchQA_top5_snippets', 'medium'),
    'quora': ('quora_duplicates', 'medium'),
    'yahoo-answers': ('yahoo_answers_qa', 'medium'),
    'msmarco-v2': ('msmarco_triples', 'medium'),
    'msmarco-passage': ('msmarco_triples', 'medium'),
    'ccnews': ('ccnews_title_text', 'medium'),
    'cnn_dailymail': ('cnn_dailymail', 'medium'),
    'xsum': ('xsum', 'medium'),
    'codesearchnet': ('codesearchnet', 'high'),
    'wikihow': ('wikihow', 'high'),
    'SimpleWiki': ('SimpleWiki', 'high'),
    'amazon_review_2018': ('amazon_review_2018', 'high'),
    'agnews': ('agnews', 'high'),
    'npr': ('npr', 'high'),
    'S2ORC_citations_abstracts': ('S2ORC_citations_abstracts', 'high'),
    'S2ORC_citations_titles': ('S2ORC_citations_titles', 'high'),
    'S2ORC_title_abstract': ('S2ORC_title_abstract', 'high'),
    'specter_train_triples': ('specter_train_triples', 'high'),
}

MEDI_TASK_MAP = {
    'NQ': ('NQ_train_pairs', 'medium'),
    'MSMARCO': ('msmarco_triples', 'medium'),
    'TriviaQA': ('TriviaQA_pairs', 'medium'),
    'SQuAD': ('squad_pairs', 'medium'),
    'GooAQ': ('gooaq_pairs', 'medium'),
    'PAQ': ('PAQ_pairs', 'medium'),
    'WikiAnswers': ('WikiAnswers', 'medium'),
}


@dataclass
class Stats:
    files_seen: int = 0
    files_converted: int = 0
    rows_seen: int = 0
    samples_written: int = 0
    skipped_no_query: int = 0
    skipped_no_positive: int = 0
    skipped_no_negative: int = 0
    skipped_source_cap: int = 0
    skipped_hash_gate: int = 0
    skipped_total_limit: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert public retrieval parquet files to MSA JSONL.')
    parser.add_argument('--input-root', type=Path, default=Path('training_data_public_sources'))
    parser.add_argument('--output-jsonl', type=Path, default=Path('converted_training_data/msa_pretrain_conservative.jsonl'))
    parser.add_argument('--manifest', type=Path, default=Path('converted_training_data/msa_pretrain_conservative_manifest.json'))
    parser.add_argument('--skipped-report', type=Path, default=Path('converted_training_data/msa_pretrain_conservative_skipped.json'))
    parser.add_argument('--max-samples-per-source', type=int, default=500000)
    parser.add_argument('--max-total-samples', type=int, default=None)
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--include-repo', action='append', default=[])
    parser.add_argument('--include-path-substring', action='append', default=[])
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--max-positives', type=int, default=1)
    parser.add_argument('--max-negatives', type=int, default=3)
    parser.add_argument('--allow-missing-negatives', action='store_true')
    parser.add_argument('--sample-rate', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--answer-format', choices=('doc_token', 'json_list', 'plain_id'), default='doc_token')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    out = []
    for item in values:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def stable_hash_int(*parts: Any) -> int:
    payload = '|'.join(str(part) for part in parts)
    return int(hashlib.sha1(payload.encode('utf-8', errors='replace')).hexdigest()[:16], 16)


def should_keep_by_hash(rate: float, seed: int, *parts: Any) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return stable_hash_int(seed, *parts) / float(16**16 - 1) < rate


def discover_parquet_files(input_root: Path) -> list[Path]:
    return sorted(p for p in input_root.rglob('*.parquet') if '.cache' not in p.parts and p.is_file())


def filter_files(input_root: Path, files: list[Path], include_repos: list[str], substrings: list[str]) -> list[Path]:
    if include_repos:
        repo_set = set(include_repos)
        files = [p for p in files if p.relative_to(input_root).parts and p.relative_to(input_root).parts[0] in repo_set]
    if substrings:
        files = [p for p in files if any(s in p.relative_to(input_root).as_posix() for s in substrings)]
    return files


def repo_and_source(input_root: Path, parquet_path: Path, row: dict[str, Any]) -> tuple[str, str]:
    rel = parquet_path.relative_to(input_root)
    repo_name = rel.parts[0] if rel.parts else 'unknown'
    if repo_name == 'medi_data_mteb_avs_triplets':
        task = str(row.get('task_name') or '').strip()
        return repo_name, task or 'medi_unknown'
    if len(rel.parts) >= 2:
        return repo_name, rel.parts[1]
    return repo_name, parquet_path.parent.name


def mapped_paper_source(repo_name: str, source_name: str) -> tuple[str | None, str]:
    if repo_name == 'kalm_embedding_finetuning_data':
        if source_name in PAPER_SOURCE_MAP:
            return PAPER_SOURCE_MAP[source_name]
        return 'kalmfinetune_data', 'low'
    if repo_name == 'kalm_embedding_pretrain_data':
        if source_name in PAPER_SOURCE_MAP:
            return PAPER_SOURCE_MAP[source_name]
        return None, 'unknown'
    if repo_name == 'medi_data_mteb_avs_triplets':
        return MEDI_TASK_MAP.get(source_name, (source_name, 'low'))
    return PAPER_SOURCE_MAP.get(source_name, (None, 'unknown'))


def read_columns(parquet_path: Path) -> list[str]:
    names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    preferred = [
        'query', 'pos', 'positive', 'positives', 'neg', 'negative', 'negatives',
        'task_name', 'query_instruct', 'pos_instruct', 'neg_instruct', 'symmetric', 'relevance'
    ]
    return [name for name in preferred if name in names]


def extract_row_fields(row: dict[str, Any]) -> tuple[str | None, list[str], list[str]]:
    query = row.get('query')
    query_text = str(query).strip() if query is not None else ''
    positives = as_text_list(row.get('pos')) or as_text_list(row.get('positive')) or as_text_list(row.get('positives'))
    negatives = as_text_list(row.get('neg')) or as_text_list(row.get('negative')) or as_text_list(row.get('negatives'))
    return query_text or None, positives, negatives


def build_answer(relevant_doc_ids: list[int], answer_format: str) -> str:
    if answer_format == 'json_list':
        return json.dumps(relevant_doc_ids, separators=(',', ':'))
    if answer_format == 'plain_id':
        return ' '.join(str(doc_id) for doc_id in relevant_doc_ids)
    return ' '.join(f'<doc_{doc_id}>' for doc_id in relevant_doc_ids)


def build_sample(
    input_root: Path,
    parquet_path: Path,
    row_index: int,
    row: dict[str, Any],
    query: str,
    positives: list[str],
    negatives: list[str],
    max_positives: int,
    max_negatives: int,
    answer_format: str,
) -> tuple[dict[str, Any], str]:
    repo_name, source_name = repo_and_source(input_root, parquet_path, row)
    mapped_source, confidence = mapped_paper_source(repo_name, source_name)
    documents = []
    relevant_doc_ids = []
    hard_negative_doc_ids = []
    doc_id = 1
    for text in positives[:max_positives]:
        documents.append({'doc_id': doc_id, 'text': text})
        relevant_doc_ids.append(doc_id)
        doc_id += 1
    for text in negatives[:max_negatives]:
        documents.append({'doc_id': doc_id, 'text': text})
        hard_negative_doc_ids.append(doc_id)
        doc_id += 1

    rel_path = parquet_path.relative_to(input_root).as_posix()
    sample_hash = hashlib.sha1(f'{rel_path}|{row_index}|{query}|{positives[:1]}'.encode('utf-8', errors='replace')).hexdigest()[:16]
    metadata = {
        'source_repo': repo_name,
        'source_name': source_name,
        'source_path': rel_path,
        'source_row_index': row_index,
        'mapped_paper_source': mapped_source,
        'mapping_confidence': confidence,
        'conversion_mode': 'conservative_explicit_positive_negative',
    }
    for key in ('task_name', 'query_instruct', 'pos_instruct', 'neg_instruct', 'symmetric', 'relevance'):
        if key in row and row[key] is not None:
            metadata[key] = row[key]

    source_key = mapped_source or f'{repo_name}:{source_name}'
    return {
        'sample_id': f'{repo_name}:{source_name}:{sample_hash}',
        'task_type': 'pretrain',
        'documents': documents,
        'query': query,
        'answer': build_answer(relevant_doc_ids, answer_format),
        'relevant_doc_ids': relevant_doc_ids,
        'hard_negative_doc_ids': hard_negative_doc_ids,
        'train_qa_sample': False,
        'metadata': metadata,
    }, source_key


def iter_rows(path: Path, columns: list[str], batch_size: int) -> Iterable[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def main() -> int:
    args = parse_args()
    if not (0.0 < args.sample_rate <= 1.0):
        raise SystemExit('--sample-rate must be in (0, 1].')
    if args.max_positives < 1:
        raise SystemExit('--max-positives must be >= 1.')
    if args.max_negatives < 0:
        raise SystemExit('--max-negatives must be >= 0.')

    files = filter_files(args.input_root, discover_parquet_files(args.input_root), args.include_repo, args.include_path_substring)
    if args.max_files is not None:
        files = files[:args.max_files]

    stats = Stats(files_seen=len(files))
    per_source_written = Counter()
    per_source_seen = Counter()
    per_file_written = Counter()
    per_file_seen = Counter()
    skipped_by_file = defaultdict(Counter)

    out_fh = None
    if not args.dry_run:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_fh = args.output_jsonl.open('w', encoding='utf-8')

    try:
        stop_conversion = False
        for parquet_path in files:
            if stop_conversion:
                break
            rel_path = parquet_path.relative_to(args.input_root).as_posix()
            columns = read_columns(parquet_path)
            if 'query' not in columns:
                skipped_by_file[rel_path]['no_query_column'] += 1
                continue
            if not any(c in columns for c in ('pos', 'positive', 'positives')):
                skipped_by_file[rel_path]['no_positive_column'] += 1
                continue

            wrote_for_file = False
            for row in iter_rows(parquet_path, columns, args.batch_size):
                row_index = per_file_seen[rel_path]
                per_file_seen[rel_path] += 1
                stats.rows_seen += 1
                query, positives, negatives = extract_row_fields(row)
                repo_name, source_name = repo_and_source(args.input_root, parquet_path, row)
                mapped_source, _ = mapped_paper_source(repo_name, source_name)
                source_key = mapped_source or f'{repo_name}:{source_name}'
                per_source_seen[source_key] += 1

                if not query:
                    stats.skipped_no_query += 1
                    skipped_by_file[rel_path]['no_query'] += 1
                    continue
                if not positives:
                    stats.skipped_no_positive += 1
                    skipped_by_file[rel_path]['no_positive'] += 1
                    continue
                if not negatives and not args.allow_missing_negatives:
                    stats.skipped_no_negative += 1
                    skipped_by_file[rel_path]['no_negative'] += 1
                    continue
                if per_source_written[source_key] >= args.max_samples_per_source:
                    stats.skipped_source_cap += 1
                    skipped_by_file[rel_path]['source_cap'] += 1
                    continue
                if not should_keep_by_hash(args.sample_rate, args.seed, rel_path, row_index, query):
                    stats.skipped_hash_gate += 1
                    skipped_by_file[rel_path]['hash_gate'] += 1
                    continue
                if args.max_total_samples is not None and stats.samples_written >= args.max_total_samples:
                    stats.skipped_total_limit += 1
                    skipped_by_file[rel_path]['total_limit'] += 1
                    stop_conversion = True
                    break

                sample, source_key = build_sample(
                    args.input_root,
                    parquet_path,
                    row_index,
                    row,
                    query,
                    positives,
                    negatives,
                    args.max_positives,
                    args.max_negatives,
                    args.answer_format,
                )
                if out_fh is not None:
                    print(json.dumps(sample, ensure_ascii=False, separators=(',', ':')), file=out_fh)
                stats.samples_written += 1
                per_source_written[source_key] += 1
                per_file_written[rel_path] += 1
                wrote_for_file = True

            if wrote_for_file:
                stats.files_converted += 1
    finally:
        if out_fh is not None:
            out_fh.close()

    manifest = {
        'converter': Path(__file__).as_posix(),
        'mode': 'conservative',
        'input_root': args.input_root.as_posix(),
        'output_jsonl': None if args.dry_run else args.output_jsonl.as_posix(),
        'args': vars(args) | {
            'input_root': args.input_root.as_posix(),
            'output_jsonl': args.output_jsonl.as_posix(),
            'manifest': args.manifest.as_posix(),
            'skipped_report': args.skipped_report.as_posix(),
        },
        'stats': asdict(stats),
        'per_source_seen': dict(per_source_seen),
        'per_source_written': dict(per_source_written),
    }
    skipped_report = {
        'stats': asdict(stats),
        'per_file_seen': dict(per_file_seen),
        'per_file_written': dict(per_file_written),
        'skipped_by_file': {k: dict(v) for k, v in skipped_by_file.items()},
    }
    write_json(args.manifest, manifest)
    write_json(args.skipped_report, skipped_report)
    print(json.dumps(asdict(stats), indent=2, ensure_ascii=False))
    print(f'manifest: {args.manifest}')
    print(f'skipped_report: {args.skipped_report}')
    if not args.dry_run:
        print(f'output_jsonl: {args.output_jsonl}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
