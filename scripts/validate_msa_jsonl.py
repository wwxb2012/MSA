#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


REQUIRED_TOP_FIELDS = {"sample_id", "documents", "query", "answer", "task_type"}
REQUIRED_DOC_FIELDS = {"doc_id", "text"}


def is_non_empty_str(x):
    return isinstance(x, str) and len(x.strip()) > 0


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def validate_one_record(
    rec: Dict[str, Any],
    target_documents: int = 64,
    require_target_documents: bool = True,
    require_relevant_doc_ids: bool = False,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors = []
    info = {"num_docs": None, "num_relevant": 0, "num_hard_negative": 0}

    missing_top = [k for k in REQUIRED_TOP_FIELDS if k not in rec]
    if missing_top:
        errors.append(f"missing_top_fields:{missing_top}")
        return False, errors, info

    if not is_non_empty_str(rec.get("sample_id", "")):
        errors.append("invalid_sample_id")
    if not is_non_empty_str(rec.get("query", "")):
        errors.append("invalid_query")
    if not is_non_empty_str(rec.get("answer", "")):
        errors.append("invalid_answer")

    task_type = rec.get("task_type")
    if task_type not in ("qa", "pretrain"):
        errors.append(f"invalid_task_type:{task_type}")

    docs = rec.get("documents")
    if not isinstance(docs, list) or len(docs) == 0:
        errors.append("invalid_documents_empty_or_not_list")
        return False, errors, info

    info["num_docs"] = len(docs)
    if require_target_documents and len(docs) != target_documents:
        errors.append(f"documents_count_not_{target_documents}:got_{len(docs)}")

    doc_ids: List[int] = []
    for i, d in enumerate(docs):
        if not isinstance(d, dict):
            errors.append(f"document_not_object_at:{i}")
            continue
        missing_doc = [k for k in REQUIRED_DOC_FIELDS if k not in d]
        if missing_doc:
            errors.append(f"document_missing_fields_at:{i}:{missing_doc}")
            continue

        did = d.get("doc_id")
        txt = d.get("text")

        if not isinstance(did, int):
            errors.append(f"doc_id_not_int_at:{i}:{did}")
        else:
            if did <= 0:
                errors.append(f"doc_id_not_positive_at:{i}:{did}")
            doc_ids.append(did)

        if not is_non_empty_str(txt):
            errors.append(f"doc_text_empty_at:{i}")

    if len(doc_ids) != len(set(doc_ids)):
        errors.append("duplicate_doc_id_in_sample")

    doc_id_set: Set[int] = set(doc_ids)

    relevant = rec.get("relevant_doc_ids", None)
    if require_relevant_doc_ids and relevant is None:
        errors.append("missing_relevant_doc_ids")

    if relevant is not None:
        if not isinstance(relevant, list):
            errors.append("relevant_doc_ids_not_list")
        else:
            info["num_relevant"] = len(relevant)
            for x in relevant:
                if not isinstance(x, int):
                    errors.append(f"relevant_doc_id_not_int:{x}")
                elif x not in doc_id_set:
                    errors.append(f"relevant_doc_id_not_in_documents:{x}")

    hard_neg = rec.get("hard_negative_doc_ids", None)
    if hard_neg is not None:
        if not isinstance(hard_neg, list):
            errors.append("hard_negative_doc_ids_not_list")
        else:
            info["num_hard_negative"] = len(hard_neg)
            for x in hard_neg:
                if not isinstance(x, int):
                    errors.append(f"hard_negative_doc_id_not_int:{x}")
                elif x not in doc_id_set:
                    errors.append(f"hard_negative_doc_id_not_in_documents:{x}")

    if isinstance(relevant, list) and isinstance(hard_neg, list):
        overlap = set(relevant) & set(hard_neg)
        if overlap:
            errors.append(f"relevant_and_hard_negative_overlap:{sorted(list(overlap))[:10]}")

    return len(errors) == 0, errors, info


def estimate_token_lengths(rec: Dict[str, Any], tokenizer, add_special_tokens: bool = False) -> Dict[str, int]:
    docs = rec.get("documents", [])
    query = rec.get("query", "")
    answer = rec.get("answer", "")

    doc_tokens = 0
    for d in docs:
        text = d.get("text", "") if isinstance(d, dict) else ""
        doc_tokens += len(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])

    q_tokens = len(tokenizer(query, add_special_tokens=add_special_tokens)["input_ids"])
    a_tokens = len(tokenizer(answer, add_special_tokens=add_special_tokens)["input_ids"])
    total = doc_tokens + q_tokens + a_tokens
    return {
        "doc_tokens": doc_tokens,
        "query_tokens": q_tokens,
        "answer_tokens": a_tokens,
        "approx_total_tokens": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to JSONL")
    ap.add_argument("--target-documents", type=int, default=64)
    ap.add_argument("--require-target-documents", action="store_true", default=True)
    ap.add_argument("--no-require-target-documents", action="store_true")
    ap.add_argument("--require-relevant-doc-ids", action="store_true", default=False)

    ap.add_argument("--max-lines", type=int, default=0, help="0 means full scan")
    ap.add_argument("--sample-errors", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--tokenizer-path", type=str, default="")
    ap.add_argument("--token-check-samples", type=int, default=0)
    ap.add_argument("--warn-over-tokens", type=int, default=65536)

    ap.add_argument("--output-report", type=str, default="")

    # progress/checkpoint args
    ap.add_argument("--progress-every", type=int, default=10000, help="update progress stats every N records")
    ap.add_argument("--checkpoint-every", type=int, default=200000, help="save checkpoint every N records (0 disable)")
    ap.add_argument("--checkpoint-path", type=str, default="", help="checkpoint json path")
    ap.add_argument("--resume", action="store_true", help="resume from checkpoint")

    args = ap.parse_args()

    random.seed(args.seed)
    path = Path(args.jsonl)
    if not path.exists():
        raise FileNotFoundError(path)

    require_target_docs = not args.no_require_target_documents

    # state
    total = 0
    ok_count = 0
    fail_count = 0
    error_counter = Counter()
    bad_examples = []
    source_counter = Counter()
    docs_num_counter = Counter()
    relevant_size_counter = Counter()
    hardneg_size_counter = Counter()
    valid_records_for_token_check = []

    line_no_start = 0

    # checkpoint load
    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    if args.resume and ckpt_path and ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        line_no_start = ckpt.get("line_no", 0)
        total = ckpt.get("total", 0)
        ok_count = ckpt.get("ok_count", 0)
        fail_count = ckpt.get("fail_count", 0)
        error_counter = Counter(ckpt.get("error_counter", {}))
        docs_num_counter = Counter({int(k): v for k, v in ckpt.get("docs_num_counter", {}).items()})
        relevant_size_counter = Counter({int(k): v for k, v in ckpt.get("relevant_size_counter", {}).items()})
        hardneg_size_counter = Counter({int(k): v for k, v in ckpt.get("hardneg_size_counter", {}).items()})
        source_counter = Counter(ckpt.get("source_counter", {}))

    start_time = time.time()
    t_read = 0.0
    t_parse = 0.0
    t_validate = 0.0

    pbar = tqdm(desc="Validating JSONL", unit="rec", dynamic_ncols=True)

    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            if ln <= line_no_start:
                continue
            if args.max_lines and total >= args.max_lines:
                break

            t0 = time.time()
            line = line.strip()
            t_read += time.time() - t0
            if not line:
                continue

            total += 1

            t1 = time.time()
            try:
                rec = json.loads(line)
            except Exception as e:
                fail_count += 1
                error_counter["json_parse_error"] += 1
                if len(bad_examples) < args.sample_errors:
                    bad_examples.append({"line": ln, "error": f"json_parse_error:{e}", "raw": line[:500]})
                t_parse += time.time() - t1
                pbar.update(1)
                continue
            t_parse += time.time() - t1

            t2 = time.time()
            ok, errs, info = validate_one_record(
                rec,
                target_documents=args.target_documents,
                require_target_documents=require_target_docs,
                require_relevant_doc_ids=args.require_relevant_doc_ids,
            )
            t_validate += time.time() - t2

            meta = rec.get("metadata", {})
            if isinstance(meta, dict):
                src = meta.get("source", None)
                if isinstance(src, str) and src:
                    source_counter[src] += 1

            if info["num_docs"] is not None:
                docs_num_counter[info["num_docs"]] += 1
            relevant_size_counter[info["num_relevant"]] += 1
            hardneg_size_counter[info["num_hard_negative"]] += 1

            if ok:
                ok_count += 1
                if args.token_check_samples > 0 and len(valid_records_for_token_check) < args.token_check_samples * 5:
                    valid_records_for_token_check.append((ln, rec))
            else:
                fail_count += 1
                for e in errs:
                    error_counter[e] += 1
                if len(bad_examples) < args.sample_errors:
                    bad_examples.append({
                        "line": ln,
                        "errors": errs[:20],
                        "sample_id": rec.get("sample_id", None),
                        "task_type": rec.get("task_type", None),
                    })

            pbar.update(1)

            if total % max(1, args.progress_every) == 0:
                elapsed = time.time() - start_time
                rate = total / elapsed if elapsed > 0 else 0.0
                pbar.set_postfix({
                    "ok": ok_count,
                    "fail": fail_count,
                    "rate_rec_s": f"{rate:.1f}",
                    "read_s": f"{t_read:.1f}",
                    "parse_s": f"{t_parse:.1f}",
                    "val_s": f"{t_validate:.1f}",
                })

            if args.checkpoint_every > 0 and ckpt_path and total % args.checkpoint_every == 0:
                ckpt_obj = {
                    "line_no": ln,
                    "total": total,
                    "ok_count": ok_count,
                    "fail_count": fail_count,
                    "error_counter": dict(error_counter),
                    "docs_num_counter": dict(docs_num_counter),
                    "relevant_size_counter": dict(relevant_size_counter),
                    "hardneg_size_counter": dict(hardneg_size_counter),
                    "source_counter": dict(source_counter),
                    "timing": {
                        "read_s": t_read,
                        "parse_s": t_parse,
                        "validate_s": t_validate,
                        "elapsed_s": time.time() - start_time,
                    },
                }
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                ckpt_path.write_text(json.dumps(ckpt_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    pbar.close()

    token_warnings = []
    token_stats = {}
    if args.tokenizer_path:
        if AutoTokenizer is None:
            token_warnings.append("transformers_not_available_skip_token_checks")
        else:
            tk_start = time.time()
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            picks = (
                random.sample(valid_records_for_token_check, min(args.token_check_samples, len(valid_records_for_token_check)))
                if args.token_check_samples > 0
                else []
            )

            over_cnt = 0
            max_seen = 0
            approx_totals = []
            sampled_details = []

            tk_bar = tqdm(picks, desc="Token length checks", unit="rec", dynamic_ncols=True)
            for ln, rec in tk_bar:
                lens = estimate_token_lengths(rec, tokenizer)
                t = lens["approx_total_tokens"]
                approx_totals.append(t)
                max_seen = max(max_seen, t)
                if t > args.warn_over_tokens:
                    over_cnt += 1
                if len(sampled_details) < 20:
                    sampled_details.append({"line": ln, "sample_id": rec.get("sample_id"), **lens})

            if approx_totals:
                approx_totals_sorted = sorted(approx_totals)
                p50 = approx_totals_sorted[len(approx_totals_sorted) // 2]
                p90 = approx_totals_sorted[int(len(approx_totals_sorted) * 0.9) - 1]
                p99 = approx_totals_sorted[max(0, int(len(approx_totals_sorted) * 0.99) - 1)]
            else:
                p50 = p90 = p99 = 0

            token_stats = {
                "sampled": len(picks),
                "warn_over_tokens": args.warn_over_tokens,
                "over_count": over_cnt,
                "max_approx_total_tokens": max_seen,
                "p50_approx_total_tokens": p50,
                "p90_approx_total_tokens": p90,
                "p99_approx_total_tokens": p99,
                "sampled_details": sampled_details,
                "token_check_elapsed_s": time.time() - tk_start,
            }

    elapsed_total = time.time() - start_time
    report = {
        "input_jsonl": str(path),
        "scanned_records": total,
        "valid_records": ok_count,
        "invalid_records": fail_count,
        "valid_ratio": (ok_count / total) if total else 0.0,
        "timing_breakdown_s": {
            "read_line_and_strip": t_read,
            "json_parse": t_parse,
            "schema_validate": t_validate,
            "other_overhead": max(0.0, elapsed_total - (t_read + t_parse + t_validate)),
            "elapsed_total": elapsed_total,
        },
        "throughput_rec_per_s": (total / elapsed_total) if elapsed_total > 0 else 0.0,
        "top_errors": error_counter.most_common(100),
        "docs_count_distribution_top": docs_num_counter.most_common(20),
        "relevant_size_distribution_top": relevant_size_counter.most_common(20),
        "hardneg_size_distribution_top": hardneg_size_counter.most_common(20),
        "source_distribution_top": source_counter.most_common(50),
        "bad_examples": bad_examples[: args.sample_errors],
        "token_checks": token_stats,
        "token_warnings": token_warnings,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_report:
        out = Path(args.output_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # final checkpoint snapshot
    if ckpt_path:
        final_ckpt = {
            "line_no": line_no_start + total,
            "total": total,
            "ok_count": ok_count,
            "fail_count": fail_count,
            "error_counter": dict(error_counter),
            "docs_num_counter": dict(docs_num_counter),
            "relevant_size_counter": dict(relevant_size_counter),
            "hardneg_size_counter": dict(hardneg_size_counter),
            "source_counter": dict(source_counter),
            "timing": report["timing_breakdown_s"],
            "done": True,
        }
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_text(json.dumps(final_ckpt, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
