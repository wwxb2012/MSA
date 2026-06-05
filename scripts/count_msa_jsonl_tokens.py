#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer


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


def tokenize_len(tokenizer, text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main():
    ap = argparse.ArgumentParser(description="Count total tokens in MSA JSONL training corpus")
    ap.add_argument("--jsonl", required=True, help="Path to MSA JSONL file")
    ap.add_argument("--tokenizer-path", required=True, help="Tokenizer path (e.g., ckpt/MSA-4B)")
    ap.add_argument("--max-lines", type=int, default=0, help="0 means full scan")
    ap.add_argument("--progress-every", type=int, default=5000, help="Update postfix every N records")
    ap.add_argument("--checkpoint-every", type=int, default=100000, help="Write checkpoint every N records; 0 disables")
    ap.add_argument("--checkpoint-path", type=str, default="", help="Checkpoint JSON path")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    ap.add_argument("--output-report", type=str, default="", help="Optional output report JSON path")
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        raise FileNotFoundError(path)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    total_records = 0
    bad_records = 0
    total_doc_tokens = 0
    total_query_tokens = 0
    total_answer_tokens = 0
    total_all_tokens = 0

    line_no_start = 0
    doc_count_counter = Counter()

    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    if args.resume and ckpt_path and ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        line_no_start = ckpt.get("line_no", 0)
        total_records = ckpt.get("total_records", 0)
        bad_records = ckpt.get("bad_records", 0)
        total_doc_tokens = ckpt.get("total_doc_tokens", 0)
        total_query_tokens = ckpt.get("total_query_tokens", 0)
        total_answer_tokens = ckpt.get("total_answer_tokens", 0)
        total_all_tokens = ckpt.get("total_all_tokens", 0)
        doc_count_counter = Counter({int(k): v for k, v in ckpt.get("doc_count_counter", {}).items()})

    start = time.time()
    parse_s = 0.0
    token_s = 0.0

    pbar = tqdm(desc="Counting tokens", unit="rec", dynamic_ncols=True)

    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            if ln <= line_no_start:
                continue
            if args.max_lines and total_records >= args.max_lines:
                break

            line = line.strip()
            if not line:
                continue

            total_records += 1

            t0 = time.time()
            try:
                rec = json.loads(line)
            except Exception:
                bad_records += 1
                parse_s += time.time() - t0
                pbar.update(1)
                continue
            parse_s += time.time() - t0

            t1 = time.time()
            docs = rec.get("documents", [])
            query = rec.get("query", "")
            answer = rec.get("answer", "")

            doc_count_counter[len(docs) if isinstance(docs, list) else -1] += 1

            doc_tokens = 0
            if isinstance(docs, list):
                for d in docs:
                    if isinstance(d, dict):
                        doc_tokens += tokenize_len(tokenizer, d.get("text", ""))

            query_tokens = tokenize_len(tokenizer, query)
            answer_tokens = tokenize_len(tokenizer, answer)

            total_doc_tokens += doc_tokens
            total_query_tokens += query_tokens
            total_answer_tokens += answer_tokens
            total_all_tokens += (doc_tokens + query_tokens + answer_tokens)
            token_s += time.time() - t1

            pbar.update(1)

            if total_records % max(1, args.progress_every) == 0:
                elapsed = time.time() - start
                rate = total_records / elapsed if elapsed > 0 else 0.0
                avg_tok = total_all_tokens / max(1, total_records)
                pbar.set_postfix({
                    "rate_rec_s": f"{rate:.1f}",
                    "avg_tok/rec": f"{avg_tok:.1f}",
                    "total_tok(B)": f"{total_all_tokens/1e9:.3f}",
                    "parse_s": f"{parse_s:.1f}",
                    "token_s": f"{token_s:.1f}",
                })

            if args.checkpoint_every > 0 and ckpt_path and total_records % args.checkpoint_every == 0:
                ckpt_obj = {
                    "line_no": ln,
                    "total_records": total_records,
                    "bad_records": bad_records,
                    "total_doc_tokens": total_doc_tokens,
                    "total_query_tokens": total_query_tokens,
                    "total_answer_tokens": total_answer_tokens,
                    "total_all_tokens": total_all_tokens,
                    "doc_count_counter": dict(doc_count_counter),
                    "timing": {
                        "elapsed_s": time.time() - start,
                        "parse_s": parse_s,
                        "token_s": token_s,
                    },
                }
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                ckpt_path.write_text(json.dumps(ckpt_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    pbar.close()

    elapsed = time.time() - start
    report = {
        "input_jsonl": str(path),
        "tokenizer_path": args.tokenizer_path,
        "records_processed": total_records,
        "bad_records": bad_records,
        "records_valid_ratio": (total_records - bad_records) / max(1, total_records),
        "total_doc_tokens": total_doc_tokens,
        "total_query_tokens": total_query_tokens,
        "total_answer_tokens": total_answer_tokens,
        "total_tokens": total_all_tokens,
        "total_tokens_billions": total_all_tokens / 1e9,
        "avg_tokens_per_record": total_all_tokens / max(1, total_records - bad_records),
        "doc_count_distribution_top": doc_count_counter.most_common(20),
        "timing_breakdown_s": {
            "elapsed_total": elapsed,
            "json_parse": parse_s,
            "tokenize": token_s,
            "other": max(0.0, elapsed - parse_s - token_s),
        },
        "throughput_records_per_s": total_records / elapsed if elapsed > 0 else 0.0,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_report:
        out = Path(args.output_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if ckpt_path:
        final_ckpt = {
            "line_no": line_no_start + total_records,
            "total_records": total_records,
            "bad_records": bad_records,
            "total_doc_tokens": total_doc_tokens,
            "total_query_tokens": total_query_tokens,
            "total_answer_tokens": total_answer_tokens,
            "total_all_tokens": total_all_tokens,
            "doc_count_counter": dict(doc_count_counter),
            "done": True,
            "timing": report["timing_breakdown_s"],
        }
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_text(json.dumps(final_ckpt, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
