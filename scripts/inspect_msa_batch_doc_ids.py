#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


class SimpleMSADataset(Dataset):
    def __init__(self, jsonl_path: str, max_samples: int = 1024):
        self.rows: List[Dict[str, Any]] = []
        p = Path(jsonl_path)
        if not p.exists():
            raise FileNotFoundError(jsonl_path)

        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def serialize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    documents = sample["documents"]
    query = sample["query"]
    answer = sample["answer"]

    doc_segments = []
    for d in documents:
        doc_id = d["doc_id"]
        text = f"[DOC {doc_id}] {d['text']}"
        doc_segments.append((doc_id, text))

    return {
        "sample_id": sample.get("sample_id", ""),
        "doc_segments": doc_segments,
        "query": query,
        "answer": answer,
    }


def build_batch_features(samples: List[Dict[str, Any]], tokenizer, max_length: int = 8192):
    template_prefix = "You are a helpful assistant.\n"

    input_ids_list = []
    attention_masks = []
    doc_ids_list = []

    for s in samples:
        x = serialize_sample(s)

        prefix_ids = tokenizer(template_prefix, add_special_tokens=False)["input_ids"]
        prefix_doc_ids = [-2] * len(prefix_ids)

        all_doc_ids = []
        all_doc_tok_doc_ids = []
        for doc_id, doc_text in x["doc_segments"]:
            tok = tokenizer(doc_text + "\n", add_special_tokens=False)["input_ids"]
            all_doc_ids.extend(tok)
            all_doc_tok_doc_ids.extend([int(doc_id)] * len(tok))

        qa_text = f"Question: {x['query']}\nAnswer: {x['answer']}"
        qa_ids = tokenizer(qa_text, add_special_tokens=False)["input_ids"]
        qa_doc_ids = [0] * len(qa_ids)

        ids = prefix_ids + all_doc_ids + qa_ids
        dids = prefix_doc_ids + all_doc_tok_doc_ids + qa_doc_ids
        assert len(ids) == len(dids)

        ids = ids[:max_length]
        dids = dids[:max_length]
        mask = [1] * len(ids)

        input_ids_list.append(torch.tensor(ids, dtype=torch.long))
        attention_masks.append(torch.tensor(mask, dtype=torch.long))
        doc_ids_list.append(torch.tensor(dids, dtype=torch.long))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_len = max(x.shape[0] for x in input_ids_list)

    def pad_1d(t, value):
        if t.shape[0] == max_len:
            return t
        out = torch.full((max_len,), value, dtype=t.dtype)
        out[: t.shape[0]] = t
        return out

    input_ids = torch.stack([pad_1d(t, pad_id) for t in input_ids_list], dim=0)
    attention_mask = torch.stack([pad_1d(t, 0) for t in attention_masks], dim=0)
    doc_ids = torch.stack([pad_1d(t, -1) for t in doc_ids_list], dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "doc_ids": doc_ids,
    }


def inspect_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    doc_ids = batch["doc_ids"]
    attn = batch["attention_mask"]

    valid = attn == 1
    vals = doc_ids[valid]

    has_neg2 = bool((vals == -2).any().item())
    has_zero = bool((vals == 0).any().item())
    has_pos = bool((vals > 0).any().item())

    c = Counter(vals.cpu().tolist())
    top_items = c.most_common(20)

    per_sample = []
    for i in range(doc_ids.shape[0]):
        v = doc_ids[i][attn[i] == 1]
        per_sample.append({
            "sample_index": i,
            "len_tokens": int(v.shape[0]),
            "has_-2": bool((v == -2).any().item()),
            "has_0": bool((v == 0).any().item()),
            "has_>0": bool((v > 0).any().item()),
            "num_unique_doc_ids_pos": int(torch.unique(v[v > 0]).shape[0]) if (v > 0).any() else 0,
        })

    return {
        "global_has_-2": has_neg2,
        "global_has_0": has_zero,
        "global_has_>0": has_pos,
        "top_doc_id_counts": top_items,
        "per_sample": per_sample,
    }


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--num-batches", type=int, default=5)
    ap.add_argument("--output-report", default="")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"

    ds = SimpleMSADataset(args.jsonl, max_samples=args.max_samples)

    def collate_fn(rows):
        return build_batch_features(rows, tokenizer=tokenizer, max_length=args.max_length)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    total_batches_available = len(dl)
    target_batches = min(args.num_batches, total_batches_available)

    all_reports = []
    start_time = time.time()

    pbar = tqdm(total=target_batches, desc="Inspecting batches", unit="batch")

    for bi, batch in enumerate(dl):
        if bi >= target_batches:
            break

        rep = inspect_batch(batch)
        rep["batch_index"] = bi
        all_reports.append(rep)

        elapsed = time.time() - start_time
        done = bi + 1
        avg_per_batch = elapsed / done
        remain = target_batches - done
        eta_sec = avg_per_batch * remain

        pbar.set_postfix({
            "avg_s/batch": f"{avg_per_batch:.2f}",
            "eta": format_eta(eta_sec),
        })
        pbar.update(1)

    pbar.close()

    total_elapsed = time.time() - start_time

    summary = {
        "jsonl": args.jsonl,
        "tokenizer_path": args.tokenizer_path,
        "checked_batches": len(all_reports),
        "planned_batches": target_batches,
        "dataset_batches_available": total_batches_available,
        "elapsed_seconds": total_elapsed,
        "avg_seconds_per_batch": (total_elapsed / len(all_reports)) if all_reports else None,
        "reports": all_reports,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_report:
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
