"""Run a 5-step local debug training loop on a tiny MSA JSONL dataset.

This script is intentionally offline and CPU-friendly. It writes a tiny JSONL
dataset, uses the real training collator and training loop, and swaps in a tiny
``save_pretrained``-compatible language model so no MSA checkpoint download is
required.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.train as train_module
from src.training.collator import MSACollatorConfig, MSATrainingCollator
from src.training.dataset import MSAJsonlDataset


DEFAULT_OUTPUT_DIR = Path("outputs/train/debug_tiny")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiny CPU debug training for the MSA training path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clean", action="store_true", help="Delete output-dir before running.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.output_dir / "tiny_train.jsonl"
    write_tiny_jsonl(dataset_path)

    tokenizer = ToyTokenizer()
    collator = MSATrainingCollator(
        tokenizer,
        MSACollatorConfig(
            max_seq_len=128,
            max_documents_per_sample=3,
            max_document_tokens=18,
            max_query_tokens=32,
            max_answer_tokens=10,
        ),
    )
    dataset = MSAJsonlDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    warmup_batch = next(iter(loader))
    model = TinyMSAForCausalLM(vocab_size=max(tokenizer.next_id + 16, int(warmup_batch["input_ids"].max()) + 16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        train_module.make_cosine_schedule(warmup_steps=0, max_steps=args.steps),
    )
    cfg = make_train_cfg(args.steps)

    train_module.set_seed(7)
    train_module.train(
        cfg=cfg,
        model=model,
        train_loader=loader,
        eval_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        autocast_context=train_module.nullcontext,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        start_step=0,
        eval_steps=args.steps,
    )

    checkpoint_dir = args.output_dir / f"checkpoint-{args.steps}"
    reloaded = TinyMSAForCausalLM.from_pretrained(checkpoint_dir)
    reloaded.eval()
    with torch.no_grad():
        inference_outputs = reloaded(
            input_ids=warmup_batch["input_ids"],
            attention_mask=warmup_batch["attention_mask"],
            position_ids=warmup_batch["position_ids"],
            doc_ids=warmup_batch["doc_ids"],
            use_cache=False,
        )

    print(
        json.dumps(
            {
                "debug_train": "ok",
                "steps": args.steps,
                "dataset": str(dataset_path),
                "checkpoint": str(checkpoint_dir),
                "logits_shape": list(inference_outputs.logits.shape),
            }
        )
    )
    return 0


def write_tiny_jsonl(path: Path) -> None:
    samples = [
        make_sample("tiny-1", "Which document mentions Paris?", "Paris appears in the city document.", 10, "Paris is a city in France."),
        make_sample("tiny-2", "Which document mentions oceans?", "The ocean document mentions oceans.", 20, "Oceans contain salt water."),
        make_sample("tiny-3", "Which document mentions apples?", "The apple document mentions apples.", 30, "Apples grow on trees."),
        make_sample("tiny-4", "Which document mentions rockets?", "The rocket document mentions rockets.", 40, "Rockets can reach orbit."),
    ]
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")


def make_sample(sample_id: str, query: str, answer: str, relevant_doc_id: int, relevant_text: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_type": "qa",
        "documents": [
            {"doc_id": relevant_doc_id, "text": relevant_text},
            {"doc_id": relevant_doc_id + 1, "text": "This distractor document is unrelated."},
            {"doc_id": relevant_doc_id + 2, "text": "Another negative passage with generic words."},
        ],
        "query": query,
        "answer": answer,
        "relevant_doc_ids": [relevant_doc_id],
        "hard_negative_doc_ids": [relevant_doc_id + 1],
        "train_qa_sample": True,
    }


class ToyTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.vocab = {self.pad_token: self.pad_token_id, self.eos_token: self.eos_token_id}
        self.next_id = 2

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        ids = []
        for token in text.split():
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.next_id += 1
            ids.append(self.vocab[token])
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text(
            json.dumps({"pad_token": self.pad_token, "eos_token": self.eos_token}),
            encoding="utf-8",
        )
        (path / "toy_vocab.json").write_text(json.dumps(self.vocab, indent=2, sort_keys=True), encoding="utf-8")


class TinyMSAForCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        del attention_mask, position_ids, doc_ids, kwargs
        logits = self.lm_head(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        (path / "tiny_config.json").write_text(
            json.dumps({"vocab_size": self.vocab_size, "hidden_size": self.hidden_size}),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs: Any) -> "TinyMSAForCausalLM":
        del kwargs
        path = Path(path)
        cfg = json.loads((path / "tiny_config.json").read_text(encoding="utf-8"))
        model = cls(**cfg)
        try:
            state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state)
        return model


def make_train_cfg(max_steps: int) -> SimpleNamespace:
    return SimpleNamespace(
        optimization=SimpleNamespace(
            gradient_accumulation_steps=1,
            max_steps=max_steps,
            max_grad_norm=1.0,
        ),
        logging=SimpleNamespace(logging_steps=1),
        checkpointing=SimpleNamespace(save_steps=max_steps, save_total_limit=1),
    )


if __name__ == "__main__":
    raise SystemExit(main())
