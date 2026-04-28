"""Minimal MSA training entrypoint.

This script is intentionally separate from inference services. It wires the
training config, JSONL dataset, tokenizer-aware collator, model, optimizer,
cosine schedule, checkpointing, resume, and periodic evaluation hooks.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.config.train_config import TrainConfig, load_train_config
from src.training.collator import MSACollatorConfig, MSATrainingCollator
from src.training.dataset import MSAJsonlDataset


TRAIN_STATE_FILE = "trainer_state.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MSA on minimal JSONL data.")
    parser.add_argument("config", type=Path, help="Path to configs/train/*.json")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override, key=value")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-steps", type=int, default=None, help="Run eval every N optimizer steps when validation data exists")
    parser.add_argument("--save-steps", type=int, default=None, help="Override checkpoint save interval")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dry-run", action="store_true", help="Build all objects and run one eval-style batch without optimizer step")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_train_config(args.config, args.override)
    if args.resume_from_checkpoint is not None:
        cfg.checkpointing.resume_from_checkpoint = str(args.resume_from_checkpoint)
    if args.save_steps is not None:
        cfg.checkpointing.save_steps = args.save_steps

    set_seed(cfg.run.seed)
    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train_config.resolved.json", cfg.to_dict())

    device = resolve_device(args.device)
    dtype = resolve_torch_dtype(cfg.model.torch_dtype)
    autocast_context = make_autocast_context(device, dtype)

    tokenizer = load_tokenizer(cfg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = MSATrainingCollator(
        tokenizer,
        MSACollatorConfig(
            max_seq_len=cfg.sequence.max_seq_len,
            max_documents_per_sample=cfg.sequence.max_documents_per_sample,
            max_document_tokens=cfg.sequence.max_document_tokens,
            max_query_tokens=cfg.sequence.max_query_tokens,
            max_answer_tokens=cfg.sequence.max_answer_tokens,
        ),
    )

    train_loader = build_loader(
        cfg.data.train_jsonl,
        collator,
        batch_size=cfg.optimization.per_device_train_batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        max_samples=cfg.data.max_train_samples,
        seed=cfg.run.seed,
    )
    eval_loader = None
    if cfg.data.validation_jsonl:
        eval_loader = build_loader(
            cfg.data.validation_jsonl,
            collator,
            batch_size=cfg.optimization.per_device_train_batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            max_samples=cfg.data.max_eval_samples,
            seed=cfg.run.seed,
        )

    resume_dir = Path(cfg.checkpointing.resume_from_checkpoint) if cfg.checkpointing.resume_from_checkpoint else None
    model_path = str(resume_dir if resume_dir else cfg.model.model_path)
    model = load_model(model_path, cfg, dtype)
    model.to(device)
    if cfg.optimization.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    apply_loss_weights(model, cfg)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimization.learning_rate,
        weight_decay=cfg.optimization.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        make_cosine_schedule(
            warmup_steps=cfg.optimization.warmup_steps,
            max_steps=cfg.optimization.max_steps,
        ),
    )

    global_step = 0
    if resume_dir:
        global_step = load_training_state(resume_dir, optimizer, scheduler)
        print(f"Resumed trainer state from {resume_dir} at global_step={global_step}")

    eval_steps = args.eval_steps if args.eval_steps is not None else cfg.checkpointing.save_steps
    if args.dry_run:
        dry_run_batch(model, train_loader, device, autocast_context)
        return 0

    train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        autocast_context=autocast_context,
        output_dir=output_dir,
        tokenizer=tokenizer,
        start_step=global_step,
        eval_steps=eval_steps,
    )
    return 0


def train(
    *,
    cfg: TrainConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader | None,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    autocast_context: Any,
    output_dir: Path,
    tokenizer: Any,
    start_step: int,
    eval_steps: int,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = start_step
    micro_step = 0
    running_loss = 0.0
    grad_accum = cfg.optimization.gradient_accumulation_steps
    last_checkpoint_step: int | None = None

    while global_step < cfg.optimization.max_steps:
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            with autocast_context():
                outputs = model(**model_batch_kwargs(batch))
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model returned loss=None; collator must provide labels.")
                scaled_loss = loss / grad_accum
            scaled_loss.backward()
            running_loss += float(loss.detach().cpu())
            micro_step += 1

            if micro_step % grad_accum != 0:
                continue

            if cfg.optimization.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimization.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % cfg.logging.logging_steps == 0:
                avg_loss = running_loss / max(1, cfg.logging.logging_steps * grad_accum)
                lr = scheduler.get_last_lr()[0]
                print(json.dumps({"step": global_step, "loss": avg_loss, "lr": lr}))
                running_loss = 0.0

            if eval_loader is not None and eval_steps > 0 and global_step % eval_steps == 0:
                metrics = evaluate(model, eval_loader, device, autocast_context)
                print(json.dumps({"step": global_step, "eval_loss": metrics["eval_loss"]}))

            if global_step % cfg.checkpointing.save_steps == 0:
                save_checkpoint(output_dir, model, tokenizer, optimizer, scheduler, global_step, cfg)
                last_checkpoint_step = global_step

            if global_step >= cfg.optimization.max_steps:
                break

    if last_checkpoint_step != global_step:
        save_checkpoint(output_dir, model, tokenizer, optimizer, scheduler, global_step, cfg)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    autocast_context: Any,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in eval_loader:
        batch = move_batch_to_device(batch, device)
        with autocast_context():
            outputs = model(**model_batch_kwargs(batch))
        if outputs.loss is not None:
            total_loss += float(outputs.loss.detach().cpu())
            total_batches += 1
    model.train()
    return {"eval_loss": total_loss / max(1, total_batches)}


def dry_run_batch(model: torch.nn.Module, loader: DataLoader, device: torch.device, autocast_context: Any) -> None:
    model.eval()
    batch = move_batch_to_device(next(iter(loader)), device)
    with torch.no_grad(), autocast_context():
        outputs = model(**model_batch_kwargs(batch))
    print(
        json.dumps(
            {
                "dry_run": True,
                "loss": None if outputs.loss is None else float(outputs.loss.detach().cpu()),
                "batch_shape": list(batch["input_ids"].shape),
            }
        )
    )


def build_loader(
    jsonl_path: str,
    collator: MSATrainingCollator,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_samples: int | None,
    seed: int,
) -> DataLoader:
    dataset = MSAJsonlDataset(jsonl_path, max_samples=max_samples)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )


def load_tokenizer(cfg: TrainConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_path,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    return tokenizer


def load_model(model_path: str, cfg: TrainConfig, dtype: torch.dtype) -> torch.nn.Module:
    from src.msa.model import MSAForCausalLM

    return MSAForCausalLM.from_pretrained(
        model_path,
        attn_implementation=cfg.model.attn_implementation,
        torch_dtype=dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )


def model_batch_kwargs(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "position_ids": batch["position_ids"],
        "doc_ids": batch["doc_ids"],
        "labels": batch["labels"],
        "batch_answer_labels": batch["batch_answer_labels"],
        "batch_aux_labels": batch["batch_aux_labels"],
        "train_qa_samples": batch["train_qa_samples"],
        "use_cache": False,
    }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def make_cosine_schedule(*, warmup_steps: int, max_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: AdamW,
    scheduler: LambdaLR,
    global_step: int,
    cfg: TrainConfig,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_random_state": random.getstate(),
        },
        checkpoint_dir / TRAIN_STATE_FILE,
    )
    last_dir = output_dir / "last"
    if last_dir.exists():
        shutil.rmtree(last_dir)
    shutil.copytree(checkpoint_dir, last_dir)
    prune_checkpoints(output_dir, cfg.checkpointing.save_total_limit)
    print(f"Saved checkpoint to {checkpoint_dir}")


def load_training_state(checkpoint_dir: Path, optimizer: AdamW, scheduler: LambdaLR) -> int:
    state_path = checkpoint_dir / TRAIN_STATE_FILE
    if not state_path.exists():
        print(f"No {TRAIN_STATE_FILE} found in {checkpoint_dir}; model weights will resume without optimizer state")
        return 0
    state = load_torch_state(state_path)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if "rng_state" in state:
        torch.set_rng_state(state["rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    return int(state.get("global_step", 0))


def prune_checkpoints(output_dir: Path, save_total_limit: int | None) -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if path.is_dir():
            try:
                step = int(path.name.split("-")[-1])
            except ValueError:
                continue
            checkpoints.append((step, path))
    checkpoints.sort()
    for _, path in checkpoints[:-save_total_limit]:
        shutil.rmtree(path)


def load_torch_state(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def apply_loss_weights(model: torch.nn.Module, cfg: TrainConfig) -> None:
    assignments = {
        "lmloss_weigth": cfg.losses.lm_loss_weight,
        "auxloss_weight": cfg.losses.aux_loss_weight,
        "ansloss_weight": cfg.losses.answer_loss_weight,
        "recloss_weight": cfg.losses.reconstruction_loss_weight,
    }
    for name, value in assignments.items():
        if hasattr(model, name):
            setattr(model, name, value)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not mps_is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


def make_autocast_context(device: torch.device, dtype: torch.dtype):
    enabled = (device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}) or (
        device.type == "cpu" and dtype is torch.bfloat16
    )
    if not enabled:
        return nullcontext
    return lambda: torch.autocast(device_type=device.type, dtype=dtype)


def mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
