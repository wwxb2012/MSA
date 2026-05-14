"""Minimal MSA training entrypoint.

This script is intentionally separate from inference services. It wires the
training config, JSONL dataset, tokenizer-aware collator, model, optimizer,
cosine schedule, checkpointing, resume, and periodic evaluation hooks.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from src.config.train_config import TrainConfig, load_train_config
from src.training.collator import MSACollatorConfig, MSATrainingCollator
from src.training.dataset import MSAJsonlDataset


TRAIN_STATE_FILE = "trainer_state.pt"
TOKEN_ACCOUNTING_FILE = "token_accounting.json"


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "none"

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MSA on minimal JSONL data.")
    parser.add_argument("config", type=Path, help="Path to configs/train/*.json")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override, key=value")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-steps", type=int, default=None, help="Run eval every N optimizer steps when validation data exists")
    parser.add_argument("--save-steps", type=int, default=None, help="Override checkpoint save interval")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dry-run", action="store_true", help="Build all objects and run one eval-style batch without optimizer step")
    parser.add_argument("--distributed-backend", default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--distributed-timeout-minutes", type=int, default=60)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("outputs/log"),
        help="Directory for teeing stdout/stderr into a timestamped training log.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_train_config(args.config, args.override)
    if args.resume_from_checkpoint is not None:
        cfg.checkpointing.resume_from_checkpoint = str(args.resume_from_checkpoint)
    if args.save_steps is not None:
        cfg.checkpointing.save_steps = args.save_steps

    distributed = init_distributed(args)
    log_file = setup_print_logging(args.log_dir, distributed)
    set_seed(cfg.run.seed + distributed.rank)
    output_dir = Path(cfg.run.output_dir)
    if distributed.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "train_config.resolved.json", cfg.to_dict())
        print(json.dumps({"log_file": str(log_file)}))
        if distributed.enabled:
            print(
                json.dumps(
                    {
                        "distributed": True,
                        "backend": distributed.backend,
                        "world_size": distributed.world_size,
                    }
                )
            )
    distributed_barrier(distributed)

    device = resolve_device(args.device, distributed)
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
        distributed=distributed,
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
            distributed=distributed,
        )

    resume_dir = Path(cfg.checkpointing.resume_from_checkpoint) if cfg.checkpointing.resume_from_checkpoint else None
    model_path = str(resume_dir if resume_dir else cfg.model.model_path)
    model = load_model(model_path, cfg, dtype)
    model.to(device)
    if cfg.optimization.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    apply_loss_weights(model, cfg)
    if distributed.enabled:
        model = wrap_distributed_model(model, device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimization.learning_rate,
        weight_decay=cfg.optimization.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        make_lr_schedule(
            scheduler_type=cfg.optimization.lr_scheduler_type,
            warmup_steps=cfg.optimization.warmup_steps,
            max_steps=cfg.optimization.max_steps,
        ),
    )

    global_step = 0
    train_tokens_seen = 0
    if resume_dir:
        global_step = load_training_state(resume_dir, optimizer, scheduler)
        train_tokens_seen = load_token_accounting(resume_dir)
        if distributed.is_main_process:
            print(
                json.dumps(
                    {
                        "resumed_from": str(resume_dir),
                        "global_step": global_step,
                        "train_tokens_seen": train_tokens_seen,
                    }
                )
            )
    prior_stage_train_tokens = detect_prior_stage_train_tokens(cfg, resume_dir)

    eval_steps = args.eval_steps if args.eval_steps is not None else cfg.checkpointing.save_steps
    if args.dry_run:
        dry_run_batch(model, train_loader, device, autocast_context, distributed)
        cleanup_distributed(distributed)
        return 0

    try:
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
            start_train_tokens=train_tokens_seen,
            prior_stage_train_tokens=prior_stage_train_tokens,
            eval_steps=eval_steps,
            distributed=distributed,
        )
        return 0
    finally:
        cleanup_distributed(distributed)


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
    start_train_tokens: int = 0,
    prior_stage_train_tokens: int = 0,
    distributed: DistributedContext | None = None,
) -> None:
    distributed = distributed or DistributedContext()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = start_step
    train_tokens_seen = int(start_train_tokens)
    prior_stage_train_tokens = int(prior_stage_train_tokens)
    micro_step = 0
    accumulated_micro_tokens = 0
    running_loss = 0.0
    grad_accum = cfg.optimization.gradient_accumulation_steps
    last_checkpoint_step: int | None = None
    epoch = 0
    train_started_at = datetime.now()
    train_start_monotonic = time.monotonic()
    if distributed.is_main_process:
        max_seq_len = getattr(getattr(cfg, "sequence", None), "max_seq_len", None)
        per_device_batch = getattr(
            cfg.optimization,
            "per_device_train_batch_size",
            getattr(train_loader, "batch_size", None) or 1,
        )
        theoretical_tokens_per_step = None
        theoretical_total_tokens = None
        if max_seq_len is not None:
            theoretical_tokens_per_step = (
                distributed.world_size
                * per_device_batch
                * cfg.optimization.gradient_accumulation_steps
                * max_seq_len
            )
            theoretical_total_tokens = theoretical_tokens_per_step * cfg.optimization.max_steps
        estimated_total_tokens = (
            None
            if theoretical_total_tokens is None
            else prior_stage_train_tokens + theoretical_total_tokens
        )
        print(
            json.dumps(
                {
                    "token_accounting": "attention_mask_non_padding_tokens",
                    "train_started_at": train_started_at.isoformat(timespec="seconds"),
                    "start_train_tokens": train_tokens_seen,
                    "prior_stage_train_tokens": prior_stage_train_tokens,
                    "total_train_tokens_seen": prior_stage_train_tokens + train_tokens_seen,
                    "theoretical_tokens_per_optimizer_step_at_max_seq_len": theoretical_tokens_per_step,
                    "theoretical_total_tokens_at_max_steps": theoretical_total_tokens,
                    "estimated_total_train_tokens_at_max_steps": estimated_total_tokens,
                }
            )
        )

    while global_step < cfg.optimization.max_steps:
        set_loader_epoch(train_loader, epoch)
        epoch += 1
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            accumulated_micro_tokens += count_batch_tokens(batch)
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
            optimizer_step_tokens = reduce_token_count(accumulated_micro_tokens, device, distributed)
            train_tokens_seen += optimizer_step_tokens
            accumulated_micro_tokens = 0

            if global_step % cfg.logging.logging_steps == 0:
                avg_loss = running_loss / max(1, cfg.logging.logging_steps * grad_accum)
                lr = scheduler.get_last_lr()[0]
                if distributed.is_main_process:
                    elapsed_seconds = time.monotonic() - train_start_monotonic
                    completed_steps = global_step - start_step
                    remaining_steps = cfg.optimization.max_steps - global_step
                    remaining_seconds = estimate_remaining_seconds(
                        start_step=start_step,
                        current_step=global_step,
                        max_steps=cfg.optimization.max_steps,
                        elapsed_seconds=elapsed_seconds,
                    )
                    estimated_finish_time = estimate_finish_time(remaining_seconds)
                    sec_per_step = elapsed_seconds / max(1, completed_steps)
                    total_train_tokens_seen = prior_stage_train_tokens + train_tokens_seen
                    estimated_total_tokens = estimate_total_train_tokens(cfg, train_loader, distributed)
                    if estimated_total_tokens is not None:
                        estimated_total_tokens += prior_stage_train_tokens
                    print(
                        json.dumps(
                            {
                                "time": datetime.now().isoformat(timespec="seconds"),
                                "step": global_step,
                                "max_steps": cfg.optimization.max_steps,
                                "loss": avg_loss,
                                "lr": lr,
                                "elapsed": format_duration_compact(elapsed_seconds),
                                "sec_per_step": round(sec_per_step, 4),
                                "remaining_steps": remaining_steps,
                                "eta": format_duration_compact(remaining_seconds),
                                "eta_time": estimated_finish_time,
                                "step_train_tokens": format_count(optimizer_step_tokens),
                                "train_tokens_seen": format_count(train_tokens_seen),
                                "prior_stage_train_tokens": format_count(prior_stage_train_tokens),
                                "total_train_tokens_seen": format_count(total_train_tokens_seen),
                                "estimated_total_train_tokens": format_count_or_none(estimated_total_tokens),
                            }
                        )
                    )
                running_loss = 0.0

            if eval_loader is not None and eval_steps > 0 and global_step % eval_steps == 0:
                metrics = evaluate(model, eval_loader, device, autocast_context, distributed)
                if distributed.is_main_process:
                    print(json.dumps({"step": global_step, "eval_loss": metrics["eval_loss"]}))

            if global_step % cfg.checkpointing.save_steps == 0:
                if distributed.is_main_process:
                    save_checkpoint(
                        output_dir,
                        model,
                        tokenizer,
                        optimizer,
                        scheduler,
                        global_step,
                        cfg,
                        train_tokens_seen=train_tokens_seen,
                        prior_stage_train_tokens=prior_stage_train_tokens,
                    )
                distributed_barrier(distributed)
                last_checkpoint_step = global_step

            if global_step >= cfg.optimization.max_steps:
                break

    if last_checkpoint_step != global_step:
        if distributed.is_main_process:
            save_checkpoint(
                output_dir,
                model,
                tokenizer,
                optimizer,
                scheduler,
                global_step,
                cfg,
                train_tokens_seen=train_tokens_seen,
                prior_stage_train_tokens=prior_stage_train_tokens,
            )
        distributed_barrier(distributed)
    if distributed.is_main_process:
        print(
            json.dumps(
                {
                    "training_complete": True,
                    "global_step": global_step,
                    "train_tokens_seen": format_count(train_tokens_seen),
                    "prior_stage_train_tokens": format_count(prior_stage_train_tokens),
                    "total_train_tokens_seen": format_count(prior_stage_train_tokens + train_tokens_seen),
                }
            )
        )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    autocast_context: Any,
    distributed: DistributedContext | None = None,
) -> dict[str, float]:
    distributed = distributed or DistributedContext()
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
    if distributed.enabled:
        totals = torch.tensor([total_loss, float(total_batches)], device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss = float(totals[0].detach().cpu())
        total_batches = int(totals[1].detach().cpu())
    return {"eval_loss": total_loss / max(1, total_batches)}


def dry_run_batch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_context: Any,
    distributed: DistributedContext | None = None,
) -> None:
    distributed = distributed or DistributedContext()
    model.eval()
    batch = move_batch_to_device(next(iter(loader)), device)
    with torch.no_grad(), autocast_context():
        outputs = model(**model_batch_kwargs(batch))
    if distributed.is_main_process:
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
    distributed: DistributedContext | None = None,
) -> DataLoader:
    distributed = distributed or DistributedContext()
    dataset = MSAJsonlDataset(jsonl_path, max_samples=max_samples)
    sampler = None
    if distributed.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=collator,
        sampler=sampler,
        generator=generator if shuffle and sampler is None else None,
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


def count_batch_tokens(batch: dict[str, Any]) -> int:
    attention_mask = batch.get("attention_mask")
    if torch.is_tensor(attention_mask):
        return int(attention_mask.detach().sum().item())
    input_ids = batch.get("input_ids")
    if torch.is_tensor(input_ids):
        return int(input_ids.numel())
    return 0


def reduce_token_count(local_count: int, device: torch.device, distributed: DistributedContext) -> int:
    if not distributed.enabled:
        return int(local_count)
    token_tensor = torch.tensor([int(local_count)], device=device, dtype=torch.long)
    dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
    return int(token_tensor.item())


def estimate_total_train_tokens(
    cfg: TrainConfig,
    train_loader: DataLoader,
    distributed: DistributedContext,
) -> int | None:
    max_seq_len = getattr(getattr(cfg, "sequence", None), "max_seq_len", None)
    if max_seq_len is None:
        return None
    per_device_batch = getattr(
        cfg.optimization,
        "per_device_train_batch_size",
        getattr(train_loader, "batch_size", None) or 1,
    )
    return int(
        distributed.world_size
        * per_device_batch
        * cfg.optimization.gradient_accumulation_steps
        * max_seq_len
        * cfg.optimization.max_steps
    )


def estimate_remaining_seconds(
    *,
    start_step: int,
    current_step: int,
    max_steps: int,
    elapsed_seconds: float,
) -> float | None:
    completed_steps = current_step - start_step
    remaining_steps = max_steps - current_step
    if completed_steps <= 0 or remaining_steps <= 0:
        return 0.0 if remaining_steps <= 0 else None
    return elapsed_seconds * remaining_steps / completed_steps


def format_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    total_seconds = max(0, int(round(seconds)))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration_compact(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    total_seconds = max(0, int(round(seconds)))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours:02d}h" if days else f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes:02d}m" if (days or hours) else f"{minutes}m")
    parts.append(f"{secs:02d}s" if parts else f"{secs}s")
    return " ".join(parts)


def format_count(value: int) -> str:
    return f"{int(value):,}"


def format_count_or_none(value: int | None) -> str | None:
    if value is None:
        return None
    return format_count(value)


def estimate_finish_time(remaining_seconds: float | None) -> str | None:
    if remaining_seconds is None:
        return None
    return (datetime.now() + timedelta(seconds=max(0.0, remaining_seconds))).isoformat(timespec="seconds")


def make_lr_schedule(*, scheduler_type: str, warmup_steps: int, max_steps: int):
    normalized = scheduler_type.lower()
    if normalized == "cosine":
        return make_cosine_schedule(warmup_steps=warmup_steps, max_steps=max_steps)
    if normalized == "constant":
        return make_constant_schedule(warmup_steps=warmup_steps)
    raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}")


def make_constant_schedule(*, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    return lr_lambda


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
    train_tokens_seen: int = 0,
    prior_stage_train_tokens: int = 0,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(checkpoint_dir)
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
    write_token_accounting(
        checkpoint_dir,
        global_step=global_step,
        train_tokens_seen=train_tokens_seen,
        prior_stage_train_tokens=prior_stage_train_tokens,
    )
    last_dir = output_dir / "last"
    if last_dir.exists():
        shutil.rmtree(last_dir)
    shutil.copytree(checkpoint_dir, last_dir)
    prune_checkpoints(output_dir, cfg.checkpointing.save_total_limit)
    print(
        json.dumps(
            {
                "saved_checkpoint": str(checkpoint_dir),
                "train_tokens_seen": format_count(train_tokens_seen),
                "prior_stage_train_tokens": format_count(prior_stage_train_tokens),
                "total_train_tokens_seen": format_count(int(prior_stage_train_tokens) + int(train_tokens_seen)),
            }
        )
    )


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


def write_token_accounting(
    checkpoint_dir: Path,
    *,
    global_step: int,
    train_tokens_seen: int,
    prior_stage_train_tokens: int = 0,
) -> None:
    total_train_tokens_seen = int(prior_stage_train_tokens) + int(train_tokens_seen)
    write_json(
        checkpoint_dir / TOKEN_ACCOUNTING_FILE,
        {
            "global_step": int(global_step),
            "train_tokens_seen": int(train_tokens_seen),
            "prior_stage_train_tokens": int(prior_stage_train_tokens),
            "total_train_tokens_seen": total_train_tokens_seen,
        },
    )


def load_token_accounting(checkpoint_dir: Path) -> int:
    path = checkpoint_dir / TOKEN_ACCOUNTING_FILE
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(json.dumps({"warning": f"Could not parse {path}; train token count resumes from 0"}))
        return 0
    return int(payload.get("train_tokens_seen", 0))


def detect_prior_stage_train_tokens(cfg: TrainConfig, resume_dir: Path | None) -> int:
    model_path = Path(cfg.model.model_path)
    if resume_dir is not None and same_path(model_path, resume_dir):
        return 0
    model_path_text = str(model_path).lower()
    run_name = getattr(cfg.run, "name", "").lower()
    if "main" not in run_name and "warmup" not in model_path_text:
        return 0
    return load_token_accounting(model_path)


def same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


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
        "lmloss_weight": cfg.losses.lm_loss_weight,
        "auxloss_weight": cfg.losses.aux_loss_weight,
        "ansloss_weight": cfg.losses.answer_loss_weight,
        "recloss_weight": cfg.losses.reconstruction_loss_weight,
    }
    for name, value in assignments.items():
        if hasattr(model, name):
            setattr(model, name, value)


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    if not hasattr(model, "gradient_checkpointing_enable"):
        return
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
        print(
            json.dumps(
                {
                    "warning": (
                        "gradient_checkpointing_enable does not accept "
                        "gradient_checkpointing_kwargs; DDP may require "
                        "disabling gradient_checkpointing for this transformers version"
                    )
                }
            )
        )


def resolve_device(requested: str, distributed: DistributedContext | None = None) -> torch.device:
    distributed = distributed or DistributedContext()
    if distributed.enabled:
        if requested == "cpu" or requested == "mps":
            raise RuntimeError("Distributed training currently requires CUDA devices")
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun distributed training was requested but CUDA is not available")
        torch.cuda.set_device(distributed.local_rank)
        return torch.device("cuda", distributed.local_rank)
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


class TeeStream:
    def __init__(self, primary: Any, log_file: Any):
        self.primary = primary
        self.log_file = log_file

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.primary.fileno()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)


def setup_print_logging(log_dir: Path, distributed: DistributedContext) -> Path:
    log_dir = log_dir.expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = make_log_timestamp(distributed)
    log_path = log_dir / f"{timestamp}.txt"

    log_fh = log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_fh)
    sys.stderr = TeeStream(original_stderr, log_fh)

    def close_log() -> None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fh.close()

    atexit.register(close_log)
    print(
        json.dumps(
            {
                "training_log_started": timestamp,
                "rank": distributed.rank,
                "world_size": distributed.world_size,
                "log_file": str(log_path),
            }
        )
    )
    return log_path


def make_log_timestamp(distributed: DistributedContext) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") if distributed.is_main_process else None
    if distributed.enabled:
        timestamp_holder = [timestamp]
        dist.broadcast_object_list(timestamp_holder, src=0)
        timestamp = timestamp_holder[0]
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return timestamp


def init_distributed(args: argparse.Namespace | None = None) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args and args.local_rank is not None else 0))
    backend = args.distributed_backend if args is not None else "nccl"
    timeout_minutes = args.distributed_timeout_minutes if args is not None else 60
    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA; launch without torchrun for CPU training")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(minutes=timeout_minutes),
    )
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
    )


def cleanup_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(distributed: DistributedContext) -> None:
    if distributed.enabled:
        dist.barrier()


def wrap_distributed_model(model: torch.nn.Module, device: torch.device) -> DistributedDataParallel:
    if device.type != "cuda":
        raise RuntimeError("DistributedDataParallel wrapping requires a CUDA device")
    kwargs = {
        "device_ids": [device.index],
        "output_device": device.index,
        "find_unused_parameters": False,
    }
    try:
        return DistributedDataParallel(model, static_graph=True, **kwargs)
    except TypeError:
        ddp_model = DistributedDataParallel(model, **kwargs)
        if hasattr(ddp_model, "_set_static_graph"):
            ddp_model._set_static_graph()
        return ddp_model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
