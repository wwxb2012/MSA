"""Minimal training config parser.

This module is deliberately independent from inference configuration. It is not
imported by benchmark or MSA service code.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints


T = TypeVar("T")


@dataclass
class RunConfig:
    name: str = "msa-train"
    seed: int = 42
    output_dir: str = "outputs/train/msa-train"


@dataclass
class TrainModelConfig:
    model_path: str = "ckpt/MSA-4B"
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"


@dataclass
class TrainDataConfig:
    train_jsonl: str = "converted_training_data/msa_pretrain_conservative.jsonl"
    validation_jsonl: str | None = None
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    shuffle: bool = True
    num_workers: int = 4


@dataclass
class SequenceConfig:
    max_seq_len: int = 8192
    max_documents_per_sample: int = 16
    max_document_tokens: int = 512
    max_query_tokens: int = 512
    max_answer_tokens: int = 64
    packing: bool = False


@dataclass
class OptimizationConfig:
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 6e-6
    weight_decay: float = 0.0
    max_steps: int = 1000
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0


@dataclass
class LossConfig:
    lm_loss_weight: float = 1.0
    aux_loss_weight: float = 0.1
    answer_loss_weight: float = 0.0
    reconstruction_loss_weight: float = 0.0


@dataclass
class CheckpointingConfig:
    save_steps: int = 100
    save_total_limit: int | None = 3
    resume_from_checkpoint: str | None = None


@dataclass
class LoggingConfig:
    logging_steps: int = 10
    report_to: list[str] | None = None


@dataclass
class TrainConfig:
    run: RunConfig
    model: TrainModelConfig
    data: TrainDataConfig
    sequence: SequenceConfig
    optimization: OptimizationConfig
    losses: LossConfig
    checkpointing: CheckpointingConfig
    logging: LoggingConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        _require_non_empty("run.name", self.run.name)
        _require_positive_int("run.seed", self.run.seed, allow_zero=True)
        _require_non_empty("run.output_dir", self.run.output_dir)
        _require_non_empty("model.model_path", self.model.model_path)
        _require_non_empty("data.train_jsonl", self.data.train_jsonl)
        _require_positive_int("data.num_workers", self.data.num_workers, allow_zero=True)
        _require_optional_positive_int("data.max_train_samples", self.data.max_train_samples)
        _require_optional_positive_int("data.max_eval_samples", self.data.max_eval_samples)
        _require_positive_int("sequence.max_seq_len", self.sequence.max_seq_len)
        _require_positive_int("sequence.max_documents_per_sample", self.sequence.max_documents_per_sample)
        _require_positive_int("sequence.max_document_tokens", self.sequence.max_document_tokens)
        _require_positive_int("sequence.max_query_tokens", self.sequence.max_query_tokens)
        _require_positive_int("sequence.max_answer_tokens", self.sequence.max_answer_tokens)
        _require_positive_int(
            "optimization.per_device_train_batch_size",
            self.optimization.per_device_train_batch_size,
        )
        _require_positive_int(
            "optimization.gradient_accumulation_steps",
            self.optimization.gradient_accumulation_steps,
        )
        _require_positive_float("optimization.learning_rate", self.optimization.learning_rate)
        _require_non_negative_float("optimization.weight_decay", self.optimization.weight_decay)
        _require_positive_int("optimization.max_steps", self.optimization.max_steps)
        _require_positive_int("optimization.warmup_steps", self.optimization.warmup_steps, allow_zero=True)
        _require_non_negative_float("optimization.max_grad_norm", self.optimization.max_grad_norm)
        for name, value in asdict(self.losses).items():
            _require_non_negative_float(f"losses.{name}", value)
        _require_positive_int("checkpointing.save_steps", self.checkpointing.save_steps)
        _require_optional_positive_int("checkpointing.save_total_limit", self.checkpointing.save_total_limit)
        _require_positive_int("logging.logging_steps", self.logging.logging_steps)


def load_train_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    """Load and validate a training config from JSON.

    Overrides use dotted keys, for example:
    `optimization.learning_rate=1e-5`.
    Values are JSON-decoded when possible, so booleans, numbers, lists, and null
    work naturally.
    """

    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Training config must be a JSON object: {config_path}")
    for override in overrides or []:
        apply_override(raw, override)
    cfg = train_config_from_dict(expand_paths(raw))
    cfg.validate()
    return cfg


def apply_override(raw: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be key=value: {override!r}")
    key, value_text = override.split("=", 1)
    parts = [p for p in key.split(".") if p]
    if not parts:
        raise ValueError(f"Override key is empty: {override!r}")
    value = _parse_override_value(value_text)
    cursor: dict[str, Any] = raw
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            raise KeyError(f"Cannot override nested key through non-object: {key!r}")
        cursor = next_value
    cursor[parts[-1]] = value


def train_config_from_dict(raw: dict[str, Any]) -> TrainConfig:
    return _dataclass_from_dict(TrainConfig, raw)


def expand_paths(raw: dict[str, Any]) -> dict[str, Any]:
    """Expand env vars and ~ in path-like string fields without resolving them."""

    path_keys = {
        "output_dir",
        "model_path",
        "train_jsonl",
        "validation_jsonl",
        "resume_from_checkpoint",
    }

    def visit(obj: Any, key: str | None = None) -> Any:
        if isinstance(obj, dict):
            return {k: visit(v, k) for k, v in obj.items()}
        if isinstance(obj, list):
            return [visit(v, key) for v in obj]
        if isinstance(obj, str) and key in path_keys:
            return os.path.expandvars(os.path.expanduser(obj))
        return obj

    return visit(raw)


def _dataclass_from_dict(cls: type[T], raw: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"Expected dataclass type, got {cls!r}")
    if not isinstance(raw, dict):
        raise TypeError(f"Expected object for {cls.__name__}, got {type(raw).__name__}")
    field_map = {f.name: f for f in fields(cls)}
    type_hints = get_type_hints(cls)
    unknown = sorted(set(raw) - set(field_map))
    if unknown:
        raise KeyError(f"Unknown key(s) for {cls.__name__}: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, field_info in field_map.items():
        if name not in raw:
            raise KeyError(f"Missing required key for {cls.__name__}: {name}")
        value = raw[name]
        field_type = type_hints[name]
        if is_dataclass(field_type):
            kwargs[name] = _dataclass_from_dict(field_type, value)
        else:
            kwargs[name] = _coerce_value(name, value, field_type)
    return cls(**kwargs)


def _coerce_value(name: str, value: Any, expected_type: Any) -> Any:
    origin = get_origin(expected_type)
    args = get_args(expected_type)
    if origin is list:
        if value is None:
            return None
        if not isinstance(value, list):
            raise TypeError(f"{name} must be a list")
        return value
    if origin is None and expected_type in (str, int, float, bool):
        if value is None:
            raise TypeError(f"{name} cannot be null")
        if expected_type is bool and not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
        if expected_type is int and not isinstance(value, int):
            raise TypeError(f"{name} must be an int")
        if expected_type is float and not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if expected_type is str and not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        return expected_type(value)
    if type(None) in args:
        non_null_args = [arg for arg in args if arg is not type(None)]
        if value is None:
            return None
        if len(non_null_args) == 1:
            return _coerce_value(name, value, non_null_args[0])
    return value


def _parse_override_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _require_non_empty(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")


def _require_positive_int(name: str, value: int, allow_zero: bool = False) -> None:
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'>= 0' if allow_zero else '> 0'}")


def _require_optional_positive_int(name: str, value: int | None) -> None:
    if value is not None:
        _require_positive_int(name, value)


def _require_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _require_non_negative_float(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def main() -> int:
    parser = argparse.ArgumentParser(description="Load and validate a training config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--print-json", action="store_true", default=True)
    args = parser.parse_args()
    cfg = load_train_config(args.config, args.override)
    if args.print_json:
        print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
