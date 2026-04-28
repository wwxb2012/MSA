"""Simple JSONL dataset for multi-document MSA QA/pretraining samples.

This module intentionally stops before tokenization. It validates and normalizes
the JSONL schema documented in ``docs/training_data_spec.md`` and exposes text
segments that a later collator can tokenize into ``input_ids``, ``doc_ids`` and
``position_ids``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


TEMPLATE_DOC_ID = -2
NON_ROUTING_DOC_ID = -1
QUERY_DOC_ID = 0


class DatasetFormatError(ValueError):
    """Raised when a JSONL training sample violates the schema."""


@dataclass(frozen=True)
class TextSegment:
    """A text span and its MSA region/document identity."""

    text: str
    doc_id: int
    role: str


def load_jsonl_samples(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield validated samples from a JSONL file."""

    dataset = MSAJsonlDataset(path)
    for i in range(len(dataset)):
        yield dataset[i]


class MSAJsonlDataset:
    """Offset-indexed JSONL dataset for MSA training samples.

    It does not depend on torch so it can be used in preprocessing scripts and
    lightweight validation jobs. ``torch.utils.data.Dataset`` only requires
    ``__len__`` and ``__getitem__``, so this object is compatible with PyTorch
    data loaders without inheriting from torch classes.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        validate: bool = True,
        max_samples: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.validate = validate
        self._offsets = _build_offsets(self.path, max_samples=max_samples)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        with self.path.open("r", encoding="utf-8") as fh:
            fh.seek(self._offsets[index])
            line = fh.readline()
        return parse_jsonl_line(line, line_no=index + 1, validate=self.validate)


def parse_jsonl_line(line: str, *, line_no: int | None = None, validate: bool = True) -> dict[str, Any]:
    """Parse one JSONL line into a normalized sample dict."""

    location = f"line {line_no}" if line_no is not None else "line"
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DatasetFormatError(f"{location}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetFormatError(f"{location}: sample must be a JSON object")
    sample = normalize_sample(raw, location=location)
    if validate:
        validate_sample(sample, location=location)
    return sample


def normalize_sample(raw: dict[str, Any], *, location: str = "sample") -> dict[str, Any]:
    """Normalize optional fields while preserving metadata."""

    sample = dict(raw)
    sample["sample_id"] = _require_str(sample, "sample_id", location)
    sample["task_type"] = _require_str(sample, "task_type", location)
    sample["query"] = _require_str(sample, "query", location)
    sample["answer"] = _require_str(sample, "answer", location)
    sample["documents"] = [_normalize_document(doc, location) for doc in _require_list(sample, "documents", location)]
    sample["relevant_doc_ids"] = _normalize_int_list(sample.get("relevant_doc_ids", []), "relevant_doc_ids", location)
    sample["hard_negative_doc_ids"] = _normalize_int_list(
        sample.get("hard_negative_doc_ids", []),
        "hard_negative_doc_ids",
        location,
    )
    sample["train_qa_sample"] = bool(sample.get("train_qa_sample", sample["task_type"] == "qa"))
    if "metadata" in sample and sample["metadata"] is not None and not isinstance(sample["metadata"], dict):
        raise DatasetFormatError(f"{location}: metadata must be an object when present")
    return sample


def validate_sample(sample: dict[str, Any], *, location: str = "sample") -> None:
    """Validate a normalized sample."""

    if sample["task_type"] not in {"qa", "pretrain"}:
        raise DatasetFormatError(f"{location}: task_type must be 'qa' or 'pretrain'")
    if not sample["query"].strip():
        raise DatasetFormatError(f"{location}: query must be non-empty")
    if not sample["answer"].strip():
        raise DatasetFormatError(f"{location}: answer must be non-empty")
    documents = sample["documents"]
    if not documents:
        raise DatasetFormatError(f"{location}: documents must be non-empty")
    doc_ids = [doc["doc_id"] for doc in documents]
    if any(doc_id <= 0 for doc_id in doc_ids):
        raise DatasetFormatError(f"{location}: document doc_id values must be positive")
    if len(set(doc_ids)) != len(doc_ids):
        raise DatasetFormatError(f"{location}: duplicate document doc_id values")
    if any(not doc["text"].strip() for doc in documents):
        raise DatasetFormatError(f"{location}: document text must be non-empty")

    doc_id_set = set(doc_ids)
    relevant = set(sample["relevant_doc_ids"])
    negatives = set(sample["hard_negative_doc_ids"])
    missing_relevant = sorted(relevant - doc_id_set)
    missing_negative = sorted(negatives - doc_id_set)
    if missing_relevant:
        raise DatasetFormatError(f"{location}: relevant_doc_ids not in documents: {missing_relevant}")
    if missing_negative:
        raise DatasetFormatError(f"{location}: hard_negative_doc_ids not in documents: {missing_negative}")
    overlap = sorted(relevant & negatives)
    if overlap:
        raise DatasetFormatError(f"{location}: relevant and hard-negative IDs overlap: {overlap}")


def sample_to_text_segments(
    sample: dict[str, Any],
    *,
    include_answer: bool = True,
    template_prefix: str | None = None,
    answer_doc_id: int = NON_ROUTING_DOC_ID,
) -> list[TextSegment]:
    """Convert a validated sample into coarse text segments.

    Tokenization later maps each segment's ``doc_id`` to token-level ``doc_ids``:
    template prefix -> -2, answer/response text -> -1, query -> 0, and
    documents -> positive doc IDs.
    """

    validate_sample(sample)
    segments: list[TextSegment] = []
    if template_prefix:
        segments.append(TextSegment(template_prefix, TEMPLATE_DOC_ID, "template"))
    for doc in sample["documents"]:
        segments.append(TextSegment(doc["text"], doc["doc_id"], "document"))
    segments.append(TextSegment(sample["query"], QUERY_DOC_ID, "query"))
    if include_answer:
        segments.append(TextSegment(sample["answer"], answer_doc_id, "answer"))
    return segments


def routing_label_vector(sample: dict[str, Any], *, doc_order: Sequence[int] | None = None) -> list[int]:
    """Return a multi-hot routing label vector over document IDs."""

    validate_sample(sample)
    order = list(doc_order) if doc_order is not None else [doc["doc_id"] for doc in sample["documents"]]
    relevant = set(sample["relevant_doc_ids"])
    missing = sorted(set(order) - {doc["doc_id"] for doc in sample["documents"]})
    if missing:
        raise DatasetFormatError(f"doc_order contains IDs absent from sample documents: {missing}")
    return [1 if doc_id in relevant else 0 for doc_id in order]


def _build_offsets(path: Path, *, max_samples: int | None = None) -> list[int]:
    offsets: list[int] = []
    with path.open("r", encoding="utf-8") as fh:
        while True:
            offset = fh.tell()
            line = fh.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
                if max_samples is not None and len(offsets) >= max_samples:
                    break
    return offsets


def _normalize_document(doc: Any, location: str) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise DatasetFormatError(f"{location}: each document must be an object")
    return {
        "doc_id": _require_int(doc, "doc_id", location),
        "text": _require_str(doc, "text", location),
    }


def _require_str(obj: dict[str, Any], key: str, location: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str):
        raise DatasetFormatError(f"{location}: {key} must be a string")
    return value


def _require_int(obj: dict[str, Any], key: str, location: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DatasetFormatError(f"{location}: {key} must be an integer")
    return value


def _require_list(obj: dict[str, Any], key: str, location: str) -> list[Any]:
    value = obj.get(key)
    if not isinstance(value, list):
        raise DatasetFormatError(f"{location}: {key} must be a list")
    return value


def _normalize_int_list(value: Any, key: str, location: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise DatasetFormatError(f"{location}: {key} must be a list")
    out: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise DatasetFormatError(f"{location}: {key} must contain only integers")
        out.append(item)
    return out
