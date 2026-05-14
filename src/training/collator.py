"""Training collator for MSA JSONL samples.

The collator mirrors the repository's inference-time layout:

- template-prefix tokens use ``doc_ids == -2``;
- document tokens use positive ``doc_ids`` and document-wise ``position_ids``
  starting from 0;
- routing query tokens use ``doc_ids == 0``;
- answer/response tokens use ``doc_ids == -1``;
- active context positions start at ``template_id_num + doc_top_k``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.training.dataset import NON_ROUTING_DOC_ID, QUERY_DOC_ID, TEMPLATE_DOC_ID, validate_sample


IGNORE_INDEX = -100
DEFAULT_PROMPT_TEMPLATE = {
    "prompt": "<|im_start|>user\n{prompt}<|im_end|>\n",
    "response": "<|im_start|>assistant\n<think>\n{think_content}\n</think>\n\n{output}<|im_end|>",
}


@dataclass(frozen=True)
class MSACollatorConfig:
    max_seq_len: int = 8192
    max_documents_per_sample: int = 16
    max_document_tokens: int = 512
    max_query_tokens: int = 512
    max_answer_tokens: int = 64
    doc_top_k: int = 16
    template_id_num: int = 3
    pad_to_multiple_of: int | None = None
    answer_suffix: str = "<|im_end|>"


class MSATrainingCollator:
    """Tokenize and batch normalized MSA JSONL samples.

    This class intentionally avoids importing model code. It returns the tensor
    fields consumed by ``MSAForCausalLM.forward`` plus list-based
    ``batch_aux_labels`` for the repository's current auxiliary-loss path.
    """

    def __init__(
        self,
        tokenizer: Any,
        config: MSACollatorConfig | None = None,
        *,
        prompt_template: dict[str, str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or MSACollatorConfig()
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.pad_token_id = _resolve_pad_token_id(tokenizer)
        self.template_head_ids, self.template_tail_ids = self._split_prompt_template()
        self.response_head_ids = self._tokenize("<|im_start|>")

    @property
    def active_context_start(self) -> int:
        return self.config.template_id_num + self.config.doc_top_k

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        encoded = [self.encode_sample(sample) for sample in samples]
        return self._pad_batch(encoded)

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Encode one validated sample without padding."""

        validate_sample(sample)
        documents = self._select_documents(sample)
        relevant_internal_ids = {
            i + 1
            for i, doc in enumerate(documents)
            if doc["source_doc_id"] in set(sample["relevant_doc_ids"])
        }

        question_ids = self._tokenize(self._format_question(sample["query"]), self.config.max_query_tokens)
        answer_text = sample["answer"] + self.config.answer_suffix
        answer_ids = self._tokenize(answer_text, self.config.max_answer_tokens)

        fixed_len = (
            len(self.template_head_ids)
            + len(question_ids)
            + len(self.template_tail_ids)
            + len(self.response_head_ids)
            + len(answer_ids)
        )
        doc_budget = self.config.max_seq_len - fixed_len
        if doc_budget < len(documents):
            raise ValueError(
                "Sample cannot fit max_seq_len even with one token per document: "
                f"sample_id={sample['sample_id']!r}, fixed_len={fixed_len}, "
                f"num_documents={len(documents)}, max_seq_len={self.config.max_seq_len}"
            )
        per_doc_limit = self.config.max_document_tokens
        if documents:
            per_doc_limit = min(per_doc_limit, max(1, doc_budget // len(documents)))

        input_ids: list[int] = []
        attention_mask: list[int] = []
        doc_ids: list[int] = []
        position_ids: list[int] = []
        labels: list[int] = []
        answer_labels: list[int] = []

        def append(
            token_ids: Sequence[int],
            region_doc_id: int,
            positions: Sequence[int],
            *,
            label_ids: Sequence[int] | None = None,
            answer_label_ids: Sequence[int] | None = None,
        ) -> None:
            input_ids.extend(token_ids)
            attention_mask.extend([1] * len(token_ids))
            doc_ids.extend([region_doc_id] * len(token_ids))
            position_ids.extend(positions)
            labels.extend(label_ids if label_ids is not None else [IGNORE_INDEX] * len(token_ids))
            answer_labels.extend(
                answer_label_ids if answer_label_ids is not None else [IGNORE_INDEX] * len(token_ids)
            )

        append(
            self.template_head_ids,
            TEMPLATE_DOC_ID,
            range(len(self.template_head_ids)),
        )

        for doc in documents:
            doc_token_ids = self._tokenize_document(
                doc["text"],
                doc["internal_doc_id"],
                max_tokens=per_doc_limit,
            )
            append(
                doc_token_ids,
                doc["internal_doc_id"],
                range(len(doc_token_ids)),
            )

        active_pos = self.active_context_start

        append(
            question_ids,
            QUERY_DOC_ID,
            range(active_pos, active_pos + len(question_ids)),
        )
        active_pos += len(question_ids)

        append(
            self.template_tail_ids,
            TEMPLATE_DOC_ID,
            range(active_pos, active_pos + len(self.template_tail_ids)),
        )
        active_pos += len(self.template_tail_ids)

        append(
            self.response_head_ids,
            NON_ROUTING_DOC_ID,
            range(active_pos, active_pos + len(self.response_head_ids)),
        )
        active_pos += len(self.response_head_ids)

        append(
            answer_ids,
            NON_ROUTING_DOC_ID,
            range(active_pos, active_pos + len(answer_ids)),
            label_ids=answer_ids,
            answer_label_ids=answer_ids if sample["train_qa_sample"] else None,
        )

        if len(input_ids) > self.config.max_seq_len:
            raise ValueError(
                f"Encoded sample exceeds max_seq_len after budgeting: "
                f"sample_id={sample['sample_id']!r}, length={len(input_ids)}, "
                f"max_seq_len={self.config.max_seq_len}"
            )

        batch_aux_labels = [
            1 if doc["internal_doc_id"] in relevant_internal_ids else 0
            for doc in documents
        ]

        return {
            "sample_id": sample["sample_id"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "doc_ids": doc_ids,
            "position_ids": position_ids,
            "labels": labels,
            "batch_answer_labels": answer_labels,
            "batch_aux_labels": batch_aux_labels,
            "train_qa_sample": bool(sample["train_qa_sample"]),
        }

    def _pad_batch(self, encoded: Sequence[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(item["input_ids"]) for item in encoded)
        if self.config.pad_to_multiple_of:
            multiple = self.config.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        def pad(values: Sequence[int], pad_value: int) -> list[int]:
            return list(values) + [pad_value] * (max_len - len(values))

        return {
            "input_ids": torch.tensor([pad(item["input_ids"], self.pad_token_id) for item in encoded], dtype=torch.long),
            "attention_mask": torch.tensor([pad(item["attention_mask"], 0) for item in encoded], dtype=torch.long),
            "doc_ids": torch.tensor([pad(item["doc_ids"], QUERY_DOC_ID) for item in encoded], dtype=torch.long),
            "position_ids": torch.tensor([pad(item["position_ids"], 0) for item in encoded], dtype=torch.long),
            "labels": torch.tensor([pad(item["labels"], IGNORE_INDEX) for item in encoded], dtype=torch.long),
            "batch_answer_labels": torch.tensor(
                [pad(item["batch_answer_labels"], IGNORE_INDEX) for item in encoded],
                dtype=torch.long,
            ),
            "batch_aux_labels": [item["batch_aux_labels"] for item in encoded],
            "train_qa_samples": torch.tensor([item["train_qa_sample"] for item in encoded], dtype=torch.bool),
            "sample_ids": [item["sample_id"] for item in encoded],
        }

    def _select_documents(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        relevant = set(sample["relevant_doc_ids"])
        hard_negative = set(sample["hard_negative_doc_ids"])
        indexed_documents = list(enumerate(sample["documents"]))

        def rank(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
            original_index, doc = item
            doc_id = doc["doc_id"]
            if doc_id in relevant:
                group = 0
            elif doc_id in hard_negative:
                group = 1
            else:
                group = 2
            return group, original_index

        selected = [doc for _, doc in sorted(indexed_documents, key=rank)[: self.config.max_documents_per_sample]]
        return [
            {
                "source_doc_id": doc["doc_id"],
                "internal_doc_id": i + 1,
                "text": doc["text"],
            }
            for i, doc in enumerate(selected)
        ]

    def _split_prompt_template(self) -> tuple[list[int], list[int]]:
        prompt = self.prompt_template["prompt"]
        if "{prompt}" not in prompt:
            raise ValueError("Prompt template must contain a {prompt} placeholder")
        head, tail = prompt.split("{prompt}", 1)
        return self._tokenize(head), self._tokenize(tail)

    def _format_question(self, query: str) -> str:
        return (
            "\nPlease answer the question based on the above historical document information\n\n"
            + query
            + "\nPlease return all documents related to the question\n"
        )

    def _tokenize_document(self, text: str, doc_id: int, *, max_tokens: int) -> list[int]:
        prefix = f"<|im_start|>[{doc_id}]. "
        suffix = f"[{doc_id}]<|im_end|>"
        prefix_ids = self._tokenize(prefix)
        suffix_ids = self._tokenize(suffix)
        content_budget = max_tokens - len(prefix_ids) - len(suffix_ids)
        if content_budget <= 0:
            return (prefix_ids + suffix_ids)[:max_tokens]
        return prefix_ids + self._tokenize(text, content_budget) + suffix_ids

    def _tokenize(self, text: str, max_tokens: int | None = None) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        token_ids = list(encoded["input_ids"])
        if max_tokens is not None:
            token_ids = token_ids[:max_tokens]
        return token_ids


def _resolve_pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
