# MSA Training Data Spec (Minimal JSONL)

This document defines a **minimal JSONL schema** for MSA training samples that supports:
- multi-document memory inputs,
- query/answer supervision,
- document-wise position reset,
- routing supervision (optional but recommended).

One line = one JSON object = one training sample.

---

## 1) File format

- Encoding: UTF-8
- Extension: `.jsonl`
- One valid JSON object per line
- No trailing commas

---

## 2) Minimal sample schema

```json
{
  "sample_id": "string",
  "documents": [
    {
      "doc_id": 1,
      "text": "string"
    }
  ],
  "query": "string",
  "answer": "string",
  "relevant_doc_ids": [1],
  "task_type": "qa"
}
```

### Required top-level fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `sample_id` | string | yes | Unique sample key for dedup/debug. |
| `documents` | array[object] | yes | Memory documents available to this sample. |
| `query` | string | yes | User question/query text. |
| `answer` | string | yes | Target answer text (for LM/answer loss). |
| `task_type` | string | yes | Minimal values: `"qa"` or `"pretrain"`. |

### Required `documents[*]` fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `doc_id` | int | yes | Positive, unique **within sample**; used as routing/memory identity. |
| `text` | string | yes | Raw document content. |

---

## 3) Routing and auxiliary-supervision fields

These are optional for plain LM, but needed to reproduce MSA auxiliary routing behavior.

| Field | Type | Required | Description |
|---|---|---:|---|
| `relevant_doc_ids` | array[int] | recommended | Positive doc IDs considered relevant for query routing supervision (`batch_aux_labels`). |
| `hard_negative_doc_ids` | array[int] | optional | Explicit negatives for routing experiments. |
| `train_qa_sample` | bool | optional | If true, include sample in answer-specific loss mask (`train_qa_samples`). |

**Constraints**
- Every ID in `relevant_doc_ids`/`hard_negative_doc_ids` must exist in `documents[*].doc_id`.
- `relevant_doc_ids` and `hard_negative_doc_ids` should be disjoint.

---

## 4) Position-reset and routing-ready layout requirements

To support MSA internals, data builders must construct token-level tensors with this semantic mapping:

- `doc_ids == -2`: template-prefix tokens
- `doc_ids == 0`: query/response tokens (routing query region)
- `doc_ids > 0`: memory document tokens (`doc_id` identity)

### Document-wise position reset

For each document segment (`doc_ids > 0`), `position_ids` should restart from 0 at the start of that document segment.

### Query/answer region

The query/answer span should be represented in the `doc_ids == 0` region so routing computes query-to-document relevance from this region.

---

## 5) Recommended extended fields (future-proof)

| Field | Type | Why useful |
|---|---|---|
| `metadata` | object | Source attribution, split, language, difficulty. |
| `answer_aliases` | array[string] | Alternative valid targets for evaluation. |
| `context_order` | array[int] | Explicit canonical order of docs for deterministic packing. |
| `max_generation_tokens` | int | Per-sample decoding cap for eval-time parity. |

---

## 6) Canonical training-object mapping

Given one JSONL sample, preprocessing should produce:

- `input_ids`, `attention_mask`, `position_ids`, `doc_ids`
- `labels` (LM)
- optional `batch_answer_labels`
- optional routing labels derived from `relevant_doc_ids` -> `batch_aux_labels`

### Minimal mapping guidance

1. Build prompt as: template prefix + serialized documents + query + answer target.
2. Assign token `doc_ids` by region:
   - template prefix -> `-2`
   - each serialized document -> that document’s positive `doc_id`
   - query/answer/user-assistant control text -> `0`
3. Build `position_ids`:
   - reset to 0 at each document start,
   - use contiguous positions for query/answer region.
4. Build routing labels:
   - binary/multi-hot over available docs using `relevant_doc_ids`.

---

## 7) JSONL example (multi-document QA)

```json
{"sample_id":"qa_000001","task_type":"qa","documents":[{"doc_id":1,"text":"Marie Curie won Nobel Prizes in Physics and Chemistry."},{"doc_id":2,"text":"Albert Einstein developed the theory of relativity."},{"doc_id":3,"text":"Curie pioneered research on radioactivity."}],"query":"Which scientist in the context won Nobel Prizes in two different sciences?","answer":"Marie Curie.","relevant_doc_ids":[1,3],"hard_negative_doc_ids":[2],"train_qa_sample":true,"metadata":{"split":"train","source":"synthetic_demo","language":"en"}}
```

---

## 8) Validation checklist

A loader should reject samples when:

1. `documents` is empty.
2. duplicate `doc_id` exists in one sample.
3. any `doc_id <= 0` in `documents`.
4. missing/empty `query` or `answer`.
5. any routing supervision ID is absent from `documents`.
6. tokenizer output exceeds configured max sequence without truncation policy.

---

## 9) Minimal interoperability contract

Any training pipeline claiming compatibility with this spec must guarantee:

- it can parse this JSONL schema,
- it materializes MSA-compatible `doc_ids` and `position_ids` semantics,
- it supports QA-only LM training with just required fields,
- it can optionally enable routing/aux losses when `relevant_doc_ids` is provided.
