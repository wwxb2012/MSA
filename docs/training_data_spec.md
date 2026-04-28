# MSA Training Data Spec (Minimal JSONL)

This document defines a minimal JSONL schema for multi-document QA and
generative-retrieval training samples.

One line = one JSON object = one training sample.

## Source Formats Observed Locally

Inference benchmark data under `data/*` uses pickle files:

- `qdata_*.pkl`: `list[dict]`
- Query records contain `query`, `answer`, and `reference_list`
- `reference_list` is the positive evidence text list for that query
- `mdata_*.pkl`: `list[str]`
- Memory records are raw document strings

Downloaded public training sources under `training_data_public_sources/*` use
parquet files:

- KaLM fine-tuning data: `query: string`, `pos: list[string]`, `neg: list[string]`
- KaLM pre-training data: `query: string`, `pos: list[string]`, `neg: []`, `relevance: float`
- MEDI/MTEB triplets: `query: string`, `pos: string`, `neg: string`, plus `task_name` and instruction fields

The JSONL format below is a normalized bridge over those shapes.

## File Format

- Encoding: UTF-8
- Extension: `.jsonl`
- One valid JSON object per line
- No trailing commas

## Minimal Sample Schema

```json
{
  "sample_id": "qa_000001",
  "task_type": "qa",
  "documents": [
    {
      "doc_id": 1,
      "text": "Marie Curie won Nobel Prizes in Physics and Chemistry."
    },
    {
      "doc_id": 2,
      "text": "Albert Einstein developed the theory of relativity."
    }
  ],
  "query": "Which scientist won Nobel Prizes in two different sciences?",
  "answer": "Marie Curie.",
  "relevant_doc_ids": [1],
  "hard_negative_doc_ids": [2],
  "train_qa_sample": true,
  "metadata": {
    "source": "synthetic_demo"
  }
}
```

## Required Fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `sample_id` | string | yes | Unique sample key for dedup/debug. |
| `task_type` | string | yes | Minimal values: `qa` or `pretrain`. |
| `documents` | array[object] | yes | Memory documents available to the sample. |
| `documents[*].doc_id` | int | yes | Positive integer, unique within the sample. |
| `documents[*].text` | string | yes | Raw document text. |
| `query` | string | yes | User question/query text. |
| `answer` | string | yes | Target answer text or generated document-id target. |

## Routing and Auxiliary-Supervision Fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `relevant_doc_ids` | array[int] | recommended | Positive docs for routing supervision. |
| `hard_negative_doc_ids` | array[int] | optional | Explicit negatives for routing experiments. |
| `train_qa_sample` | bool | optional | If true, include sample in answer-specific loss masking. |

Constraints:

- Every ID in `relevant_doc_ids` and `hard_negative_doc_ids` must exist in `documents[*].doc_id`.
- `relevant_doc_ids` and `hard_negative_doc_ids` should be disjoint.
- `documents[*].doc_id` must be positive; reserved token-region IDs are not valid document IDs.

## Mapping From Local Source Formats

### Benchmark QA Pickles

For benchmark-style `qdata` records:

- `query` -> `query`
- `answer` -> `answer`
- `reference_list` -> positive `documents`
- `relevant_doc_ids` -> doc IDs assigned to `reference_list`
- `train_qa_sample` -> `true`

Optional hard negatives can be sampled from the matching `mdata` memory corpus,
but this must be done outside the basic JSONL loader.

### Public Retrieval Parquets

For retrieval-style parquet rows:

- `query` -> `query`
- `pos` -> positive `documents`
- `neg` -> hard-negative `documents`
- `answer` -> generated document-id target such as `<doc_1>`
- `relevant_doc_ids` -> doc IDs assigned to `pos`
- `train_qa_sample` -> `false`

This is the format produced by
`scripts/convert_public_sources_to_msa_jsonl.py`.

## Token-Level Layout Contract

The JSONL loader returns structured text; a tokenizer/collator must later
materialize token-level tensors with these region semantics:

- `doc_ids == -2`: template-prefix tokens
- `doc_ids == -1`: active non-routing prompt/answer/response tokens
- `doc_ids == 0`: query tokens used as the routing query region
- `doc_ids > 0`: memory document tokens

For each document segment (`doc_ids > 0`), `position_ids` should restart from
0 at the start of the document. Query tokens should be the only generated
sample text marked as `doc_ids == 0`; answer/response tokens should normally be
marked as `doc_ids == -1` so they remain active context without becoming router
queries.

## Dataset Loader Contract

`src/training/dataset.py` provides a simple JSONL reader/validator. It does not
tokenize. It returns normalized Python dictionaries and can produce text
segments with `doc_id` region labels for a future collator.

The loader rejects samples when:

1. the JSON line is not an object,
2. `documents` is empty,
3. document IDs are duplicated or non-positive,
4. `query` or `answer` is empty,
5. routing IDs are absent from the document set,
6. relevant and hard-negative IDs overlap.

## Recommended Extended Fields

| Field | Type | Why useful |
|---|---|---|
| `metadata` | object | Source attribution, split, language, difficulty. |
| `answer_aliases` | array[string] | Alternative valid targets for evaluation. |
| `context_order` | array[int] | Explicit canonical document order for deterministic packing. |
| `max_generation_tokens` | int | Per-sample decoding cap for eval-time parity. |
