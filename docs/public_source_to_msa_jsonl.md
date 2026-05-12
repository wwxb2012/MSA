# Public Source to MSA JSONL Converter

`scripts/convert_public_sources_to_msa_jsonl.py` converts downloaded public retrieval-style parquet files into the minimal MSA JSONL schema.

The converter is intentionally conservative:

- It requires `query` and explicit positive documents.
- It requires explicit negatives by default.
- It writes `relevant_doc_ids` and `hard_negative_doc_ids` for routing supervision.
- It does not invent answers from unrelated fields.
- It records source provenance and approximate paper-source mapping in `metadata`.

Default output:

```bash
.venv/bin/python scripts/convert_public_sources_to_msa_jsonl.py
```

This reads from `training_data_public_sources/` and writes:

- `converted_training_data/msa_pretrain_conservative.jsonl`
- `converted_training_data/msa_pretrain_conservative_manifest.json`
- `converted_training_data/msa_pretrain_conservative_skipped.json`

Smoke test:

```bash
.venv/bin/python scripts/convert_public_sources_to_msa_jsonl.py \
  --max-files 2 \
  --max-total-samples 10 \
  --output-jsonl converted_training_data/smoke.jsonl \
  --manifest converted_training_data/smoke_manifest.json \
  --skipped-report converted_training_data/smoke_skipped.json
```

Rows without negatives are skipped unless `--allow-missing-negatives` is set. This means many KaLM pretrain rows are skipped in default conservative mode because they contain `query` and `pos` but no explicit `neg`.

Use `--sample-rate` and `--seed` for deterministic downsampling, and `--max-samples-per-source` to cap each normalized source. The default source cap is `500000`, mirroring the paper's non-KALM cap.

Expanded candidate-document mode:

```bash
.venv/bin/python scripts/convert_public_sources_to_msa_jsonl.py \
  --num-workers 32 \
  --target-documents-per-sample 64 \
  --require-target-documents \
  --supplemental-negatives \
  --supplemental-negative-pool-size 100000 \
  --allow-missing-negatives-with-supplemental \
  --max-negatives 16 \
  --output-jsonl converted_training_data/msa_pretrain_64docs.jsonl \
  --manifest converted_training_data/msa_pretrain_64docs_manifest.json \
  --skipped-report converted_training_data/msa_pretrain_64docs_skipped.json
```

This keeps the explicit positives/negatives from each row, then fills remaining
candidate slots with deterministic same-source negatives sampled from documents
seen earlier in the same normalized source. The supplemental negatives are
recorded in `hard_negative_doc_ids`, and per-sample metadata records both
`explicit_negative_count` and `supplemental_negative_count`.
Use `--require-target-documents` when generating long-context training data so
early rows without enough same-source candidates are skipped instead of producing
short samples.
Use `--num-workers` to parallelize parquet scanning/conversion across CPU cores;
expanded mode performs a parallel candidate-pool pass before writing JSONL shards.

## Paper-Aligned CPT Approximation

The paper reports a continuous-pretraining corpus of 17,852,825 queries and
158.95B tokens. It does not publish an exact public manifest, so this converter
builds a public proxy dataset rather than an official reproduction of the
authors' data.

For the closest public-source approximation currently supported, use expanded
candidate-document mode:

```bash
python scripts/convert_public_sources_to_msa_jsonl.py \
  --input-root training_data_public_sources \
  --output-jsonl converted_training_data/msa_pretrain_64docs.jsonl \
  --manifest converted_training_data/msa_pretrain_64docs_manifest.json \
  --skipped-report converted_training_data/msa_pretrain_64docs_skipped.json \
  --num-workers 32 \
  --target-documents-per-sample 64 \
  --require-target-documents \
  --supplemental-negatives \
  --supplemental-negative-pool-size 100000 \
  --allow-missing-negatives-with-supplemental \
  --max-negatives 16
```

Important interpretation notes:

- `--target-documents-per-sample 64` means 64 candidate documents per query; it
  does not by itself mean 64k training tokens.
- `--max-negatives 16` is chosen to align with the paper's top-16 routing
  design, but it is still a proxy for the unpublished training recipe.
- `--require-target-documents` keeps the generated JSONL shape consistent by
  skipping rows that cannot be filled to 64 documents.
- The final training token count is determined by the tokenizer, collator
  truncation, sequence config, world size, batch size, gradient accumulation,
  and max steps. Use the `train.py` token accounting log
  (`train_tokens_seen`, `total_train_tokens_seen`,
  `estimated_total_train_tokens`) to check whether the actual run is near the
  paper's 158.95B-token CPT budget.

If disk space is limited, add `--max-total-samples 1000000` for a first 64-doc
dataset shard. Full 64-doc conversion can be around the terabyte scale and can
require substantially more peak disk when parallel temp shards are present.

Run only one downloaded source repository:

```bash
.venv/bin/python scripts/convert_public_sources_to_msa_jsonl.py \
  --include-repo medi_data_mteb_avs_triplets
```

Run only paths containing a source name:

```bash
.venv/bin/python scripts/convert_public_sources_to_msa_jsonl.py \
  --include-path-substring WikiAnswers
```
