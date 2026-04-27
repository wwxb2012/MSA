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
