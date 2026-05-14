# Training Configs

This directory contains minimal training-side configuration files. They are not
used by inference scripts.

The parser utility lives at `src/config/train_config.py` and currently supports
JSON configs plus optional dotted overrides:

```bash
python -m src.config.train_config configs/train/smoke.json \
  --override run.seed=123 \
  --override optimization.learning_rate=1e-5
```

The default examples are intentionally small and conservative:

- `smoke.json`: tiny local smoke-test shape for validating data loading and a
  short training loop once one exists.
- `minimal_pretrain.json`: a minimal continued-pretraining shape for the
  conservative MSA JSONL data produced by
  `scripts/convert_public_sources_to_msa_jsonl.py`.

