# MSA Training Guide

This guide covers the current minimal training path added for approximate MSA
reproduction work: public-source data conversion, JSONL validation, config
selection, launch commands, checkpoint handling, and evaluation.

The training path is separate from inference code. Existing benchmark and
service behavior should continue to use the original inference entrypoints.

## 1. Environment

Install the repository dependencies first:

```bash
python -m pip install -r requirements.txt
```

For CPU-only smoke/debug runs, PyTorch is enough. For real MSA fine-tuning you
need a CUDA PyTorch build and enough GPU memory for the selected base model.
Do not call `.cuda()` directly in scripts; use the `--device` option or let
`src/train.py` choose the device.

## 2. Data Preparation

### 2.1 Expected JSONL Format

Training consumes one JSON object per line. The schema is documented in
`docs/training_data_spec.md`.

Minimal example:

```json
{"sample_id":"qa_000001","task_type":"qa","documents":[{"doc_id":1,"text":"Paris is the capital of France."},{"doc_id":2,"text":"Berlin is the capital of Germany."}],"query":"Which city is the capital of France?","answer":"Paris.","relevant_doc_ids":[1],"hard_negative_doc_ids":[2],"train_qa_sample":true}
```

Important constraints:

- `documents[*].doc_id` must be positive and unique within the sample.
- `relevant_doc_ids` and `hard_negative_doc_ids` must refer to existing
  documents.
- Relevant and hard-negative IDs must not overlap.
- `task_type` is currently `qa` or `pretrain`.
- `train_qa_sample=false` is useful for retrieval-style pretraining rows where
  the `answer` is a generated document target such as `<doc_1>`.

The loader is `src/training/dataset.py`. The tokenizer-aware collator is
`src/training/collator.py`.

### 2.2 Convert Public Parquet Sources

If the public parquet sources are already under `training_data_public_sources/`,
convert them into the minimal JSONL format:

```bash
python scripts/convert_public_sources_to_msa_jsonl.py \
  --input-root training_data_public_sources \
  --output-jsonl converted_training_data/msa_pretrain_conservative.jsonl \
  --manifest converted_training_data/msa_pretrain_conservative_manifest.json \
  --skipped-report converted_training_data/msa_pretrain_conservative_skipped.json
```

Useful smaller runs:

```bash
python scripts/convert_public_sources_to_msa_jsonl.py \
  --input-root training_data_public_sources \
  --output-jsonl converted_training_data/smoke.jsonl \
  --manifest converted_training_data/smoke_manifest.json \
  --skipped-report converted_training_data/smoke_skipped.json \
  --max-total-samples 128 \
  --sample-rate 0.01
```

Conservative defaults:

- Requires explicit query and positive text.
- Requires explicit negatives unless `--allow-missing-negatives` is set.
- Caps each normalized source with `--max-samples-per-source`.
- Uses deterministic hash sampling with `--sample-rate` and `--seed`.

### 2.3 Tiny Debug Dataset

For a fully offline pipeline check, run:

```bash
python scripts/debug_tiny_train.py --steps 5 --clean
```

This script writes `outputs/train/debug_tiny/tiny_train.jsonl`, trains a tiny
CPU model for 5 steps through the real training collator and loop, saves
`outputs/train/debug_tiny/checkpoint-5`, and reloads that checkpoint for an
inference-style forward pass.

## 3. Training Configs

Configs live under `configs/train/`.

Available examples:

- `configs/train/smoke.json`: small shape for quick training-loop checks.
- `configs/train/minimal_pretrain.json`: conservative continued-pretraining
  shape for `converted_training_data/msa_pretrain_conservative.jsonl`.

Validate config parsing and overrides:

```bash
python -m src.config.train_config configs/train/minimal_pretrain.json \
  --override run.seed=123 \
  --override optimization.learning_rate=1e-5
```

Key options:

- `run.output_dir`: root for `train_config.resolved.json`, `checkpoint-N/`, and
  `last/`.
- `model.model_path`: MSA-compatible base checkpoint or a previous training
  checkpoint.
- `model.torch_dtype`: `bfloat16`, `float16`, or `float32`.
- `model.attn_implementation`: passed to `MSAForCausalLM.from_pretrained`.
- `data.train_jsonl`: minimal MSA JSONL training file.
- `data.validation_jsonl`: optional JSONL for periodic eval loss.
- `sequence.max_seq_len`: total token budget after document/query/answer
  serialization.
- `sequence.max_documents_per_sample`: maximum documents retained per sample.
- `optimization.gradient_accumulation_steps`: accumulation factor before one
  optimizer step.
- `optimization.max_steps`: optimizer steps, not raw micro-batches.
- `optimization.warmup_steps`: warmup steps for the cosine schedule.
- `losses.*`: copied onto matching model attributes when present.
- `checkpointing.save_steps`: checkpoint interval.
- `checkpointing.resume_from_checkpoint`: optional checkpoint directory.

Override any JSON field from the command line with dotted keys:

```bash
python src/train.py configs/train/minimal_pretrain.json \
  --override optimization.max_steps=200 \
  --override checkpointing.save_steps=50
```

## 4. Checkpoint Conversion

### 4.1 Prepare an MSA-Compatible Base Model

If your base model directory already contains an MSA config and loads with
`MSAForCausalLM.from_pretrained`, use it directly as `model.model_path`.

If you need to convert/resave a base checkpoint with MSA settings, use:

```bash
bash scripts/resave_model.sh /path/to/origin_model /path/to/msa_base_model
```

`scripts/resave_model.sh` sets the MSA hyperparameters through environment
variables and calls `src/utils/resave_model.py`. Important knobs include:

- `TOP_K_DOCS`
- `POOLING_KERNEL_SIZE`
- `ROUTER_LAYER_IDX`
- `REWRITE_POSITION`
- `AUX_LOSS`
- `AUX_LOSS_METHOD`
- `LMLOSS_WEIGHT`
- `AUX_LOSS_WEIGHT`
- `ANS_LOSS_WEIGHT`

After conversion, set:

```json
"model": {
  "model_path": "/path/to/msa_base_model"
}
```

### 4.2 Training Checkpoints

`src/train.py` saves HuggingFace-style checkpoints:

```text
outputs/train/<run-name>/
  train_config.resolved.json
  checkpoint-100/
    config.json
    pytorch_model.bin or model shards
    tokenizer files
    trainer_state.pt
  last/
```

Use `checkpoint-N/` or `last/` directly as `--model_path` for evaluation. The
`trainer_state.pt` file is only for optimizer/scheduler/RNG resume and is
ignored by inference code.

Resume training:

```bash
python src/train.py configs/train/minimal_pretrain.json \
  --resume-from-checkpoint outputs/train/msa-minimal-pretrain/last
```

## 5. Training Launch

### 5.1 Debug Run

```bash
python scripts/debug_tiny_train.py --steps 5 --clean
```

Expected final line:

```json
{"debug_train":"ok","steps":5,"dataset":"outputs/train/debug_tiny/tiny_train.jsonl","checkpoint":"outputs/train/debug_tiny/checkpoint-5","logits_shape":[2,55,70]}
```

The exact loss values can differ by PyTorch version.

### 5.2 Short Smoke Run With Real Training Entry

Prepare a small JSONL first, then run:

```bash
python src/train.py configs/train/smoke.json \
  --device cpu \
  --override data.train_jsonl=converted_training_data/smoke.jsonl \
  --override optimization.max_steps=5 \
  --override checkpointing.save_steps=5 \
  --override model.torch_dtype=float32 \
  --override model.attn_implementation=eager
```

This command still loads the configured MSA model, so it requires
`model.model_path` to point at a valid local MSA checkpoint.

### 5.3 Real Fine-Tuning

Example single-GPU launch:

```bash
python src/train.py configs/train/minimal_pretrain.json \
  --device cuda \
  --override model.model_path=ckpt/MSA-4B \
  --override data.train_jsonl=converted_training_data/msa_pretrain_conservative.jsonl \
  --override run.output_dir=outputs/train/msa-minimal-pretrain \
  --override optimization.max_steps=1000 \
  --override checkpointing.save_steps=100
```

Example single-node multi-GPU launch:

```bash
torchrun --standalone --nproc_per_node=8 src/train.py configs/train/minimal_pretrain.json \
  --device cuda \
  --override model.model_path=ckpt/MSA-4B \
  --override data.train_jsonl=converted_training_data/msa_pretrain_conservative.jsonl \
  --override run.output_dir=outputs/train/msa-minimal-pretrain \
  --override optimization.max_steps=1000 \
  --override checkpointing.save_steps=100
```

Notes:

- Plain `python src/train.py ...` is single-process and will use one CUDA
  device. Use `torchrun` to enable DistributedDataParallel and spread batches
  across multiple GPUs.
- `optimization.per_device_train_batch_size` is per GPU. Effective batch size is
  `per_device_train_batch_size * gradient_accumulation_steps * world_size`.
- Checkpoint saving, config writing, and logging are performed by rank 0 only.
- Use smaller `sequence.*` limits and larger gradient accumulation when memory
  is tight.
- `logging.logging_steps` controls JSON progress lines printed to stdout.
- Eval loss is printed when `data.validation_jsonl` is set and `--eval-steps`
  or `checkpointing.save_steps` triggers it.

## 6. Evaluation

The inference benchmark entrypoint is unchanged. Point `--model_path` to the
trained checkpoint directory:

```bash
python -u src/app/benchmark.py \
  --benchmark hotpotqa \
  --model_path outputs/train/msa-minimal-pretrain/last \
  --top_p 0.9 \
  --temperature 0.0 \
  --max_length 2048 \
  --template QWEN3_INSTRUCT_TEMPLATE \
  --output_file src/evaluation/outputs/hotpotqa_train_eval.json \
  --max_batch_size 16 \
  --max_chunk_per_block 16384 \
  --block_size 2048
```

To run the repository benchmark suite, edit `model_path` in
`scripts/run_benchmarks.sh` or temporarily override it in the script, then run:

```bash
bash scripts/run_benchmarks.sh trained_checkpoint_eval
```

Benchmark data is resolved by `src/benchmarks.py`. If a benchmark pickle is not
already present under `data/`, the code will try to download it from
`EverMind-AI/MSA-RAG-BENCHMARKS` on Hugging Face.

## 7. Recommended End-to-End Flow

1. Convert or prepare JSONL:

```bash
python scripts/convert_public_sources_to_msa_jsonl.py \
  --input-root training_data_public_sources \
  --output-jsonl converted_training_data/msa_pretrain_conservative.jsonl
```

2. Ensure the base model is MSA-compatible:

```bash
bash scripts/resave_model.sh /path/to/origin_model ckpt/MSA-4B
```

3. Run a tiny pipeline check:

```bash
python scripts/debug_tiny_train.py --steps 5 --clean
```

4. Launch training:

```bash
python src/train.py configs/train/minimal_pretrain.json \
  --override model.model_path=ckpt/MSA-4B \
  --override data.train_jsonl=converted_training_data/msa_pretrain_conservative.jsonl
```

5. Evaluate:

```bash
python -u src/app/benchmark.py \
  --benchmark hotpotqa \
  --model_path outputs/train/msa-minimal-pretrain/last \
  --output_file src/evaluation/outputs/hotpotqa_after_training.json \
  --max_batch_size 16 \
  --top_p 0.9 \
  --temperature 0.0 \
  --max_length 2048 \
  --template QWEN3_INSTRUCT_TEMPLATE \
  --max_chunk_per_block 16384 \
  --block_size 2048
```

## 8. Troubleshooting

- `ModuleNotFoundError: pyarrow`: install `pyarrow` before running the converter.
- `FileNotFoundError` for JSONL: check `data.train_jsonl` after env-var and `~`
  expansion.
- `Model returned loss=None`: the collator must provide `labels`; use
  `MSATrainingCollator`.
- CUDA OOM: lower `sequence.max_seq_len`, `sequence.max_documents_per_sample`,
  `sequence.max_document_tokens`, or batch size; increase gradient
  accumulation.
- Inference cannot load a checkpoint: point `--model_path` to `checkpoint-N/` or
  `last/`, not the parent run directory.
- Benchmark data downloads unexpectedly: pre-place benchmark files under
  `data/<bench_name>/` or allow Hugging Face dataset access.
