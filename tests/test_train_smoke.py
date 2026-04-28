"""Smoke tests for the minimal MSA training loop.

These tests use a tiny local model with a ``from_pretrained``/``save_pretrained``
API so they do not require downloading MSA weights. They verify the train-time
contracts around forward/backward, checkpointing, resume state, and inference
style checkpoint loading.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - exercised only in missing-dep envs
    torch = None


@unittest.skipIf(torch is None, "torch is required for training smoke tests")
class TrainingSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        from src.training.collator import MSACollatorConfig, MSATrainingCollator

        self.tokenizer = ToyTokenizer()
        self.collator = MSATrainingCollator(
            self.tokenizer,
            MSACollatorConfig(
                max_seq_len=128,
                max_documents_per_sample=4,
                max_document_tokens=16,
                max_query_tokens=32,
                max_answer_tokens=8,
            ),
        )
        self.samples = [make_sample("s1"), make_sample("s2")]
        self.batch = self.collator(self.samples)
        self.model = TinyMSAForCausalLM(vocab_size=len(self.tokenizer.vocab) + 16)

    def test_forward_and_backward_pass_work(self) -> None:
        import src.train as train_module

        outputs = self.model(**train_module.model_batch_kwargs(self.batch))
        self.assertIsNotNone(outputs.loss)
        self.assertTrue(torch.isfinite(outputs.loss))

        outputs.loss.backward()
        grad_norm = sum(
            param.grad.detach().abs().sum().item()
            for param in self.model.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_training_loop_runs_eval_and_checkpoint(self) -> None:
        import src.train as train_module

        cfg = make_train_cfg(max_steps=1, save_steps=1)
        loader = DataLoader(self.samples, batch_size=2, collate_fn=self.collator)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            train_module.make_cosine_schedule(warmup_steps=0, max_steps=1),
        )

        with tempfile.TemporaryDirectory() as tmp:
            train_module.train(
                cfg=cfg,
                model=self.model,
                train_loader=loader,
                eval_loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=torch.device("cpu"),
                autocast_context=train_module.nullcontext,
                output_dir=Path(tmp),
                tokenizer=self.tokenizer,
                start_step=0,
                eval_steps=1,
            )
            self.assertTrue((Path(tmp) / "checkpoint-1" / train_module.TRAIN_STATE_FILE).exists())
            self.assertTrue((Path(tmp) / "last" / train_module.TRAIN_STATE_FILE).exists())

    def test_checkpoint_save_and_load_are_consistent(self) -> None:
        import src.train as train_module

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            train_module.make_cosine_schedule(warmup_steps=0, max_steps=2),
        )
        outputs = self.model(**train_module.model_batch_kwargs(self.batch))
        outputs.loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        cfg = make_train_cfg(max_steps=2, save_steps=1)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            train_module.save_checkpoint(
                output_dir,
                self.model,
                self.tokenizer,
                optimizer,
                scheduler,
                global_step=2,
                cfg=cfg,
            )
            checkpoint_dir = output_dir / "checkpoint-2"

            reloaded = TinyMSAForCausalLM.from_pretrained(checkpoint_dir)
            for name, tensor in self.model.state_dict().items():
                self.assertTrue(torch.allclose(tensor, reloaded.state_dict()[name]))

            new_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
            new_scheduler = torch.optim.lr_scheduler.LambdaLR(
                new_optimizer,
                train_module.make_cosine_schedule(warmup_steps=0, max_steps=2),
            )
            step = train_module.load_training_state(checkpoint_dir, new_optimizer, new_scheduler)
            self.assertEqual(step, 2)
            self.assertEqual(new_scheduler.state_dict()["last_epoch"], scheduler.state_dict()["last_epoch"])

    def test_saved_checkpoint_loads_for_inference_style_forward(self) -> None:
        import src.train as train_module

        cfg = make_train_cfg(max_steps=1, save_steps=1)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            train_module.make_cosine_schedule(warmup_steps=0, max_steps=1),
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            train_module.save_checkpoint(
                output_dir,
                self.model,
                self.tokenizer,
                optimizer,
                scheduler,
                global_step=1,
                cfg=cfg,
            )
            checkpoint_dir = output_dir / "checkpoint-1"

            inference_model = TinyMSAForCausalLM.from_pretrained(checkpoint_dir)
            inference_model.eval()
            with torch.no_grad():
                outputs = inference_model(
                    input_ids=self.batch["input_ids"],
                    attention_mask=self.batch["attention_mask"],
                    position_ids=self.batch["position_ids"],
                    doc_ids=self.batch["doc_ids"],
                    use_cache=False,
                )
            self.assertEqual(outputs.logits.shape[:2], self.batch["input_ids"].shape)


class ToyTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.vocab = {self.pad_token: self.pad_token_id, self.eos_token: self.eos_token_id}

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        tokens = text.split()
        ids = []
        for token in tokens:
            self.vocab.setdefault(token, len(self.vocab))
            ids.append(self.vocab[token])
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text(
            json.dumps({"pad_token": self.pad_token, "eos_token": self.eos_token}),
            encoding="utf-8",
        )


if torch is not None:

    class TinyMSAForCausalLM(torch.nn.Module):
        def __init__(self, vocab_size: int = 128, hidden_size: int = 16) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.embed = torch.nn.Embedding(vocab_size, hidden_size)
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            doc_ids: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            **kwargs,
        ) -> SimpleNamespace:
            del attention_mask, position_ids, doc_ids, kwargs
            logits = self.lm_head(self.embed(input_ids))
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            return SimpleNamespace(loss=loss, logits=logits)

        def save_pretrained(self, path: str | Path) -> None:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), path / "pytorch_model.bin")
            (path / "tiny_config.json").write_text(
                json.dumps({"vocab_size": self.vocab_size, "hidden_size": self.hidden_size}),
                encoding="utf-8",
            )

        @classmethod
        def from_pretrained(cls, path: str | Path, **kwargs) -> "TinyMSAForCausalLM":
            del kwargs
            path = Path(path)
            cfg = json.loads((path / "tiny_config.json").read_text(encoding="utf-8"))
            model = cls(**cfg)
            try:
                state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state)
            return model
else:

    class TinyMSAForCausalLM:
        pass


def make_sample(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "task_type": "qa",
        "documents": [
            {"doc_id": 10, "text": "positive document"},
            {"doc_id": 20, "text": "negative document"},
        ],
        "query": "Which document is positive?",
        "answer": "The positive document.",
        "relevant_doc_ids": [10],
        "hard_negative_doc_ids": [20],
        "train_qa_sample": True,
    }


def make_train_cfg(max_steps: int, save_steps: int) -> SimpleNamespace:
    return SimpleNamespace(
        optimization=SimpleNamespace(
            gradient_accumulation_steps=1,
            max_steps=max_steps,
            max_grad_norm=1.0,
        ),
        logging=SimpleNamespace(logging_steps=1),
        checkpointing=SimpleNamespace(save_steps=save_steps, save_total_limit=2),
    )


if __name__ == "__main__":
    unittest.main()
