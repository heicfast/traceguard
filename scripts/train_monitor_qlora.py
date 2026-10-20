#!/usr/bin/env python3
"""Train a secondary TRACEGUARD monitor adapter with T4-safe QLoRA.

This arm is intentionally secondary.  The zero-shot monitor remains primary;
the adapter may be frozen only after development-family training and before
validation/holdout scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    args = parser.parse_args()

    if not args.revision or args.revision.startswith("PIN"):
        raise SystemExit("--revision must be an immutable model commit")
    corpus_dir = Path(args.corpus_dir)
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    train_path = corpus_dir / "train.jsonl"
    if sha256_file(train_path) != manifest["train_jsonl_sha256"]:
        raise SystemExit("training corpus hash mismatch")
    if manifest["split_policy"] != "development_only; validation/holdout/shift forbidden":
        raise SystemExit("unsafe split policy in tuning manifest")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA training requires CUDA")
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) != (7, 5):
        print(f"NOTICE: recipe was designed for T4 sm_75; detected sm_{major}{minor}")
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise SystemExit("training corpus is empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        quantization_config=quant,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    ))

    def tokenize(row):
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        completion = row["completion"] + (tokenizer.eos_token or "")
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        full = tokenizer(
            prefix + completion,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )
        labels = [-100] * min(len(prefix_ids), len(full["input_ids"]))
        labels += full["input_ids"][len(labels):]
        full["labels"] = labels
        return full

    dataset = Dataset.from_list(rows).map(
        tokenize,
        remove_columns=list(rows[0]),
        desc="Tokenizing prompt-masked tuning examples",
    )
    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100
        ),
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    artifacts = {
        str(path.relative_to(output_dir)): {
            "sha256": sha256_file(path), "bytes": path.stat().st_size
        }
        for path in sorted(output_dir.rglob("*")) if path.is_file()
    }
    training_manifest = {
        "schema_version": "1.0.0",
        "arm": "secondary_development_only_qlora",
        "base_model_id": args.model_id,
        "base_model_revision": args.revision,
        "corpus_manifest_sha256": manifest["manifest_sha256"],
        "corpus_sha256": manifest["train_jsonl_sha256"],
        "seed": args.seed,
        "hyperparameters": vars(args),
        "train_metrics": result.metrics,
        "quantization": quant.to_dict(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "accelerate", "bitsandbytes", "peft", "datasets")
        },
        "artifacts": artifacts,
    }
    (output_dir / "traceguard_training_manifest.json").write_text(
        json.dumps(training_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
