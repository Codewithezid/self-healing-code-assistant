from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a small local coding model with LoRA on the prepared JSONL dataset."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Base local model to fine-tune.",
    )
    parser.add_argument(
        "--train-file",
        default=str(ROOT / "data" / "finetune" / "train.jsonl"),
        help="Training JSONL file.",
    )
    parser.add_argument(
        "--validation-file",
        default=str(ROOT / "data" / "finetune" / "validation.jsonl"),
        help="Validation JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "local-lora"),
        help="Directory for the trained adapter.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--max-length", type=int, default=1024, help="Sequence length.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="If set, limit the number of training samples (0 = no limit).",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="If set, limit the number of eval samples (0 = no limit).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce VRAM usage.",
    )
    return parser


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render_messages(messages: list[dict], tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return "\n".join(f"{item['role']}: {item['content']}" for item in messages)


def main() -> int:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Local LoRA training requires torch, datasets, transformers, and peft."
        ) from exc

    args = build_parser().parse_args()
    train_path = Path(args.train_file)
    validation_path = Path(args.validation_file)
    output_dir = Path(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Let Trainer/Accelerate place the model; avoid device_map here for fewer edge-cases.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_rows = load_jsonl(train_path)
    validation_rows = load_jsonl(validation_path)
    if args.max_train_samples and args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_eval_samples and args.max_eval_samples > 0:
        validation_rows = validation_rows[: args.max_eval_samples]
    train_texts = [render_messages(row["messages"], tokenizer) for row in train_rows]
    validation_texts = [render_messages(row["messages"], tokenizer) for row in validation_rows]

    train_dataset = Dataset.from_dict({"text": train_texts})
    validation_dataset = Dataset.from_dict({"text": validation_texts})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    validation_dataset = validation_dataset.map(tokenize, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
