from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent

HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
DEFAULT_DATASET = "HuggingFaceH4/CodeAlpaca_20K"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "train"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Mistral fine-tuning JSONL files from public coding data and local failures."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Source dataset name.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Dataset config name.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split name.")
    parser.add_argument("--seed-count", type=int, default=120, help="Number of seed examples to fetch.")
    parser.add_argument(
        "--failure-log",
        default=str(ROOT / "data" / "runtime" / "failure_log.jsonl"),
        help="Optional failure log to fold into the training set.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "finetune"),
        help="Directory for train/validation JSONL output.",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--random-seed", type=int, default=42, help="Shuffle seed.")
    return parser


def fetch_rows(dataset: str, config: str, split: str, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < count:
        length = min(100, count - len(rows))
        response = requests.get(
            HF_ROWS_API,
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        chunk = payload.get("rows", [])
        if not chunk:
            break
        rows.extend(item.get("row", {}) for item in chunk)
        offset += length
    return rows


def codealpaca_row_to_messages(row: dict[str, Any]) -> dict[str, Any]:
    prompt = (row.get("prompt") or "").strip()
    completion = (row.get("completion") or "").strip()
    if not prompt or not completion:
        raise ValueError("Dataset row is missing prompt or completion.")
    return {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": completion,
            },
        ]
    }


def failure_record_to_messages(record: dict[str, Any]) -> dict[str, Any] | None:
    question = (record.get("question") or "").strip()
    generation = record.get("generation") or {}
    if not question or not isinstance(generation, dict):
        return None
    imports = (generation.get("imports") or "").strip()
    code = (generation.get("code") or "").strip()
    prefix = (generation.get("prefix") or "").strip()
    assistant_text = "\n\n".join(
        part
        for part in [
            prefix,
            f"Imports:\n{imports}" if imports else "",
            f"Code:\n{code}" if code else "",
        ]
        if part
    ).strip()
    if not assistant_text:
        return None
    return {
        "messages": [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": assistant_text,
            },
        ]
    }


def load_failure_examples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    examples: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        mapped = failure_record_to_messages(record)
        if mapped:
            examples.append(mapped)
    return examples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    failure_log = Path(args.failure_log)

    print(f"Fetching {args.seed_count} seed examples from {args.dataset}...")
    seed_rows = fetch_rows(args.dataset, args.config, args.split, args.seed_count)
    seed_examples = [codealpaca_row_to_messages(row) for row in seed_rows]
    failure_examples = load_failure_examples(failure_log)

    combined = seed_examples + failure_examples
    if len(combined) < 10:
        print("Not enough examples to prepare a fine-tuning dataset.", file=sys.stderr)
        return 1

    random.Random(args.random_seed).shuffle(combined)
    validation_count = max(1, int(len(combined) * args.validation_ratio))
    validation_rows = combined[:validation_count]
    training_rows = combined[validation_count:]

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    write_jsonl(train_path, training_rows)
    write_jsonl(val_path, validation_rows)

    summary = {
        "dataset": args.dataset,
        "seed_examples": len(seed_examples),
        "failure_examples": len(failure_examples),
        "training_examples": len(training_rows),
        "validation_examples": len(validation_rows),
        "train_file": str(train_path),
        "validation_file": str(val_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
