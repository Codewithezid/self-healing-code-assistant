from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload fine-tuning files to Mistral and create a fine-tuning job."
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
        "--model",
        default="codestral-latest",
        help="Base Mistral model to fine-tune.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create and validate the job without starting training.",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Actually start the fine-tuning job after creation. This can incur charges.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate hyperparameter.",
    )
    return parser


def require_api_key() -> str:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set.")
    return api_key


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def upload_file(api_key: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    content_type = mimetypes.guess_type(path.name)[0] or "application/jsonl"
    with path.open("rb") as handle:
        response = requests.post(
            "https://api.mistral.ai/v1/files",
            headers=auth_headers(api_key),
            data={"purpose": "fine-tune"},
            files={"file": (path.name, handle, content_type)},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


def create_job(
    api_key: str,
    *,
    model: str,
    training_file_id: str,
    validation_file_id: str,
    learning_rate: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "training_files": [{"file_id": training_file_id, "weight": 1}],
        "validation_files": [validation_file_id],
        "hyperparameters": {
            "training_steps": 10,
            "learning_rate": learning_rate,
        },
        "auto_start": False,
        "invalid_sample_skip_percentage": 0.05,
    }
    response = requests.post(
        "https://api.mistral.ai/v1/fine_tuning/jobs",
        headers={**auth_headers(api_key), "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(
            f"Fine-tuning job creation failed ({response.status_code}): {response.text}"
        )
    return response.json()


def start_job(api_key: str, job_id: str) -> dict[str, Any]:
    response = requests.post(
        f"https://api.mistral.ai/v1/fine_tuning/jobs/{job_id}/start",
        headers=auth_headers(api_key),
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(f"Fine-tuning job start failed ({response.status_code}): {response.text}")
    return response.json()


def main() -> int:
    args = build_parser().parse_args()
    api_key = require_api_key()
    train_path = Path(args.train_file)
    validation_path = Path(args.validation_file)

    print(f"Uploading training file: {train_path}")
    train_file = upload_file(api_key, train_path)
    print(f"Uploading validation file: {validation_path}")
    validation_file = upload_file(api_key, validation_path)

    job = create_job(
        api_key,
        model=args.model,
        training_file_id=train_file["id"],
        validation_file_id=validation_file["id"],
        learning_rate=args.learning_rate,
    )

    output = {
        "train_file_id": train_file["id"],
        "validation_file_id": validation_file["id"],
        "job": job,
    }

    if args.start:
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError("Job ID missing from fine-tuning response.")
        output["start_response"] = start_job(api_key, job_id)

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
