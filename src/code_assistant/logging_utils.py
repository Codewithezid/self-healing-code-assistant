from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .platform_utils import UpstashRedis


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def append_failure_record(
    *,
    payload: dict[str, Any],
    file_path: Path,
    destination: str,
    upstash_url: str = "",
    upstash_token: str = "",
    upstash_key: str = "code-assistant:failures",
) -> None:
    mode = destination.strip().lower()
    if mode in {"", "none"}:
        return
    if mode == "file":
        append_jsonl(file_path, payload)
        return
    if mode == "upstash":
        redis = UpstashRedis(base_url=upstash_url, token=upstash_token)
        redis.command("LPUSH", upstash_key, json.dumps(payload, ensure_ascii=True))
        return
    if os.getenv("CODE_ASSISTANT_LOG_DESTINATION"):
        raise RuntimeError(f"Unsupported log destination: {destination}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
