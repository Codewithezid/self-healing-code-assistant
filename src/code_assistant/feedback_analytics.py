from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeedbackRecord:
    created_at: str
    thread_id: str
    verdict: str
    rating: int
    provider: str
    model: str
    runtime_profile: str
    rag_enabled: bool
    corrective_rag_mode: str
    confidence_score: float
    hallucination_risk: float
    comment: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "thread_id": self.thread_id,
            "verdict": self.verdict,
            "rating": self.rating,
            "provider": self.provider,
            "model": self.model,
            "runtime_profile": self.runtime_profile,
            "rag_enabled": self.rag_enabled,
            "corrective_rag_mode": self.corrective_rag_mode,
            "confidence_score": self.confidence_score,
            "hallucination_risk": self.hallucination_risk,
            "comment": self.comment,
        }


def append_feedback(path: Path, record: FeedbackRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")


def load_feedback(path: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows[-limit:]


def summarize_feedback(rows: list[dict[str, Any]], *, last_days: int = 30) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=last_days)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        stamp = str(row.get("created_at", "")).strip()
        if not stamp:
            continue
        try:
            when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when >= cutoff:
            filtered.append(row)

    verdicts = Counter(str(row.get("verdict", "unknown")) for row in filtered)
    profiles = Counter(str(row.get("runtime_profile", "custom")) for row in filtered)
    rag_modes = Counter(
        "rag_on" if bool(row.get("rag_enabled", False)) else "rag_off"
        for row in filtered
    )
    ratings = [int(row.get("rating", 0) or 0) for row in filtered]
    confidences = [float(row.get("confidence_score", 0.0) or 0.0) for row in filtered]
    hallucinations = [float(row.get("hallucination_risk", 0.0) or 0.0) for row in filtered]

    return {
        "window_days": last_days,
        "total_feedback": len(filtered),
        "verdict_counts": dict(verdicts),
        "profile_counts": dict(profiles),
        "rag_usage_counts": dict(rag_modes),
        "average_rating": round(sum(ratings) / len(ratings), 3) if ratings else 0.0,
        "average_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        "average_hallucination_risk": round(sum(hallucinations) / len(hallucinations), 3) if hallucinations else 0.0,
    }
