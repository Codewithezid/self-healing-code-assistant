from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.code_assistant.feedback_analytics import FeedbackRecord, append_feedback, load_feedback, summarize_feedback


def test_feedback_append_and_summary(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    append_feedback(
        path,
        FeedbackRecord(
            created_at=now,
            thread_id="t1",
            verdict="correct",
            rating=5,
            provider="mistral",
            model="mistral-medium-latest",
            runtime_profile="balanced",
            rag_enabled=True,
            corrective_rag_mode="balanced",
            confidence_score=0.8,
            hallucination_risk=0.2,
            comment="good",
        ),
    )
    rows = load_feedback(path, limit=10)
    assert len(rows) == 1
    summary = summarize_feedback(rows, last_days=30)
    assert summary["total_feedback"] == 1
    assert summary["verdict_counts"]["correct"] == 1
