from __future__ import annotations

import json
from pathlib import Path

from src.code_assistant.benchmarking import compare_latest_by_profile, load_report_files


def _write_report(path: Path, *, profile: str, accuracy: float, latency: float) -> None:
    payload = {
        "generated_at": "2026-05-13T12:00:00",
        "config": {
            "runtime_profile": profile,
            "provider": "mistral",
            "model": "mistral-medium-latest",
            "rag_enabled": profile != "fast",
            "corrective_rag_mode": "balanced",
        },
        "summary": {
            "total_cases": 8,
            "pipeline_passes": 7,
            "semantic_passes": 6,
            "average_latency_seconds": latency,
            "semantic_accuracy_percent": accuracy,
        },
        "cases": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_report_files_reads_and_sorts(tmp_path: Path) -> None:
    first = tmp_path / "benchmark_report_fast_20260513_120000.json"
    second = tmp_path / "benchmark_report_balanced_20260513_130000.json"
    _write_report(first, profile="fast", accuracy=50.0, latency=1.2)
    _write_report(second, profile="balanced", accuracy=75.0, latency=2.1)

    rows = load_report_files(tmp_path, limit=10)
    assert len(rows) == 2
    profiles = {row["runtime_profile"] for row in rows}
    assert profiles == {"fast", "balanced"}
    assert all("semantic_accuracy_percent" in row for row in rows)


def test_compare_latest_by_profile_filters_requested(tmp_path: Path) -> None:
    _write_report(tmp_path / "benchmark_report_fast_20260513_120000.json", profile="fast", accuracy=50.0, latency=1.2)
    _write_report(tmp_path / "benchmark_report_balanced_20260513_130000.json", profile="balanced", accuracy=75.0, latency=2.1)
    _write_report(tmp_path / "benchmark_report_accurate_20260513_140000.json", profile="accurate", accuracy=87.5, latency=3.0)

    compared = compare_latest_by_profile(tmp_path, profiles=["balanced", "accurate"])
    assert set(compared.keys()) == {"balanced", "accurate"}
    assert compared["accurate"]["semantic_accuracy_percent"] == 87.5
