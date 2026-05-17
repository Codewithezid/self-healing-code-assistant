from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .assistant import CodeAssistant, CodeSolution
from .profiles import get_runtime_profile


@dataclass(frozen=True)
class BenchmarkRunResult:
    json_path: Path
    markdown_path: Path
    report: dict[str, Any]


def assistant_for_profile(profile_name: str) -> tuple[CodeAssistant, dict[str, object]]:
    profile = get_runtime_profile(profile_name)
    if profile is None:
        assistant = CodeAssistant(
            max_iterations=3,
            validation_timeout_seconds=5,
            runtime_profile="custom",
        )
        return assistant, {
            "runtime_profile": "custom",
            "provider": assistant.provider,
            "model": assistant.model_name,
            "rag_enabled": False,
            "corrective_rag_mode": "balanced",
            "max_iterations": assistant.max_iterations,
            "validation_timeout": assistant.validation_timeout_seconds,
        }

    assistant = CodeAssistant(
        provider=profile.provider,
        model_name=profile.model,
        max_iterations=profile.max_iterations,
        validation_timeout_seconds=profile.validation_timeout,
        rag_enabled=profile.rag_enabled,
        corrective_rag_mode=profile.corrective_rag_mode,
        runtime_profile=profile.name,
    )
    return assistant, {
        "runtime_profile": profile.name,
        "provider": profile.provider,
        "model": profile.model,
        "rag_enabled": profile.rag_enabled,
        "corrective_rag_mode": profile.corrective_rag_mode,
        "max_iterations": profile.max_iterations,
        "validation_timeout": profile.validation_timeout,
    }


def run_case(
    assistant: CodeAssistant,
    case: dict[str, str],
    *,
    root_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    row: dict[str, object] = {
        "name": case["name"],
        "iterations": 0,
        "pipeline_ok": False,
        "semantic_ok": False,
        "latency_seconds": 0.0,
        "failure_category": "none",
        "failure_stage": "none",
        "failure_summary": "",
        "confidence_score": 0.0,
        "hallucination_risk": 0.5,
        "regression_test_passed": False,
    }
    try:
        result = assistant.run(case["prompt"])
    except Exception as exc:
        row["latency_seconds"] = round(time.perf_counter() - started, 3)
        row["failure_category"] = "benchmark_runtime_error"
        row["failure_stage"] = "assistant_run"
        row["failure_summary"] = str(exc)
        return row

    solution = result.get("generation")
    diagnostics = CodeAssistant.classify_failure(result)
    row.update(
        {
            "iterations": int(result.get("iterations", 0) or 0),
            "latency_seconds": round(time.perf_counter() - started, 3),
            "failure_category": diagnostics.category,
            "failure_stage": diagnostics.stage,
            "failure_summary": diagnostics.summary,
            "confidence_score": float(result.get("confidence_score", 0.0) or 0.0),
            "hallucination_risk": float(result.get("hallucination_risk", 0.5) or 0.5),
            "regression_test_passed": bool(result.get("regression_test_passed", False)),
        }
    )
    if not isinstance(solution, CodeSolution) or result.get("error") == "yes":
        return row

    row["pipeline_ok"] = True
    snippet = "\n\n".join(
        part
        for part in [solution.imports.strip(), solution.code.strip(), case["tests"].strip()]
        if part
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", snippet],
        cwd=root_dir,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    row["semantic_ok"] = completed.returncode == 0
    if completed.returncode != 0:
        row["failure_category"] = "semantic_assertion_failure"
        row["failure_stage"] = "benchmark_assertion"
        row["failure_summary"] = (
            completed.stderr.strip() or completed.stdout.strip() or "Semantic assertion failed."
        )
    return row


def build_report(
    *,
    runtime_profile: str,
    cases: list[dict[str, str]],
    root_dir: Path,
) -> dict[str, Any]:
    assistant, config = assistant_for_profile(runtime_profile)
    rows = [run_case(assistant, case, root_dir=root_dir) for case in cases]
    semantic_passes = sum(int(bool(row["semantic_ok"])) for row in rows)
    pipeline_passes = sum(int(bool(row["pipeline_ok"])) for row in rows)
    average_latency = round(
        sum(float(row["latency_seconds"]) for row in rows) / len(rows),
        3,
    ) if rows else 0.0
    average_confidence = round(
        sum(float(row["confidence_score"]) for row in rows) / len(rows),
        3,
    ) if rows else 0.0
    average_hallucination_risk = round(
        sum(float(row["hallucination_risk"]) for row in rows) / len(rows),
        3,
    ) if rows else 0.0
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "summary": {
            "total_cases": len(rows),
            "pipeline_passes": pipeline_passes,
            "semantic_passes": semantic_passes,
            "average_latency_seconds": average_latency,
            "average_confidence_score": average_confidence,
            "average_hallucination_risk": average_hallucination_risk,
            "semantic_accuracy_percent": round((semantic_passes / len(rows)) * 100, 2) if rows else 0.0,
        },
        "cases": rows,
    }


def write_report(output_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile = str(report.get("config", {}).get("runtime_profile", "custom"))
    safe_profile = "".join(ch for ch in profile if ch.isalnum() or ch in {"-", "_"}).strip("_") or "custom"
    json_path = output_dir / f"benchmark_report_{safe_profile}_{stamp}.json"
    md_path = output_dir / f"benchmark_report_{safe_profile}_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Benchmark Report",
        "",
        f"- Runtime profile: `{report['config']['runtime_profile']}`",
        f"- Provider: `{report['config']['provider']}`",
        f"- Model: `{report['config']['model']}`",
        f"- Semantic accuracy: `{report['summary']['semantic_accuracy_percent']}%`",
        f"- Average latency: `{report['summary']['average_latency_seconds']}s`",
        f"- Average confidence: `{report['summary']['average_confidence_score']}`",
        f"- Average hallucination risk: `{report['summary']['average_hallucination_risk']}`",
        f"- Pipeline pass rate: `{report['summary']['pipeline_passes']}/{report['summary']['total_cases']}`",
        "",
        "| Case | Pipeline | Semantic | Iterations | Latency | Failure |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["cases"]:
        lines.append(
            f"| `{row['name']}` | {'PASS' if row['pipeline_ok'] else 'FAIL'} | "
            f"{'PASS' if row['semantic_ok'] else 'FAIL'} | {row['iterations']} | {row['latency_seconds']}s | "
            f"{row['failure_category'] or 'none'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def run_benchmark(
    *,
    runtime_profile: str,
    cases: list[dict[str, str]],
    output_dir: Path,
    root_dir: Path,
) -> BenchmarkRunResult:
    report = build_report(runtime_profile=runtime_profile, cases=cases, root_dir=root_dir)
    json_path, markdown_path = write_report(output_dir, report)
    return BenchmarkRunResult(
        json_path=json_path,
        markdown_path=markdown_path,
        report=report,
    )


def load_report_files(output_dir: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    if not output_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    json_files = sorted(output_dir.glob("benchmark_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in json_files[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = payload.get("summary", {})
        config = payload.get("config", {})
        rows.append(
            {
                "filename": path.name,
                "generated_at": payload.get("generated_at", ""),
                "runtime_profile": config.get("runtime_profile", "custom"),
                "provider": config.get("provider", ""),
                "model": config.get("model", ""),
                "rag_enabled": bool(config.get("rag_enabled", False)),
                "corrective_rag_mode": config.get("corrective_rag_mode", "balanced"),
                "semantic_accuracy_percent": float(summary.get("semantic_accuracy_percent", 0.0) or 0.0),
                "pipeline_passes": int(summary.get("pipeline_passes", 0) or 0),
                "semantic_passes": int(summary.get("semantic_passes", 0) or 0),
                "total_cases": int(summary.get("total_cases", 0) or 0),
                "average_latency_seconds": float(summary.get("average_latency_seconds", 0.0) or 0.0),
                "json_path": str(path),
            }
        )
    return rows


def compare_latest_by_profile(output_dir: Path, *, profiles: list[str]) -> dict[str, dict[str, Any]]:
    reports = load_report_files(output_dir, limit=200)
    latest: dict[str, dict[str, Any]] = {}
    for row in reports:
        profile = str(row.get("runtime_profile", "custom"))
        if profile not in profiles:
            continue
        if profile not in latest:
            latest[profile] = row
    return latest
