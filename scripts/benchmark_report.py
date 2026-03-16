from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from scripts.complex_benchmark import BENCHMARK_CASES
from src.code_assistant.assistant import CodeAssistant, CodeSolution
from src.code_assistant.profiles import get_runtime_profile


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the semantic benchmark suite and write a structured report.",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=["custom", "fast", "balanced", "accurate"],
        default="balanced",
        help="Runtime profile to benchmark.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on benchmark cases. Use 0 for all cases.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmark_reports",
        help="Directory where the benchmark report files will be written.",
    )
    return parser


def _assistant_for_profile(profile_name: str) -> tuple[CodeAssistant, dict[str, object]]:
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


def _run_case(assistant: CodeAssistant, case: dict[str, str]) -> dict[str, object]:
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
        }
    )
    if not isinstance(solution, CodeSolution) or result.get("error") != "no":
        return row

    row["pipeline_ok"] = True
    snippet = "\n\n".join(
        part
        for part in [solution.imports.strip(), solution.code.strip(), case["tests"].strip()]
        if part
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", snippet],
        cwd=ROOT,
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


def _write_reports(output_dir: Path, report: dict[str, object]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"benchmark_report_{stamp}.json"
    md_path = output_dir / f"benchmark_report_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Benchmark Report",
        "",
        f"- Runtime profile: `{report['config']['runtime_profile']}`",
        f"- Provider: `{report['config']['provider']}`",
        f"- Model: `{report['config']['model']}`",
        f"- Semantic accuracy: `{report['summary']['semantic_accuracy_percent']}%`",
        f"- Average latency: `{report['summary']['average_latency_seconds']}s`",
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


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    assistant, config = _assistant_for_profile(args.runtime_profile)
    cases = BENCHMARK_CASES[: args.limit] if args.limit > 0 else BENCHMARK_CASES
    rows = [_run_case(assistant, case) for case in cases]
    semantic_passes = sum(int(bool(row["semantic_ok"])) for row in rows)
    pipeline_passes = sum(int(bool(row["pipeline_ok"])) for row in rows)
    average_latency = round(
        sum(float(row["latency_seconds"]) for row in rows) / len(rows),
        3,
    ) if rows else 0.0
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "summary": {
            "total_cases": len(rows),
            "pipeline_passes": pipeline_passes,
            "semantic_passes": semantic_passes,
            "average_latency_seconds": average_latency,
            "semantic_accuracy_percent": round((semantic_passes / len(rows)) * 100, 2) if rows else 0.0,
        },
        "cases": rows,
    }
    json_path, md_path = _write_reports(Path(args.output_dir), report)
    print(f"Wrote JSON report to {json_path}")
    print(f"Wrote Markdown report to {md_path}")
    print(
        f"Semantic accuracy: {report['summary']['semantic_passes']}/"
        f"{report['summary']['total_cases']} = {report['summary']['semantic_accuracy_percent']}%"
    )
    print(f"Average latency: {report['summary']['average_latency_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
