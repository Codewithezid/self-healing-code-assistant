from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .assistant import CodeAssistant
from .benchmarking import run_case


@dataclass(frozen=True)
class AblationVariant:
    name: str
    rag_enabled: bool
    corrective_enabled: bool
    corrective_mode: str


DEFAULT_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant("no_rag", rag_enabled=False, corrective_enabled=False, corrective_mode="fast"),
    AblationVariant("rag_no_corrective", rag_enabled=True, corrective_enabled=False, corrective_mode="fast"),
    AblationVariant("rag_balanced", rag_enabled=True, corrective_enabled=True, corrective_mode="balanced"),
    AblationVariant("rag_aggressive", rag_enabled=True, corrective_enabled=True, corrective_mode="aggressive"),
)


def _summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    semantic_passes = sum(int(bool(row["semantic_ok"])) for row in rows)
    pipeline_passes = sum(int(bool(row["pipeline_ok"])) for row in rows)
    total = len(rows)
    latency = round(sum(float(row["latency_seconds"]) for row in rows) / max(1, total), 3)
    return {
        "total_cases": total,
        "semantic_passes": semantic_passes,
        "pipeline_passes": pipeline_passes,
        "semantic_accuracy_percent": round((semantic_passes / max(1, total)) * 100, 2),
        "average_latency_seconds": latency,
    }


def run_ablation(
    *,
    cases: list[dict[str, str]],
    root_dir: Path,
    provider: str,
    model: str,
    max_iterations: int,
    validation_timeout: int,
    variants: list[AblationVariant] | None = None,
) -> dict[str, Any]:
    active_variants = variants or list(DEFAULT_VARIANTS)
    rows: list[dict[str, Any]] = []
    for variant in active_variants:
        assistant = CodeAssistant(
            provider=provider,
            model_name=model,
            max_iterations=max_iterations,
            validation_timeout_seconds=validation_timeout,
            rag_enabled=variant.rag_enabled,
            corrective_rag_enabled=variant.corrective_enabled,
            corrective_rag_mode=variant.corrective_mode,
            runtime_profile=f"ablation-{variant.name}",
        )
        case_rows = [run_case(assistant, case, root_dir=root_dir) for case in cases]
        rows.append(
            {
                "variant": variant.name,
                "config": {
                    "rag_enabled": variant.rag_enabled,
                    "corrective_enabled": variant.corrective_enabled,
                    "corrective_mode": variant.corrective_mode,
                },
                "summary": _summary(case_rows),
                "cases": case_rows,
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "model": model,
        "max_iterations": max_iterations,
        "validation_timeout": validation_timeout,
        "variants": rows,
    }


def write_ablation_report(output_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"rag_ablation_{stamp}.json"
    md_path = output_dir / f"rag_ablation_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# RAG Ablation Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Provider/model: `{report['provider']}` / `{report['model']}`",
        f"- Max iterations: `{report['max_iterations']}`",
        f"- Validation timeout: `{report['validation_timeout']}s`",
        "",
        "| Variant | Accuracy | Semantic Passes | Pipeline Passes | Avg Latency |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in report["variants"]:
        summary = variant["summary"]
        lines.append(
            f"| `{variant['variant']}` | {summary['semantic_accuracy_percent']}% | "
            f"{summary['semantic_passes']}/{summary['total_cases']} | "
            f"{summary['pipeline_passes']}/{summary['total_cases']} | "
            f"{summary['average_latency_seconds']}s |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
