from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.complex_benchmark import BENCHMARK_CASES
from src.code_assistant.ablation import run_ablation, write_ablation_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAG ablation experiments and write a report.")
    parser.add_argument("--provider", default="mistral", help="LLM provider (mistral/openai/openrouter/local).")
    parser.add_argument("--model", default="mistral-medium-latest", help="Model name.")
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit. 0 uses all benchmark cases.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Assistant retry limit.")
    parser.add_argument("--validation-timeout", type=int, default=5, help="Validation timeout in seconds.")
    parser.add_argument("--output-dir", default="artifacts/ablation_reports", help="Where report files are written.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    cases = BENCHMARK_CASES[: args.limit] if args.limit > 0 else BENCHMARK_CASES
    report = run_ablation(
        cases=cases,
        root_dir=ROOT,
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iterations,
        validation_timeout=args.validation_timeout,
    )
    json_path, md_path = write_ablation_report(Path(args.output_dir), report)
    print(f"Wrote JSON ablation report to {json_path}")
    print(f"Wrote Markdown ablation report to {md_path}")
    for variant in report["variants"]:
        summary = variant["summary"]
        print(
            f"{variant['variant']}: "
            f"{summary['semantic_passes']}/{summary['total_cases']} "
            f"({summary['semantic_accuracy_percent']}%), "
            f"avg latency {summary['average_latency_seconds']}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
