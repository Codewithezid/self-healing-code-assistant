from __future__ import annotations

import argparse
import sys
import warnings
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
from src.code_assistant.benchmarking import run_benchmark


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the semantic benchmark suite and write a structured report.",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=["custom", "fast", "balanced", "accurate", "goated"],
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


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    cases = BENCHMARK_CASES[: args.limit] if args.limit > 0 else BENCHMARK_CASES
    outcome = run_benchmark(
        runtime_profile=args.runtime_profile,
        cases=cases,
        output_dir=Path(args.output_dir),
        root_dir=ROOT,
    )
    print(f"Wrote JSON report to {outcome.json_path}")
    print(f"Wrote Markdown report to {outcome.markdown_path}")
    print(
        f"Semantic accuracy: {outcome.report['summary']['semantic_passes']}/"
        f"{outcome.report['summary']['total_cases']} = {outcome.report['summary']['semantic_accuracy_percent']}%"
    )
    print(f"Average latency: {outcome.report['summary']['average_latency_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
