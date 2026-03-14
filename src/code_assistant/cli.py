from __future__ import annotations

import argparse
import json
from typing import Any

from .assistant import CodeAssistant, CodeSolution


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph self-correcting code assistant."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Coding prompt to send to the assistant.",
    )
    parser.add_argument(
        "--model",
        default="mistral-large-latest",
        help="Remote model name to use when provider=mistral.",
    )
    parser.add_argument(
        "--provider",
        choices=["mistral", "local"],
        default="mistral",
        help="Model provider to use.",
    )
    parser.add_argument(
        "--local-model",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Local Hugging Face model to use when provider=local.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum self-correction attempts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final structured output as JSON.",
    )
    parser.add_argument(
        "--show-events",
        action="store_true",
        help="Print each graph event while the assistant runs.",
    )
    parser.add_argument(
        "--validation-timeout",
        type=int,
        default=5,
        help="Timeout in seconds for validating generated code.",
    )
    return parser


def _event_preview(event: dict[str, Any]) -> str:
    generation = event.get("generation")
    iterations = event.get("iterations")
    error = event.get("error", "pending")
    if isinstance(generation, CodeSolution):
        return f"iteration={iterations} error={error} summary={generation.prefix}"
    return f"iteration={iterations} error={error}"


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.question:
        parser.error("A coding question is required.")

    try:
        assistant = CodeAssistant(
            model_name=args.model,
            max_iterations=args.max_iterations,
            validation_timeout_seconds=args.validation_timeout,
            provider=args.provider,
            local_model_name=args.local_model,
        )
    except RuntimeError as exc:
        print(f"Configuration error: {exc}")
        return 1

    if args.show_events:
        try:
            final_event: dict[str, Any] | None = None
            for event in assistant.stream(args.question):
                final_event = event
                print(_event_preview(event))
            result = final_event or {}
        except Exception as exc:
            print(f"Runtime error: {exc}")
            return 1
    else:
        try:
            result = assistant.run(args.question)
        except Exception as exc:
            print(f"Runtime error: {exc}")
            return 1

    solution = result.get("generation")
    if not isinstance(solution, CodeSolution):
        print("The assistant did not return a code solution.")
        return 1

    if args.json:
        print(json.dumps(solution.model_dump(), indent=2))
    else:
        print(CodeAssistant.format_solution(solution))
    return 0
