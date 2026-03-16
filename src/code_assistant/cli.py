from __future__ import annotations

import argparse
import json
from typing import Any

from .assistant import CodeAssistant, CodeSolution
from .profiles import get_runtime_profile
from .sandbox_utils import parse_sandbox_cmd


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
        "--runtime-profile",
        choices=["custom", "fast", "balanced", "accurate"],
        default="custom",
        help="Named runtime preset for model, RAG, and retry settings.",
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
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Retrieve project context from the local Qdrant index before generation.",
    )
    parser.add_argument(
        "--rag-auto-index",
        action="store_true",
        help="Build the local Qdrant index automatically if it does not exist.",
    )
    parser.add_argument(
        "--sandbox-cmd",
        default="",
        help="Optional sandbox command prefix for validation (e.g., 'firejail --quiet').",
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
        profile = get_runtime_profile(args.runtime_profile)
        provider = profile.provider if profile is not None else args.provider
        model_name = profile.model if profile is not None else args.model
        max_iterations = profile.max_iterations if profile is not None else args.max_iterations
        validation_timeout = profile.validation_timeout if profile is not None else args.validation_timeout
        rag_enabled = profile.rag_enabled if profile is not None else args.rag
        assistant = CodeAssistant(
            model_name=model_name,
            max_iterations=max_iterations,
            validation_timeout_seconds=validation_timeout,
            provider=provider,
            local_model_name=args.local_model,
            rag_enabled=rag_enabled,
            rag_auto_index=args.rag_auto_index,
            runtime_profile=args.runtime_profile,
            corrective_rag_mode=(profile.corrective_rag_mode if profile is not None else "balanced"),
            sandbox_cmd=(parse_sandbox_cmd(args.sandbox_cmd) if args.sandbox_cmd.strip() else None),
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
