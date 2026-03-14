from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, message_to_dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .logging_utils import append_failure_record, utc_now_iso
from .local_backend import LocalCodeGenerator
from .models import CodeSolution

load_dotenv()


class GraphState(TypedDict):
    """State passed between LangGraph nodes."""

    error: str
    events: list[dict[str, Any]]
    messages: Annotated[list[AnyMessage], add_messages]
    generation: CodeSolution
    iterations: int


class CodeAssistant:
    """Self-correcting code assistant backed by Mistral and LangGraph."""

    def __init__(
        self,
        *,
        model_name: str = "mistral-large-latest",
        temperature: float = 0.0,
        max_iterations: int = 3,
        validation_timeout_seconds: int = 5,
        failure_log_path: str | None = None,
        log_destination: str | None = None,
        upstash_redis_rest_url: str = "",
        upstash_redis_rest_token: str = "",
        failure_log_key: str = "code-assistant:failures",
        provider: str = "mistral",
        local_model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        local_max_new_tokens: int = 768,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.validation_timeout_seconds = validation_timeout_seconds
        self.log_destination = log_destination or os.getenv(
            "CODE_ASSISTANT_LOG_DESTINATION",
            "file",
        )
        self.upstash_redis_rest_url = upstash_redis_rest_url
        self.upstash_redis_rest_token = upstash_redis_rest_token
        self.failure_log_key = failure_log_key
        self.provider = provider
        self.local_model_name = local_model_name
        self.local_max_new_tokens = local_max_new_tokens
        self.failure_log_path = Path(
            failure_log_path
            or os.getenv("CODE_ASSISTANT_FAILURE_LOG", "data/runtime/failure_log.jsonl")
        )
        self._graph = self._build_graph()

    @staticmethod
    def _require_api_key() -> str:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY is not set. Add it to your environment or a local .env file."
            )
        return api_key

    def _build_chain(self):
        if self.provider == "local":
            return LocalCodeGenerator(
                model_name=self.local_model_name,
                max_new_tokens=self.local_max_new_tokens,
            )
        self._require_api_key()

        llm = ChatMistralAI(
            model=self.model_name,
            temperature=self.temperature,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a careful coding assistant. Return code that can run as-is. "
                        "Always provide: "
                        "1) a short explanation, "
                        "2) the complete import block, "
                        "3) the executable code block. "
                        "If the user asks for a demo, include one."
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        )
        return prompt | llm.with_structured_output(CodeSolution)

    def _build_fallback_components(self):
        if self.provider == "local":
            return None, None
        self._require_api_key()

        llm = ChatMistralAI(
            model=self.model_name,
            temperature=self.temperature,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a careful coding assistant. Respond with valid JSON only. "
                        'The JSON object must contain exactly these string keys: "prefix", '
                        '"imports", and "code". '
                        "The code value must be a single string containing the full executable code body. "
                        "Do not wrap the JSON in markdown fences."
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        )
        return prompt, llm

    @staticmethod
    def _parse_fallback_response(content: Any) -> CodeSolution:
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        text = str(content).strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Fallback response was not valid JSON.")
            payload = json.loads(text[start : end + 1])
        return CodeSolution.model_validate(payload)

    def _build_graph(self):
        chain = self._build_chain()
        fallback_prompt, fallback_llm = self._build_fallback_components()
        builder = StateGraph(GraphState)

        def run_validation(snippet: str, *, filename: str) -> tuple[bool, str]:
            safe_env = {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": "",
            }
            for key in ["SystemRoot", "WINDIR", "TEMP", "TMP"]:
                value = os.environ.get(key)
                if value:
                    safe_env[key] = value

            with tempfile.TemporaryDirectory() as temp_dir:
                script_path = os.path.join(temp_dir, filename)
                with open(script_path, "w", encoding="utf-8") as handle:
                    handle.write(snippet)
                try:
                    completed = subprocess.run(
                        [sys.executable, "-I", script_path],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=self.validation_timeout_seconds,
                        check=False,
                        env=safe_env,
                    )
                except subprocess.TimeoutExpired:
                    return (
                        False,
                        f"Validation timed out after {self.validation_timeout_seconds} seconds.",
                    )

            if completed.returncode != 0:
                error_output = completed.stderr.strip() or completed.stdout.strip() or (
                    "The generated script exited with a non-zero status."
                )
                return False, error_output
            return True, ""

        def generate(state: GraphState) -> dict[str, Any]:
            messages = state["messages"]
            iterations = state["iterations"]
            events = state.get("events", [])
            try:
                code_solution = chain.invoke({"messages": messages})
            except Exception:
                if self.provider == "local" or fallback_prompt is None or fallback_llm is None:
                    raise
                fallback_messages = fallback_prompt.invoke({"messages": messages})
                fallback_response = fallback_llm.invoke(fallback_messages)
                code_solution = self._parse_fallback_response(fallback_response.content)
            messages = messages + [
                (
                    "assistant",
                    "Attempted solution:\n"
                    f"Summary: {code_solution.prefix}\n"
                    f"Imports:\n{code_solution.imports}\n"
                    f"Code:\n{code_solution.code}",
                )
            ]
            return {
                "events": events
                + [
                    {
                        "stage": "generate_code",
                        "status": "done",
                        "iteration": iterations + 1,
                        "detail": code_solution.prefix or "Generated a candidate solution.",
                    }
                ],
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations + 1,
            }

        def code_check(state: GraphState) -> dict[str, Any]:
            messages = state["messages"]
            code_solution = state["generation"]
            iterations = state["iterations"]
            events = state.get("events", [])
            imports = code_solution.imports.strip()
            code = code_solution.code.strip()
            combined_code = "\n\n".join(part for part in [imports, code] if part)

            imports_ok, imports_error = run_validation(
                imports or "pass\n",
                filename="imports_check.py",
            )
            if not imports_ok:
                return {
                    "events": events
                    + [
                        {
                            "stage": "execute_code",
                            "status": "error",
                            "iteration": iterations,
                            "detail": "Import execution failed before the full script could run.",
                        },
                        {
                            "stage": "check_result",
                            "status": "error",
                            "iteration": iterations,
                            "detail": f"Import validation failed: {imports_error}",
                        },
                        {
                            "stage": "retry_or_end",
                            "status": (
                                "done" if iterations >= self.max_iterations else "running"
                            ),
                            "iteration": iterations,
                            "detail": (
                                "Stopped after reaching the retry limit."
                                if iterations >= self.max_iterations
                                else "Queued a corrected retry after the import failure."
                            ),
                        },
                    ],
                    "generation": code_solution,
                    "messages": messages
                    + [
                        (
                            "user",
                            "Your previous solution failed during import execution with the "
                            f"following error: {imports_error}. Explain what went wrong briefly, then "
                            "return a full corrected solution.",
                        )
                    ],
                    "iterations": iterations,
                    "error": "yes",
                }

            code_ok, code_error = run_validation(
                combined_code or "pass\n",
                filename="code_check.py",
            )
            if not code_ok:
                return {
                    "events": events
                    + [
                        {
                            "stage": "execute_code",
                            "status": "done",
                            "iteration": iterations,
                            "detail": "Imports succeeded. Running the generated Python script.",
                        },
                        {
                            "stage": "check_result",
                            "status": "error",
                            "iteration": iterations,
                            "detail": f"Runtime validation failed: {code_error}",
                        },
                        {
                            "stage": "retry_or_end",
                            "status": (
                                "done" if iterations >= self.max_iterations else "running"
                            ),
                            "iteration": iterations,
                            "detail": (
                                "Stopped after reaching the retry limit."
                                if iterations >= self.max_iterations
                                else "Queued a corrected retry after the runtime failure."
                            ),
                        },
                    ],
                    "generation": code_solution,
                    "messages": messages
                    + [
                        (
                            "user",
                            "Your previous solution failed during code execution with the "
                            f"following error: {code_error}. Explain what went wrong briefly, then "
                            "return a full corrected solution.",
                        )
                    ],
                    "iterations": iterations,
                    "error": "yes",
                }

            return {
                "events": events
                + [
                    {
                        "stage": "execute_code",
                        "status": "done",
                        "iteration": iterations,
                        "detail": "Imports succeeded. Running the generated Python script.",
                    },
                    {
                        "stage": "check_result",
                        "status": "done",
                        "iteration": iterations,
                        "detail": "The generated code passed isolated validation.",
                    },
                    {
                        "stage": "retry_or_end",
                        "status": "done",
                        "iteration": iterations,
                        "detail": "Finished because the generated code validated successfully.",
                    },
                ],
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "no",
            }

        def decide_to_finish(state: GraphState) -> Literal["end", "generate"]:
            if state["error"] == "no" or state["iterations"] >= self.max_iterations:
                return "end"
            return "generate"

        builder.add_node("generate", generate)
        builder.add_node("check_code", code_check)
        builder.add_edge(START, "generate")
        builder.add_edge("generate", "check_code")
        builder.add_conditional_edges(
            "check_code",
            decide_to_finish,
            {"end": END, "generate": "generate"},
        )

        memory = InMemorySaver()
        return builder.compile(checkpointer=memory)

    def run(self, question: str, *, thread_id: str | None = None) -> dict[str, Any]:
        """Run the assistant and return the final graph state."""

        resolved_thread_id = thread_id or str(uuid.uuid4())
        config = {
            "configurable": {
                "thread_id": resolved_thread_id,
            }
        }
        result = self._graph.invoke(
            {"error": "pending", "events": [], "messages": [("user", question)], "iterations": 0},
            config=config,
        )
        self._log_failure_if_needed(question, resolved_thread_id, result)
        return result

    def stream(self, question: str, *, thread_id: str | None = None):
        """Yield graph events for streaming/debug output."""

        config = {
            "configurable": {
                "thread_id": thread_id or str(uuid.uuid4()),
            }
        }
        return self._graph.stream(
            {"error": "pending", "events": [], "messages": [("user", question)], "iterations": 0},
            config=config,
            stream_mode="values",
        )

    @staticmethod
    def format_solution(solution: CodeSolution) -> str:
        return (
            f"{solution.prefix}\n\n"
            f"Imports:\n{solution.imports}\n\n"
            f"Code:\n{solution.code}\n"
        )

    def _log_failure_if_needed(
        self,
        question: str,
        thread_id: str,
        result: dict[str, Any],
    ) -> None:
        if result.get("error") != "yes":
            return

        generation = result.get("generation")
        record = {
            "timestamp": utc_now_iso(),
            "thread_id": thread_id,
            "model_name": self.model_name,
            "provider": self.provider,
            "question": question,
            "iterations": result.get("iterations"),
            "error": result.get("error"),
            "messages": [
                message_to_dict(message) if hasattr(message, "content") else message
                for message in result.get("messages", [])
            ],
        }
        if isinstance(generation, CodeSolution):
            record["generation"] = generation.model_dump()
        try:
            append_failure_record(
                payload=record,
                file_path=self.failure_log_path,
                destination=self.log_destination,
                upstash_url=self.upstash_redis_rest_url,
                upstash_token=self.upstash_redis_rest_token,
                upstash_key=self.failure_log_key,
            )
        except Exception:
            return
