from __future__ import annotations

import ast
import difflib
import importlib.util
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import certifi
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, message_to_dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .logging_utils import append_failure_record, utc_now_iso
from .local_backend import LocalCodeGenerator
from .models import CodeSolution, FailureDiagnostics
from .rag import ProjectRAG
from .secure_executor import SandboxPolicy, run_python_snippet

load_dotenv()


def _configure_tls_cert_bundle() -> None:
    """Configure TLS roots for hosted HTTPS provider calls."""
    insecure_ssl = os.getenv("CODE_ASSISTANT_INSECURE_SSL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if insecure_ssl:
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
        os.environ["PYTHONHTTPSVERIFY"] = "0"
        return

    # Prefer native OS trust stores when available (works well on Windows).
    try:
        import truststore  # type: ignore

        truststore.inject_into_ssl()
    except Exception:
        pass

    ca_bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
    os.environ.setdefault("CURL_CA_BUNDLE", ca_bundle)


_configure_tls_cert_bundle()


class GraphState(TypedDict):
    """State passed between LangGraph nodes."""

    error: str
    events: list[dict[str, Any]]
    messages: Annotated[list[AnyMessage], add_messages]
    generation: CodeSolution
    iterations: int
    question: str
    rag_context: str
    rag_sources: list[dict[str, str]]
    failure_memory: list[str]
    retry_plan: str
    last_error: str
    error_fingerprints: dict[str, int]
    previous_combined_code: str


class CodeAssistant:
    """Self-correcting code assistant backed by hosted LLMs and LangGraph."""

    def __init__(
        self,
        *,
        model_name: str = "mistral-medium-latest",
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
        rag_enabled: bool = False,
        rag_auto_index: bool = False,
        rag_project_root: str | None = None,
        rag_qdrant_path: str = "data/qdrant",
        rag_collection_name: str = "code-assistant-project",
        rag_embedding_model: str = "mistral-embed",
        rag_retrieval_k: int = 4,
        rag_retrieval_fetch_k: int = 10,
        rag_max_chunks_per_source: int = 2,
        rag_chunk_size: int = 1200,
        rag_chunk_overlap: int = 200,
        corrective_rag_enabled: bool = True,
        corrective_rag_model: str = "mistral-small-latest",
        corrective_rag_mode: str = "balanced",
        corrective_rag_min_score: int = 3,
        corrective_rag_retry_k: int = 6,
        runtime_profile: str = "custom",
        sandbox_cmd: list[str] | tuple[str, ...] | None = None,
        api_key: str | None = None,
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
        self.runtime_profile = runtime_profile
        self.sandbox_cmd = list(sandbox_cmd) if sandbox_cmd else []
        self.api_key = (api_key or "").strip()
        self.local_model_name = local_model_name
        self.local_max_new_tokens = local_max_new_tokens
        self.rag = (
            ProjectRAG(
                project_root=rag_project_root or Path(__file__).resolve().parents[2],
                qdrant_path=rag_qdrant_path,
                collection_name=rag_collection_name,
                embedding_model=rag_embedding_model,
                retrieval_k=rag_retrieval_k,
                retrieval_fetch_k=rag_retrieval_fetch_k,
                max_chunks_per_source=rag_max_chunks_per_source,
                chunk_size=rag_chunk_size,
                chunk_overlap=rag_chunk_overlap,
                auto_index=rag_auto_index,
                corrective_enabled=corrective_rag_enabled,
                corrective_model=corrective_rag_model,
                corrective_mode=corrective_rag_mode,
                corrective_min_score=corrective_rag_min_score,
                corrective_retry_k=corrective_rag_retry_k,
            )
            if rag_enabled
            else None
        )
        self.failure_log_path = Path(
            failure_log_path
            or os.getenv("CODE_ASSISTANT_FAILURE_LOG", "data/runtime/failure_log.jsonl")
        )
        self._graph = self._build_graph()

    def _resolve_api_key(self, provider: str) -> str:
        if self.api_key:
            return self.api_key
        key_var_by_provider = {
            "mistral": "MISTRAL_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        key_var = key_var_by_provider.get(provider)
        if not key_var:
            raise RuntimeError(
                f"Unsupported provider '{provider}'. Supported providers are: mistral, openai, openrouter, local."
            )
        api_key = os.getenv(key_var)
        if not api_key:
            raise RuntimeError(
                f"{key_var} is not set. Add it to your environment or a local .env file."
            )
        return api_key

    def _build_remote_llm(self):
        if self.provider == "mistral":
            api_key = self._resolve_api_key("mistral")
            insecure_ssl = os.getenv("CODE_ASSISTANT_INSECURE_SSL", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if insecure_ssl:
                try:
                    import langchain_mistralai.chat_models as mistral_chat_models

                    mistral_chat_models.global_ssl_context = False
                except Exception:
                    pass
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "temperature": self.temperature,
                "api_key": api_key,
            }
            try:
                return ChatMistralAI(**kwargs)
            except TypeError:
                kwargs.pop("api_key", None)
                kwargs["mistral_api_key"] = api_key
                return ChatMistralAI(**kwargs)
        if self.provider == "openai":
            api_key = self._resolve_api_key("openai")
            try:
                from langchain_openai import ChatOpenAI
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "OpenAI provider requires langchain-openai. Install dependencies from requirements.txt."
                ) from exc
            kwargs = {
                "model": self.model_name,
                "temperature": self.temperature,
                "api_key": api_key,
            }
            try:
                return ChatOpenAI(**kwargs)
            except TypeError:
                kwargs.pop("api_key", None)
                kwargs["openai_api_key"] = api_key
                return ChatOpenAI(**kwargs)
        if self.provider == "openrouter":
            api_key = self._resolve_api_key("openrouter")
            try:
                from langchain_openai import ChatOpenAI
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "OpenRouter provider requires langchain-openai. Install dependencies from requirements.txt."
                ) from exc
            kwargs = {
                "model": self.model_name,
                "temperature": self.temperature,
                "api_key": api_key,
                "base_url": "https://openrouter.ai/api/v1",
            }
            try:
                return ChatOpenAI(**kwargs)
            except TypeError:
                kwargs.pop("api_key", None)
                kwargs["openai_api_key"] = api_key
                if "base_url" in kwargs:
                    kwargs["openai_api_base"] = kwargs.pop("base_url")
                return ChatOpenAI(**kwargs)
        raise RuntimeError(
            f"Unsupported provider '{self.provider}'. Supported providers are: mistral, openai, openrouter, local."
        )

    def _build_chain(self):
        if self.provider == "local":
            return LocalCodeGenerator(
                model_name=self.local_model_name,
                max_new_tokens=self.local_max_new_tokens,
            )
        llm = self._build_remote_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a careful coding assistant. Return code that can run as-is. "
                        "When project context is supplied, use it to stay consistent with the codebase "
                        "and prefer the local project patterns over generic advice. "
                        "Infer function signatures and expected input types from the user prompt. "
                        "Add lightweight runtime input validation for critical parameters (TypeError/ValueError) "
                        "when invalid types or impossible constraints are provided. "
                        "Handle positive, negative, and edge conditions in implementation logic. "
                        "Always provide: "
                        "1) a short explanation, "
                        "2) the complete import block, "
                        "3) the executable code block. "
                        "If the user asks for a demo, include one. "
                        "The imports field must contain only valid Python import statements, "
                        "comments, or be empty. Never write prose such as 'None required' in imports."
                    ),
                ),
                (
                    "system",
                    "Relevant project context:\n{project_context}",
                ),
                ("placeholder", "{messages}"),
            ]
        )
        return prompt | llm.with_structured_output(CodeSolution)

    def _build_fallback_components(self):
        if self.provider == "local":
            return None, None
        llm = self._build_remote_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a careful coding assistant. Respond with valid JSON only. "
                        "When project context is supplied, use it to stay consistent with the "
                        "existing codebase. "
                        'The JSON object must contain exactly these string keys: "prefix", '
                        '"imports", and "code". '
                        "The code value must be a single string containing the full executable code body. "
                        "The imports value must contain only valid Python import statements, comments, or be empty. "
                        "Do not wrap the JSON in markdown fences."
                    ),
                ),
                (
                    "system",
                    "Relevant project context:\n{project_context}",
                ),
                ("placeholder", "{messages}"),
            ]
        )
        return prompt, llm

    @staticmethod
    def _normalize_imports(imports: str) -> str:
        normalized_lines: list[str] = []
        for raw_line in imports.splitlines():
            line = raw_line.strip()
            if not line or line == "```" or line.lower() == "python":
                continue
            lowered = line.lower()
            if lowered in {"none", "none required", "no imports", "no imports required"}:
                continue
            if any(phrase in lowered for phrase in ("none required", "no import", "not required")):
                normalized_lines.append(f"# {line}")
                continue
            if line.startswith(("import ", "from ", "#")):
                normalized_lines.append(line)
                continue
            normalized_lines.append(f"# {line}")
        return "\n".join(normalized_lines).strip()

    @classmethod
    def _normalize_solution(cls, solution: CodeSolution) -> CodeSolution:
        return CodeSolution(
            prefix=solution.prefix.strip(),
            imports=cls._normalize_imports(solution.imports),
            code=solution.code.strip(),
        )

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
        return CodeAssistant._normalize_solution(CodeSolution.model_validate(payload))

    @staticmethod
    def _project_context_text(rag_context: str) -> str:
        if rag_context.strip():
            return rag_context
        return "No project context was retrieved for this request."

    @staticmethod
    def _extract_python_fence(text: str) -> str:
        match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text

    @staticmethod
    def _extract_import_lines(text: str) -> list[str]:
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("import ") or line.startswith("from "):
                lines.append(line)
        return lines

    @staticmethod
    def _is_ast_valid(imports: str, code: str) -> bool:
        combined = "\n\n".join(part for part in [imports.strip(), code.strip()] if part).strip()
        if not combined:
            return False
        try:
            ast.parse(combined)
            return True
        except Exception:
            return False

    @classmethod
    def _attempt_ast_patch(cls, solution: CodeSolution) -> tuple[CodeSolution, list[str]]:
        notes: list[str] = []
        original_imports = solution.imports.strip()
        original_code = solution.code.strip()
        imports = cls._normalize_imports(cls._extract_python_fence(original_imports))
        code = cls._extract_python_fence(original_code)
        code = code.replace("\r\n", "\n").replace("\t", "    ")
        if code and not code.endswith("\n"):
            code = f"{code}\n"
        candidate = cls._normalize_solution(
            CodeSolution(prefix=solution.prefix, imports=imports, code=code)
        )

        # Guard 1: never drop all imports when original solution had concrete imports.
        original_import_lines = cls._extract_import_lines(original_imports)
        candidate_import_lines = cls._extract_import_lines(candidate.imports)
        if original_import_lines and not candidate_import_lines:
            candidate = cls._normalize_solution(
                CodeSolution(prefix=candidate.prefix, imports=original_imports, code=candidate.code)
            )
            notes.append("import_drop_guard")

        # Guard 2: if patching makes AST invalid while original was valid, rollback.
        original_valid = cls._is_ast_valid(original_imports, original_code)
        candidate_valid = cls._is_ast_valid(candidate.imports, candidate.code)
        if original_valid and not candidate_valid:
            candidate = cls._normalize_solution(
                CodeSolution(prefix=solution.prefix, imports=original_imports, code=original_code)
            )
            notes.append("ast_regression_guard")

        if candidate.imports != original_imports:
            notes.append("imports_normalized")
        if candidate.code.strip() != original_code:
            notes.append("code_sanitized")
        return candidate, notes

    @staticmethod
    def _static_analyze(imports: str, code: str) -> tuple[bool, str]:
        combined = "\n\n".join(part for part in [imports.strip(), code.strip()] if part).strip()
        if not combined:
            return False, "Generated output was empty."
        try:
            ast.parse(combined)
        except SyntaxError as exc:
            return False, f"AST parse failed: {exc.msg} at line {exc.lineno}, column {exc.offset}."
        if "eval(" in combined or "exec(" in combined:
            return True, "Static warning: dynamic execution detected (eval/exec)."
        return True, "Static analysis passed."

    @staticmethod
    def _classify_runtime_error(error_text: str) -> str:
        lowered = error_text.lower()
        if "syntaxerror" in lowered or "indentationerror" in lowered:
            return "syntax_error"
        if "modulenotfounderror" in lowered or "importerror" in lowered:
            return "import_error"
        if "timeout" in lowered:
            return "timeout"
        if "asyncio" in lowered or "event loop" in lowered:
            return "async_error"
        if "typeerror" in lowered:
            return "type_error"
        if "valueerror" in lowered:
            return "value_error"
        if "assertionerror" in lowered:
            return "assertion_error"
        return "runtime_error"

    @staticmethod
    def _retry_plan(error_text: str, iteration: int) -> str:
        category = CodeAssistant._classify_runtime_error(error_text)
        if category == "import_error":
            return f"Retry {iteration}: fix imports and package/module usage, then revalidate."
        if category == "syntax_error":
            return f"Retry {iteration}: fix syntax/indentation and produce runnable code only."
        if category == "async_error":
            return f"Retry {iteration}: fix async lifecycle (await, event loop, task handling)."
        return f"Retry {iteration}: address {category}, add minimal deterministic self-check in code."

    @staticmethod
    def _error_fingerprint(error_text: str) -> str:
        normalized = re.sub(r"\d+", "#", error_text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized[:220]

    @staticmethod
    def _summarize_traceback(error_text: str) -> str:
        lines = [line.strip() for line in error_text.splitlines() if line.strip()]
        if not lines:
            return ""
        tail = lines[-3:]
        return " | ".join(tail)

    @staticmethod
    def _annotation_to_name(annotation: ast.expr | None) -> str:
        if annotation is None:
            return ""
        if isinstance(annotation, ast.Name):
            return annotation.id.lower()
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id.lower()
        if isinstance(annotation, ast.Attribute):
            return annotation.attr.lower()
        return ""

    @staticmethod
    def _sample_arg_for_param(param_name: str, annotation_name: str, *, edge: bool = False) -> str:
        name = param_name.lower()
        ann = annotation_name.lower()
        if ann in {"list", "sequence", "tuple"} or name in {"arr", "array", "nums", "numbers", "items", "values"}:
            return "[]" if edge else "[1, 2, 3]"
        if ann in {"dict", "mapping"} or name in {"mapping", "table", "lookup"}:
            return "{}" if edge else "{'a': 1, 'b': 2}"
        if ann in {"set"}:
            return "set()" if edge else "{1, 2, 3}"
        if ann in {"str", "string"} or name in {"s", "text", "word", "name"}:
            return "''" if edge else "'abc'"
        if ann in {"bool"} or name.startswith("is_") or name.startswith("has_"):
            return "False" if edge else "True"
        if ann in {"float"}:
            return "0.0" if edge else "2.5"
        if ann in {"int", "integer"} or name in {"n", "k", "i", "j", "target", "index"}:
            return "0" if edge else "2"
        return "None" if edge else "1"

    @staticmethod
    def _negative_arg_for_param(param_name: str, annotation_name: str) -> str:
        name = param_name.lower()
        ann = annotation_name.lower()
        if ann in {"list", "sequence", "tuple"} or name in {"arr", "array", "nums", "numbers", "items", "values"}:
            return "'not-a-list'"
        if ann in {"int", "integer", "float"} or name in {"n", "k", "i", "j", "target", "index"}:
            return "'bad'"
        if ann in {"str", "string"}:
            return "123"
        if ann in {"dict", "mapping"}:
            return "[]"
        return "object()"

    @staticmethod
    def _constraint_hints_from_signature(fn: ast.FunctionDef) -> list[str]:
        hints: list[str] = []
        for arg in fn.args.args:
            ann = CodeAssistant._annotation_to_name(arg.annotation)
            pname = arg.arg.lower()
            if ann in {"list", "sequence", "tuple"} or pname in {"arr", "array", "nums", "numbers"}:
                hints.append(f"{arg.arg} should be an indexable sequence")
            if pname in {"arr", "array", "nums", "numbers"} and any(key in fn.name.lower() for key in {"binary_search", "bisect"}):
                hints.append(f"{arg.arg} should be sorted for correct behavior")
            if ann in {"int", "integer"}:
                hints.append(f"{arg.arg} should be an integer")
        return list(dict.fromkeys(hints))

    @staticmethod
    def _behavior_reasoning_lines(fn: ast.FunctionDef) -> list[str]:
        name = fn.name.lower()
        lines: list[str] = []
        if "search" in name:
            lines.append("Search-style function should return an index/flag and handle missing targets.")
        if "sort" in name:
            lines.append("Sort-style function output should preserve ordering invariants.")
        if "merge" in name:
            lines.append("Merge-style function should preserve all input elements without loss.")
        if "parse" in name:
            lines.append("Parse-style function should handle malformed inputs predictably.")
        if not lines:
            lines.append("Function should be deterministic for identical inputs.")
        return lines

    @staticmethod
    def _smart_assertion_snippet(fn: ast.FunctionDef, result_var: str, call_expr: str) -> str:
        name = fn.name.lower()
        ann = CodeAssistant._annotation_to_name(fn.returns)
        if any(key in name for key in {"search", "find", "index"}):
            return (
                f"        self.assertIsInstance({result_var}, int)\n"
                f"        self.assertTrue({result_var} >= -1)\n"
            )
        if "sort" in name:
            return (
                f"        self.assertIsInstance({result_var}, list)\n"
                f"        self.assertEqual({result_var}, sorted({result_var}))\n"
            )
        if ann in {"bool"}:
            return f"        self.assertIsInstance({result_var}, bool)\n"
        if ann in {"list", "tuple", "set", "dict"}:
            return f"        self.assertIsInstance({result_var}, {ann})\n"
        if ann in {"int", "float", "str"}:
            return f"        self.assertIsInstance({result_var}, {ann})\n"
        return f"        self.assertIsNotNone({result_var})\n"

    @staticmethod
    def _find_param_index(fn: ast.FunctionDef, candidates: set[str]) -> int:
        for idx, arg in enumerate(fn.args.args):
            if arg.arg.lower() in candidates:
                return idx
        return -1

    @staticmethod
    def _behavior_specific_tests(fn: ast.FunctionDef) -> str:
        name = fn.name
        lowered = name.lower()
        args = fn.args.args
        if any(key in lowered for key in {"search", "find", "index"}) and len(args) >= 2:
            seq_idx = CodeAssistant._find_param_index(fn, {"arr", "array", "nums", "numbers", "items", "values", "seq"})
            target_idx = CodeAssistant._find_param_index(fn, {"target", "x", "key", "value", "needle"})
            if seq_idx == -1:
                seq_idx = 0
            if target_idx == -1:
                target_idx = 1 if len(args) > 1 else 0
            call_args = ["None"] * len(args)
            call_args[seq_idx] = "arr"
            call_args[target_idx] = "target"
            call_expr = f"{name}({', '.join(call_args)})"
            return (
                "    def _oracle_linear_search(self, arr, target):\n"
                "        for i, value in enumerate(arr):\n"
                "            if value == target:\n"
                "                return i\n"
                "        return -1\n\n"
                f"    def test_{name}_oracle_alignment(self):\n"
                "        # Expected behavior: if index returned, it should point to target; otherwise target must be absent.\n"
                "        arr = [1, 2, 2, 3, 5, 8]\n"
                "        target = 2\n"
                f"        result = {call_expr}\n"
                "        self.assertIsInstance(result, int)\n"
                "        if result == -1:\n"
                "            self.assertEqual(self._oracle_linear_search(arr, target), -1)\n"
                "        else:\n"
                "            self.assertTrue(0 <= result < len(arr))\n"
                "            self.assertEqual(arr[result], target)\n\n"
                f"    def test_{name}_edge_empty_sequence(self):\n"
                "        # Edge case: empty input sequence should not produce an invalid index.\n"
                "        arr = []\n"
                "        target = 9\n"
                f"        result = {call_expr}\n"
                "        self.assertTrue(result in (-1, None) or (isinstance(result, int) and result < 0))\n\n"
                f"    def test_{name}_edge_missing_target(self):\n"
                "        # Edge case: missing target should map to not-found behavior.\n"
                "        arr = [1, 3, 5, 7]\n"
                "        target = 4\n"
                f"        result = {call_expr}\n"
                "        self.assertTrue(result in (-1, None) or (isinstance(result, int) and (result < 0 or result >= len(arr))))\n\n"
                f"    def test_{name}_edge_duplicates(self):\n"
                "        # Edge case: duplicate targets should return any valid matching index.\n"
                "        arr = [1, 2, 2, 2, 3]\n"
                "        target = 2\n"
                f"        result = {call_expr}\n"
                "        self.assertIsInstance(result, int)\n"
                "        self.assertTrue(result == -1 or arr[result] == target)\n\n"
            )

        if "sort" in lowered and len(args) >= 1:
            first = args[0].arg
            return (
                "    def _normalize_sort_result(self, original, result):\n"
                "        return original if result is None else result\n\n"
                f"    def test_{name}_order_and_multiset(self):\n"
                "        # Expected behavior: sorted order with same elements preserved.\n"
                "        arr = [3, 1, 2, 2, 5]\n"
                "        before = list(arr)\n"
                f"        result = {name}({first}=arr)\n"
                "        normalized = self._normalize_sort_result(arr, result)\n"
                "        self.assertEqual(normalized, sorted(before))\n"
                "        self.assertEqual(sorted(normalized), sorted(before))\n\n"
            )
        return ""

    @staticmethod
    def _negative_exception_for_signature(fn: ast.FunctionDef) -> str:
        # Prefer explicit type errors for input-contract violations.
        for arg in fn.args.args:
            ann = CodeAssistant._annotation_to_name(arg.annotation)
            if ann in {"list", "sequence", "tuple", "dict", "int", "float", "str", "bool"}:
                return "TypeError"
        return "Exception"

    @staticmethod
    def _estimate_confidence(
        *,
        validation_passed: bool,
        iterations: int,
        max_iterations: int,
        had_static_failure: bool,
        had_runtime_failure: bool,
        regression_test_passed: bool,
        generated_tests_present: bool,
    ) -> float:
        score = 0.9 if validation_passed else 0.3
        score -= min(0.4, 0.08 * max(0, iterations - 1))
        if had_static_failure:
            score -= 0.1
        if had_runtime_failure and validation_passed:
            score -= 0.1
        if not validation_passed and iterations >= max_iterations:
            score -= 0.1
        if generated_tests_present and regression_test_passed:
            score += 0.08
        if generated_tests_present and not regression_test_passed:
            score -= 0.25
        return round(max(0.0, min(1.0, score)), 3)

    @staticmethod
    def _generate_unit_tests(solution: CodeSolution) -> str:
        imports = solution.imports.strip()
        code = solution.code.strip()
        combined = "\n\n".join(part for part in [imports, code] if part)
        try:
            tree = ast.parse(combined)
        except Exception:
            return ""
        fns = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if not fns:
            return ""
        fn = fns[0]
        name = fn.name
        args = fn.args.args
        if any(arg.arg in {"self", "cls"} for arg in args):
            return ""
        primary_args = [
            CodeAssistant._sample_arg_for_param(
                arg.arg,
                CodeAssistant._annotation_to_name(arg.annotation),
            )
            for arg in args
        ]
        edge_args = [
            CodeAssistant._sample_arg_for_param(
                arg.arg,
                CodeAssistant._annotation_to_name(arg.annotation),
                edge=True,
            )
            for arg in args
        ]
        negative_args = [
            CodeAssistant._negative_arg_for_param(
                arg.arg,
                CodeAssistant._annotation_to_name(arg.annotation),
            )
            for arg in args
        ]
        primary_call = f"{name}({', '.join(primary_args)})"
        edge_call = f"{name}({', '.join(edge_args)})"
        negative_call = f"{name}({', '.join(negative_args)})"
        constraint_hints = CodeAssistant._constraint_hints_from_signature(fn)
        behavior_reasons = CodeAssistant._behavior_reasoning_lines(fn)
        expected_exception = CodeAssistant._negative_exception_for_signature(fn)
        specialized_block = CodeAssistant._behavior_specific_tests(fn)
        hints_line = ""
        if constraint_hints:
            hints_line = "    # Signature constraints: " + "; ".join(constraint_hints[:4]) + "\n"
        reason_line = "    # Expected behavior: " + "; ".join(behavior_reasons[:3]) + "\n"
        primary_asserts = CodeAssistant._smart_assertion_snippet(fn, "result", primary_call)
        edge_asserts = CodeAssistant._smart_assertion_snippet(fn, "result", edge_call)
        test_code = (
            "import unittest\n\n"
            f"class TestGenerated(unittest.TestCase):\n"
            f"{hints_line}"
            f"{reason_line}"
            f"    def test_{name}_positive_case(self):\n"
            "        # Positive case: canonical valid input should produce a stable meaningful result.\n"
            f"        result = {primary_call}\n"
            f"{primary_asserts}\n"
            f"    def test_{name}_edge_case(self):\n"
            "        # Edge case: minimal/empty-like input should be handled safely.\n"
            f"        result = {edge_call}\n"
            f"{edge_asserts}\n"
            f"    def test_{name}_negative_input_type(self):\n"
            f"        # Exception-aware test: invalid input types should raise {expected_exception}.\n"
            f"        with self.assertRaises({expected_exception}):\n"
            f"            {negative_call}\n\n"
            f"{specialized_block}"
            f"    def test_{name}_deterministic_behavior(self):\n"
            "        # Expected behavior: identical input should produce identical output.\n"
            f"        first = {primary_call}\n"
            f"        second = {primary_call}\n"
            "        self.assertEqual(first, second)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        return test_code

    @staticmethod
    def _verify_environment(imports: str) -> tuple[bool, str]:
        missing: list[str] = []
        for line in imports.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("import "):
                module = stripped.split()[1].split(".")[0]
                if importlib.util.find_spec(module) is None:
                    missing.append(module)
            elif stripped.startswith("from "):
                parts = stripped.split()
                if len(parts) >= 2:
                    module = parts[1].split(".")[0]
                    if importlib.util.find_spec(module) is None:
                        missing.append(module)
        if missing:
            return False, "Missing modules: " + ", ".join(sorted(set(missing)))
        return True, "Environment verification passed."

    @staticmethod
    def _dependency_graph_hint(code: str) -> str:
        try:
            tree = ast.parse(code)
        except Exception:
            return ""
        function_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        class_names = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        calls: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        return (
            f"Functions: {', '.join(function_names[:8]) or 'none'}; "
            f"Classes: {', '.join(class_names[:6]) or 'none'}; "
            f"Calls: {', '.join(sorted(calls)[:14]) or 'none'}"
        )

    @staticmethod
    def _unified_diff(before: str, after: str) -> str:
        if not before.strip():
            return ""
        diff_lines = list(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile="previous.py",
                tofile="repaired.py",
                lineterm="",
            )
        )
        return "\n".join(diff_lines[:2200])

    @staticmethod
    def _hallucination_risk(
        solution: CodeSolution,
        rag_context: str,
        rag_enabled: bool,
        *,
        source_count: int = 0,
        validation_passed: bool = False,
        regression_passed: bool = False,
    ) -> float:
        if not rag_enabled or not rag_context.strip():
            return 0.5
        answer_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", solution.code.lower()))
        ctx_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", rag_context.lower()))
        if not answer_tokens:
            return 0.7
        overlap = len(answer_tokens & ctx_tokens) / max(1, len(answer_tokens))
        risk = 1.0 - min(1.0, overlap * 1.8)
        if source_count >= 2:
            risk -= 0.12
        if validation_passed:
            risk -= 0.08
        if regression_passed:
            risk -= 0.08
        if source_count >= 2 and validation_passed and regression_passed and overlap >= 0.2:
            risk = min(risk, 0.1)
        return round(max(0.0, min(1.0, risk)), 3)

    def _run_regression_tests(self, snippet: str) -> tuple[bool, str]:
        policy = SandboxPolicy(
            timeout_seconds=max(8, self.validation_timeout_seconds + 3),
            block_unsafe_imports=False,
        )
        result = run_python_snippet(
            snippet,
            filename="generated_regression_test.py",
            policy=policy,
            sandbox_cmd=self.sandbox_cmd,
        )
        if result.ok:
            return True, "Regression tests passed."
        output = result.output or "Regression tests failed."
        return False, output

    def _build_graph(self):
        chain = self._build_chain()
        fallback_prompt, fallback_llm = self._build_fallback_components()
        builder = StateGraph(GraphState)

        def run_validation(snippet: str, *, filename: str) -> tuple[bool, str]:
            policy = SandboxPolicy(timeout_seconds=self.validation_timeout_seconds)
            result = run_python_snippet(
                snippet,
                filename=filename,
                policy=policy,
                sandbox_cmd=self.sandbox_cmd,
            )
            if not result.ok:
                error_output = result.output or "The generated script exited with a non-zero status."
                return False, error_output
            return True, ""

        def generate(state: GraphState) -> dict[str, Any]:
            messages = state["messages"]
            iterations = state["iterations"]
            events = state.get("events", [])
            project_context = self._project_context_text(state.get("rag_context", ""))
            retry_plan = state.get("retry_plan", "").strip()
            failure_memory = state.get("failure_memory", [])
            planning_prefix = ""
            if retry_plan:
                planning_prefix += f"Retry plan:\n{retry_plan}\n\n"
            if failure_memory:
                planning_prefix += "Recent failure memory:\n" + "\n".join(f"- {item}" for item in failure_memory[-4:]) + "\n\n"
            try:
                code_solution = self._normalize_solution(
                    chain.invoke(
                        {
                            "messages": ([("system", planning_prefix)] + messages) if planning_prefix else messages,
                            "project_context": project_context,
                        }
                    )
                )
            except Exception:
                if self.provider == "local" or fallback_prompt is None or fallback_llm is None:
                    raise
                fallback_messages = fallback_prompt.invoke(
                    {
                        "messages": ([("system", planning_prefix)] + messages) if planning_prefix else messages,
                        "project_context": project_context,
                    }
                )
                fallback_response = fallback_llm.invoke(fallback_messages)
                code_solution = self._parse_fallback_response(fallback_response.content)
            code_solution, patch_notes = self._attempt_ast_patch(code_solution)
            dependency_hint = self._dependency_graph_hint(code_solution.code)
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
                ]
                + (
                    [
                        {
                            "stage": "ast_patch",
                            "status": "done",
                            "iteration": iterations + 1,
                            "detail": f"Applied patches: {', '.join(patch_notes)}",
                        }
                    ]
                    if patch_notes
                    else []
                )
                + (
                    [
                        {
                            "stage": "dependency_graph",
                            "status": "done",
                            "iteration": iterations + 1,
                            "detail": dependency_hint or "No dependency hint available.",
                        }
                    ]
                    if dependency_hint
                    else []
                ),
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations + 1,
                "retry_plan": "",
            }

        def retrieve_context(state: GraphState) -> dict[str, Any]:
            events = state.get("events", [])
            if self.rag is None:
                return {"rag_context": "", "rag_sources": [], "events": events}

            try:
                bundle = self.rag.retrieve(state["question"])
            except Exception as exc:
                return {
                    "rag_context": "",
                    "rag_sources": [],
                    "events": events
                    + [
                        {
                            "stage": "retrieve_context",
                            "status": "error",
                            "iteration": 0,
                            "detail": str(exc),
                        }
                    ],
                }

            status = "done" if bundle.context else "skipped"
            return {
                "rag_context": bundle.context,
                "rag_sources": bundle.sources,
                "events": events
                + [
                    {
                        "stage": "retrieve_context",
                        "status": status,
                        "iteration": 0,
                        "detail": bundle.detail,
                    }
                ],
            }

        def code_check(state: GraphState) -> dict[str, Any]:
            messages = state["messages"]
            code_solution = state["generation"]
            iterations = state["iterations"]
            events = state.get("events", [])
            imports = code_solution.imports.strip()
            code = code_solution.code.strip()
            combined_code = "\n\n".join(part for part in [imports, code] if part)
            static_ok, static_detail = self._static_analyze(imports, code)
            if not static_ok:
                plan = self._retry_plan(static_detail, iterations + 1)
                memory = state.get("failure_memory", []) + [static_detail]
                return {
                    "events": events
                    + [
                        {
                            "stage": "static_analysis",
                            "status": "error",
                            "iteration": iterations,
                            "detail": static_detail,
                        },
                        {
                            "stage": "retry_or_end",
                            "status": ("done" if iterations >= self.max_iterations else "running"),
                            "iteration": iterations,
                            "detail": (
                                "Stopped after reaching the retry limit."
                                if iterations >= self.max_iterations
                                else "Queued retry after static analysis failure."
                            ),
                        },
                    ],
                    "generation": code_solution,
                    "messages": messages
                    + [
                        (
                            "user",
                            f"Static analysis failed: {static_detail}. Return corrected runnable code and include a tiny self-check.",
                        )
                    ],
                    "iterations": iterations,
                    "error": "yes",
                    "retry_plan": plan,
                    "failure_memory": memory,
                    "last_error": static_detail,
                }

            env_ok, env_detail = self._verify_environment(imports)
            if not env_ok:
                plan = self._retry_plan(env_detail, iterations + 1)
                memory = state.get("failure_memory", []) + [env_detail]
                return {
                    "events": events
                    + [
                        {
                            "stage": "environment_verification",
                            "status": "error",
                            "iteration": iterations,
                            "detail": env_detail,
                        },
                        {
                            "stage": "retry_or_end",
                            "status": ("done" if iterations >= self.max_iterations else "running"),
                            "iteration": iterations,
                            "detail": (
                                "Stopped after reaching the retry limit."
                                if iterations >= self.max_iterations
                                else "Queued retry after environment verification failure."
                            ),
                        },
                    ],
                    "generation": code_solution,
                    "messages": messages
                    + [
                        ("user", f"Environment verification failed: {env_detail}. Use only available modules or stdlib."),
                    ],
                    "iterations": iterations,
                    "error": "yes",
                    "retry_plan": plan,
                    "failure_memory": memory,
                    "last_error": env_detail,
                    "error_fingerprints": dict(state.get("error_fingerprints", {})),
                }

            imports_ok, imports_error = run_validation(
                imports or "pass\n",
                filename="imports_check.py",
            )
            if not imports_ok:
                trace_context = ""
                if self.rag is not None:
                    try:
                        trace_bundle = self.rag.retrieve(f"{state['question']}\nTraceback:\n{imports_error}")
                        trace_context = trace_bundle.context[:2500]
                    except Exception:
                        trace_context = ""
                error_category = self._classify_runtime_error(imports_error)
                plan = self._retry_plan(imports_error, iterations + 1)
                memory = state.get("failure_memory", []) + [f"{error_category}: {imports_error[:300]}"]
                fingerprint = self._error_fingerprint(imports_error)
                fingerprints = dict(state.get("error_fingerprints", {}))
                fingerprints[fingerprint] = fingerprints.get(fingerprint, 0) + 1
                should_terminate = fingerprints[fingerprint] >= 2
                return {
                    "events": events
                    + [
                        {
                            "stage": "static_analysis",
                            "status": "done",
                            "iteration": iterations,
                            "detail": static_detail,
                        },
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
                                else (
                                    "Terminated early due to repeated identical import failure."
                                    if should_terminate
                                    else "Queued a corrected retry after the import failure."
                                )
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
                            "return a full corrected solution.\n"
                            f"Retry plan: {plan}\n"
                            + (f"Traceback-aware context:\n{trace_context}\n" if trace_context else ""),
                        )
                    ],
                    "iterations": self.max_iterations if should_terminate else iterations,
                    "error": "yes",
                    "retry_plan": plan,
                    "failure_memory": memory,
                    "last_error": imports_error,
                    "error_fingerprints": fingerprints,
                }

            code_ok, code_error = run_validation(
                combined_code or "pass\n",
                filename="code_check.py",
            )
            if not code_ok:
                trace_context = ""
                if self.rag is not None:
                    try:
                        trace_bundle = self.rag.retrieve(f"{state['question']}\nTraceback:\n{code_error}")
                        trace_context = trace_bundle.context[:2500]
                    except Exception:
                        trace_context = ""
                error_category = self._classify_runtime_error(code_error)
                plan = self._retry_plan(code_error, iterations + 1)
                memory = state.get("failure_memory", []) + [f"{error_category}: {code_error[:300]}"]
                fingerprint = self._error_fingerprint(code_error)
                fingerprints = dict(state.get("error_fingerprints", {}))
                fingerprints[fingerprint] = fingerprints.get(fingerprint, 0) + 1
                should_terminate = fingerprints[fingerprint] >= 2
                return {
                    "events": events
                    + [
                        {
                            "stage": "static_analysis",
                            "status": "done",
                            "iteration": iterations,
                            "detail": static_detail,
                        },
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
                                else (
                                    "Terminated early due to repeated identical runtime failure."
                                    if should_terminate
                                    else "Queued a corrected retry after the runtime failure."
                                )
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
                            "return a full corrected solution.\n"
                            f"Retry plan: {plan}\n"
                            + (f"Traceback-aware context:\n{trace_context}\n" if trace_context else ""),
                        )
                    ],
                    "iterations": self.max_iterations if should_terminate else iterations,
                    "error": "yes",
                    "retry_plan": plan,
                    "failure_memory": memory,
                    "last_error": code_error,
                    "error_fingerprints": fingerprints,
                }

            return {
                "events": events
                + [
                    {
                        "stage": "static_analysis",
                        "status": "done",
                        "iteration": iterations,
                        "detail": static_detail,
                    },
                    {
                        "stage": "environment_verification",
                        "status": "done",
                        "iteration": iterations,
                        "detail": env_detail,
                    },
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
                "retry_plan": "",
                "error_fingerprints": state.get("error_fingerprints", {}),
                "previous_combined_code": combined_code,
            }

        def decide_to_finish(state: GraphState) -> Literal["end", "generate"]:
            if state["error"] == "no" or state["iterations"] >= self.max_iterations:
                return "end"
            return "generate"

        builder.add_node("retrieve_context", retrieve_context)
        builder.add_node("generate", generate)
        builder.add_node("check_code", code_check)
        builder.add_edge(START, "retrieve_context")
        builder.add_edge("retrieve_context", "generate")
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
            {
                "error": "pending",
                "events": [],
                "messages": [("user", question)],
                "iterations": 0,
                "question": question,
                "rag_context": "",
                "rag_sources": [],
                "failure_memory": [],
                "retry_plan": "",
                "last_error": "",
                "error_fingerprints": {},
                "previous_combined_code": "",
            },
            config=config,
        )
        events = result.get("events", [])
        had_static_failure = any(
            str(event.get("stage", "")) == "static_analysis" and str(event.get("status", "")) == "error"
            for event in events
        )
        had_runtime_failure = any(
            str(event.get("stage", "")) == "check_result" and str(event.get("status", "")) == "error"
            for event in events
        )
        validation_passed = result.get("error") != "yes"
        last_error = str(result.get("last_error", "") or "")
        result["traceback_summary"] = self._summarize_traceback(last_error)
        generation = result.get("generation")
        previous = str(result.get("previous_combined_code", "") or "")
        code_diff = ""
        if isinstance(generation, CodeSolution) and validation_passed:
            result["generated_tests"] = self._generate_unit_tests(generation)
            combined = "\n\n".join(part for part in [generation.imports.strip(), generation.code.strip()] if part)
            code_diff = self._unified_diff(previous, combined)
        else:
            result["generated_tests"] = ""
        result["repair_diff"] = code_diff
        regression_passed = False
        if isinstance(generation, CodeSolution):
            result["hallucination_risk"] = self._hallucination_risk(
                generation,
                str(result.get("rag_context", "") or ""),
                bool(self.rag is not None),
                source_count=len(result.get("rag_sources", []) or []),
                validation_passed=validation_passed,
                regression_passed=False,
            )
        else:
            result["hallucination_risk"] = 0.5
        generated_tests_present = bool(result.get("generated_tests"))
        if generated_tests_present and isinstance(generation, CodeSolution):
            snippet = "\n\n".join(
                part
                for part in [
                    generation.imports.strip(),
                    generation.code.strip(),
                    str(result.get("generated_tests", "")).strip(),
                ]
                if part
            )
            ok, err = self._run_regression_tests(snippet)
            result["regression_test_passed"] = ok
            result["regression_test_output"] = err
            regression_passed = ok
        else:
            result["regression_test_passed"] = False
            result["regression_test_output"] = ""
        semantic_validation_passed = bool(validation_passed and ((not generated_tests_present) or regression_passed))
        result["semantic_validation_passed"] = semantic_validation_passed
        if generated_tests_present and not regression_passed:
            result["error"] = "yes"
            result["events"] = list(result.get("events", [])) + [
                {
                    "stage": "semantic_validation",
                    "status": "error",
                    "iteration": int(result.get("iterations", 0) or 0),
                    "detail": "Generated unit tests did not pass. Marking result as failed.",
                }
            ]
        result["confidence_score"] = self._estimate_confidence(
            validation_passed=semantic_validation_passed,
            iterations=int(result.get("iterations", 0) or 0),
            max_iterations=self.max_iterations,
            had_static_failure=had_static_failure,
            had_runtime_failure=had_runtime_failure,
            regression_test_passed=regression_passed,
            generated_tests_present=generated_tests_present,
        )
        if isinstance(generation, CodeSolution):
            result["hallucination_risk"] = self._hallucination_risk(
                generation,
                str(result.get("rag_context", "") or ""),
                bool(self.rag is not None),
                source_count=len(result.get("rag_sources", []) or []),
                validation_passed=semantic_validation_passed,
                regression_passed=regression_passed,
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
            {
                "error": "pending",
                "events": [],
                "messages": [("user", question)],
                "iterations": 0,
                "question": question,
                "rag_context": "",
                "rag_sources": [],
                "failure_memory": [],
                "retry_plan": "",
                "last_error": "",
                "error_fingerprints": {},
                "previous_combined_code": "",
            },
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

    @staticmethod
    def classify_failure(result: dict[str, Any]) -> FailureDiagnostics:
        if result.get("error") != "yes":
            return FailureDiagnostics(category="none", stage="none", summary="")

        events = result.get("events", [])
        for event in events:
            if event.get("stage") == "retrieve_context" and event.get("status") == "error":
                detail = str(event.get("detail", "")).strip()
                return FailureDiagnostics(
                    category="retrieval_error",
                    stage="retrieve_context",
                    summary=detail or "Project retrieval failed before generation.",
                )

        for event in reversed(events):
            detail = str(event.get("detail", "")).strip()
            stage = str(event.get("stage", "")).strip() or "unknown"
            if "Import validation failed:" in detail or "Import execution failed" in detail:
                return FailureDiagnostics(
                    category="import_validation_error",
                    stage=stage,
                    summary=detail,
                )
            if "AST parse failed:" in detail or "Static analysis failed:" in detail:
                return FailureDiagnostics(
                    category="static_analysis_error",
                    stage=stage,
                    summary=detail,
                )
            if "Runtime validation failed:" in detail:
                inner = detail.split("Runtime validation failed:", 1)[-1].strip()
                cat = CodeAssistant._classify_runtime_error(inner)
                return FailureDiagnostics(
                    category=cat,
                    stage=stage,
                    summary=detail,
                )
            if "timed out" in detail.lower():
                return FailureDiagnostics(
                    category="timeout",
                    stage=stage,
                    summary=detail,
                )

        iterations = int(result.get("iterations", 0) or 0)
        return FailureDiagnostics(
            category="retry_limit_reached",
            stage="retry_or_end",
            summary=f"The assistant stopped after {iterations} iteration(s) without a validated solution.",
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
            "runtime_profile": self.runtime_profile,
            "question": question,
            "iterations": result.get("iterations"),
            "error": result.get("error"),
            "failure_diagnostics": self.classify_failure(result).model_dump(),
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
