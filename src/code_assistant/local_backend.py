from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import CodeSolution


class LocalCodeGenerator:
    """Generate structured code solutions with a local Hugging Face model."""

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        max_new_tokens: int = 768,
    ) -> None:
        try:
            import torch
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local model support requires the packages in requirements-local.txt."
            ) from exc

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_path = Path(model_name)
        if model_path.exists() and (model_path / "adapter_config.json").exists():
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )

    @staticmethod
    def _extract_code_fence(text: str) -> str | None:
        match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip()

    @classmethod
    def _parse_structured(cls, text: str) -> dict[str, Any]:
        cleaned = text.strip()

        # 1) Strict JSON (possibly wrapped in ```json fences).
        if cleaned.startswith("```"):
            unfenced = cleaned.strip("`").strip()
            if unfenced.lower().startswith("json"):
                unfenced = unfenced[4:].strip()
            cleaned = unfenced
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2) Best-effort JSON extraction from a larger response.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass

        # 3) Parse "Imports:" / "Code:" style output.
        # Example:
        #   <prefix>
        #   Imports:
        #   ...
        #   Code:
        #   ...
        lower = cleaned.lower()
        imports_index = lower.find("imports:")
        code_index = lower.find("code:")
        if code_index != -1:
            prefix = cleaned[:imports_index].strip() if imports_index != -1 else cleaned[:code_index].strip()
            imports_block = ""
            if imports_index != -1 and imports_index < code_index:
                imports_block = cleaned[imports_index + len("imports:") : code_index].strip()
            code_block = cleaned[code_index + len("code:") :].strip()
            fenced = cls._extract_code_fence(code_block)
            if fenced:
                code_block = fenced
            return {"prefix": prefix, "imports": imports_block, "code": code_block}

        # 4) Parse a python code fence if present.
        fenced = cls._extract_code_fence(cleaned)
        if fenced:
            return {"prefix": "", "imports": "", "code": fenced}

        # 5) Heuristic: treat leading import lines as imports and the rest as code.
        lines = cleaned.splitlines()
        import_lines: list[str] = []
        code_lines: list[str] = []
        in_imports = True
        for line in lines:
            stripped = line.strip()
            if in_imports and (stripped.startswith("import ") or stripped.startswith("from ")):
                import_lines.append(line)
                continue
            in_imports = False
            code_lines.append(line)

        if not code_lines and import_lines:
            code_lines = ["pass"]

        if not code_lines:
            raise ValueError("Local model response could not be parsed into code.")

        return {"prefix": "", "imports": "\n".join(import_lines).strip(), "code": "\n".join(code_lines).strip()}

    @staticmethod
    def _normalize_messages(
        messages: list[Any],
        *,
        project_context: str = "",
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a careful coding assistant. Respond with valid JSON only. "
                    'The JSON object must contain exactly these keys: "prefix", "imports", and "code". '
                    "The first character must be '{' and the last character must be '}'. "
                    "Do not wrap the JSON in markdown."
                ),
            }
        ]
        if project_context.strip():
            normalized.append(
                {
                    "role": "system",
                    "content": f"Relevant project context:\n{project_context}",
                }
            )

        for message in messages:
            if isinstance(message, tuple) and len(message) == 2:
                role, content = message
                normalized.append({"role": role, "content": str(content)})
                continue

            role = getattr(message, "type", "user")
            content = getattr(message, "content", str(message))
            mapped_role = {
                "human": "user",
                "ai": "assistant",
                "system": "system",
            }.get(role, "user")
            normalized.append({"role": mapped_role, "content": str(content)})
        return normalized

    def invoke(self, payload: dict[str, Any]) -> CodeSolution:
        messages = self._normalize_messages(
            payload.get("messages", []),
            project_context=str(payload.get("project_context", "")),
        )
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        generated = outputs[0][prompt_length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        payload = self._parse_structured(text)
        return CodeSolution.model_validate(payload)
