from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from pydantic import field_validator


class CodeSolution(BaseModel):
    """Structured model output for a generated code solution."""

    prefix: str = Field(description="Short explanation of the approach.")
    imports: str = Field(description="The import statements needed for the solution.")
    code: str = Field(description="The executable code body without the import statements.")

    @field_validator("prefix", "imports", "code", mode="before")
    @classmethod
    def _coerce_to_string(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        if isinstance(value, dict):
            return "\n".join(f"{key}: {item}" for key, item in value.items())
        return str(value)


class FailureDiagnostics(BaseModel):
    category: str = "none"
    stage: str = "none"
    summary: str = ""
