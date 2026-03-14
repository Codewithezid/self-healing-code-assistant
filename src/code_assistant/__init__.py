"""LangGraph code assistant package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .assistant import CodeAssistant
    from .models import CodeSolution

__all__ = ["CodeAssistant", "CodeSolution"]


def __getattr__(name: str):
    if name == "CodeAssistant":
        from .assistant import CodeAssistant

        return CodeAssistant
    if name == "CodeSolution":
        from .models import CodeSolution

        return CodeSolution
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
