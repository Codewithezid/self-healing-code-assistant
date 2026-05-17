from __future__ import annotations

from src.code_assistant.secure_executor import SandboxPolicy, run_python_snippet


def test_secure_executor_blocks_unsafe_imports() -> None:
    result = run_python_snippet(
        "import socket\nprint('x')\n",
        filename="blocked.py",
        policy=SandboxPolicy(timeout_seconds=2),
    )
    assert result.ok is False
    assert "Blocked unsafe import" in result.output


def test_secure_executor_runs_safe_snippet() -> None:
    result = run_python_snippet(
        "print('hello')\n",
        filename="ok.py",
        policy=SandboxPolicy(timeout_seconds=2),
    )
    assert result.ok is True
