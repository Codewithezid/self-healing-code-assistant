from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

if os.name != "nt":
    import resource


@dataclass(frozen=True)
class SandboxPolicy:
    timeout_seconds: int
    max_output_chars: int = 12000
    block_unsafe_imports: bool = True
    memory_limit_mb: int = 512
    cpu_seconds: int = 8
    nofile_limit: int = 64


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    output: str
    exit_code: int
    timed_out: bool = False


UNSAFE_IMPORT_PATTERN = re.compile(
    r"^\s*(?:import|from)\s+(socket|requests|httpx|urllib|subprocess|multiprocessing|ctypes|winreg)\b",
    flags=re.MULTILINE,
)


def _truncate(text: str, *, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars]}...[truncated]"


def _sandbox_preexec(policy: SandboxPolicy):
    def _apply() -> None:
        # CPU + memory/file guardrails for POSIX runtimes.
        resource.setrlimit(resource.RLIMIT_CPU, (policy.cpu_seconds, policy.cpu_seconds + 1))
        mem_bytes = policy.memory_limit_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (policy.nofile_limit, policy.nofile_limit))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
        except Exception:
            pass

    return _apply


def run_python_snippet(
    snippet: str,
    *,
    filename: str,
    policy: SandboxPolicy,
    sandbox_cmd: list[str] | None = None,
) -> SandboxResult:
    if policy.block_unsafe_imports and UNSAFE_IMPORT_PATTERN.search(snippet):
        return SandboxResult(
            ok=False,
            output="Blocked unsafe import for sandbox policy. Network/process/native imports are not allowed.",
            exit_code=126,
            timed_out=False,
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = Path(temp_dir) / filename
        script_path.write_text(snippet, encoding="utf-8")

        safe_env = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HOME": temp_dir,
            "TMPDIR": temp_dir,
            "TEMP": temp_dir,
            "TMP": temp_dir,
        }
        for key in ("SystemRoot", "WINDIR"):
            value = os.environ.get(key)
            if value:
                safe_env[key] = value

        cmd = [sys.executable, "-I", str(script_path)]
        if sandbox_cmd:
            cmd = [*sandbox_cmd, sys.executable, "-I", str(script_path)]

        kwargs: dict[str, object] = {
            "cwd": temp_dir,
            "capture_output": True,
            "text": True,
            "timeout": policy.timeout_seconds,
            "check": False,
            "env": safe_env,
        }
        if os.name != "nt":
            kwargs["preexec_fn"] = _sandbox_preexec(policy)

        try:
            completed = subprocess.run(cmd, **kwargs)
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False,
                output=f"Validation timed out after {policy.timeout_seconds} seconds.",
                exit_code=124,
                timed_out=True,
            )
        except OSError as exc:
            return SandboxResult(
                ok=False,
                output=f"Validation sandbox failed to start: {exc}",
                exit_code=125,
                timed_out=False,
            )

    output = completed.stderr.strip() or completed.stdout.strip() or ""
    output = _truncate(output, max_chars=policy.max_output_chars)
    return SandboxResult(
        ok=completed.returncode == 0,
        output=output,
        exit_code=completed.returncode,
        timed_out=False,
    )
