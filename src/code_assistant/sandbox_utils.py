from __future__ import annotations

import os
import shlex


def _merge_windows_executable(tokens: list[str]) -> list[str]:
    if os.name != "nt" or len(tokens) < 2:
        return tokens
    first = tokens[0]
    if ":" not in first or first.startswith("-") or " " in first:
        return tokens

    merged = first
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-"):
            break
        merged = f"{merged} {token}"
        index += 1
        if merged.lower().endswith((".exe", ".cmd", ".bat", ".com", ".ps1")):
            break

    return [merged, *tokens[index:]]


def _strip_wrapping_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


def parse_sandbox_cmd(raw: str) -> tuple[str, ...]:
    value = raw.strip()
    if not value:
        return ()
    try:
        tokens = shlex.split(value, posix=(os.name != "nt"))
    except ValueError:
        return ()
    if os.name == "nt":
        tokens = _merge_windows_executable(tokens)
    cleaned = [_strip_wrapping_quotes(token) for token in tokens]
    return tuple(token for token in cleaned if token)
