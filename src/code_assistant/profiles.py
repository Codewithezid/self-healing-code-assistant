from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    provider: str
    model: str
    rag_enabled: bool
    corrective_rag_mode: str
    max_iterations: int
    validation_timeout: int


RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    "fast": RuntimeProfile(
        name="fast",
        provider="mistral",
        model="codestral-latest",
        rag_enabled=False,
        corrective_rag_mode="fast",
        max_iterations=1,
        validation_timeout=3,
    ),
    "balanced": RuntimeProfile(
        name="balanced",
        provider="mistral",
        model="mistral-large-latest",
        rag_enabled=True,
        corrective_rag_mode="balanced",
        max_iterations=2,
        validation_timeout=5,
    ),
    "accurate": RuntimeProfile(
        name="accurate",
        provider="mistral",
        model="mistral-large-latest",
        rag_enabled=True,
        corrective_rag_mode="aggressive",
        max_iterations=3,
        validation_timeout=5,
    ),
}


def get_runtime_profile(name: str | None) -> RuntimeProfile | None:
    if not name:
        return None
    return RUNTIME_PROFILES.get(name.strip().lower())
