from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _split_csv(value: str | None, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if not value:
        return default
    parts = [item.strip() for item in value.split(",")]
    return tuple(item for item in parts if item)


def _int_env(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    value = max(minimum, value)
    if maximum is not None:
        value = min(value, maximum)
    return value


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BackendSettings:
    project_root: Path
    public_dir: Path
    allowed_origins: tuple[str, ...]
    allowed_providers: tuple[str, ...]
    default_provider: str
    default_runtime_profile: str
    auth_token: str
    max_iterations_cap: int
    validation_timeout_cap: int
    rate_limit_requests: int
    rate_limit_window_seconds: int
    log_destination: str
    failure_log_path: Path
    upstash_redis_rest_url: str
    upstash_redis_rest_token: str
    failure_log_key: str
    allow_credentials: bool
    rag_enabled: bool
    rag_auto_index: bool
    rag_qdrant_path: Path
    rag_collection_name: str
    rag_embedding_model: str
    rag_retrieval_k: int
    rag_chunk_size: int
    rag_chunk_overlap: int
    corrective_rag_enabled: bool
    corrective_rag_model: str
    corrective_rag_mode: str
    corrective_rag_min_score: int
    corrective_rag_retry_k: int


@lru_cache(maxsize=1)
def get_settings() -> BackendSettings:
    project_root = Path(__file__).resolve().parents[2]
    public_dir = project_root / "public"
    allowed_providers = _split_csv(
        os.getenv("CODE_ASSISTANT_ALLOWED_PROVIDERS"),
        default=("mistral",),
    )
    default_provider = os.getenv("CODE_ASSISTANT_DEFAULT_PROVIDER", allowed_providers[0]).strip() or allowed_providers[0]
    if default_provider not in allowed_providers:
        default_provider = allowed_providers[0]
    default_runtime_profile = os.getenv("CODE_ASSISTANT_DEFAULT_RUNTIME_PROFILE", "custom").strip().lower() or "custom"
    if default_runtime_profile not in {"custom", "fast", "balanced", "accurate"}:
        default_runtime_profile = "custom"
    corrective_rag_mode = os.getenv("CODE_ASSISTANT_CORRECTIVE_RAG_MODE", "balanced").strip().lower() or "balanced"
    if corrective_rag_mode not in {"fast", "balanced", "aggressive"}:
        corrective_rag_mode = "balanced"

    require_access_token = _bool_env("CODE_ASSISTANT_REQUIRE_ACCESS_TOKEN", False)

    return BackendSettings(
        project_root=project_root,
        public_dir=public_dir,
        allowed_origins=_split_csv(os.getenv("CODE_ASSISTANT_ALLOWED_ORIGINS")),
        allowed_providers=allowed_providers,
        default_provider=default_provider,
        default_runtime_profile=default_runtime_profile,
        auth_token=(
            os.getenv("CODE_ASSISTANT_ACCESS_TOKEN", "").strip()
            if require_access_token
            else ""
        ),
        max_iterations_cap=_int_env("CODE_ASSISTANT_MAX_ITERATIONS_CAP", 3, minimum=1, maximum=10),
        validation_timeout_cap=_int_env("CODE_ASSISTANT_VALIDATION_TIMEOUT_CAP", 5, minimum=1, maximum=30),
        rate_limit_requests=_int_env("CODE_ASSISTANT_RATE_LIMIT_REQUESTS", 8, minimum=1, maximum=200),
        rate_limit_window_seconds=_int_env("CODE_ASSISTANT_RATE_LIMIT_WINDOW_SECONDS", 300, minimum=10, maximum=86400),
        log_destination=os.getenv("CODE_ASSISTANT_LOG_DESTINATION", "none").strip().lower() or "none",
        failure_log_path=project_root / os.getenv("CODE_ASSISTANT_FAILURE_LOG", "data/runtime/failure_log.jsonl"),
        upstash_redis_rest_url=os.getenv("UPSTASH_REDIS_REST_URL", "").strip(),
        upstash_redis_rest_token=os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip(),
        failure_log_key=os.getenv("CODE_ASSISTANT_FAILURE_LOG_KEY", "code-assistant:failures").strip() or "code-assistant:failures",
        allow_credentials=_bool_env("CODE_ASSISTANT_ALLOW_CREDENTIALS", False),
        rag_enabled=_bool_env("CODE_ASSISTANT_RAG_ENABLED", False),
        rag_auto_index=_bool_env("CODE_ASSISTANT_RAG_AUTO_INDEX", False),
        rag_qdrant_path=project_root / os.getenv("CODE_ASSISTANT_RAG_QDRANT_PATH", "data/qdrant"),
        rag_collection_name=os.getenv("CODE_ASSISTANT_RAG_COLLECTION", "code-assistant-project").strip() or "code-assistant-project",
        rag_embedding_model=os.getenv("CODE_ASSISTANT_RAG_EMBED_MODEL", "mistral-embed").strip() or "mistral-embed",
        rag_retrieval_k=_int_env("CODE_ASSISTANT_RAG_RETRIEVAL_K", 4, minimum=1, maximum=12),
        rag_chunk_size=_int_env("CODE_ASSISTANT_RAG_CHUNK_SIZE", 1200, minimum=200, maximum=4000),
        rag_chunk_overlap=_int_env("CODE_ASSISTANT_RAG_CHUNK_OVERLAP", 200, minimum=0, maximum=1000),
        corrective_rag_enabled=_bool_env("CODE_ASSISTANT_CORRECTIVE_RAG_ENABLED", True),
        corrective_rag_model=os.getenv("CODE_ASSISTANT_CORRECTIVE_RAG_MODEL", "mistral-small-latest").strip() or "mistral-small-latest",
        corrective_rag_mode=corrective_rag_mode,
        corrective_rag_min_score=_int_env("CODE_ASSISTANT_CORRECTIVE_RAG_MIN_SCORE", 3, minimum=1, maximum=5),
        corrective_rag_retry_k=_int_env("CODE_ASSISTANT_CORRECTIVE_RAG_RETRY_K", 6, minimum=1, maximum=12),
    )
