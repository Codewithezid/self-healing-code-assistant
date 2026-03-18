from __future__ import annotations

import os
import uuid
import warnings
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .assistant import CodeAssistant
from .key_vault import EncryptedKeyVault, StoredKeyPublic
from .models import CodeSolution, FailureDiagnostics
from .platform_utils import InMemoryRateLimiter, UpstashRateLimiter, UpstashRedis
from .profiles import RUNTIME_PROFILES, get_runtime_profile
from .provider_clients import (
    ProviderClientError,
    list_models_for_provider,
    supports_hosted_provider,
)
from .settings import BackendSettings, get_settings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    runtime_profile: Literal["custom", "fast", "balanced", "accurate"] = "custom"
    provider: str = "mistral"
    model: str = "mistral-medium-latest"
    local_model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    max_iterations: int = Field(default=3, ge=1, le=10)
    validation_timeout: int = Field(default=5, ge=1, le=30)
    show_events: bool = False
    json_mode: bool = False
    tracing: bool = False
    provider_key_id: str | None = None
    rag_enabled: bool | None = None
    corrective_rag_mode: Literal["fast", "balanced", "aggressive"] | None = None


class RunEvent(BaseModel):
    stage: str
    status: str
    iteration: int | None = None
    detail: str


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["success", "error"]
    provider: str
    model: str
    iterations: int
    max_iterations: int
    validation_timeout: int
    solution: CodeSolution
    combined_code: str
    validation_passed: bool
    validation_message: str
    events: list[RunEvent]
    tracing_requested: bool
    json_mode: bool
    rag_enabled: bool
    rag_sources: list[str]
    corrective_rag_mode: str
    runtime_profile: str
    failure_diagnostics: FailureDiagnostics


class BackendConfigResponse(BaseModel):
    allowed_providers: list[str]
    default_provider: str
    auth_required: bool
    max_iterations_cap: int
    validation_timeout_cap: int
    rate_limit_requests: int
    rate_limit_window_seconds: int
    rag_available: bool
    rag_default_enabled: bool
    corrective_rag_modes: list[str]
    corrective_rag_default_mode: str
    runtime_profiles: list[str]
    default_runtime_profile: str
    user_keys_enabled: bool
    user_keys_persistent: bool
    user_keys_max_entries: int


class StoredKeyResponse(BaseModel):
    key_id: str
    provider: str
    label: str
    masked_key: str
    created_at: str


class SaveKeyRequest(BaseModel):
    provider: str = Field(min_length=2, max_length=50)
    api_key: str = Field(min_length=8, max_length=2048)
    label: str = Field(default="", max_length=120)


class SaveKeyResponse(BaseModel):
    key: StoredKeyResponse
    models: list[str]


class DeleteKeyResponse(BaseModel):
    deleted: bool


class ProviderModelsResponse(BaseModel):
    provider: str
    models: list[str]
    source: Literal["environment", "saved_key"]
    key_id: str | None = None


def _combined_code(solution: CodeSolution) -> str:
    return "\n\n".join(
        part.strip() for part in [solution.imports, solution.code] if part.strip()
    )


def _extract_validation_message(
    events: list[dict[str, Any]],
    *,
    passed: bool,
    iterations: int,
    max_iterations: int,
) -> str:
    for event in reversed(events):
        if event.get("stage") == "check_result":
            return str(event.get("detail", "")).strip() or (
                "The generated code passed validation."
                if passed
                else "Validation failed."
            )
    if passed:
        return "The generated code passed isolated validation."
    return (
        f"The assistant reached the retry limit after {iterations} iteration(s) "
        f"out of {max_iterations}."
    )


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _extract_access_token(request: Request) -> str:
    auth_header = request.headers.get("authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return request.headers.get("x-access-token", "").strip()


def _build_rate_limiter(settings: BackendSettings):
    if settings.upstash_redis_rest_url and settings.upstash_redis_rest_token:
        return UpstashRateLimiter(
            UpstashRedis(
                base_url=settings.upstash_redis_rest_url,
                token=settings.upstash_redis_rest_token,
            )
        )
    return InMemoryRateLimiter()


def _as_key_response(record: StoredKeyPublic) -> StoredKeyResponse:
    return StoredKeyResponse(
        key_id=record.key_id,
        provider=record.provider,
        label=record.label,
        masked_key=record.masked_key,
        created_at=record.created_at,
    )


def create_app() -> FastAPI:
    settings = get_settings()
    rate_limiter = _build_rate_limiter(settings)
    key_vault = (
        EncryptedKeyVault(
            file_path=settings.user_keys_path,
            secret=settings.user_keys_secret,
            max_entries=settings.user_keys_max_entries,
        )
        if settings.user_keys_enabled
        else None
    )

    app = FastAPI(
        title="LangGraph Code Assistant",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_credentials=settings.allow_credentials,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Access-Token"],
        )

    @app.get("/api/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config", response_model=BackendConfigResponse)
    def backend_config() -> BackendConfigResponse:
        return BackendConfigResponse(
            allowed_providers=list(settings.allowed_providers),
            default_provider=settings.default_provider,
            auth_required=bool(settings.auth_token),
            max_iterations_cap=settings.max_iterations_cap,
            validation_timeout_cap=settings.validation_timeout_cap,
            rate_limit_requests=settings.rate_limit_requests,
            rate_limit_window_seconds=settings.rate_limit_window_seconds,
            rag_available=True,
            rag_default_enabled=settings.rag_enabled,
            corrective_rag_modes=["fast", "balanced", "aggressive"],
            corrective_rag_default_mode=settings.corrective_rag_mode,
            runtime_profiles=["custom", *RUNTIME_PROFILES.keys()],
            default_runtime_profile=settings.default_runtime_profile,
            user_keys_enabled=bool(settings.user_keys_enabled),
            user_keys_persistent=bool(key_vault.persistent if key_vault else False),
            user_keys_max_entries=settings.user_keys_max_entries,
        )

    def require_auth(request: Request) -> None:
        if not settings.auth_token:
            return
        supplied_token = _extract_access_token(request)
        if supplied_token != settings.auth_token:
            raise HTTPException(status_code=401, detail="Missing or invalid access token.")

    def enforce_rate_limit(
        request: Request,
        *,
        scope: str,
        limit: int | None = None,
    ) -> None:
        client_ip = _client_ip(request)
        allowed, retry_after = rate_limiter.allow(
            f"{scope}:{client_ip}",
            limit=limit or settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        if allowed:
            return
        raise HTTPException(
            status_code=429,
            detail=(
                "Rate limit exceeded. "
                f"Try again in about {retry_after} second(s)."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    def require_user_keys_enabled() -> EncryptedKeyVault:
        if not settings.user_keys_enabled or key_vault is None:
            raise HTTPException(
                status_code=403,
                detail="User-managed API keys are disabled on this deployment.",
            )
        return key_vault

    def require_provider_allowed(provider: str) -> str:
        normalized = provider.strip().lower()
        if normalized not in settings.allowed_providers:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Provider '{normalized}' is disabled for this deployment. "
                    f"Allowed providers: {', '.join(settings.allowed_providers)}."
                ),
            )
        return normalized

    def resolve_provider_api_key(
        *,
        provider: str,
        key_id: str | None,
    ) -> tuple[str, str, str | None]:
        normalized_provider = provider.strip().lower()
        selected_key_id = (key_id or "").strip()
        if selected_key_id:
            vault = require_user_keys_enabled()
            key = vault.get_api_key(key_id=selected_key_id, provider=normalized_provider)
            if not key:
                raise HTTPException(
                    status_code=404,
                    detail="Saved API key not found for the selected provider.",
                )
            return key, "saved_key", selected_key_id

        env_key_name = {
            "openai": "OPENAI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }.get(normalized_provider, "")
        env_key = ""
        if env_key_name:
            env_key = os.getenv(env_key_name, "").strip()
        if not env_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No server key configured for this provider. "
                    "Add a user key first."
                ),
            )
        return env_key, "environment", None

    @app.get("/api/keys", response_model=list[StoredKeyResponse])
    def list_keys(
        request: Request,
        provider: str | None = Query(default=None),
    ) -> list[StoredKeyResponse]:
        require_auth(request)
        vault = require_user_keys_enabled()
        normalized_provider = provider.strip().lower() if provider else None
        if normalized_provider:
            require_provider_allowed(normalized_provider)
            if normalized_provider == "local":
                return []
        records = vault.list_keys(provider=normalized_provider)
        return [_as_key_response(record) for record in records]

    @app.post("/api/keys", response_model=SaveKeyResponse)
    def save_key(request_body: SaveKeyRequest, request: Request) -> SaveKeyResponse:
        require_auth(request)
        enforce_rate_limit(
            request,
            scope="keys",
            limit=max(3, min(settings.rate_limit_requests, 10)),
        )
        vault = require_user_keys_enabled()
        provider = require_provider_allowed(request_body.provider)
        if provider == "local":
            raise HTTPException(
                status_code=400,
                detail="Local provider does not require an API key.",
            )
        if not supports_hosted_provider(provider):
            raise HTTPException(
                status_code=400,
                detail=f"Provider '{provider}' is not supported for key testing yet.",
            )
        try:
            models = list_models_for_provider(provider, request_body.api_key)
            record = vault.add_key(
                provider=provider,
                api_key=request_body.api_key,
                label=request_body.label,
            )
        except ProviderClientError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return SaveKeyResponse(
            key=_as_key_response(record),
            models=models[:200],
        )

    @app.delete("/api/keys/{key_id}", response_model=DeleteKeyResponse)
    def delete_key(key_id: str, request: Request) -> DeleteKeyResponse:
        require_auth(request)
        vault = require_user_keys_enabled()
        return DeleteKeyResponse(deleted=vault.delete_key(key_id=key_id))

    @app.get("/api/providers/{provider}/models", response_model=ProviderModelsResponse)
    def list_provider_models(
        provider: str,
        request: Request,
        key_id: str | None = Query(default=None),
    ) -> ProviderModelsResponse:
        require_auth(request)
        enforce_rate_limit(
            request,
            scope="models",
            limit=max(5, min(settings.rate_limit_requests, 20)),
        )
        normalized_provider = require_provider_allowed(provider)
        if normalized_provider == "local":
            return ProviderModelsResponse(
                provider="local",
                models=[],
                source="environment",
                key_id=None,
            )
        if not supports_hosted_provider(normalized_provider):
            raise HTTPException(
                status_code=400,
                detail=f"Provider '{normalized_provider}' is not supported for model listing yet.",
            )

        api_key, source, resolved_key_id = resolve_provider_api_key(
            provider=normalized_provider,
            key_id=key_id,
        )
        try:
            models = list_models_for_provider(normalized_provider, api_key)
        except ProviderClientError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return ProviderModelsResponse(
            provider=normalized_provider,
            models=models[:200],
            source=source,
            key_id=resolved_key_id,
        )

    @app.post("/api/chat", response_model=ChatResponse)
    def run_chat(request_body: ChatRequest, request: Request) -> ChatResponse:
        require_auth(request)
        enforce_rate_limit(request, scope="chat")

        runtime_profile = request_body.runtime_profile
        if "runtime_profile" not in request_body.model_fields_set:
            runtime_profile = settings.default_runtime_profile
        profile = get_runtime_profile(runtime_profile)
        request_provider = request_body.provider
        resolved_model = request_body.model
        max_iterations = min(request_body.max_iterations, settings.max_iterations_cap)
        validation_timeout = min(
            request_body.validation_timeout,
            settings.validation_timeout_cap,
        )
        rag_enabled = settings.rag_enabled if request_body.rag_enabled is None else request_body.rag_enabled
        corrective_rag_mode = request_body.corrective_rag_mode or settings.corrective_rag_mode
        if profile is not None:
            request_provider = profile.provider  # profiles are authoritative presets
            resolved_model = profile.model
            max_iterations = min(profile.max_iterations, settings.max_iterations_cap)
            validation_timeout = min(profile.validation_timeout, settings.validation_timeout_cap)
            rag_enabled = profile.rag_enabled
            corrective_rag_mode = profile.corrective_rag_mode
        request_provider = require_provider_allowed(request_provider)
        provider_key_id = (request_body.provider_key_id or "").strip() or None
        selected_api_key = None
        if request_provider != "local" and provider_key_id:
            vault = require_user_keys_enabled()
            selected_api_key = vault.get_api_key(
                key_id=provider_key_id,
                provider=request_provider,
            )
            if not selected_api_key:
                raise HTTPException(
                    status_code=404,
                    detail="Saved API key not found for the selected provider.",
                )
        thread_id = str(uuid.uuid4())

        try:
            assistant = CodeAssistant(
                model_name=resolved_model,
                max_iterations=max_iterations,
                validation_timeout_seconds=validation_timeout,
                failure_log_path=str(settings.failure_log_path),
                log_destination=settings.log_destination,
                upstash_redis_rest_url=settings.upstash_redis_rest_url,
                upstash_redis_rest_token=settings.upstash_redis_rest_token,
                failure_log_key=settings.failure_log_key,
                provider=request_provider,
                local_model_name=request_body.local_model,
                rag_enabled=rag_enabled,
                rag_auto_index=settings.rag_auto_index,
                rag_project_root=str(settings.project_root),
                rag_qdrant_path=str(settings.rag_qdrant_path),
                rag_collection_name=settings.rag_collection_name,
                rag_embedding_model=settings.rag_embedding_model,
                rag_retrieval_k=settings.rag_retrieval_k,
                rag_chunk_size=settings.rag_chunk_size,
                rag_chunk_overlap=settings.rag_chunk_overlap,
                corrective_rag_enabled=settings.corrective_rag_enabled,
                corrective_rag_model=settings.corrective_rag_model,
                corrective_rag_mode=corrective_rag_mode,
                corrective_rag_min_score=settings.corrective_rag_min_score,
                corrective_rag_retry_k=settings.corrective_rag_retry_k,
                runtime_profile=runtime_profile,
                sandbox_cmd=settings.sandbox_cmd,
                api_key=selected_api_key,
            )
            result = assistant.run(request_body.prompt, thread_id=thread_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        solution = result.get("generation")
        if not isinstance(solution, CodeSolution):
            raise HTTPException(
                status_code=500,
                detail="The assistant did not return a structured solution.",
            )

        if request_provider == "local":
            resolved_model = request_body.local_model

        raw_events = result.get("events", [])
        events = [RunEvent.model_validate(event) for event in raw_events]
        rag_sources = [
            str(item.get("source", "")).strip()
            for item in result.get("rag_sources", [])
            if str(item.get("source", "")).strip()
        ]
        validation_passed = result.get("error") != "yes"
        validation_message = _extract_validation_message(
            raw_events,
            passed=validation_passed,
            iterations=int(result.get("iterations", 0) or 0),
            max_iterations=max_iterations,
        )

        failure_diagnostics = CodeAssistant.classify_failure(result)
        return ChatResponse(
            thread_id=thread_id,
            status="success" if validation_passed else "error",
            provider=request_provider,
            model=resolved_model,
            iterations=int(result.get("iterations", 0) or 0),
            max_iterations=max_iterations,
            validation_timeout=validation_timeout,
            solution=solution,
            combined_code=_combined_code(solution),
            validation_passed=validation_passed,
            validation_message=validation_message,
            events=events,
            tracing_requested=request_body.tracing,
            json_mode=request_body.json_mode,
            rag_enabled=rag_enabled,
            rag_sources=list(dict.fromkeys(rag_sources)),
            corrective_rag_mode=corrective_rag_mode,
            runtime_profile=runtime_profile,
            failure_diagnostics=failure_diagnostics,
        )

    if settings.public_dir.exists():
        app.mount("/", StaticFiles(directory=settings.public_dir, html=True), name="site")

    return app


app = create_app()
