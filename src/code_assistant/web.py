from __future__ import annotations

import os
import re
import uuid
import warnings
import zipfile
from datetime import datetime
from io import BytesIO
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .assistant import CodeAssistant
from .key_vault import EncryptedKeyVault, StoredKeyPublic
from .models import CodeSolution, FailureDiagnostics
from .platform_utils import InMemoryRateLimiter, UpstashRateLimiter, UpstashRedis
from .profiles import RUNTIME_PROFILES, get_runtime_profile
from .benchmarking import compare_latest_by_profile, load_report_files, run_benchmark
from .ablation import run_ablation, write_ablation_report
from .feedback_analytics import FeedbackRecord, append_feedback, load_feedback, summarize_feedback
from .provider_clients import (
    ProviderClientError,
    list_models_for_provider,
    supports_hosted_provider,
)
from .settings import BackendSettings, get_settings
from .rag import ProjectRAG

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    runtime_profile: Literal["custom", "fast", "balanced", "accurate", "goated"] = "custom"
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
    attachment_ids: list[str] = Field(default_factory=list)
    attachment_mode: Literal["rag_only", "both"] = "rag_only"


class AttachmentSummary(BaseModel):
    attachment_id: str
    filename: str
    kind: str
    size_bytes: int
    char_count: int
    preview: str
    indexed_chunks: int = 0
    indexed_to_qdrant: bool = False


class AttachmentUploadResponse(BaseModel):
    attachments: list[AttachmentSummary]


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
    semantic_validation_passed: bool
    validation_message: str
    events: list[RunEvent]
    tracing_requested: bool
    json_mode: bool
    rag_enabled: bool
    rag_sources: list[str]
    corrective_rag_mode: str
    runtime_profile: str
    failure_diagnostics: FailureDiagnostics
    confidence_score: float
    traceback_summary: str
    generated_tests: str
    repair_diff: str
    hallucination_risk: float
    regression_test_passed: bool
    regression_test_output: str


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


class BenchmarkRunRequest(BaseModel):
    profiles: list[Literal["custom", "fast", "balanced", "accurate", "goated"]] = Field(
        default_factory=lambda: ["fast", "balanced", "accurate"]
    )
    limit_cases: int = Field(default=0, ge=0, le=1000)


class BenchmarkReportSummary(BaseModel):
    filename: str
    generated_at: str
    runtime_profile: str
    provider: str
    model: str
    rag_enabled: bool
    corrective_rag_mode: str
    semantic_accuracy_percent: float
    pipeline_passes: int
    semantic_passes: int
    total_cases: int
    average_latency_seconds: float


class BenchmarkReportsResponse(BaseModel):
    reports: list[BenchmarkReportSummary]


class BenchmarkCompareResponse(BaseModel):
    profiles: dict[str, BenchmarkReportSummary]


class BenchmarkRunItem(BaseModel):
    runtime_profile: str
    semantic_accuracy_percent: float
    average_latency_seconds: float
    pipeline_passes: int
    semantic_passes: int
    total_cases: int
    report_filename: str


class BenchmarkRunResponse(BaseModel):
    runs: list[BenchmarkRunItem]


class AblationRunRequest(BaseModel):
    provider: str = "mistral"
    model: str = "mistral-medium-latest"
    limit_cases: int = Field(default=0, ge=0, le=1000)
    max_iterations: int = Field(default=3, ge=1, le=10)
    validation_timeout: int = Field(default=5, ge=1, le=30)


class AblationVariantSummary(BaseModel):
    variant: str
    semantic_accuracy_percent: float
    semantic_passes: int
    total_cases: int
    average_latency_seconds: float


class AblationRunResponse(BaseModel):
    report_file: str
    variants: list[AblationVariantSummary]


class FeedbackSubmitRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    verdict: Literal["correct", "partially_correct", "wrong"]
    rating: int = Field(ge=1, le=5)
    provider: str = Field(default="unknown", min_length=2, max_length=60)
    model: str = Field(default="unknown", min_length=2, max_length=120)
    runtime_profile: str = Field(default="custom", min_length=2, max_length=30)
    rag_enabled: bool = False
    corrective_rag_mode: str = Field(default="balanced", min_length=2, max_length=20)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    comment: str = Field(default="", max_length=2000)


class FeedbackSubmitResponse(BaseModel):
    saved: bool


class FeedbackListResponse(BaseModel):
    items: list[dict[str, Any]]


class AnalyticsSummaryResponse(BaseModel):
    summary: dict[str, Any]


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


def _guess_attachment_kind(filename: str) -> str:
    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        return "pdf"
    if lowered.endswith(".docx"):
        return "docx"
    if lowered.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".cpp", ".c", ".cs", ".rb", ".php", ".swift", ".kt", ".kts", ".sql", ".sh", ".yaml", ".yml", ".json", ".md", ".txt")):
        return "code"
    return "text"


def _extract_docx_text(raw: bytes) -> str:
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        try:
            xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
        except KeyError as exc:
            raise ValueError("DOCX content is missing word/document.xml.") from exc
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return re.sub(r"\n{3,}", "\n\n", xml).strip()


def _extract_pdf_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF parsing requires 'pypdf'. Please install it on the backend.") from exc
    reader = PdfReader(BytesIO(raw))
    chunks: list[str] = []
    for page in reader.pages:
        chunks.append((page.extract_text() or "").strip())
    return "\n\n".join(part for part in chunks if part).strip()


def _extract_attachment_text(filename: str, raw: bytes) -> tuple[str, str]:
    kind = _guess_attachment_kind(filename)
    if kind == "pdf":
        return _extract_pdf_text(raw), kind
    if kind == "docx":
        return _extract_docx_text(raw), kind
    return raw.decode("utf-8", errors="ignore"), kind


def _compose_prompt_with_attachments(prompt: str, attachments: list[dict[str, str]]) -> str:
    if not attachments:
        return prompt
    parts = [prompt.strip(), "", "Attachment Context (user-provided files):"]
    for item in attachments:
        parts.append(f"[{item['filename']}]")
        parts.append(item["content"])
        parts.append("")
    parts.append(
        "Use attachment context when relevant. If attachment context conflicts with your assumptions, prefer the attachment."
    )
    return "\n".join(parts).strip()


def _compose_prompt_with_attachment_refs(prompt: str, attachments: list[dict[str, str]]) -> str:
    if not attachments:
        return prompt
    names = ", ".join(item["filename"] for item in attachments if item.get("filename"))
    return (
        f"{prompt.strip()}\n\n"
        f"User attached files indexed in RAG: {names}\n"
        "Prefer retrieving relevant chunks from indexed attachments and project files."
    ).strip()


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
    attachment_store: dict[str, dict[str, str | int]] = {}
    benchmark_reports_dir = settings.project_root / "artifacts" / "benchmark_reports"
    ablation_reports_dir = settings.project_root / "artifacts" / "ablation_reports"
    feedback_log_path = settings.project_root / "data" / "runtime" / "feedback_log.jsonl"
    max_attachment_bytes = 20 * 1024 * 1024
    max_attachment_chars = 120_000
    max_attachment_count = 8
    rag_indexer = ProjectRAG(
        project_root=str(settings.project_root),
        qdrant_path=str(settings.rag_qdrant_path),
        collection_name=settings.rag_collection_name,
        embedding_model=settings.rag_embedding_model,
        retrieval_k=settings.rag_retrieval_k,
        retrieval_fetch_k=settings.rag_retrieval_fetch_k,
        max_chunks_per_source=settings.rag_max_chunks_per_source,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        auto_index=settings.rag_auto_index,
        corrective_enabled=False,
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
        attachment_ids = [item.strip() for item in request_body.attachment_ids if item.strip()]
        if len(attachment_ids) > max_attachment_count:
            raise HTTPException(
                status_code=400,
                detail=f"Too many attachments. Maximum allowed is {max_attachment_count}.",
            )
        attachments_for_prompt: list[dict[str, str]] = []
        for attachment_id in attachment_ids:
            payload = attachment_store.get(attachment_id)
            if not payload:
                continue
            content = str(payload.get("content", ""))
            if not content:
                continue
            attachments_for_prompt.append(
                {
                    "filename": str(payload.get("filename", "attachment")),
                    "content": content,
                }
            )
        if request_body.attachment_mode == "both":
            composed_prompt = _compose_prompt_with_attachments(
                request_body.prompt,
                attachments_for_prompt,
            )
        else:
            composed_prompt = _compose_prompt_with_attachment_refs(
                request_body.prompt,
                attachments_for_prompt,
            )
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
                rag_retrieval_fetch_k=settings.rag_retrieval_fetch_k,
                rag_max_chunks_per_source=settings.rag_max_chunks_per_source,
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
            result = assistant.run(composed_prompt, thread_id=thread_id)
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
        semantic_validation_passed = bool(result.get("semantic_validation_passed", validation_passed))
        validation_message = _extract_validation_message(
            raw_events,
            passed=semantic_validation_passed,
            iterations=int(result.get("iterations", 0) or 0),
            max_iterations=max_iterations,
        )

        failure_diagnostics = CodeAssistant.classify_failure(result)
        return ChatResponse(
            thread_id=thread_id,
            status="success" if semantic_validation_passed else "error",
            provider=request_provider,
            model=resolved_model,
            iterations=int(result.get("iterations", 0) or 0),
            max_iterations=max_iterations,
            validation_timeout=validation_timeout,
            solution=solution,
            combined_code=_combined_code(solution),
            validation_passed=validation_passed,
            semantic_validation_passed=semantic_validation_passed,
            validation_message=validation_message,
            events=events,
            tracing_requested=request_body.tracing,
            json_mode=request_body.json_mode,
            rag_enabled=rag_enabled,
            rag_sources=list(dict.fromkeys(rag_sources)),
            corrective_rag_mode=corrective_rag_mode,
            runtime_profile=runtime_profile,
            failure_diagnostics=failure_diagnostics,
            confidence_score=float(result.get("confidence_score", 0.0) or 0.0),
            traceback_summary=str(result.get("traceback_summary", "") or ""),
            generated_tests=str(result.get("generated_tests", "") or ""),
            repair_diff=str(result.get("repair_diff", "") or ""),
            hallucination_risk=float(result.get("hallucination_risk", 0.5) or 0.5),
            regression_test_passed=bool(result.get("regression_test_passed", False)),
            regression_test_output=str(result.get("regression_test_output", "") or ""),
        )

    @app.get("/api/benchmark/reports", response_model=BenchmarkReportsResponse)
    def list_benchmark_reports(
        request: Request,
        limit: int = Query(default=20, ge=1, le=200),
    ) -> BenchmarkReportsResponse:
        require_auth(request)
        reports = [
            BenchmarkReportSummary.model_validate(item)
            for item in load_report_files(benchmark_reports_dir, limit=limit)
        ]
        return BenchmarkReportsResponse(reports=reports)

    @app.get("/api/benchmark/compare", response_model=BenchmarkCompareResponse)
    def compare_benchmark_profiles(
        request: Request,
        profiles: str = Query(default="fast,balanced,accurate"),
    ) -> BenchmarkCompareResponse:
        require_auth(request)
        requested = [
            part.strip().lower()
            for part in profiles.split(",")
            if part.strip()
        ]
        if not requested:
            requested = ["fast", "balanced", "accurate"]
        latest = compare_latest_by_profile(benchmark_reports_dir, profiles=requested)
        payload = {
            name: BenchmarkReportSummary.model_validate(row)
            for name, row in latest.items()
        }
        return BenchmarkCompareResponse(profiles=payload)

    @app.post("/api/benchmark/run", response_model=BenchmarkRunResponse)
    def run_benchmark_profiles(
        request_body: BenchmarkRunRequest,
        request: Request,
    ) -> BenchmarkRunResponse:
        require_auth(request)
        enforce_rate_limit(
            request,
            scope="benchmark",
            limit=max(1, min(settings.rate_limit_requests, 4)),
        )
        from scripts.complex_benchmark import BENCHMARK_CASES

        profiles = list(dict.fromkeys(request_body.profiles))
        cases = (
            BENCHMARK_CASES[: request_body.limit_cases]
            if request_body.limit_cases > 0
            else BENCHMARK_CASES
        )
        runs: list[BenchmarkRunItem] = []
        for profile in profiles:
            try:
                outcome = run_benchmark(
                    runtime_profile=profile,
                    cases=cases,
                    output_dir=benchmark_reports_dir,
                    root_dir=settings.project_root,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Benchmark failed for profile '{profile}': {exc}",
                ) from exc
            summary = outcome.report["summary"]
            runs.append(
                BenchmarkRunItem(
                    runtime_profile=profile,
                    semantic_accuracy_percent=float(summary["semantic_accuracy_percent"]),
                    average_latency_seconds=float(summary["average_latency_seconds"]),
                    pipeline_passes=int(summary["pipeline_passes"]),
                    semantic_passes=int(summary["semantic_passes"]),
                    total_cases=int(summary["total_cases"]),
                    report_filename=outcome.json_path.name,
                )
            )
        return BenchmarkRunResponse(runs=runs)

    @app.post("/api/ablation/run", response_model=AblationRunResponse)
    def run_ablation_experiment(
        request_body: AblationRunRequest,
        request: Request,
    ) -> AblationRunResponse:
        require_auth(request)
        enforce_rate_limit(
            request,
            scope="ablation",
            limit=max(1, min(settings.rate_limit_requests, 3)),
        )
        from scripts.complex_benchmark import BENCHMARK_CASES

        cases = (
            BENCHMARK_CASES[: request_body.limit_cases]
            if request_body.limit_cases > 0
            else BENCHMARK_CASES
        )
        report = run_ablation(
            cases=cases,
            root_dir=settings.project_root,
            provider=request_body.provider,
            model=request_body.model,
            max_iterations=request_body.max_iterations,
            validation_timeout=request_body.validation_timeout,
        )
        json_path, _ = write_ablation_report(ablation_reports_dir, report)
        variants: list[AblationVariantSummary] = []
        for item in report["variants"]:
            summary = item["summary"]
            variants.append(
                AblationVariantSummary(
                    variant=str(item["variant"]),
                    semantic_accuracy_percent=float(summary["semantic_accuracy_percent"]),
                    semantic_passes=int(summary["semantic_passes"]),
                    total_cases=int(summary["total_cases"]),
                    average_latency_seconds=float(summary["average_latency_seconds"]),
                )
            )
        return AblationRunResponse(report_file=json_path.name, variants=variants)

    @app.post("/api/feedback", response_model=FeedbackSubmitResponse)
    def submit_feedback(
        request_body: FeedbackSubmitRequest,
        request: Request,
    ) -> FeedbackSubmitResponse:
        require_auth(request)
        enforce_rate_limit(
            request,
            scope="feedback",
            limit=max(5, min(settings.rate_limit_requests, 40)),
        )
        record = FeedbackRecord(
            created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            thread_id=request_body.thread_id.strip(),
            verdict=request_body.verdict,
            rating=request_body.rating,
            provider=request_body.provider.strip().lower(),
            model=request_body.model.strip(),
            runtime_profile=request_body.runtime_profile.strip().lower(),
            rag_enabled=bool(request_body.rag_enabled),
            corrective_rag_mode=request_body.corrective_rag_mode.strip().lower(),
            confidence_score=float(request_body.confidence_score),
            hallucination_risk=float(request_body.hallucination_risk),
            comment=request_body.comment.strip(),
        )
        append_feedback(feedback_log_path, record)
        return FeedbackSubmitResponse(saved=True)

    @app.get("/api/analytics/feedback/recent", response_model=FeedbackListResponse)
    def list_recent_feedback(
        request: Request,
        limit: int = Query(default=50, ge=1, le=500),
    ) -> FeedbackListResponse:
        require_auth(request)
        rows = load_feedback(feedback_log_path, limit=limit)
        return FeedbackListResponse(items=list(reversed(rows)))

    @app.get("/api/analytics/feedback/summary", response_model=AnalyticsSummaryResponse)
    def feedback_summary(
        request: Request,
        window_days: int = Query(default=30, ge=1, le=365),
    ) -> AnalyticsSummaryResponse:
        require_auth(request)
        rows = load_feedback(feedback_log_path, limit=5000)
        return AnalyticsSummaryResponse(summary=summarize_feedback(rows, last_days=window_days))

    @app.post("/api/attachments", response_model=AttachmentUploadResponse)
    async def upload_attachments(
        request: Request,
        files: list[UploadFile] = File(...),
    ) -> AttachmentUploadResponse:
        require_auth(request)
        enforce_rate_limit(
            request,
            scope="attachments",
            limit=max(5, min(settings.rate_limit_requests, 20)),
        )
        if not files:
            raise HTTPException(status_code=400, detail="No files were uploaded.")
        if len(files) > max_attachment_count:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Maximum {max_attachment_count} files per upload.",
            )

        summaries: list[AttachmentSummary] = []
        to_index: list[dict[str, str]] = []
        for upload in files:
            filename = (upload.filename or "attachment.txt").strip()
            raw = await upload.read()
            size_bytes = len(raw)
            if size_bytes == 0:
                raise HTTPException(status_code=400, detail=f"'{filename}' is empty.")
            if size_bytes > max_attachment_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{filename}' exceeds {max_attachment_bytes // (1024 * 1024)}MB limit.",
                )
            try:
                extracted_text, kind = _extract_attachment_text(filename, raw)
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(status_code=400, detail=f"{filename}: {exc}") from exc
            except Exception as exc:  # pragma: no cover - defensive parse guard
                raise HTTPException(status_code=400, detail=f"Failed to parse '{filename}'.") from exc

            cleaned = extracted_text.strip()
            if not cleaned:
                raise HTTPException(status_code=400, detail=f"'{filename}' has no readable text.")
            clipped = cleaned[:max_attachment_chars]
            attachment_id = str(uuid.uuid4())
            estimated_chunks = len(rag_indexer._split_text(settings.project_root / filename, clipped))
            preview = clipped[:180].replace("\n", " ").strip()
            attachment_store[attachment_id] = {
                "filename": filename,
                "kind": kind,
                "size_bytes": size_bytes,
                "char_count": len(clipped),
                "content": clipped,
                "indexed_chunks": estimated_chunks,
                "indexed_to_qdrant": 0,
                "preview": preview,
            }
            to_index.append(
                {
                    "attachment_id": attachment_id,
                    "filename": filename,
                    "content": clipped,
                    "preview": preview,
                }
            )
        try:
            stats = rag_indexer.index_attachment_texts(attachments=to_index)
            indexed_ok = stats["chunks"] > 0
        except Exception:
            indexed_ok = False
            stats = {"chunks": 0}

        for item in to_index:
            record = attachment_store.get(item["attachment_id"])
            if record is not None:
                record["indexed_to_qdrant"] = 1 if indexed_ok else 0
            summaries.append(
                AttachmentSummary(
                    attachment_id=item["attachment_id"],
                    filename=str(record.get("filename", "")) if record else "",
                    kind=str(record.get("kind", "text")) if record else "text",
                    size_bytes=int(record.get("size_bytes", 0)) if record else 0,
                    char_count=int(record.get("char_count", 0)) if record else 0,
                    preview=str(record.get("preview", "")) if record else "",
                    indexed_chunks=int(record.get("indexed_chunks", 0)) if record else 0,
                    indexed_to_qdrant=bool(int(record.get("indexed_to_qdrant", 0))) if record else False,
                )
            )
        return AttachmentUploadResponse(attachments=summaries)

    if settings.public_dir.exists():
        app.mount("/", StaticFiles(directory=settings.public_dir, html=True), name="site")

    return app


app = create_app()
