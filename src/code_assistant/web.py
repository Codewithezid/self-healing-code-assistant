from __future__ import annotations

import uuid
import warnings
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .assistant import CodeAssistant
from .models import CodeSolution
from .platform_utils import InMemoryRateLimiter, UpstashRateLimiter, UpstashRedis
from .settings import BackendSettings, get_settings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    provider: Literal["mistral", "local"] = "mistral"
    model: str = "mistral-large-latest"
    local_model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    max_iterations: int = Field(default=3, ge=1, le=10)
    validation_timeout: int = Field(default=5, ge=1, le=30)
    show_events: bool = False
    json_mode: bool = False
    tracing: bool = False
    rag_enabled: bool | None = None


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


def create_app() -> FastAPI:
    settings = get_settings()
    rate_limiter = _build_rate_limiter(settings)

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
        )

    @app.post("/api/chat", response_model=ChatResponse)
    def run_chat(request_body: ChatRequest, request: Request) -> ChatResponse:
        if settings.auth_token:
            supplied_token = _extract_access_token(request)
            if supplied_token != settings.auth_token:
                raise HTTPException(status_code=401, detail="Missing or invalid access token.")

        client_ip = _client_ip(request)
        allowed, retry_after = rate_limiter.allow(
            f"chat:{client_ip}",
            limit=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Rate limit exceeded. "
                    f"Try again in about {retry_after} second(s)."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        if request_body.provider not in settings.allowed_providers:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Provider '{request_body.provider}' is disabled for this deployment. "
                    f"Allowed providers: {', '.join(settings.allowed_providers)}."
                ),
            )

        max_iterations = min(request_body.max_iterations, settings.max_iterations_cap)
        validation_timeout = min(
            request_body.validation_timeout,
            settings.validation_timeout_cap,
        )
        rag_enabled = settings.rag_enabled if request_body.rag_enabled is None else request_body.rag_enabled
        thread_id = str(uuid.uuid4())
        resolved_model = request_body.model

        try:
            assistant = CodeAssistant(
                model_name=request_body.model,
                max_iterations=max_iterations,
                validation_timeout_seconds=validation_timeout,
                failure_log_path=str(settings.failure_log_path),
                log_destination=settings.log_destination,
                upstash_redis_rest_url=settings.upstash_redis_rest_url,
                upstash_redis_rest_token=settings.upstash_redis_rest_token,
                failure_log_key=settings.failure_log_key,
                provider=request_body.provider,
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
                corrective_rag_min_score=settings.corrective_rag_min_score,
                corrective_rag_retry_k=settings.corrective_rag_retry_k,
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

        if request_body.provider == "local":
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

        return ChatResponse(
            thread_id=thread_id,
            status="success" if validation_passed else "error",
            provider=request_body.provider,
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
        )

    if settings.public_dir.exists():
        app.mount("/", StaticFiles(directory=settings.public_dir, html=True), name="site")

    return app


app = create_app()
