# Deployment Guide

## Recommended production shape

Use one backend web service that also serves the frontend:

1. Deploy this repo to Render as a Python Web Service
2. Serve frontend and API from the same domain
3. Optionally add Upstash Redis for rate limiting and failure-log storage

This avoids exposing backend URL/token fields in the UI and keeps app wiring simple.

## Backend deployment

Install the backend and start it with:

```powershell
pip install -r requirements.txt
python web_main.py
```

The included `render.yaml` is a minimal starter for Render.

## Backend environment variables

Minimum production set:

```powershell
MISTRAL_API_KEY=...
CODE_ASSISTANT_ALLOWED_PROVIDERS=mistral
CODE_ASSISTANT_DEFAULT_PROVIDER=mistral
CODE_ASSISTANT_REQUIRE_ACCESS_TOKEN=false
CODE_ASSISTANT_MAX_ITERATIONS_CAP=3
CODE_ASSISTANT_VALIDATION_TIMEOUT_CAP=5
CODE_ASSISTANT_RATE_LIMIT_REQUESTS=8
CODE_ASSISTANT_RATE_LIMIT_WINDOW_SECONDS=300
CODE_ASSISTANT_LOG_DESTINATION=none
```

Optional project RAG settings:

```powershell
CODE_ASSISTANT_RAG_ENABLED=false
CODE_ASSISTANT_RAG_AUTO_INDEX=false
CODE_ASSISTANT_RAG_QDRANT_PATH=data/qdrant
CODE_ASSISTANT_RAG_COLLECTION=code-assistant-project
CODE_ASSISTANT_RAG_EMBED_MODEL=mistral-embed
CODE_ASSISTANT_RAG_RETRIEVAL_K=4
CODE_ASSISTANT_CORRECTIVE_RAG_ENABLED=true
CODE_ASSISTANT_CORRECTIVE_RAG_MODEL=mistral-small-latest
```

## Operational recommendations

- Keep `CODE_ASSISTANT_ALLOWED_PROVIDERS=mistral` in hosted environments.
- Keep frontend and API on the same hosted service/domain.
- Keep retry counts and validation timeouts small on free tiers.
- The current RAG implementation uses Mistral embeddings, so `MISTRAL_API_KEY` is still required when RAG is enabled.
- If you enable project RAG, pre-build the index during deployment or startup with `python scripts/index_project_rag.py`.
- Use Upstash only if you need shared limits across multiple backend instances.
- Treat the backend as a controlled tool, not a public anonymous endpoint.
