# LangGraph Code Assistant

A Python code assistant that uses LangGraph to generate code, validate it in an isolated subprocess, and retry with corrections when execution fails.

The repository supports two ways to use it:

- a local CLI for development and experiments
- a web app where backend and frontend are served together from one domain

## Highlights

- Generates structured code solutions with Mistral or a local Hugging Face model
- Validates generated code in a subprocess with a timeout
- Retries failed generations up to a configurable limit
- Optionally retrieves project-aware context from a local Qdrant RAG index
- Supports runtime profiles for fast, balanced, and accuracy-focused runs
- Classifies failed runs so retrieval, import, timeout, and runtime errors are easier to debug
- Writes structured benchmark reports for accuracy and latency tracking
- Exposes a FastAPI backend with auth, provider allowlists, request caps, CORS, and rate limiting
- Serves the bundled frontend from the same backend app

## Repository layout

- `main.py`: CLI entrypoint
- `web_main.py`: local backend entrypoint
- `public/`: static frontend
- `src/code_assistant/`: core assistant, API, settings, and utilities
- `scripts/`: audits, benchmarks, and fine-tuning helpers
- `scripts/index_project_rag.py`: build the local project RAG index
- `docs/ARCHITECTURE.md`: system design
- `docs/DEPLOYMENT.md`: recommended hosting setups
- `render.yaml`: starter backend service manifest

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `MISTRAL_API_KEY` in `.env`.
If you want project-aware retrieval, build the local index once:

```powershell
python scripts/index_project_rag.py
```

Project RAG currently uses Mistral embeddings, so it still needs `MISTRAL_API_KEY` even if code generation runs with `--provider local`.

Recommended Python version: `3.12` or `3.13`.

## Run locally

CLI:

```powershell
python main.py "Write a Python function that returns the Fibonacci sequence up to n."
```

With project RAG:

```powershell
python main.py --rag "Add a retrieval step that matches this codebase."
```

Backend + frontend together:

```powershell
python web_main.py
```

Then open `http://localhost:8000`.

Useful routes:

- `/`
- `/api/health`
- `/api/config`
- `/api/docs`

Optional RAG environment flags:

- `CODE_ASSISTANT_RAG_ENABLED=true`
- `CODE_ASSISTANT_RAG_AUTO_INDEX=true`
- `CODE_ASSISTANT_RAG_QDRANT_PATH=data/qdrant`
- `CODE_ASSISTANT_CORRECTIVE_RAG_ENABLED=true`
- `CODE_ASSISTANT_CORRECTIVE_RAG_MODEL=mistral-small-latest`
- `CODE_ASSISTANT_CORRECTIVE_RAG_MODE=balanced`
- `CODE_ASSISTANT_DEFAULT_RUNTIME_PROFILE=custom`

Named runtime profiles:

- `fast`
- `balanced`
- `accurate`

Generate a structured benchmark report:

```powershell
python scripts/benchmark_report.py --runtime-profile balanced
```

## Deployment

Recommended low-cost setup:

1. Deploy this repo as one Python web service on Render (or similar).
2. Keep frontend and API on the same domain (`/` and `/api/*`).
3. Optionally add Upstash Redis for shared rate limits and failure-log storage.

See `docs/DEPLOYMENT.md` for the full setup.

## Security note

Generated code is validated in an isolated Python subprocess, not inline in the API process. That is safer than direct `exec(...)`, but it is still not a full sandbox. Do not expose the backend publicly without authentication, rate limits, and careful operational controls.

## Validation

Run the built-in audit suite:

```powershell
python scripts/audit_project.py
python scripts/complex_benchmark.py
```
