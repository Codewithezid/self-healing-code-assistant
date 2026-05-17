# LangGraph Code Assistant

A Python code assistant that uses LangGraph to generate code, validate it in an isolated subprocess, and retry with corrections when execution fails.

The repository supports two ways to use it:

- a local CLI for development and experiments
- a web app where backend and frontend are served together from one domain

## Highlights

- Generates structured code solutions with OpenAI, Mistral, or a local Hugging Face model
  (and OpenRouter when enabled)
- Validates generated code in a subprocess with a timeout
- Retries failed generations up to a configurable limit
- Optionally retrieves project-aware context from a local Qdrant RAG index
- Supports runtime profiles for fast, balanced, and accuracy-focused runs
- Supports user-managed provider API keys via encrypted key IDs (BYOK)
- Classifies failed runs so retrieval, import, timeout, and runtime errors are easier to debug
- Writes structured benchmark reports for accuracy and latency tracking
- Optional sandbox prefix for code validation (e.g., firejail/nsjail) for safer execution
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

Set at least one hosted key in `.env` (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or `MISTRAL_API_KEY`).
To persist user-added keys safely, set `CODE_ASSISTANT_USER_KEYS_SECRET` to a strong secret value.
If you want project-aware retrieval, build the local index once:

```powershell
python scripts/index_project_rag.py
```

Project RAG currently uses Mistral embeddings, so it still needs `MISTRAL_API_KEY` even if code generation runs with `--provider local`.
To ingest external PDF documents into project RAG context, place PDFs under `data/pdfs/` and run:

```powershell
python scripts/ingest_project_pdfs.py --reindex-rag
```

This converts PDFs into `docs/pdf_ingested/` as Markdown/JSON and refreshes the local Qdrant index.

Recommended Python version: `3.12` or `3.13`.

## Run locally

Fastest (single command):

```powershell
.\run_project.cmd
```

If port `8000` is already in use:

```powershell
.\run_project.cmd -ForceRestart
```

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

## Coding Arena mode (RAG vs Normal)

The web UI includes a topbar `Coding Arena` toggle with a crossed-swords icon.

When enabled:

- the center panel switches to dual-screen mode
- the same prompt is sent to two assistants in parallel
- left panel: `RAG Mode` (`rag_enabled=true`)
- right panel: `Normal Mode` (`rag_enabled=false`)

This is useful for project demos where you want to show your teacher how project-aware retrieval compares against direct model generation for the same coding task.

### Attach files with `+` in the input bar

Use the `+` button next to the prompt box to upload context files:

- `.pdf`
- `.docx`
- code/text files (`.py`, `.js`, `.ts`, `.java`, `.go`, `.md`, `.txt`, and more)

You can choose whether attachments are used in:

- `RAG only` (default, recommended for fair Arena comparison)
- `Both sides` (RAG + Normal)

Useful routes:

- `/`
- `/benchmark.html`
- `/analytics.html`
- `/api/health`
- `/api/config`
- `/api/docs`

Benchmark APIs:

- `GET /api/benchmark/reports?limit=30`
- `GET /api/benchmark/compare?profiles=fast,balanced,accurate`
- `POST /api/benchmark/run`
- `POST /api/ablation/run`
- `POST /api/feedback`
- `GET /api/analytics/feedback/recent?limit=100`
- `GET /api/analytics/feedback/summary?window_days=30`

## PDF ingestion for RAG

The repository includes first-party OpenDataLoader PDF integration.

Default workflow:

```powershell
python scripts/ingest_project_pdfs.py --reindex-rag
```

Advanced usage:

```powershell
python scripts/ingest_project_pdfs.py `
  --input data/pdfs `
  --input archive/customer_docs `
  --output-dir docs/pdf_ingested `
  --format markdown,json `
  --reindex-rag
```

Expected output:

```text
PDF ingestion complete: 12 PDF(s) converted from 2 input path(s) into C:\...\docs\pdf_ingested.
RAG index refreshed: 48 file(s), 630 chunk(s) at C:\...\data\qdrant.
```

Failure example:

```text
PDF ingestion failed: No PDF files were found in the configured input path(s). Add PDF files first, then retry ingestion.
```

Optional RAG environment flags:

- `CODE_ASSISTANT_RAG_ENABLED=true`
- `CODE_ASSISTANT_RAG_AUTO_INDEX=true`
- `CODE_ASSISTANT_RAG_QDRANT_PATH=data/qdrant`
- `CODE_ASSISTANT_RAG_RETRIEVAL_K=6`
- `CODE_ASSISTANT_RAG_RETRIEVAL_FETCH_K=14`
- `CODE_ASSISTANT_RAG_MAX_CHUNKS_PER_SOURCE=2`
- `CODE_ASSISTANT_CORRECTIVE_RAG_ENABLED=true`
- `CODE_ASSISTANT_CORRECTIVE_RAG_MODEL=mistral-small-latest`
- `CODE_ASSISTANT_CORRECTIVE_RAG_MODE=balanced`
- `CODE_ASSISTANT_DEFAULT_RUNTIME_PROFILE=custom`
- `CODE_ASSISTANT_SANDBOX_CMD=` (optional, e.g., `firejail --quiet --private`; on Windows quote paths with spaces)

Named runtime profiles:

- `fast`
- `balanced`
- `accurate`

Generate a structured benchmark report:

```powershell
python scripts/benchmark_report.py --runtime-profile balanced
```

Run and compare multiple profiles from API (used by dashboard):

```powershell
curl -X POST http://localhost:8000/api/benchmark/run `
  -H "Content-Type: application/json" `
  -d "{\"profiles\":[\"fast\",\"balanced\",\"accurate\"],\"limit_cases\":0}"
```

Run RAG ablation experiments:

```powershell
python scripts/rag_ablation_report.py --provider mistral --model mistral-medium-latest --limit 4
```

### Secure sandboxed executor

Validation and regression test execution now use a centralized secure executor:

- Isolated temp working directory per execution
- Minimal environment variables (`PYTHONPATH` cleared, usersite disabled)
- Timeout enforcement and output truncation
- Unsafe import blocking policy for validation (`socket`, `subprocess`, `requests`, etc.)
- Optional external sandbox prefix via `CODE_ASSISTANT_SANDBOX_CMD`

This is integrated into LangGraph validation and regression checks in `CodeAssistant`.

### Validation pipeline upgrades (hallucination reduction)

The assistant now applies a stricter pipeline:

`Generate -> AST analyze -> sandbox runtime validation -> signature-aware unit test generation -> semantic regression run -> success`

Key improvements:

- AST-driven function signature parsing for test generation
- Type-aware sample/edge/negative test inputs (instead of naive positional integers)
- Constraint hints derived from signatures (e.g., sequence/sorted expectations for search-like functions)
- Semantic success gating: API success status now depends on semantic validation pass (unit/regression checks), not syntax/runtime-only checks
- Confidence scoring now includes generated-test presence and pass/fail outcomes

Improve RAG quality quickly:

```powershell
$env:CODE_ASSISTANT_RAG_ENABLED="true"
$env:CODE_ASSISTANT_RAG_RETRIEVAL_K="6"
$env:CODE_ASSISTANT_RAG_RETRIEVAL_FETCH_K="14"
$env:CODE_ASSISTANT_RAG_MAX_CHUNKS_PER_SOURCE="2"
$env:CODE_ASSISTANT_CORRECTIVE_RAG_MODE="aggressive"
python scripts/index_project_rag.py
```

Build a larger fine-tuning dataset:

```powershell
python scripts/prepare_finetune_dataset.py `
  --seed-count 2000 `
  --extra-dataset "iamtarun/python_code_instructions_18k_alpaca:default:train" `
  --extra-jsonl data/external/kaggle_code_data.jsonl `
  --validation-ratio 0.1
```

Start a longer Mistral fine-tuning job:

```powershell
python scripts/mistral_finetune.py `
  --model codestral-latest `
  --training-steps 500 `
  --learning-rate 1e-4 `
  --start
```

Run offline audit (CI-friendly):

```powershell
python scripts/audit_project.py
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

## Contributing

See `CONTRIBUTING.md` for local workflow and pre-PR checks.
