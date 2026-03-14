# LangGraph Code Assistant

A Python code assistant that uses LangGraph to generate code, validate it in an isolated subprocess, and retry with corrections when execution fails.

The repository supports two ways to use it:

- a local CLI for development and experiments
- a split web deployment with a static frontend and a separate Python backend

## Highlights

- Generates structured code solutions with Mistral or a local Hugging Face model
- Validates generated code in a subprocess with a timeout
- Retries failed generations up to a configurable limit
- Exposes a FastAPI backend with auth, provider allowlists, request caps, CORS, and rate limiting
- Ships a static frontend ready for Vercel-style hosting

## Repository layout

- `main.py`: CLI entrypoint
- `web_main.py`: local backend entrypoint
- `public/`: static frontend
- `src/code_assistant/`: core assistant, API, settings, and utilities
- `scripts/`: audits, benchmarks, and fine-tuning helpers
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

Recommended Python version: `3.12` or `3.13`.

## Run locally

CLI:

```powershell
python main.py "Write a Python function that returns the Fibonacci sequence up to n."
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

## Deployment

The most practical low-cost production setup is:

1. Deploy `public/` to Vercel Hobby.
2. Deploy the Python backend separately on a host that supports subprocess execution.
3. Use Upstash Redis for shared rate limits and failure-log storage.

See `docs/DEPLOYMENT.md` for the full setup.

## Security note

Generated code is validated in an isolated Python subprocess, not inline in the API process. That is safer than direct `exec(...)`, but it is still not a full sandbox. Do not expose the backend publicly without authentication, rate limits, and careful operational controls.

## Validation

Run the built-in audit suite:

```powershell
python scripts/audit_project.py
python scripts/complex_benchmark.py
```
