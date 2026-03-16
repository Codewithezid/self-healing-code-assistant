# Contributing

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `MISTRAL_API_KEY` in `.env` for live model checks.

## Run locally

One-command start:

```powershell
.\run_project.cmd
```

If port 8000 is in use:

```powershell
.\run_project.cmd -ForceRestart
```

## Verification before PR

```powershell
python -m compileall src scripts main.py web_main.py
python scripts/audit_project.py
python scripts/complex_benchmark.py
```

## Pull request checklist

- Keep changes scoped to one concern.
- Update docs for any new env vars, scripts, or routes.
- Keep generated/runtime files out of commits (`artifacts/`, `data/runtime/`, logs).
- Include benchmark or audit output when behavior changes.
