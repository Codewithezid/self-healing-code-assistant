# Architecture

## Overview

The project is split into four layers:

1. `public/`
   A static frontend that collects prompts, displays validation results, and talks to the backend over HTTP.
2. `src/code_assistant/web.py`
   A FastAPI service that exposes `/api/health`, `/api/config`, and `/api/chat`.
3. `src/code_assistant/assistant.py`
   The LangGraph workflow that generates code, validates it in a subprocess, and retries on failure.
4. `src/code_assistant/rag.py`
   An optional local Qdrant-backed retriever for project-aware context.

## Runtime flow

```mermaid
flowchart TD
    A[Browser UI] --> B[/api/config]
    A --> C[/api/chat]
    C --> D[FastAPI backend]
    D --> E[CodeAssistant]
    E --> F[LangGraph retrieve_context node]
    F --> G[Local Qdrant project index]
    E --> H[LangGraph generate node]
    H --> I[Mistral or local model]
    E --> J[LangGraph check_code node]
    J --> K[Isolated Python subprocess]
    K --> L{Validation passed?}
    L -- No --> M[Append correction message]
    M --> H
    L -- Yes --> N[Return validated result]
```

## Main components

### Frontend

Files:

- `public/index.html`
- `public/app.js`
- `public/styles.css`
- `public/config.js`

Responsibilities:

- collect prompt and runtime options
- expose runtime profiles without changing the underlying graph
- fetch backend capabilities from `/api/config`
- enforce deployment-driven frontend behavior such as hidden providers
- render solution text, code, validation events, and failure diagnostics

### Backend API

File:

- `src/code_assistant/web.py`

Responsibilities:

- serve health and config endpoints
- enforce access-token authentication when configured
- apply CORS rules for allowed frontend origins
- enforce provider allowlists, iteration caps, timeout caps, and rate limits
- resolve named runtime profiles into safe backend defaults
- invoke `CodeAssistant` and normalize the response for the frontend

### Assistant workflow

Files:

- `src/code_assistant/assistant.py`
- `src/code_assistant/rag.py`

Responsibilities:

- build the model chain
- optionally retrieve matching project context from local Qdrant
- grade weak retrievals and retry them with corrective query rewriting
- support fast, balanced, and aggressive corrective-retrieval modes
- run the LangGraph state machine
- validate imports and code in an isolated subprocess
- retry with corrective feedback until success or retry exhaustion
- classify failed runs without changing the main retry pipeline
- emit structured execution events for the UI

### Deployment settings

Files:

- `src/code_assistant/settings.py`
- `src/code_assistant/platform_utils.py`
- `src/code_assistant/logging_utils.py`
- `src/code_assistant/profiles.py`

Responsibilities:

- load backend configuration from environment variables
- expose reusable runtime presets for CLI, API, and frontend calls
- implement local or Upstash-backed rate limiting
- route failure logging to file storage, Upstash, or no-op mode

## Storage model

- Local development can write failure logs to `data/runtime/failure_log.jsonl`.
- Hosted deployments should prefer Upstash-backed failure logging.
- Frontend deployment settings such as backend URL and access token are stored in browser local storage.

## Important constraints

- The backend can execute model-generated Python. This is operationally sensitive.
- The isolated subprocess reduces risk but is not a complete security sandbox.
- The static frontend is safe to host on Vercel; the Python execution backend is better suited to a separate service.
