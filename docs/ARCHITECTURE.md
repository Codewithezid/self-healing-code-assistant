# Architecture

## Overview

The project is split into three layers:

1. `public/`
   A static frontend that collects prompts, displays validation results, and talks to the backend over HTTP.
2. `src/code_assistant/web.py`
   A FastAPI service that exposes `/api/health`, `/api/config`, and `/api/chat`.
3. `src/code_assistant/assistant.py`
   The LangGraph workflow that generates code, validates it in a subprocess, and retries on failure.

## Runtime flow

```mermaid
flowchart TD
    A[Browser UI] --> B[/api/config]
    A --> C[/api/chat]
    C --> D[FastAPI backend]
    D --> E[CodeAssistant]
    E --> F[LangGraph generate node]
    F --> G[Mistral or local model]
    E --> H[LangGraph check_code node]
    H --> I[Isolated Python subprocess]
    I --> J{Validation passed?}
    J -- No --> K[Append correction message]
    K --> F
    J -- Yes --> L[Return validated result]
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
- fetch backend capabilities from `/api/config`
- enforce deployment-driven frontend behavior such as hidden providers
- render solution text, code, and validation events

### Backend API

File:

- `src/code_assistant/web.py`

Responsibilities:

- serve health and config endpoints
- enforce access-token authentication when configured
- apply CORS rules for allowed frontend origins
- enforce provider allowlists, iteration caps, timeout caps, and rate limits
- invoke `CodeAssistant` and normalize the response for the frontend

### Assistant workflow

File:

- `src/code_assistant/assistant.py`

Responsibilities:

- build the model chain
- run the LangGraph state machine
- validate imports and code in an isolated subprocess
- retry with corrective feedback until success or retry exhaustion
- emit structured execution events for the UI

### Deployment settings

Files:

- `src/code_assistant/settings.py`
- `src/code_assistant/platform_utils.py`
- `src/code_assistant/logging_utils.py`

Responsibilities:

- load backend configuration from environment variables
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
