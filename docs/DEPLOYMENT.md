# Deployment Guide

## Recommended production shape

Use a split deployment:

1. `public/` on Vercel Hobby
2. Python backend on Render, Koyeb, or another service that supports subprocess execution
3. Upstash Redis for rate limiting and failure-log storage

This keeps the fast static frontend on Vercel while moving the risky Python execution path to a more appropriate backend host.

## Frontend deployment

Deploy the `public/` directory as a static site.

Before deploying, update `public/config.js`:

```js
window.APP_CONFIG = {
  apiBaseUrl: "https://your-backend-service.example.com",
  defaultProvider: "mistral",
  allowedProviders: ["mistral"],
  authRequired: true,
  maxIterationsCap: 3,
  validationTimeoutCap: 5
};
```

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
CODE_ASSISTANT_ALLOWED_ORIGINS=https://your-frontend.vercel.app
CODE_ASSISTANT_ALLOWED_PROVIDERS=mistral
CODE_ASSISTANT_DEFAULT_PROVIDER=mistral
CODE_ASSISTANT_ACCESS_TOKEN=change_this_private_token
CODE_ASSISTANT_MAX_ITERATIONS_CAP=3
CODE_ASSISTANT_VALIDATION_TIMEOUT_CAP=5
CODE_ASSISTANT_RATE_LIMIT_REQUESTS=8
CODE_ASSISTANT_RATE_LIMIT_WINDOW_SECONDS=300
CODE_ASSISTANT_LOG_DESTINATION=upstash
UPSTASH_REDIS_REST_URL=...
UPSTASH_REDIS_REST_TOKEN=...
```

## Operational recommendations

- Keep `CODE_ASSISTANT_ALLOWED_PROVIDERS=mistral` in hosted environments.
- Do not expose `/api/chat` without an access token.
- Keep retry counts and validation timeouts small on free tiers.
- Use Upstash for shared rate limits if you expect more than one backend instance.
- Treat the backend as a controlled tool, not a public anonymous endpoint.
