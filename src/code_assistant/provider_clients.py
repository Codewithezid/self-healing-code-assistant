from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class ProviderClientError(RuntimeError):
    pass


SUPPORTED_HOSTED_PROVIDERS = {"openai", "mistral"}


def supports_hosted_provider(provider: str) -> bool:
    return provider.strip().lower() in SUPPORTED_HOSTED_PROVIDERS


def list_models_for_provider(provider: str, api_key: str) -> list[str]:
    normalized = provider.strip().lower()
    key = api_key.strip()
    if not key:
        raise ProviderClientError("API key is required.")

    if normalized == "openai":
        return _list_openai_models(key)
    if normalized == "mistral":
        return _list_mistral_models(key)
    raise ProviderClientError(
        f"Provider '{provider}' is not supported for hosted model discovery yet."
    )


def _list_openai_models(api_key: str) -> list[str]:
    payload = _fetch_json(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise ProviderClientError("Unexpected OpenAI response format.")
    models = sorted(
        {
            str(item.get("id", "")).strip()
            for item in rows
            if isinstance(item, dict)
            and _is_openai_chat_model(str(item.get("id", "")).strip())
        }
    )
    if not models:
        raise ProviderClientError("No compatible OpenAI chat models were returned.")
    return models


def _list_mistral_models(api_key: str) -> list[str]:
    payload = _fetch_json(
        "https://api.mistral.ai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise ProviderClientError("Unexpected Mistral response format.")
    models = sorted(
        {
            str(item.get("id", "")).strip()
            for item in rows
            if isinstance(item, dict)
            and _is_mistral_chat_model(str(item.get("id", "")).strip())
        }
    )
    if not models:
        raise ProviderClientError("No compatible Mistral chat models were returned.")
    return models


def _fetch_json(url: str, *, headers: dict[str, str], timeout_seconds: int = 20) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = _extract_error_message(body) or f"HTTP {exc.code}"
        raise ProviderClientError(detail) from exc
    except urllib.error.URLError as exc:
        raise ProviderClientError(f"Network error: {exc.reason}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderClientError("Provider returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ProviderClientError("Provider returned an unexpected payload.")
    return parsed


def _extract_error_message(body: str) -> str:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()[:300]
    if not isinstance(parsed, dict):
        return ""
    error = parsed.get("error")
    if isinstance(error, dict):
        message = str(error.get("message", "")).strip()
        if message:
            return message
    return str(parsed.get("message", "")).strip()


def _is_openai_chat_model(model_id: str) -> bool:
    if not model_id:
        return False
    prefixes = ("gpt-", "o1", "o3", "o4", "o5")
    lowered = model_id.lower()
    if not lowered.startswith(prefixes):
        return False
    blocked_terms = (
        "audio",
        "realtime",
        "transcribe",
        "tts",
        "search",
        "embedding",
        "moderation",
        "image",
    )
    return not any(term in lowered for term in blocked_terms)


def _is_mistral_chat_model(model_id: str) -> bool:
    if not model_id:
        return False
    lowered = model_id.lower()
    blocked_terms = ("embed", "moderation")
    return not any(term in lowered for term in blocked_terms)
