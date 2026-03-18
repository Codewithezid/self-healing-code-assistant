from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _derive_key(secret: str) -> bytes:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _mask_key(api_key: str) -> str:
    cleaned = api_key.strip()
    if len(cleaned) <= 4:
        return "****"
    return f"****{cleaned[-4:]}"


@dataclass(frozen=True)
class StoredKeyPublic:
    key_id: str
    provider: str
    label: str
    masked_key: str
    created_at: str


class EncryptedKeyVault:
    """Encrypted key storage with optional persistence."""

    def __init__(
        self,
        *,
        file_path: Path,
        secret: str,
        max_entries: int = 50,
    ) -> None:
        self.file_path = file_path
        self.max_entries = max(1, max_entries)
        self._lock = threading.Lock()
        resolved_secret = secret.strip()
        self.persistent = bool(resolved_secret)
        if not resolved_secret:
            resolved_secret = secrets.token_urlsafe(48)
        self._fernet = Fernet(_derive_key(resolved_secret))
        self._records: list[dict[str, str]] = []
        if self.persistent:
            self._load()

    def list_keys(self, *, provider: str | None = None) -> list[StoredKeyPublic]:
        with self._lock:
            rows = [
                record
                for record in self._records
                if provider is None or record.get("provider") == provider
            ]
            return [
                StoredKeyPublic(
                    key_id=str(record.get("key_id", "")).strip(),
                    provider=str(record.get("provider", "")).strip(),
                    label=str(record.get("label", "")).strip(),
                    masked_key=str(record.get("masked_key", "")).strip() or "****",
                    created_at=str(record.get("created_at", "")).strip(),
                )
                for record in rows
            ]

    def add_key(self, *, provider: str, api_key: str, label: str = "") -> StoredKeyPublic:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ValueError("API key cannot be empty.")
        normalized_provider = provider.strip().lower()
        if not normalized_provider:
            raise ValueError("Provider cannot be empty.")

        with self._lock:
            encrypted = self._fernet.encrypt(normalized_key.encode("utf-8")).decode("utf-8")
            record = {
                "key_id": str(uuid.uuid4()),
                "provider": normalized_provider,
                "label": label.strip() or f"{normalized_provider}-key",
                "masked_key": _mask_key(normalized_key),
                "created_at": _now_iso(),
                "ciphertext": encrypted,
            }
            self._records.insert(0, record)
            if len(self._records) > self.max_entries:
                self._records = self._records[: self.max_entries]
            self._save()
            return StoredKeyPublic(
                key_id=record["key_id"],
                provider=record["provider"],
                label=record["label"],
                masked_key=record["masked_key"],
                created_at=record["created_at"],
            )

    def get_api_key(self, *, key_id: str, provider: str | None = None) -> str | None:
        lookup = key_id.strip()
        if not lookup:
            return None
        with self._lock:
            for record in self._records:
                if record.get("key_id") != lookup:
                    continue
                if provider and record.get("provider") != provider:
                    continue
                ciphertext = str(record.get("ciphertext", "")).strip()
                if not ciphertext:
                    return None
                try:
                    return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
                except (InvalidToken, ValueError):
                    return None
        return None

    def delete_key(self, *, key_id: str) -> bool:
        lookup = key_id.strip()
        if not lookup:
            return False
        with self._lock:
            before = len(self._records)
            self._records = [record for record in self._records if record.get("key_id") != lookup]
            changed = len(self._records) != before
            if changed:
                self._save()
            return changed

    def _load(self) -> None:
        if not self.file_path.exists():
            return
        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        rows = payload.get("records", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return
        cleaned: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            key_id = str(row.get("key_id", "")).strip()
            provider = str(row.get("provider", "")).strip().lower()
            ciphertext = str(row.get("ciphertext", "")).strip()
            if not key_id or not provider or not ciphertext:
                continue
            cleaned.append(
                {
                    "key_id": key_id,
                    "provider": provider,
                    "label": str(row.get("label", "")).strip() or f"{provider}-key",
                    "masked_key": str(row.get("masked_key", "")).strip() or "****",
                    "created_at": str(row.get("created_at", "")).strip() or _now_iso(),
                    "ciphertext": ciphertext,
                }
            )
        self._records = cleaned[: self.max_entries]

    def _save(self) -> None:
        if not self.persistent:
            return
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "records": self._records,
        }
        self.file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
