from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class UpstashRedis:
    def __init__(self, *, base_url: str, token: str) -> None:
        if not base_url or not token:
            raise ValueError("Upstash Redis credentials are required.")
        self.base_url = base_url.rstrip("/")
        self.token = token

    def command(self, *parts: Any) -> Any:
        path = "/".join(quote(str(part), safe="") for part in parts)
        request = Request(
            f"{self.base_url}/{path}",
            headers={"Authorization": f"Bearer {self.token}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Upstash request failed: {exc}") from exc
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return payload.get("result")


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, *, limit: int, window_seconds: int) -> tuple[bool, int]:
        now = time.time()
        bucket = self._buckets[key]
        cutoff = now - window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(window_seconds - (now - bucket[0])))
            return False, retry_after
        bucket.append(now)
        return True, 0


class UpstashRateLimiter:
    def __init__(self, redis: UpstashRedis) -> None:
        self.redis = redis

    def allow(self, key: str, *, limit: int, window_seconds: int) -> tuple[bool, int]:
        bucket = int(time.time() // window_seconds)
        redis_key = f"{key}:{bucket}"
        count = int(self.redis.command("INCR", redis_key) or 0)
        if count == 1:
            self.redis.command("EXPIRE", redis_key, window_seconds + 5)
        if count > limit:
            ttl = int(self.redis.command("TTL", redis_key) or window_seconds)
            return False, max(1, ttl)
        return True, 0
