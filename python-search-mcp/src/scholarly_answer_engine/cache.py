"""Client-side response cache, checked *before* retrieval.

A router-side synthesis cache would run after engines and fetch had already
executed, so a hit could only save the synthesis call. This one short-circuits
the expensive part, which is why the TTL is short (15 min default) and the age
is reported so the caller can judge freshness.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .config import Config


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _normalize_question(question: str) -> str:
    """Case-fold and collapse whitespace; drop trailing sentence punctuation.

    Without the punctuation strip, "what is attention?" and "what is attention"
    would be different cache keys for the same question.
    """
    text = " ".join(question.split()).lower()
    return text.rstrip("?!.,;:").strip()


class ResponseCache:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._store: dict[str, tuple[float, dict[str, Any]]] = {}
        self.enabled = config.cache_enabled
        self.ttl = config.cache_ttl

    def key(self, question: str, max_sources: int, follow_up: str | None) -> str:
        payload = json.dumps(
            {
                "question": _normalize_question(question),
                "max_sources": max_sources,
                "follow_up": (follow_up or "").strip().lower(),
            },
            sort_keys=True,
        )
        return _hash(payload)

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_at, payload = entry
        age = time.time() - stored_at
        if age > self.ttl:
            self._store.pop(key, None)
            return None
        result = dict(payload)
        result["cached"] = True
        result["cache_age"] = round(age, 1)
        return result

    def put(self, key: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._store[key] = (time.time(), payload)

    def stats(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "ttl_seconds": self.ttl, "entries": len(self._store)}
