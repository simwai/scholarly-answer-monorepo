"""Tests for the client-side TTL cache."""

from __future__ import annotations

import time

from scholarly_answer_engine.cache import ResponseCache
from scholarly_answer_engine.config import Config


def build(ttl: float = 60.0, enabled: bool = True) -> ResponseCache:
    config = Config().sanitized()
    config.cache_ttl = ttl
    config.cache_enabled = enabled
    return ResponseCache(config)


def test_key_is_stable_for_same_question():
    cache = build()
    assert cache.key("What is attention?", 5, None) == cache.key("what is attention", 5, None)
    assert cache.key("What is attention?", 5, None) != cache.key("What is attention?", 6, None)
    assert cache.key("What is attention?", 5, None) != cache.key("What is attention?", 5, "prior turn")
    assert cache.key("Attention?", 5, None) != cache.key("Vision?", 5, None)


def test_hit_within_ttl_reports_cached_and_age():
    cache = build(ttl=60)
    key = cache.key("Q", 5, None)
    cache.put(key, {"answer": "hello", "model_used": "x"})

    hit = cache.get(key)
    assert hit is not None
    assert hit["cached"] is True
    assert hit["answer"] == "hello"
    assert hit["cache_age"] is not None and hit["cache_age"] >= 0


def test_miss_after_ttl():
    cache = build(ttl=0)
    key = cache.key("Q", 5, None)
    cache.put(key, {"answer": "hello"})
    time.sleep(0.01)
    assert cache.get(key) is None


def test_disabled_cache_never_hits():
    cache = build(enabled=False)
    key = cache.key("Q", 5, None)
    cache.put(key, {"answer": "hello"})
    assert cache.get(key) is None


def test_unknown_key_misses():
    assert build().get("nope") is None


def test_stats_reports_entries():
    cache = build()
    cache.put(cache.key("Q", 5, None), {"answer": "x"})
    assert cache.stats()["entries"] == 1
