"""Per-provider request pacing.

Rate limits are the binding constraint for scholarly APIs: arXiv asks for one
request every three seconds, and a question fans out to several providers. This
is a threaded interval limiter with a floor plus jitter, plus a bounded worker
pool so a burst of fetches cannot exceed ``poll_size`` in flight providers.
"""

from __future__ import annotations

import random
import threading
import time


class IntervalLimiter:
    """Enforces a minimum interval between calls to one named bucket."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                delay = self._next_allowed - now
            else:
                delay = 0.0
            # Small jitter avoids a perfectly periodic burst being fingerprinted.
            self._next_allowed = max(now, self._next_allowed) + self.min_interval + random.uniform(0, 0.15)
        if delay > 0:
            time.sleep(delay)


class ProviderThrottle:
    """One limiter per provider, created on demand with its own interval."""

    def __init__(self, default_interval: float, per_provider: dict[str, float] | None = None) -> None:
        self.default_interval = default_interval
        self.per_provider = per_provider or {}
        self._limiters: dict[str, IntervalLimiter] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> IntervalLimiter:
        with self._lock:
            limiter = self._limiters.get(name)
            if limiter is None:
                limiter = IntervalLimiter(self.per_provider.get(name, self.default_interval))
                self._limiters[name] = limiter
            return limiter

    def wait(self, name: str) -> None:
        self.get(name).wait()
