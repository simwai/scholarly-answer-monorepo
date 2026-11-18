"""Environment-driven configuration. Every value has a working default.

The retrieval path is intentionally keyless: Crossref, arXiv and Europe PMC need
no credentials. The only settings that carry a credential are the LLM endpoints,
and the router is expected to serve the anonymous Kilo lane so that no key is
required there either.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


# Ordered LLM tier list, walked when a tier is unusable.
#
# VERIFIED against live catalogs (2026-11-11).
#
# `stepfun-ai/step-5-preview` IS a free endpoint - on NVIDIA NIM, under the
# `stepfun-ai/` publisher prefix, priced $0/$0 per 1M with no credit card.
# An earlier revision of this config wrongly concluded no free StepFun model
# existed: that check only looked at Kilo, where the same model appears under
# the `stepfun/` prefix and is PAID (~$1/M in). The publisher prefix is the
# difference: stepfun-ai/* on NIM is free, stepfun/* on Kilo is not.
#
# Step 5 Preview: 600B-total / 27B-active MoE, 1M context, reasoning + tool
# calling. NIM free tier is prototyping-only, non-commercial, ~1,000 credits.
NIM_STEP5 = "stepfun-ai/step-5-preview"
KILO_FREE_NEMOTRON = "nvidia/nemotron-3.5-super-vl:free"

DEFAULT_PIN = (
    f"{NIM_STEP5},"
    f"{KILO_FREE_NEMOTRON},"
    "kilo-auto/free"
)

# The NIM tier needs a free key (build.nvidia.com/settings). Without it the
# chain skips straight to the Kilo free tiers, which work anonymously.
DEFAULT_PIN_NO_NIM_KEY = (
    f"{KILO_FREE_NEMOTRON},"
    "kilo-auto/free"
)


@dataclass
class Config:
    """All runtime configuration, resolved once at startup."""

    # --- LLM transport -------------------------------------------------
    # The router is the primary transport. The inline chain is a contingency.
    llm_transport: str = field(default_factory=lambda: os.getenv("SEARCH_LLM_TRANSPORT", "router"))
    router_url: str = field(default_factory=lambda: os.getenv("ROUTER_URL", "http://127.0.0.1:8787"))
    router_pin: list[str] = field(
        default_factory=lambda: [p.strip() for p in os.getenv("ROUTER_PIN", DEFAULT_PIN).split(",") if p.strip()]
    )
    router_allow_fallback: bool = field(default_factory=lambda: _bool("ROUTER_ALLOW_FALLBACK", True))
    router_deadline: float = field(default_factory=lambda: _float("ROUTER_DEADLINE", 120.0))
    # Start the router automatically when it is not already listening.
    router_autostart: bool = field(default_factory=lambda: _bool("ROUTER_AUTOSTART", True))
    router_dir: str = field(default_factory=lambda: os.getenv("ROUTER_DIR", ""))
    router_port: int = field(default_factory=lambda: _int("ROUTER_PORT", 0))
    # Contingency only: direct provider chain when SEARCH_LLM_TRANSPORT=inline.
    kilo_base_url: str = field(default_factory=lambda: os.getenv("KILO_BASE_URL", "https://ai-gateway.kilo.ai/api/v1"))
    kilo_api_key: str = field(default_factory=lambda: os.getenv("KILO_API_KEY", ""))
    nim_base_url: str = field(default_factory=lambda: os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    nim_api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    reasoning: str = field(default_factory=lambda: os.getenv("SEARCH_LLM_REASONING", "high"))
    contact_email: str = field(default_factory=lambda: os.getenv("SEARCH_CONTACT_EMAIL", "research@example.invalid"))

    # --- retrieval -----------------------------------------------------
    engines: list[str] = field(
        default_factory=lambda: [e.strip() for e in os.getenv("SEARCH_ENGINES", "crossref,arxiv,europepmc").split(",") if e.strip()]
    )
    enable_semantic_scholar: bool = field(default_factory=lambda: _bool("ENABLE_SEMANTIC_SCHOLAR", False))
    per_provider_limit: int = field(default_factory=lambda: _int("PER_PROVIDER_LIMIT", 15))
    max_sources: int = field(default_factory=lambda: _int("SEARCH_MAX_SOURCES", 5))
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))

    # --- conditional expansion ----------------------------------------
    expansion_enabled: bool = field(default_factory=lambda: _bool("ENABLE_EXPANSION", True))
    min_works: int = field(default_factory=lambda: _int("EXPANSION_MIN_WORKS", 3))
    min_providers: int = field(default_factory=lambda: _int("EXPANSION_MIN_PROVIDERS", 2))
    expansion_max_queries: int = field(default_factory=lambda: _int("EXPANSION_MAX_QUERIES", 3))

    # --- full text -----------------------------------------------------
    fulltext_enabled: bool = field(default_factory=lambda: _bool("ENABLE_FULLTEXT", True))
    fulltext_chars: int = field(default_factory=lambda: _int("FULLTEXT_CHARS", 12000))
    fulltext_timeout: float = field(default_factory=lambda: _float("FULLTEXT_TIMEOUT", 30.0))
    fulltext_max_bytes: int = field(default_factory=lambda: _int("FULLTEXT_MAX_BYTES", 12_000_000))
    fetch_concurrency: int = field(default_factory=lambda: _int("FETCH_CONCURRENCY", 4))

    # --- pacing --------------------------------------------------------
    arxiv_interval: float = field(default_factory=lambda: _float("ARXIV_INTERVAL", 3.0))
    provider_min_interval: float = field(default_factory=lambda: _float("PROVIDER_MIN_INTERVAL", 0.25))
    request_timeout: float = field(default_factory=lambda: _float("REQUEST_TIMEOUT", 25.0))

    # --- cache ---------------------------------------------------------
    cache_enabled: bool = field(default_factory=lambda: _bool("ENABLE_CACHE", True))
    cache_ttl: float = field(default_factory=lambda: _float("CACHE_TTL_SECONDS", 900.0))

    # --- openalex (singleton only) ------------------------------------
    openalex_enabled: bool = field(default_factory=lambda: _bool("ENABLE_OPENALEX", True))
    openalex_enrich_top: int = field(default_factory=lambda: _int("OPENALEX_ENRICH_TOP", 10))

    def sanitized(self) -> Config:
        """Clamp obviously invalid values rather than failing later."""
        object.__setattr__(self, "per_provider_limit", max(1, self.per_provider_limit))
        object.__setattr__(self, "max_sources", max(1, min(self.max_sources, 20)))
        object.__setattr__(self, "fulltext_chars", max(500, self.fulltext_chars))
        object.__setattr__(self, "engines", [e for e in self.engines if e])
        return self
