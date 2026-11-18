"""Provider registry: builds the configured providers and fans a query out."""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Callable

from ..config import Config
from ..models import Work
from ..throttle import ProviderThrottle
from .arxiv import ArxivProvider
from .base import BaseProvider, ProviderError, make_session
from .crossref import CrossrefProvider
from .europepmc import EuropePmcProvider
from .openalex import OpenAlexProvider
from .semanticscholar import SemanticScholarProvider
from .unpaywall import UnpaywallProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    def __init__(self, config: Config) -> None:
        self.config = config
        per_provider = {"arxiv": config.arxiv_interval}
        self.throttle = ProviderThrottle(config.provider_min_interval, per_provider)
        self.session = make_session(config.contact_email)
        self.timeout = config.request_timeout

        self.providers: dict[str, BaseProvider] = {}
        for name in config.engines:
            provider = self._build(name)
            if provider is not None:
                self.providers[name] = provider

        # Named accessors for the two providers other layers reach into directly.
        self.europepmc = self.providers.get("europepmc")
        if not isinstance(self.europepmc, EuropePmcProvider):
            self.europepmc = None

        if config.enable_semantic_scholar:
            self.providers["semanticscholar"] = SemanticScholarProvider(self.session, self.throttle, self.timeout)

        self.unpaywall = UnpaywallProvider(self.session, self.throttle, self.timeout, config.contact_email)
        self.openalex: OpenAlexProvider | None = (
            OpenAlexProvider(self.session, self.throttle, self.timeout) if config.openalex_enabled else None
        )

    def _build(self, name: str) -> BaseProvider | None:
        maker: dict[str, Callable[[], BaseProvider]] = {
            "crossref": lambda: CrossrefProvider(self.session, self.throttle, self.timeout, self.config.contact_email),
            "arxiv": lambda: ArxivProvider(self.session, self.throttle, self.timeout),
            "europepmc": lambda: EuropePmcProvider(self.session, self.throttle, self.timeout),
        }
        factory = maker.get(name)
        if factory is None:
            logger.warning("unknown provider %r requested via SEARCH_ENGINES; ignored", name)
            return None
        return factory()

    def search_all(self, query: str, limit: int) -> tuple[list[Work], list[str]]:
        """Run one query against every provider, in parallel.

        Returns (works, failed_providers). A provider failure is recorded but
        never raised: it degrades corroboration instead of the whole retrieval.
        """
        if not self.providers:
            raise ProviderError("no providers are enabled; check SEARCH_ENGINES")

        query_text = query
        results: list[Work] = []
        failed: list[str] = []

        def run(name: str, provider: BaseProvider) -> tuple[str, list[Work] | None, str | None]:
            try:
                # Each provider adapts the question to its own query language.
                query = provider.build_query(query_text)
                return name, provider.search(query, limit), None
            except ProviderError as exc:
                logger.warning("provider %s failed: %s", name, exc)
                return name, None, str(exc)
            except Exception as exc:  # defensive: a parser bug must not kill the request
                logger.exception("provider %s crashed", name)
                return name, None, f"{type(exc).__name__}: {exc}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(self.providers))) as pool:
            futures = {pool.submit(run, name, provider): name for name, provider in self.providers.items()}
            for future in concurrent.futures.as_completed(futures):
                # A provider failing is recorded, never raised; third slot unused.
                name, works, _error = future.result()
                if works is None:
                    failed.append(name)
                    continue
                if not works:
                    logger.info("provider %s returned no results", name)
                    continue
                results.extend(works)

        return results, failed

    def close(self) -> None:
        self.session.close()
