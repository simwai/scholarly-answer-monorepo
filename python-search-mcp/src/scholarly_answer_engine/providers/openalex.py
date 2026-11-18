"""OpenAlex - metered since Feb 2026.

Only the DOI singleton endpoint is used: it costs $0, while ``search=``/list
calls are metered (~$0.001 each against a ~$0.10/day keyless budget). This module
therefore exposes no search method at all, so the metered surface is unreachable
by construction rather than by convention.
"""

from __future__ import annotations

import logging

from .base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAlexProvider(BaseProvider):
    name = "openalex"
    endpoint = "https://api.openalex.org"

    def get_citation_count(self, doi: str) -> int | None:
        """Citation count for one DOI. Returns None on any failure."""
        try:
            data = self._get_json(f"{self.endpoint}/works/doi:{doi}", {"select": "id,cited_by_count,doi"})
        except Exception as exc:  # noqa: BLE001 - enrichment is best-effort, never fatal
            logger.debug("openalex lookup failed for %s: %s", doi, exc)
            return None
        if not isinstance(data, dict):
            return None
        count = data.get("cited_by_count")
        return count if isinstance(count, int) else None

    def search(self, query: str, limit: int) -> list:  # pragma: no cover - unreachable by design
        raise NotImplementedError(
            "OpenAlex search is metered and deliberately unavailable; "
            "citation counts come from the $0 DOI singleton endpoint only."
        )
