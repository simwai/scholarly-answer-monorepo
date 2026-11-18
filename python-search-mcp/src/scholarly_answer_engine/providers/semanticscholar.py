"""Semantic Scholar - optional, keyless shared pool (~1 req/s).

Enabled only when ``ENABLE_SEMANTIC_SCHOLAR`` is set, because the keyless pool
is shared with all anonymous traffic and rate-limits aggressively. When it 429s
it is dropped for that request rather than failing the retrieval.
"""

from __future__ import annotations

from typing import Any

from ..models import Work
from .base import BaseProvider, clean_text


class SemanticScholarProvider(BaseProvider):
    name = "semanticscholar"
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(self, query: str, limit: int) -> list[Work]:
        data = self._get_json(
            self.endpoint,
            {
                "query": query,
                "limit": str(limit),
                "fields": "title,authors,year,venue,abstract,externalIds,openAccessPdf,citationCount,tldr",
            },
        )
        results = (data or {}).get("data", []) or []
        works: list[Work] = []
        for rank, item in enumerate(results):
            work = self._to_work(item)
            if work is None:
                continue
            works.append(self._record(work, rank))
        return works

    def _to_work(self, item: dict[str, Any]) -> Work | None:
        title = clean_text(item.get("title"))
        if not title:
            return None

        authors = [clean_text(a.get("name")) or "" for a in item.get("authors") or []]
        authors = [a for a in authors if a]

        ids = item.get("externalIds") or {}
        oa_pdf = item.get("openAccessPdf") or {}

        return Work(
            title=title,
            authors=authors,
            year=item.get("year") if isinstance(item.get("year"), int) else None,
            venue=clean_text(item.get("venue")),
            doi=ids.get("DOI"),
            arxiv_id=ids.get("ArXiv"),
            url=(oa_pdf.get("url") if oa_pdf.get("url") else None),
            abstract=clean_text(item.get("abstract"), limit=4000),
            oa_url=oa_pdf.get("url"),
            open_access=bool(oa_pdf.get("url")),
            cited_by=item.get("citationCount") if isinstance(item.get("citationCount"), int) else None,
        )
