"""Crossref - authoritative DOI metadata, no key, polite pool via mailto."""

from __future__ import annotations

from typing import Any

from ..models import Work
from .base import BaseProvider, clean_text


class CrossrefProvider(BaseProvider):
    name = "crossref"
    endpoint = "https://api.crossref.org/works"

    def __init__(self, session, throttle, timeout, contact_email: str) -> None:
        super().__init__(session, throttle, timeout)
        self.contact_email = contact_email

    def search(self, query: str, limit: int) -> list[Work]:
        data = self._get_json(
            self.endpoint,
            {
                "query": query,
                "rows": str(limit),
                "select": "DOI,title,author,issued,container-title,abstract,URL,is-referenced-by-count,type",
                "mailto": self.contact_email,
            },
        )
        items = (data or {}).get("message", {}).get("items", []) or []
        works: list[Work] = []
        for rank, item in enumerate(items):
            work = self._to_work(item)
            if work is None:
                continue
            works.append(self._record(work, rank))
        return works

    def _to_work(self, item: dict[str, Any]) -> Work | None:
        titles = item.get("title") or []
        title = clean_text(titles[0] if titles else None)
        if not title:
            return None

        authors: list[str] = []
        for author in item.get("author") or []:
            name = " ".join(filter(None, [author.get("given"), author.get("family")])).strip()
            if name:
                authors.append(name)

        year: int | None = None
        date_parts = (item.get("issued", {}).get("date-parts") or [[None]])[0]
        if date_parts and isinstance(date_parts[0], int):
            year = date_parts[0]

        venue = clean_text((item.get("container-title") or [None])[0])
        doi = item.get("DOI")
        cited = item.get("is-referenced-by-count")
        abstract = clean_text(item.get("abstract"), limit=4000)

        return Work(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            url=item.get("URL") or (f"https://doi.org/{doi}" if doi else None),
            abstract=abstract,
            cited_by=cited if isinstance(cited, int) else None,
        )
