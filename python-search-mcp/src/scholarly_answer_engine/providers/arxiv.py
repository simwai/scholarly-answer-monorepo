"""arXiv - Atom API, no key, one request every three seconds.

arXiv is the tightest rate limit in the system, so its interval is configured
separately in config.
"""

from __future__ import annotations

import re

import feedparser

from ..models import Work
from ..retrieval.keywords import build_boolean_query
from .base import BaseProvider, clean_text


class ArxivProvider(BaseProvider):
    name = "arxiv"
    endpoint = "http://export.arxiv.org/api/query"

    def build_query(self, question: str) -> str:
        """arXiv needs salient terms, not a verbatim phrase.

        ``all:"<full question>"`` matches only when the whole phrase appears
        literally, which no paper does for a natural-language question, so
        sending the raw question reliably returns zero results.
        """
        return build_boolean_query(question)

    def search(self, query: str, limit: int) -> list[Work]:
        search_query = query or self.build_query(query)
        xml = self._get_text(
            self.endpoint,
            {
                "search_query": search_query,
                "start": "0",
                "max_results": str(limit),
                "sortBy": "relevance",
            },
        )
        feed = feedparser.parse(xml)
        works: list[Work] = []
        for rank, entry in enumerate(feed.entries or []):
            work = self._to_work(entry)
            if work is None:
                continue
            works.append(self._record(work, rank))
        return works

    def _to_work(self, entry) -> Work | None:
        title = clean_text(entry.get("title"))
        if not title:
            return None

        authors: list[str] = []
        for author in entry.get("authors") or []:
            name = clean_text(author.get("name"))
            if name:
                authors.append(name)

        arxiv_id = self._extract_id(entry)
        year: int | None = None
        published = entry.get("published") or entry.get("updated") or ""
        match = re.search(r"(19|20)\d{2}", published)
        if match:
            year = int(match.group(0))

        pdf_url = None
        for link in entry.get("links") or []:
            if getattr(link, "get", lambda *_: None)("type") == "application/pdf":
                pdf_url = link.get("href")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        doi = None
        for key in ("arxiv_doi", "doi"):
            if entry.get(key):
                doi = entry.get(key)
                break

        return Work(
            title=title,
            authors=authors,
            year=year,
            venue="arXiv preprint",
            doi=doi,
            arxiv_id=arxiv_id,
            url=(f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else entry.get("link")),
            abstract=clean_text(entry.get("summary"), limit=4000),
            pmcid=None,
            oa_url=pdf_url,
            open_access=True,
        )

    @staticmethod
    def _extract_id(entry) -> str | None:
        raw_id = entry.get("id") or ""
        match = re.search(r"abs/([^?#]+)", raw_id)
        if match:
            return match.group(1)
        return None
