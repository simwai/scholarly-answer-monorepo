"""Europe PMC - no key, generous limits, and the only provider here that hands
back open-access full text directly (fullTextXML for the OA subset)."""

from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup

from ..models import Work
from .base import BaseProvider, clean_text


def _split_authors(author_string: str) -> list[str]:
    """Split a Europe PMC ``authorString`` into individual authors.

    Records use one of two conventions: ``"Vaswani A, Shazeer N"`` (comma
    separated, surname first) or ``"Vaswani, Ashish; Shazeer, Noam"`` (semicolon
    separated, with an internal comma). Semicolons are unambiguous, so they take
    precedence; falling back to a comma split is the standard case.
    """
    value = author_string.strip()
    if not value:
        return []
    if ";" in value:
        parts = [part.strip() for part in value.split(";")]
    else:
        parts = [part.strip() for part in value.split(",")]
    return [part for part in parts if part]


class EuropePmcProvider(BaseProvider):
    name = "europepmc"
    endpoint = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def search(self, query: str, limit: int) -> list[Work]:
        data = self._get_json(
            self.endpoint,
            {
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": str(limit),
            },
        )
        results = (data or {}).get("resultList", {}).get("result", []) or []
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

        authors: list[str] = []
        for author in _split_authors(item.get("authorString") or ""):
            name = clean_text(author)
            if name:
                authors.append(name)

        year: int | None = None
        raw_year = item.get("pubYear")
        if isinstance(raw_year, int):
            year = raw_year
        elif isinstance(raw_year, str) and raw_year.isdigit():
            year = int(raw_year)

        doi = item.get("doi")
        pmcid = item.get("pmcid")
        venue = clean_text(item.get("journalTitle") or item.get("bookOrReportDetails", {}).get("publisher"))
        is_oa = bool(item.get("isOpenAccess"))

        oa_url = self._best_oa_url(item, pmcid)

        return Work(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            pmcid=pmcid,
            url=(f"https://doi.org/{doi}" if doi else (f"https://europepmc.org/article/PMC/{pmcid}" if pmcid else None)),
            abstract=clean_text(item.get("abstractText"), limit=4000),
            oa_url=oa_url,
            open_access=is_oa or bool(oa_url),
        )

    @staticmethod
    def _best_oa_url(item: dict[str, Any], pmcid: str | None) -> str | None:
        urls = item.get("fullTextUrlList", {}).get("fullTextUrl", []) or []
        for url in urls:
            if url.get("documentStyle") == "pdf" and url.get("url"):
                return url.get("url")
        for url in urls:
            if url.get("availability") == "Open access" and url.get("url"):
                return url.get("url")
        if pmcid:
            return f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        return None

    def full_text_xml(self, pmcid: str, max_chars: int) -> str | None:
        """Fetch and flatten full-text XML for an open-access PMCID."""
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        xml = self._get_text(url)
        if not xml.strip():
            return None
        soup = BeautifulSoup(xml, "xml")
        for tag in soup(["xref", "table-wrap", "fig", "graphic", "inline-formula", "disp-formula"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ", strip=True).split())
        return text[:max_chars] or None
