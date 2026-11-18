"""Citation-count enrichment via OpenAlex DOI singletons (the $0 endpoint).

Crossref already returns citation counts for its own records, so this only fills
gaps - and only for the top-M works, because every call is one request against a
metered account.
"""

from __future__ import annotations

import logging

from ..models import Work

logger = logging.getLogger(__name__)


def enrich_citations(works: list[Work], openalex, top: int) -> int:
    """Fill missing citation counts for the top ``top`` works. Returns fill count."""
    if openalex is None or not works:
        return 0
    filled = 0
    for work in works[:top]:
        if work.cited_by is not None:
            continue
        doi = work.doi
        if not doi:
            continue
        count = openalex.get_citation_count(doi)
        if count is not None:
            work.cited_by = count
            filled += 1
    return filled
