"""Identity normalisation and cross-provider deduplication.

Three independent keys, tried in order of reliability:
DOI -> arXiv ID -> normalised title fingerprint.

Providers overlap (the same paper is in Crossref, arXiv and Europe PMC), so
merging is what makes cross-provider corroboration meaningful.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable

from .models import Work

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize_doi(doi: str | None) -> str | None:
    """Lowercase a DOI and strip the resolver prefixes providers like to add."""
    if not doi:
        return None
    value = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        value = value.removeprefix(prefix)
    return value or None


def normalize_arxiv_id(raw: str | None) -> str | None:
    """Normalise arXiv identifiers, both old-style (cs/0501001v2) and new (2501.01234v3).

    Accepts bare IDs, /abs/ URLs and /pdf/ URLs with or without a .pdf suffix.
    Strips leading path segments explicitly rather than with ``lstrip``, whose
    character-set semantics would mangle IDs such as ``pdf/2103.00020``.
    """
    if not raw:
        return None
    value = raw.strip()
    if "arxiv.org" in value:
        value = value.split("arxiv.org", 1)[-1]
    value = value.strip("/")
    value = value.removesuffix(".pdf")
    for segment in ("abs", "pdf"):
        if value.startswith(segment + "/"):
            value = value[len(segment) + 1 :]
    match = re.search(r"((?:[a-z-]+/\d{7})|(\d{4}\.\d{4,5}))(v\d+)?$", value)
    if not match:
        return None
    return f"{match.group(1)}{match.group(3) or ''}".lower()


def normalize_title(title: str | None) -> str:
    """Aggressive normalisation for fingerprinting: unicode fold, lowercase, punctuation-stripped."""
    if not title:
        return ""
    text = unicodedata.normalize("NFKD", title)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = _NON_ALNUM.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def title_fingerprint(title: str | None) -> str | None:
    """Order-independent token hash, so 'attention is all you need' == 'all you need is attention'."""
    normalized = normalize_title(title)
    if not normalized:
        return None
    return hashlib.sha1(" ".join(sorted(normalized.split())).encode("utf-8")).hexdigest()


def _identity_keys(work: Work) -> list[str]:
    keys: list[str] = []
    doi = normalize_doi(work.doi)
    if doi:
        keys.append(f"doi:{doi}")
    arxiv = normalize_arxiv_id(work.arxiv_id)
    if arxiv:
        keys.append(f"arxiv:{arxiv}")
    if normalize_title(work.title):
        keys.append(f"title:{title_fingerprint(work.title)}")
    return keys


def _preferred(values: Iterable[str | None], current: str | None) -> str | None:
    if current:
        return current
    for value in values:
        if value:
            return value
    return current


def _fill_missing(existing: Work, incoming: Work) -> None:
    """Merge metadata and provenance into ``existing``.

    Field values keep the first non-empty one. Providers accumulate, so a work
    returned by both Crossref and arXiv carries both - which is what makes the
    corroboration count and the RRF sum meaningful.
    """
    if not existing.authors and incoming.authors:
        existing.authors = incoming.authors
    if existing.year is None and incoming.year is not None:
        existing.year = incoming.year
    existing.venue = _preferred([incoming.venue], existing.venue)
    existing.doi = _preferred([incoming.doi], existing.doi)
    existing.arxiv_id = _preferred([incoming.arxiv_id], existing.arxiv_id)
    existing.url = _preferred([incoming.url], existing.url)
    if not existing.abstract and incoming.abstract:
        existing.abstract = incoming.abstract
    existing.pmcid = _preferred([incoming.pmcid], existing.pmcid)
    existing.oa_url = _preferred([incoming.oa_url], existing.oa_url)
    existing.open_access = existing.open_access or incoming.open_access
    if existing.cited_by is None and incoming.cited_by is not None:
        existing.cited_by = incoming.cited_by

    for provider in incoming.providers:
        if provider not in existing.providers:
            existing.providers.append(provider)
    for provider, rank in incoming.ranks.items():
        # If one provider reports the same work twice, keep the better rank.
        existing.ranks[provider] = min(existing.ranks.get(provider, rank), rank)


def merge_works(works: list[Work]) -> list[Work]:
    """Merge works from all providers into one canonical list.

    Returns works with ``providers`` populated, which is what RRF scores and the
    corroboration summary depend on.
    """
    by_key: dict[str, Work] = {}
    order: list[str] = []

    for work in works:
        keys = _identity_keys(work)
        if not keys:
            keys = [f"raw:{work.title}"]
        # Find an existing entry via any identity key (a paper discovered by DOI
        # in one provider and by title in another must still merge).
        existing: Work | None = None
        for key in keys:
            if key in by_key:
                existing = by_key[key]
                break

        if existing is None:
            existing = work
            order.append(keys[0])
            for key in keys:
                by_key.setdefault(key, existing)
            continue

        _fill_missing(existing, work)
        for key in keys:
            by_key[key] = existing

    return [by_key[key] for key in order]


def dedupe(works: list[Work]) -> list[Work]:
    """Merge and return works sorted by first appearance."""
    return merge_works(works)
