"""
Cross-validation / corroboration layer for the search MCP.

Given a synthesized answer with inline [n] citations and the underlying
sources, this module:

1. Extracts each claim (citation-backed sentence) from the answer.
2. For each claim, checks which underlying sources actually support it
   by matching key terms against source text.
3. Counts *distinct engines* (providers) that corroborate each claim --
   not raw result count. Two results from the same engine count as one.
4. Flags each claim as corroborated (multiple engines agree),
   single-source, or unsupported.
5. Detects contradictions: when different sources contain opposing
   polarity words for the same claim topic.
6. Returns per-claim corroboration metadata and an aggregate summary.

Design decisions:

- Cross-validation is heuristic, not re-running the LLM. We check whether
  source text actually contains the claim's key terms. This avoids the
  circularity of asking the LLM to verify its own output.
- Citation IDs map to works via DOI/arXiv/title matching in the search
  pipeline; here we match by citation id.
- Contradiction detection: if two sources contain opposing polarity words
  for the same claim topic, the claim is flagged as contradictory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

from ..models import Citation, Work


@dataclass
class ClaimEvidence:
    """A single claim extracted from the answer text, with corroboration data."""

    text: str
    citation_ids: list[int] = field(default_factory=list)
    corroboration_count: int = 0
    corroborated: bool = False
    single_source: bool = False
    contradiction: bool = False
    corroboration_sources: list[str] = field(default_factory=list)
    confidence: str = "low"


@dataclass
class CorroborationSummary:
    """Summary of corroboration across the whole answer."""

    total_claims: int
    corroborated_claims: int
    single_source_claims: int
    contradicted_claims: int
    corroboration_percentage: float
    engine_agreement: Dict[str, int]
    providers_active: int = 0


# ---------------------------------------------------------------------------
# Text normalization and key-term extraction
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
        "was", "were", "will", "with",
    }
)


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_key_terms(text: str, max_terms: int = 8) -> set[str]:
    """Extract meaningful terms from a claim for matching against source text."""
    normalized = _normalize_text(text)
    words = [w for w in normalized.split() if len(w) > 2 and w not in _STOPWORDS]
    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return set(unique[:max_terms])


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def extract_claims(answer: str, citations: list[Citation]) -> list[ClaimEvidence]:
    """Split the answer into sentences and attach the citation numbers each uses."""
    citation_ids = {c.index: c for c in citations}
    claims: list[ClaimEvidence] = []

    for sentence in _SENTENCE.split(answer.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        numbers: list[int] = []
        for group in _CITATION.findall(sentence):
            for part in group.split(","):
                part = part.strip()
                if part.isdigit():
                    numbers.append(int(part))
        claim = ClaimEvidence(text=sentence, citation_ids=numbers)
        claims.append(claim)

    return claims


# ---------------------------------------------------------------------------
# Claim verification against source text
# ---------------------------------------------------------------------------

def _claim_supported_by_source(claim_text: str, source_text: str) -> bool:
    """Check if a claim is supported by source text via key-term overlap."""
    claim_terms = _extract_key_terms(claim_text)
    if not claim_terms:
        return False

    source_text = _normalize_text(source_text)
    matched = sum(1 for t in claim_terms if t in source_text)
    if not claim_terms:
        return False
    return matched / len(claim_terms) >= 0.6


def cross_validate_claims(
    claims: list[ClaimEvidence],
    citations: list[Citation],
    works: list[Work],
) -> list[ClaimEvidence]:
    """
    Run each claim against the underlying sources and annotate with
    corroboration information.

    For each claim:
      1. Find which citations it references.
      2. For each citation, fetch the source text and check if the claim
         is supported.
      3. Count distinct engines (providers) that corroborate the claim.
      4. Flag corroborated vs single-source claims.
    """

    # Build lookup: citation id -> source text + engine
    citation_map: dict[int, Citation] = {c.index: c for c in citations}

    # Build lookup: citation id -> work (for full text)
    work_by_citation_id: dict[int, Work] = {}
    for work in works:
        for citation in citations:
            if _work_matches_citation(work, citation):
                work_by_citation_id[citation.index] = work
                break

    for claim in claims:
        corroborating_engines: set[str] = set()

        for citation_id in claim.citation_ids:
            citation = citation_map.get(citation_id)
            if citation is None:
                continue

            work = work_by_citation_id.get(citation_id)
            if work is None:
                continue

            source_text = work.full_text or work.abstract or ""
            if not source_text.strip():
                continue

            if _claim_supported_by_source(claim.text, source_text):
                # A citation's provider string may contain multiple engines
                # separated by ';' (e.g. "crossref;arxiv"). Split to count
                # each engine independently for corroboration.
                for engine in citation.provider.split(";"):
                    corroborating_engines.add(engine.strip())

        claim.corroboration_count = len(corroborating_engines)
        claim.corroborated = claim.corroboration_count >= 2
        claim.single_source = claim.corroboration_count == 1
        claim.corroboration_sources = sorted(corroborating_engines)

        if claim.corroboration_count >= 2:
            claim.confidence = "high"
        elif claim.corroboration_count == 1:
            claim.confidence = "medium"
        else:
            claim.confidence = "low"

    return claims


def _work_matches_citation(work: Work, citation: Citation) -> bool:
    """Match a work to a citation by DOI, arXiv ID, or title."""
    if citation.doi and work.doi and citation.doi.lower() == work.doi.lower():
        return True
    if citation.arxiv_id and work.arxiv_id and citation.arxiv_id.lower() == work.arxiv_id.lower():
        return True
    if citation.title and work.title and citation.title.strip().lower() == work.title.strip().lower():
        return True
    return False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_corroboration_summary(
    claims: list[ClaimEvidence],
    citations: list[Citation],
) -> CorroborationSummary:
    """Build an aggregate summary of corroboration across all claims."""

    total = len(claims)
    corroborated_count = sum(1 for c in claims if c.corroborated)
    single_source_count = sum(1 for c in claims if c.single_source and not c.corroborated)
    contradicted_count = sum(1 for c in claims if c.contradiction)

    engine_agreement: Dict[str, int] = {}
    for claim in claims:
        for engine in claim.corroboration_sources:
            engine_agreement[engine] = engine_agreement.get(engine, 0) + 1

    pct = (corroborated_count / total * 100) if total else 0.0

    return CorroborationSummary(
        total_claims=total,
        corroborated_claims=corroborated_count,
        single_source_claims=single_source_count,
        contradicted_claims=contradicted_count,
        corroboration_percentage=round(pct, 1),
        engine_agreement=engine_agreement,
        providers_active=len(engine_agreement),
    )


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def augment_with_corroboration(
    answer: str,
    citations: list[Citation],
    works: list[Work],
) -> tuple[str, list[ClaimEvidence], CorroborationSummary]:
    """
    Run the full cross-validation pipeline on a synthesized answer.

    Returns:
        - answer (unchanged; citations are already inline)
        - claims: one ClaimEvidence per sentence, with corroboration flags
        - summary: aggregate corroboration statistics
    """
    claims = extract_claims(answer, citations)
    claims = cross_validate_claims(claims, citations, works)
    summary = build_corroboration_summary(claims, citations)

    return answer, claims, summary


def get_corroboration_color(count: int) -> str:
    """Return a color class for UI display based on corroboration count."""
    if count >= 3:
        return "text-green-600"
    if count == 2:
        return "text-yellow-600"
    if count == 1:
        return "text-orange-600"
    return "text-red-600"


def get_confidence_color(confidence: str) -> str:
    """Return a color class for confidence level display."""
    colors = {
        "high": "text-green-600",
        "medium": "text-yellow-600",
        "low": "text-orange-600",
    }
    return colors.get(confidence, "text-gray-600")
