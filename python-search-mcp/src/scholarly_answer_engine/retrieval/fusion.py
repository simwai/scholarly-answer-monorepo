"""RRF fusion and the corroboration summary.

RRF is used because Crossref, arXiv and Europe PMC each return relevance-ranked
lists on opaque, incomparable scales - there is nothing to normalise. RRF needs
no scores, only ranks.

Corroboration falls out of the fusion for free: the RRF sum has one term per
provider list containing a document, so the term count *is* the corroboration
count.
"""

from __future__ import annotations

from ..models import Work


def score_rrf(works: list[Work], k: int = 60) -> list[Work]:
    """Attach ``rrf_score`` and return works sorted best-first.

    ``works`` must already be merged (``providers``/``ranks`` populated).
    """
    for work in works:
        work.rrf_score = sum(1.0 / (k + rank) for rank in work.ranks.values())
    return sorted(works, key=lambda w: (-w.rrf_score, -w.corroboration, w.title.lower()))


def apply_priors(works: list[Work], citation_weight: float = 0.0001, oa_bonus: float = 0.0005) -> list[Work]:
    """Small additive priors on top of RRF.

    Citation count is a real quality signal for literature but is temporally
    biased (newer papers have fewer citations), so its weight is deliberately
    tiny - enough to break ties, never to reorder. The OA bonus prefers works
    whose full text we can actually read.
    """
    for work in works:
        bonus = 0.0
        if work.cited_by:
            bonus += min(citation_weight * work.cited_by, 0.01)
        if work.oa_url:
            bonus += oa_bonus
        work.rrf_score += bonus
    return sorted(works, key=lambda w: -w.rrf_score)


def corroboration_summary(works: list[Work]) -> dict:
    """Summarise how independent the evidence base is."""
    total = len(works)
    multi = sum(1 for w in works if w.corroboration >= 2)
    all_three = sum(1 for w in works if w.corroboration >= 3)
    per_provider: dict[str, int] = {}
    for work in works:
        for provider in work.providers:
            per_provider[provider] = per_provider.get(provider, 0) + 1
    return {
        "works": total,
        "corroborated": multi,
        "corroborated_by_three": all_three,
        "providers": per_provider,
        "providers_active": len(per_provider),
    }


def needs_expansion(works: list[Work], config, max_sources: int | None = None) -> bool:
    """Conditional-expansion trigger, computed from data already in hand.

    No LLM call is spent deciding. Expansion fires when the raw retrieval was
    weak: too few works, too few providers answering, or nothing corroborated
    across providers.

    ``max_sources`` matters: asking for 2 sources and getting 2 is not weak, so
    the works threshold is capped at what the caller actually requested.
    """
    if not config.expansion_enabled:
        return False
    if not works:
        return True

    min_works = config.min_works
    if max_sources is not None:
        min_works = min(min_works, max(1, max_sources))

    active_providers = {p for w in works for p in w.providers}
    return (
        len(works) < min_works
        or len(active_providers) < config.min_providers
        or max((w.corroboration for w in works), default=1) < 2
    )
