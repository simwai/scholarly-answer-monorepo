"""Tests for RRF fusion, corroboration and the conditional-expansion trigger."""

from __future__ import annotations

from scholarly_answer_engine.config import Config
from scholarly_answer_engine.models import Work
from scholarly_answer_engine.retrieval.fusion import (
    apply_priors,
    corroboration_summary,
    needs_expansion,
    score_rrf,
)


def ranked(title: str, provider: str, rank: int, extras: dict | None = None) -> Work:
    work = Work(title=title, **(extras or {}))
    work.add_provider(provider, rank)
    return work


def test_rrf_prefers_rank_agreeing_across_providers():
    # A appears in both lists; B and C appear once each.
    a = ranked("A", "crossref", 0)
    a.add_provider("arxiv", 1)
    b = ranked("B", "arxiv", 0)
    c = ranked("C", "crossref", 1)

    scored = score_rrf([a, b, c], k=60)
    assert scored[0].title == "A"
    assert scored[0].corroboration == 2


def test_rrf_score_is_sum_of_reciprocal_ranks():
    work = ranked("X", "crossref", 0)
    work.add_provider("arxiv", 2)
    scored = score_rrf([work], k=60)[0]
    assert abs(scored.rrf_score - (1 / 60 + 1 / 62)) < 1e-9


def test_single_provider_highest_rank_beats_later_multi_provider_rank():
    # B is rank 0 in one list; A is rank 5 in two lists. RRF: A ~= 2/65 > 1/60.
    a = ranked("A", "crossref", 5)
    a.add_provider("arxiv", 5)
    b = ranked("B", "crossref", 0)
    scored = score_rrf([a, b], k=60)
    assert scored[0].title == "A"


def test_corroboration_summary_counts():
    a = ranked("A", "crossref", 0)
    a.add_provider("arxiv", 1)
    a.add_provider("europepmc", 2)
    b = ranked("B", "crossref", 1)

    summary = corroboration_summary([a, b])
    assert summary["works"] == 2
    assert summary["corroborated"] == 1
    assert summary["corroborated_by_three"] == 1
    assert summary["providers"] == {"crossref": 2, "arxiv": 1, "europepmc": 1}
    assert summary["providers_active"] == 3


def test_priors_only_break_ties():
    base = ranked("A", "crossref", 0)
    cited = ranked("B", "crossref", 0, extras={"cited_by": 5000})

    result = apply_priors([base, cited])
    # B outranks A purely on the citation prior, but by a tiny margin.
    assert result[0].title == "B"
    assert abs(result[0].rrf_score - result[1].rrf_score - min(0.0001 * 5000, 0.01)) < 1e-9


def test_priors_prefer_open_access():
    closed = ranked("A", "crossref", 0)
    open_work = ranked("B", "crossref", 0, extras={"oa_url": "https://example.org/x.pdf"})
    result = apply_priors([closed, open_work])
    assert result[0].title == "B"


def _config(**overrides) -> Config:
    base = Config().sanitized()
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_needs_expansion_when_too_few_works():
    works = [ranked("A", "crossref", 0)]
    assert needs_expansion(works, _config(min_works=3)) is True


def test_needs_expansion_when_no_cross_provider_corroboration():
    works = [ranked("A", "crossref", 0), ranked("B", "arxiv", 0)]
    assert needs_expansion(works, _config(min_works=3, min_providers=2)) is True


def test_needs_expansion_when_single_provider_answers():
    works = [ranked("A", "crossref", 0), ranked("B", "crossref", 1), ranked("C", "crossref", 2)]
    assert needs_expansion(works, _config(min_providers=2)) is True


def test_no_expansion_when_retrieval_is_healthy():
    a = ranked("A", "crossref", 0)
    a.add_provider("arxiv", 0)
    b = ranked("B", "crossref", 1)
    b.add_provider("europepmc", 0)
    works = [a, b, ranked("C", "crossref", 2)]
    assert needs_expansion(works, _config()) is False


def test_needs_expansion_with_no_results_at_all():
    assert needs_expansion([], _config(expansion_enabled=True)) is True


def test_expansion_can_be_disabled():
    assert needs_expansion([], _config(expansion_enabled=False)) is False
