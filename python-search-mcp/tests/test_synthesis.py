"""Tests for citation parsing and the post-check invariant."""

from __future__ import annotations

from scholarly_answer_engine.models import Work
from scholarly_answer_engine.synthesis import add_citation_numbers, parse_claims, post_check


def make(index: int, corroboration: int, provider="crossref") -> Work:
    work = Work(title=f"Paper {index}", providers=[provider] if corroboration < 2 else [provider, "arxiv"])
    return work


def test_parse_claims_extracts_multiple_citation_numbers():
    answer = "First claim [1, 3]. Second claim [2]."
    works = [make(1, 2), make(2, 1), make(3, 1)]
    claims = parse_claims(answer, {i: w.to_citation(i) for i, w in enumerate(works, 1)})
    assert claims[0].citations == [1, 3]
    assert claims[1].citations == [2]


def test_claim_is_corroborated_when_any_source_is_multi_provider():
    works = [make(1, 2), make(2, 1)]
    claims = parse_claims("A claim citing only the single-source paper [2].", 
                          {i: w.to_citation(i) for i, w in enumerate(works, 1)})
    assert claims[0].corroborated is False

    claims = parse_claims("A claim citing the corroborated paper [1].",
                          {i: w.to_citation(i) for i, w in enumerate(works, 1)})
    assert claims[0].corroborated is True


def test_post_check_flags_citation_out_of_range():
    works = [make(1, 1)]
    _, warnings = post_check("Something [1] and something impossible [7].", works)
    assert any("[7]" in w for w in warnings)


def test_post_check_flags_uncited_assertion():
    works = [make(1, 1)]
    long_sentence = "This is a fairly long assertion that carries no citation marker at all."
    _, warnings = post_check(long_sentence, works)
    assert any("no citation" in w for w in warnings)


def test_post_check_flags_never_cited_sources():
    works = [make(1, 1), make(2, 1)]
    _, warnings = post_check("Only the first source is cited here [1].", works)
    assert any("never cited" in w for w in warnings)


def test_post_check_clean_answer_has_no_warnings():
    works = [make(1, 1)]
    _, warnings = post_check("Short cited claim [1].", works)
    assert warnings == []


def test_add_citation_numbers_appends_source_list_when_absent():
    works = [make(1, 1), make(2, 1)]
    result = add_citation_numbers("An answer with no markers at all.", works)
    assert "Sources:" in result
    assert "[1] Paper 1" in result
    assert "[2] Paper 2" in result


def test_add_citation_numbers_leaves_marked_answer_alone():
    works = [make(1, 1)]
    original = "This answer already cites [1]."
    assert add_citation_numbers(original, works) == original


def test_add_citation_numbers_no_works_returns_original():
    assert add_citation_numbers("No sources.", []) == "No sources."
