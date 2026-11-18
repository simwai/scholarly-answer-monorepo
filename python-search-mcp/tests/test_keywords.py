"""Tests for salient-term extraction and arXiv's boolean query construction."""

from __future__ import annotations

from scholarly_answer_engine.providers.arxiv import ArxivProvider
from scholarly_answer_engine.retrieval.keywords import build_boolean_query, extract_keywords


def test_extract_keywords_drops_stopwords():
    terms = extract_keywords("how does low-rank adaptation reduce the cost of fine-tuning")
    assert "how" not in terms
    assert "does" not in terms
    assert "the" not in terms
    assert "low-rank" in terms
    assert "adaptation" in terms
    assert "fine-tuning" in terms


def test_extract_keywords_keeps_hyphenated_compounds_whole():
    terms = extract_keywords("what is low-rank adaptation for large language models")
    assert "low-rank" in terms
    assert "large" in terms


def test_extract_keywords_preserves_order_and_deduplicates():
    terms = extract_keywords("attention attention mechanism transformer attention is all you need")
    assert terms.count("attention") == 1
    assert terms == ["attention", "mechanism", "transformer", "need"]


def test_extract_keywords_respects_max_terms():
    terms = extract_keywords("alpha beta gamma delta epsilon zeta eta theta iota kappa", max_terms=4)
    assert len(terms) == 4
    assert terms[0] == "alpha"


def test_extract_keywords_empty_when_only_stopwords():
    assert extract_keywords("what is the of a and to for") == []


def test_build_boolean_query_uses_or_and_field_prefix():
    query = build_boolean_query("how does low-rank adaptation reduce cost")
    assert query.startswith("all:(")
    assert query.endswith(")")
    assert " OR " in query
    assert "low-rank" in query


def test_build_boolean_query_empty_for_stopword_only_question():
    assert build_boolean_query("what is the") == ""


def test_build_boolean_query_respects_max_terms():
    query = build_boolean_query("alpha beta gamma delta epsilon zeta eta theta", max_terms=3)
    assert query.count(" OR ") == 2


def test_arxiv_builds_keyword_query_not_verbatim_phrase():
    """Regression: all:"<full question>" can never match a paper."""
    provider = ArxivProvider.build_query(object.__new__(ArxivProvider), "does low-rank adaptation reduce costs")  # type: ignore[arg-type]
    assert provider.startswith("all:(")
    assert '"' not in provider
    assert "low-rank" in provider

    question = "how does low-rank adaptation reduce the cost of fine-tuning large language models"
    query = ArxivProvider.build_query(object.__new__(ArxivProvider), question)  # type: ignore[arg-type]
    assert "how" not in query
    assert "reduce" in query
    assert "low-rank" in query
