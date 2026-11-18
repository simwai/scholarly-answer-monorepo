"""Tests for identity normalisation and cross-provider merging."""

from __future__ import annotations

from scholarly_answer_engine.models import Work
from scholarly_answer_engine.normalize import (
    merge_works,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
    title_fingerprint,
)


def make(title, provider, rank, doi=None, arxiv=None, year=None):
    work = Work(title=title, year=year, doi=doi, arxiv_id=arxiv)
    work.add_provider(provider, rank)
    return work


def test_normalize_doi_strips_resolver():
    assert normalize_doi("https://doi.org/10.1038/nature12373") == "10.1038/nature12373"
    assert normalize_doi("DOI:10.1000/XYZ") == "10.1000/xyz"
    assert normalize_doi(None) is None
    assert normalize_doi("  10.1/a ") == "10.1/a"


def test_normalize_arxiv_id_handles_both_styles():
    assert normalize_arxiv_id("2501.01234v3") == "2501.01234v3"
    assert normalize_arxiv_id("http://arxiv.org/abs/cs/0501001v2") == "cs/0501001v2"
    assert normalize_arxiv_id("https://arxiv.org/pdf/2103.00020v1.pdf") == "2103.00020v1"
    assert normalize_arxiv_id("not an id") is None


def test_title_fingerprint_is_order_independent():
    assert title_fingerprint("Attention Is All You Need") == title_fingerprint("all you need is attention")
    assert title_fingerprint("Attention Is All You Need") != title_fingerprint("A different paper")
    assert title_fingerprint(None) is None


def test_merge_unifies_same_paper_across_providers():
    crossref = make("Attention Is All You Need", "crossref", 0, doi="10.48550/arXiv.1706.03762")
    arxiv = make("Attention Is All You Need", "arxiv", 2, arxiv="1706.03762", year=2017)
    europe = make("Attention is all you need", "europepmc", 1, doi="10.48550/ARXIV.1706.03762")

    merged = merge_works([crossref, arxiv, europe])

    assert len(merged) == 1
    work = merged[0]
    assert work.corroboration == 3
    assert set(work.providers) == {"crossref", "arxiv", "europepmc"}
    assert normalize_doi(work.doi) == "10.48550/arxiv.1706.03762"
    assert work.arxiv_id == "1706.03762"
    assert work.year == 2017


def test_merge_keeps_distinct_papers_apart():
    a = make("Paper one", "crossref", 0, doi="10.1/a")
    b = make("Paper two", "crossref", 1, doi="10.1/b")
    merged = merge_works([a, b])
    assert len(merged) == 2


def test_merge_matches_by_title_when_no_ids():
    shared = make("A study of small things", "crossref", 0)
    other = make("A Study of Small Things", "europepmc", 3)
    merged = merge_works([shared, other])
    assert len(merged) == 1
    assert merged[0].corroboration == 2


def test_merge_fills_missing_metadata_first_wins():
    first = make("Paper", "crossref", 0, year=None)
    second = make("Paper", "arxiv", 1, year=2022)
    merged = merge_works([first, second])
    assert len(merged) == 1
    # Merge fills gaps but does not overwrite an existing value.
    assert merged[0].year == 2022


def test_normalize_title_is_unicode_folded():
    assert normalize_title("Café") == "cafe"
