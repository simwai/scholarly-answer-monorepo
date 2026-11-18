"""End-to-end pipeline test with stub providers and a stub LLM.

Network-free: the provider registry is replaced with stubs returning fixed
works, so the fusion, grounding, synthesis and citation post-check are exercised
deterministically.
"""

from __future__ import annotations

import asyncio

import pytest

import scholarly_answer_engine.server as server_mod
from scholarly_answer_engine.config import Config
from scholarly_answer_engine.models import Work


class StubProvider:
    """Returns a fixed list of works, recording provenance like the real providers."""

    def __init__(self, name: str, works: list[Work]) -> None:
        self.name = name
        self.works = works
        self.calls: list[str] = []

    def build_query(self, question: str) -> str:
        return question

    def search(self, query: str, limit: int) -> list[Work]:
        self.calls.append(query)
        for rank, work in enumerate(self.works[:limit]):
            work.add_provider(self.name, rank)
        return self.works[:limit]


class StubLLM:
    """Answers deterministically; expansion returns two sub-queries."""

    def __init__(self) -> None:
        self.last_model_used = "stepfun/step-3.7-flash:free"
        self.last_from_fallback = False
        self.purposes: list[str] = []

    def complete(self, system: str, user: str, purpose: str = "synthesis") -> str:
        self.purposes.append(purpose)
        if purpose == "expansion":
            return '["keyword one", "keyword two"]'
        # Cites 1 and 3, plus an uncited long claim the post-check must flag.
        return (
            "Low-rank adaptation freezes pretrained weights and learns a low-rank update [1]. "
            "ALoRA allocates rank unevenly across layers [3]. "
            "This sentence is long enough to count as an assertion and carries no citation."
        )


def _work(title: str, **kwargs) -> Work:
    abstract = kwargs.pop("abstract", None)
    if abstract is None:
        # Generate a simple abstract from the title words
        words = title.split()
        abstract = f"This paper presents {title.lower()}. It discusses the key concepts and methods related to {' '.join(words[:3])}."
    return Work(title=title, abstract=abstract, **kwargs)


def build_engine(works_by_provider: dict[str, list[Work]], llm: StubLLM, **overrides):
    config = Config().sanitized()
    config.openalex_enabled = False
    config.fulltext_enabled = False  # no network for full text in this test
    # Autostart would spawn a real router and block the test for its timeout.
    config.router_autostart = False
    for key, value in overrides.items():
        setattr(config, key, value)

    mcp, engine = server_mod.create_server(config)
    # Replace the live registry with deterministic stubs.
    engine.registry.providers = {
        name: StubProvider(name, works) for name, works in works_by_provider.items()
    }
    engine.registry.europepmc = None
    engine.llm = llm
    return mcp, engine, config


def unwrap(result):
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[1]
    first = result[0] if isinstance(result, list) else result
    if isinstance(first, dict):
        return first
    if hasattr(first, "text"):
        import json

        return json.loads(first.text)
    raise TypeError(f"unexpected return {type(result)}")


def test_pipeline_produces_cited_answer_with_provenance():
    shared = _work("ALoRA: Allocating Low-Rank Adaptation", doi="10.1/alora", cited_by=30, open_access=True,
        abstract="Low-rank adaptation freezes pretrained weights and learns a low-rank update. "
                 "This paper presents ALoRA: Allocating Low-Rank Adaptation, a method for "
                 "parameter-efficient fine-tuning of large language models.")
    crossref_only = _work("MiLoRA: Efficient Mixture", doi="10.1/milora", cited_by=35,
        abstract="MiLoRA: Efficient Mixture presents a mixture of low-rank adapters.")
    arxiv_only = _work("RefLoRA: Refactored Low-Rank", arxiv_id="2505.18877", doi="10.1/reflora",
        abstract="RefLoRA: Refactored Low-Rank allocates rank unevenly across layers "
                 "for efficient parameter-efficient fine-tuning.")

    mcp, _engine, _ = build_engine(
        {
            "crossref": [shared, crossref_only],
            "arxiv": [shared, arxiv_only],
        },
        StubLLM(),
        expansion_enabled=False,
        cache_enabled=False,
    )

    payload = unwrap(
        asyncio.run(
            mcp.call_tool(
                "perplexity-search",
                {"question": "how does low-rank adaptation work", "max_sources": 3},
            )
        )
    )

    assert "[1]" in payload["answer"]
    assert payload["model_used"] == "stepfun/step-3.7-flash:free"
    assert payload["from_fallback"] is False
    assert payload["cached"] is False

    citations = payload["citations"]
    # The shared work is first and is corroborated by two providers.
    assert citations[0]["title"] == "ALoRA: Allocating Low-Rank Adaptation"
    assert citations[0]["corroborated"] is True
    assert set(citations[0]["provider"].split(";")) == {"crossref", "arxiv"}
    assert citations[0]["cited_by"] == 30

    summary = payload["corroboration"]
    assert summary["corroborated_claims"] == 1
    assert summary["providers_active"] == 2


def test_uncited_assertion_is_flagged_by_post_check():
    llm = StubLLM()
    mcp, _engine, _ = build_engine(
        {"crossref": [_work("Paper A", doi="10.1/a")]},
        llm,
        expansion_enabled=False,
        cache_enabled=False,
    )
    payload = unwrap(
        asyncio.run(mcp.call_tool("perplexity-search", {"question": "anything", "max_sources": 1}))
    )
    assert any("no citation" in w for w in payload["warnings"])


def test_abstract_only_citations_are_labelled():
    works = [_work("No OA copy", doi="10.1/x", open_access=False)]
    mcp, _engine, _ = build_engine(
        {"crossref": works}, StubLLM(), expansion_enabled=False, cache_enabled=False
    )
    payload = unwrap(
        asyncio.run(mcp.call_tool("perplexity-search", {"question": "q", "max_sources": 1}))
    )
    assert payload["citations"][0]["grounding"] == "abstract"
    assert payload["sources"][0]["has_full_text"] is False


def test_expansion_fires_only_on_weak_retrieval():
    llm = StubLLM()
    # Only one provider, only one work, no corroboration -> weak.
    mcp, engine, _ = build_engine(
        {"crossref": [_work("Lonely paper", doi="10.1/only")]},
        llm,
        expansion_enabled=True,
        expansion_max_queries=2,
        cache_enabled=False,
    )
    payload = unwrap(
        asyncio.run(mcp.call_tool("perplexity-search", {"question": "obscure thing", "max_sources": 1}))
    )
    assert payload["expanded"] is True
    assert payload["sub_queries"] == ["keyword one", "keyword two"]
    assert "expansion" in llm.purposes
    # The sub-queries were actually sent to the providers.
    stubs = engine.registry.providers.values()
    assert any(len(s.calls) > 1 for s in stubs)


def test_expansion_does_not_fire_on_healthy_retrieval():
    llm = StubLLM()
    shared_a = _work("Paper A", doi="10.1/a")
    shared_b = _work("Paper B", doi="10.1/b")
    mcp, _engine, _ = build_engine(
        {
            "crossref": [shared_a, shared_b],
            "arxiv": [shared_a, shared_b],
        },
        llm,
        expansion_enabled=True,
        cache_enabled=False,
    )
    payload = unwrap(
        asyncio.run(mcp.call_tool("perplexity-search", {"question": "common thing", "max_sources": 2}))
    )
    assert payload["expanded"] is False
    assert payload["sub_queries"] == []
    assert "expansion" not in llm.purposes


def test_cache_hit_avoids_a_second_retrieval():
    works = [_work("Cached paper", doi="10.1/cached")]
    llm = StubLLM()
    mcp, _engine, _ = build_engine(
        {"crossref": works},
        llm,
        expansion_enabled=False,
        cache_enabled=True,
        cache_ttl=900.0,
    )
    first = unwrap(asyncio.run(mcp.call_tool("perplexity-search", {"question": "same question"})))
    assert first["cached"] is False

    second = unwrap(asyncio.run(mcp.call_tool("perplexity-search", {"question": "same question"})))
    assert second["cached"] is True
    assert second["cache_age"] is not None and second["cache_age"] >= 0
    # Only one synthesis call happened.
    assert llm.purposes.count("synthesis") == 1

    # A different question misses.
    third = unwrap(asyncio.run(mcp.call_tool("perplexity-search", {"question": "different"})))
    assert third["cached"] is False


def test_papers_tool_returns_raw_retrieval_without_synthesis():
    llm = StubLLM()
    mcp, _engine, _ = build_engine(
        {"crossref": [_work("Paper A", doi="10.1/a")], "arxiv": [_work("Paper A", doi="10.1/a")]},
        llm,
        expansion_enabled=False,
    )
    payload = unwrap(asyncio.run(mcp.call_tool("papers", {"question": "q", "limit": 5})))
    assert len(payload["works"]) == 1
    assert payload["works"][0]["providers"] == ["crossref", "arxiv"]
    assert payload["works"][0]["corroboration"] == 2
    assert llm.purposes == []  # no LLM call at all


def test_no_sources_produces_a_clear_error():
    from scholarly_answer_engine.synth.client import LLMError

    _mcp, engine, _ = build_engine({"crossref": []}, StubLLM(), expansion_enabled=False, cache_enabled=False)
    with pytest.raises(LLMError, match="academic literature"):
        engine.search("some question")
