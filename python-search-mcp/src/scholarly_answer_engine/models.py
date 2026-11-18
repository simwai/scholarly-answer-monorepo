"""Normalised domain types shared across providers, fusion and synthesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Work:
    """A single scholarly work in a normalised shape.

    ``providers`` accumulates every provider that returned this work, so its
    length is the cross-provider corroboration count used for RRF fusion.
    """

    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    abstract: str | None = None
    pmcid: str | None = None
    oa_url: str | None = None
    open_access: bool = False
    cited_by: int | None = None
    providers: list[str] = field(default_factory=list)
    # Retrieval stage bookkeeping (not part of the citation payload).
    ranks: dict[str, int] = field(default_factory=dict)
    rrf_score: float = 0.0
    # Full-text stage output.
    full_text: str | None = None
    grounding: str = "abstract"

    def add_provider(self, name: str, rank: int) -> None:
        if name not in self.providers:
            self.providers.append(name)
            self.ranks[name] = rank

    @property
    def corroboration(self) -> int:
        return len(self.providers)

    def to_citation(self, index: int) -> Citation:
        return Citation(
            index=index,
            title=self.title,
            authors=list(self.authors),
            year=self.year,
            venue=self.venue,
            doi=self.doi,
            arxiv_id=self.arxiv_id,
            url=self.url,
            provider=";".join(self.providers),
            corroborated=self.corroboration >= 2,
            grounding=self.grounding,
            cited_by=self.cited_by,
        )


@dataclass
class Citation:
    """A citation entry emitted in the tool response."""

    index: int
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    provider: str = ""
    corroborated: bool = False
    grounding: str = "abstract"
    cited_by: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchOutcome:
    """Everything the caller needs to interpret an answer."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    corroboration: dict[str, Any] = field(default_factory=dict)
    model_used: str = ""
    from_fallback: bool = False
    cached: bool = False
    cache_age: float | None = None
    expanded: bool = False
    sub_queries: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "sources": self.sources,
            "corroboration": self.corroboration,
            "model_used": self.model_used,
            "from_fallback": self.from_fallback,
            "cached": self.cached,
            "cache_age": self.cache_age,
            "expanded": self.expanded,
            "sub_queries": self.sub_queries,
            "warnings": self.warnings,
        }
