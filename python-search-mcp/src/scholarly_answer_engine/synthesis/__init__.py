"""Citation parsing and the post-check that every synthesis passes.

The post-check guarantees the invariant the plan requires: every ``[n]`` in the
answer maps to a real source. It also detects claims that carry no citation at
all - a fabricated method is *not* an unsupported reference, so the post-check
cannot catch that class of error, but an uncited claim is still surfaced as a
warning rather than silently trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import Citation, Work

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Claim:
    text: str
    citations: list[int] = field(default_factory=list)

    @property
    def corroborated(self) -> bool:
        return any(self._corroborated_indices.get(n) for n in self.citations)

    _corroborated_indices: dict[int, bool] = field(default_factory=dict)


def parse_claims(answer: str, citations: dict[int, Citation]) -> list[Claim]:
    """Split the answer into sentences and attach the citation numbers each uses."""
    claims: list[Claim] = []
    for sentence in _SENTENCE.split(answer.strip()):
        if not sentence.strip():
            continue
        numbers: list[int] = []
        for group in _CITATION.findall(sentence):
            for part in group.split(","):
                part = part.strip()
                if part.isdigit():
                    numbers.append(int(part))
        claim = Claim(text=sentence, citations=numbers)
        claim._corroborated_indices = {n: citations[n].corroborated for n in numbers if n in citations}
        claims.append(claim)
    return claims


def post_check(answer: str, works: list[Work]) -> tuple[list[Claim], list[str]]:
    """Validate citations. Returns (claims, warnings). Never mutates the answer."""
    citations: dict[int, Citation] = {i: w.to_citation(i) for i, w in enumerate(works, start=1)}
    warnings: list[str] = []
    claims = parse_claims(answer, citations)

    total = len(citations)
    used: set[int] = set()
    for claim in claims:
        for n in claim.citations:
            if n in citations:
                used.add(n)
            else:
                warnings.append(f"answer cites [{n}] but only {total} sources exist")

    uncited = [c.text for c in claims if not c.citations and len(c.text) > 40]
    if uncited:
        warnings.append(f"{len(uncited)} assertion(s) carry no citation")

    unused = [n for n in sorted(citations) if n not in used]
    if unused and total:
        warnings.append(f"sources never cited: {unused}")

    return claims, warnings


def add_citation_numbers(answer: str, works: list[Work]) -> str:
    """If the model emitted no [n] markers at all, append a source list.

    The prompt asks for inline citations, but a model may answer without them;
    this keeps the citation payload actionable instead of silently disconnected.
    """
    if _CITATION.search(answer):
        return answer
    if not works:
        return answer
    lines = [answer.rstrip(), "", "Sources:"]
    for index, work in enumerate(works, start=1):
        lines.append(f"[{index}] {work.title}")
    return "\n".join(lines)
