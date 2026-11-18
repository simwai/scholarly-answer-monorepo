"""Conditional query expansion.

Vocabulary mismatch is the dominant failure mode of academic keyword search: a
user asks about "attention", the paper says "self-attention". Expansion fixes
that, but it costs one LLM call and multiplies provider calls - and arXiv's
1 req/3s interval makes every extra sub-query expensive.

So expansion is *conditional*: raw retrieval runs first, and expansion fires
only when that retrieval was weak (see ``fusion.needs_expansion``).

The Exp4Fuse pattern is followed rather than replacement: the original query's
result list is retained and fused together with the sub-query lists, which is
what guards against expansion drift.
"""

from __future__ import annotations

import json
import logging

from ..models import Work

logger = logging.getLogger(__name__)

EXPANSION_PROMPT = """You are a literature-search assistant. Rewrite the user's question as 2-3
short keyword search queries suitable for scholarly APIs such as Crossref, arXiv
and Europe PMC.

Rules:
- Use field terminology the papers would use, not lay phrasing.
- Each query is 3-8 keywords, no punctuation, no explanatory text.
- Do not invent constraints, dates or venues the user did not ask for.
- Output ONLY a JSON array of strings.

Question: {question}
Output:"""


def _parse_sub_queries(text: str, limit: int) -> list[str]:
    """Extract a JSON array of queries from model output, tolerantly."""
    candidate = text.strip()
    start = candidate.find("[")
    end = candidate.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    queries: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            cleaned = item.strip().strip('"').strip()
            if cleaned and cleaned.lower() != "output:":
                queries.append(" ".join(cleaned.split()))
    return queries[:limit]


def expand_question(llm_client, question: str, limit: int) -> list[str]:
    """Generate up to ``limit`` sub-queries. Returns [] on any failure."""
    try:
        response = llm_client.complete(
            system="You rewrite research questions into scholarly search queries.",
            user=EXPANSION_PROMPT.format(question=question),
            purpose="expansion",
        )
    except Exception as exc:  # noqa: BLE001 - expansion is optional; retrieval still succeeds
        logger.warning("query expansion failed: %s", exc)
        return []
    queries = _parse_sub_queries(response, limit)
    logger.info("expansion produced %d sub-queries", len(queries))
    return queries


def fuse_rounds(rounds: list[list[Work]], k: int = 60) -> list[Work]:
    """Fuse several provider rounds, keeping the original round's ranks.

    ``rounds`` is a list of per-round provider result lists (expanded queries
    produce additional rounds). Ranks already recorded on each work are used, so
    a work returned by both the original and an expanded query keeps both
    contributions and therefore a higher score.
    """
    merged: dict[str, Work] = {}
    for round_index, round_works in enumerate(rounds):
        for work in round_works:
            # Re-key by identity so repeats within/across rounds merge.
            from scholarly_answer_engine.normalize import _identity_keys

            keys = _identity_keys(work) or [f"raw:{work.title}"]
            existing = None
            for key in keys:
                if key in merged:
                    existing = merged[key]
                    break
            if existing is None:
                merged[keys[0]] = work
                for key in keys:
                    merged[key] = work
            else:
                existing.ranks.update(work.ranks)

    works = list({id(w): w for w in merged.values()}.values())
    from scholarly_answer_engine.retrieval.fusion import score_rrf

    return score_rrf(works, k=k)


def sub_queries_payload(llm_client, question: str, limit: int) -> list[str]:
    """Convenience wrapper used by the engine."""
    return expand_question(llm_client, question, limit)
