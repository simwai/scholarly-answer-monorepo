"""Prompts for expansion and synthesis.

The synthesis prompt is the compensating control for the abstract-only
hallucination risk: a fabricated method is *not* an unsupported citation, so the
citation post-check cannot catch it. The prompt therefore restricts claims to
what is actually present in the supplied text and names the failure modes
explicitly.
"""

SUPPLEMENT_SYSTEM = (
    "You are a precise literature-synthesis assistant. You answer questions using "
    "ONLY the numbered sources provided. You never use outside knowledge."
)

SUPPLEMENT_RULES = """Rules:
- Answer the question directly and thoroughly.
- Support each factual claim with an inline citation [n] where n is the source number.
- A claim may only use evidence you can actually see in that source's text.
- If a source is only an abstract, do NOT assert its methods, sample sizes, datasets,
  baselines or numeric results - the abstract does not state them. Say so instead.
- Do not extrapolate, generalise, or reconcile contradictions silently: note them.
- If the sources do not answer the question, say what is missing rather than guessing.
- End with a short list of the sources you actually relied on, by number."""

EXPANSION_SYSTEM = "You rewrite research questions into scholarly search queries. You output only a JSON array."


def build_synthesis_user(question: str, works) -> str:
    """Assemble the numbered source block for the synthesis prompt."""
    lines = [f"Question: {question}", "", "Sources:"]
    for index, work in enumerate(works, start=1):
        lines.append(f"[{index}] {work.title}")
        meta = ", ".join(
            filter(
                None,
                [
                    ", ".join(work.authors[:6]) or None,
                    str(work.year) if work.year else None,
                    work.venue or None,
                    f"doi:{work.doi}" if work.doi else None,
                ],
            )
        )
        if meta:
            lines.append(f"    {meta}")
        body = (work.full_text or work.abstract or "").strip()
        if body:
            lines.append(f"    {body}")
        else:
            lines.append("    (no text available)")
        lines.append("")
    lines.append(SUPPLEMENT_RULES)
    return "\n".join(lines)


__all__ = [
    "EXPANSION_SYSTEM",
    "SUPPLEMENT_RULES",
    "SUPPLEMENT_SYSTEM",
    "build_synthesis_user",
]
