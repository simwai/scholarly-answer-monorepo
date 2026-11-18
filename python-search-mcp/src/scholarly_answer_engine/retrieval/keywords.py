"""Salient-term extraction for providers whose query language is keyword-based.

arXiv's ``all:"..."`` matches a verbatim phrase, so a full natural-language
question ("how does low-rank adaptation reduce the cost of fine-tuning large
language models") can never match any paper. It needs the salient terms.

Crossref and Europe PMC accept free text and work well enough with the question
as-is, so keyword extraction is opt-in per provider rather than global.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won't", "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves"]
)

_WORD = re.compile(r"[a-z0-9][a-z0-9\-\.]*")


def _tokens(question: str) -> list[str]:
    return _WORD.findall(question.lower())


def extract_keywords(question: str, max_terms: int = 8) -> list[str]:
    """Return salient terms from a question, preserving order.

    Stopwords and very short tokens are dropped. Hyphenated compounds such as
    "low-rank" are kept whole because they carry the technical meaning.
    """
    tokens = [t for t in _tokens(question) if t not in STOPWORDS and len(t) > 2]
    # Deduplicate while preserving first-appearance order.
    seen: set[str] = set()
    unique: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        unique.append(token)
    return unique[:max_terms]


def build_boolean_query(question: str, field: str = "all", max_terms: int = 6) -> str:
    """Build an arXiv-style boolean query from the question's salient terms.

    OR is used rather than AND: a paper is relevant if it matches any of the
    key terms, and relevance ranking then orders them. AND across every term
    would under-match badly.
    """
    terms = extract_keywords(question, max_terms)
    if not terms:
        return ""
    quoted = [f'{term}' for term in terms]
    return f"{field}:(" + " OR ".join(quoted) + ")"
