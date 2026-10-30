"""Tokenisation shared by the BM25 index and the hashing embedder.

The tokenizer is deliberately small and predictable: lower-casing, a word regex
that keeps dotted/underscored identifiers (``retry.max_attempts``) together *and*
emits their parts, an English stop-word list, and the classic "S-stemmer"
(Harman, 1991) which only folds plural endings. It is not a linguistic analyzer;
it is good enough for technical documentation and easy to reason about in tests.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_PART_RE = re.compile(r"[._-]")

# A word list reads better than 130 quoted strings.
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours ourselves out over own same
    she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours yourself
    yourselves
    """.split()  # noqa: SIM905
)


def stem(token: str) -> str:
    """Fold plural endings ("queues" -> "queue", "policies" -> "policy")."""
    if len(token) <= 3 or not token.isalpha():
        return token
    if token.endswith("ies") and not token.endswith(("eies", "aies")):
        return token[:-3] + "y"
    if token.endswith("es") and not token.endswith(("aes", "ees", "oes")):
        return token[:-1]
    if token.endswith("s") and not token.endswith(("us", "ss")):
        return token[:-1]
    return token


def _normalise(token: str) -> str | None:
    if token in STOPWORDS:
        return None
    if len(token) == 1 and not token.isdigit():
        return None
    return stem(token)


def tokenize(text: str) -> list[str]:
    """Split ``text`` into normalised index terms (order preserved, duplicates kept)."""
    terms: list[str] = []
    for match in _WORD_RE.finditer(text.lower()):
        word = match.group()
        parts = _PART_RE.split(word)
        candidates = [word, *parts] if len(parts) > 1 else [word]
        for candidate in candidates:
            term = _normalise(candidate)
            if term:
                terms.append(term)
    return terms


def word_count(text: str) -> int:
    """Whitespace-separated word count, used as the chunk length unit."""
    return len(text.split())
