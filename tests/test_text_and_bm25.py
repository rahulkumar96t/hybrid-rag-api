import math

import pytest

from hybrid_rag_api.bm25 import BM25Index
from hybrid_rag_api.text import stem, tokenize, word_count


def test_tokenize_lowercases_drops_stopwords_and_folds_plurals():
    assert tokenize("The Queues are FULL of jobs") == ["queue", "full", "job"]


def test_tokenize_keeps_identifiers_and_their_parts():
    tokens = tokenize("Set retry.max_attempts to 5")
    assert tokens[:4] == ["set", "retry.max_attempts", "retry", "max"]
    assert "attempt" in tokens  # "attempts" folded by the S-stemmer
    assert "5" in tokens


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("policies", "policy"),
        ("queues", "queue"),
        ("status", "status"),
        ("class", "class"),
        ("bus", "bus"),
        ("goes", "goe"),
        ("v2s", "v2s"),
    ],
)
def test_s_stemmer(word, expected):
    assert stem(word) == expected


def test_word_count():
    assert word_count("  one two\nthree\t four ") == 4


def corpus_index() -> BM25Index:
    docs = [
        "retry backoff jitter retry",
        "schedule cron timezone",
        "retry policy max attempts",
        "metrics prometheus gauge histogram counter queue depth alert window",
    ]
    return BM25Index.build(tokenize(d) for d in docs)


def test_idf_is_positive_and_rarer_terms_score_higher():
    index = corpus_index()
    assert index.idf("retry") > 0
    assert index.idf("cron") > index.idf("retry")
    assert index.idf("unknown") == pytest.approx(math.log(1 + (4 + 0.5) / 0.5))


def test_top_n_ranks_by_term_frequency_and_skips_non_matching_docs():
    index = corpus_index()
    hits = index.top_n(tokenize("retry"), 10)
    assert [doc for doc, _ in hits] == [0, 2]  # doc 0 mentions "retry" twice
    assert all(score > 0 for _, score in hits)


def test_length_normalisation_prefers_shorter_documents():
    index = BM25Index.build([["queue", "a", "b"], ["queue", *["x"] * 30]])
    (best, _), (second, _) = index.top_n(["queue"], 2)
    assert (best, second) == (0, 1)


def test_ties_break_on_document_position():
    index = BM25Index.build([["alpha"], ["alpha"], ["alpha"]])
    assert [doc for doc, _ in index.top_n(["alpha"], 3)] == [0, 1, 2]


def test_duplicate_query_terms_count_once():
    index = corpus_index()
    assert index.scores(["cron", "cron"]) == index.scores(["cron"])


def test_empty_index_and_zero_n():
    assert BM25Index().top_n(["x"], 5) == []
    assert corpus_index().top_n(["retry"], 0) == []


def test_invalid_parameters():
    with pytest.raises(ValueError):
        BM25Index(b=1.5)
