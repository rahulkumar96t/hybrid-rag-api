import numpy as np
import pytest

from hybrid_rag_api.chunking import chunk_text
from hybrid_rag_api.embeddings import Embedder, HashingEmbedder, make_embedder
from hybrid_rag_api.loaders import load_bytes
from hybrid_rag_api.retrieval import HybridIndex
from hybrid_rag_api.store import IndexMismatchError, IndexStore


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dim=256)
    a = embedder.embed_documents(["retry with exponential backoff", "cron schedules"])
    b = HashingEmbedder(dim=256).embed_documents(
        ["retry with exponential backoff", "cron schedules"]
    )
    assert a.dtype == np.float32 and a.shape == (2, 256)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    assert isinstance(embedder, Embedder)


def test_hashing_embedder_rewards_shared_words_and_subwords():
    embedder = HashingEmbedder(dim=1024)
    query = embedder.embed_query("retrying failed jobs")
    close, far = embedder.embed_documents(["jobs are retried when they fail", "cron time zones"])
    assert float(close @ query) > float(far @ query)


def test_empty_inputs():
    embedder = HashingEmbedder(dim=64)
    assert embedder.embed_documents([]).shape == (0, 64)
    assert float(np.linalg.norm(embedder.embed_query("the of and"))) == 0.0  # only stop-words


def test_make_embedder_specs():
    assert make_embedder("hashing:128").dim == 128
    with pytest.raises(ValueError, match="unknown embedder"):
        make_embedder("word2vec")


def _ingest(store: IndexStore, embedder: HashingEmbedder, name: str, text: str) -> None:
    doc = load_bytes(name, text.encode())
    chunks = chunk_text(doc.text, doc.kind, max_words=20, overlap_words=0)
    store.upsert_document(doc, chunks, embedder.embed_documents([c.text for c in chunks]))


def test_store_round_trip_on_disk(tmp_path):
    path = tmp_path / "idx" / "index.sqlite"
    embedder = HashingEmbedder(dim=64)
    store = IndexStore(path, embedder.name, embedder.dim)
    _ingest(store, embedder, "notes.md", "# Notes\n\nalpha beta gamma\n\n## More\n\ndelta epsilon")
    store.close()

    reopened = IndexStore(path, embedder.name, embedder.dim)
    records, vectors = reopened.load_chunks()
    assert [r.id for r in records] == ["notes:0", "notes:1"]
    assert [r.section for r in records] == ["Notes", "Notes > More"]
    assert records[0].title == "Notes" and records[0].source == "notes.md"
    assert vectors.shape == (2, 64)
    expected = embedder.embed_documents([r.text for r in records])
    np.testing.assert_allclose(vectors, expected)


def test_store_refuses_a_different_embedder(tmp_path):
    path = tmp_path / "index.sqlite"
    IndexStore(path, "hashing-64-c4", 64).close()
    with pytest.raises(IndexMismatchError, match="embedder"):
        IndexStore(path, "fastembed:other", 64)
    with pytest.raises(IndexMismatchError, match="dim"):
        IndexStore(path, "hashing-64-c4", 128)


def test_upsert_replaces_chunks_and_delete_cascades():
    embedder = HashingEmbedder(dim=64)
    store = IndexStore(":memory:", embedder.name, embedder.dim)
    _ingest(store, embedder, "a.txt", " ".join(f"w{i}" for i in range(50)))
    _ingest(store, embedder, "b.txt", "other document")
    assert len(store.load_chunks()[0]) == 4
    _ingest(store, embedder, "a.txt", "short now")
    records, _ = store.load_chunks()
    assert [r.id for r in records] == ["b:0", "a:0"]
    assert store.delete_document("a") is True
    assert store.delete_document("a") is False
    assert [r.id for r in store.load_chunks()[0]] == ["b:0"]


def test_upsert_validates_vector_shape():
    store = IndexStore(":memory:", "x", 8)
    doc = load_bytes("a.txt", b"hello world")
    chunks = chunk_text(doc.text, doc.kind)
    with pytest.raises(ValueError, match="shape"):
        store.upsert_document(doc, chunks, np.zeros((len(chunks), 4), dtype=np.float32))


def test_hybrid_index_reports_per_retriever_ranks():
    embedder = HashingEmbedder(dim=512)
    store = IndexStore(":memory:", embedder.name, embedder.dim)
    _ingest(store, embedder, "retries.md", "# Retries\n\nFailed jobs are retried with backoff.")
    _ingest(store, embedder, "cron.md", "# Cron\n\nSchedules use cron expressions.")
    index = HybridIndex(store, embedder)

    hybrid = index.search("how are failed jobs retried", k=2)
    assert hybrid[0].chunk.doc_id == "retries"
    assert hybrid[0].bm25_rank == 1 and hybrid[0].dense_rank == 1
    assert hybrid[0].score > hybrid[1].score

    lexical = index.search("cron", k=2, mode="bm25")
    assert [r.chunk.doc_id for r in lexical] == ["cron"]  # BM25 drops zero scores
    assert lexical[0].dense_rank is None

    dense = index.search("cron", k=2, mode="dense")
    assert len(dense) == 2 and dense[0].bm25_rank is None


def test_support_gate():
    embedder = HashingEmbedder(dim=512)
    store = IndexStore(":memory:", embedder.name, embedder.dim)
    _ingest(store, embedder, "retries.md", "# Retries\n\nFailed jobs are retried with backoff.")
    index = HybridIndex(store, embedder)
    assert index.has_support(index.search("backoff"))
    assert not index.has_support(index.search("photosynthesis in green plants"))
    assert index.search("anything", k=0) == []
