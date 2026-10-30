import json

import pytest

from hybrid_rag_api.cli import main
from hybrid_rag_api.embeddings import HashingEmbedder
from hybrid_rag_api.evaluation import (
    EvalQuestion,
    Target,
    build_index,
    evaluate,
    format_markdown,
    load_dataset,
)
from tests.conftest import CORPUS

DATASET = CORPUS.parent / "qa.jsonl"


@pytest.fixture(scope="module")
def tiny_index(tmp_path_factory):
    corpus = tmp_path_factory.mktemp("corpus")
    (corpus / "cats.md").write_text("# Cats\n\n## Food\n\nCats eat fish.\n\n## Sleep\n\nCats nap.")
    (corpus / "dogs.md").write_text("# Dogs\n\n## Food\n\nDogs eat bones.\n")
    return build_index(corpus, HashingEmbedder(dim=256), chunk_words=50, overlap_words=0)


def test_recall_and_mrr_on_a_controlled_corpus(tiny_index):
    questions = [
        EvalQuestion("a", "what do cats eat fish", (Target("cats", "Food"),)),
        EvalQuestion("b", "dogs bones", (Target("dogs", "Food"),)),
        # two targets, only one can be in the top 1
        EvalQuestion("c", "cats", (Target("cats", "Food"), Target("cats", "Sleep"))),
        EvalQuestion("d", "quantum chromodynamics", ()),
    ]
    report = evaluate(tiny_index, questions, ks=(1, 3), mode="hybrid")
    assert report.answerable == 3 and report.unanswerable == 1
    assert report.recall[1] == pytest.approx((1 + 1 + 0.5) / 3)
    assert report.recall[3] == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)
    assert report.gated_unanswerable == 1 and report.gated_answerable == 0
    assert report.misses == []


def test_misses_are_listed(tiny_index):
    questions = [EvalQuestion("x", "dogs bones", (Target("cats", "Sleep"),))]
    report = evaluate(tiny_index, questions, ks=(1,), mode="bm25")
    assert report.recall[1] == 0.0 and report.mrr == 0.0 and report.misses == ["x"]


def test_bundled_dataset_is_well_formed():
    questions = load_dataset(DATASET)
    assert len(questions) >= 30
    doc_ids = {p.stem for p in CORPUS.iterdir()}
    for q in questions:
        assert all(t.doc in doc_ids for t in q.targets), q.id
    assert sum(not q.answerable for q in questions) >= 5


def test_dataset_rejects_answerable_rows_without_targets(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "x"}\n')
    with pytest.raises(ValueError, match="relevant"):
        load_dataset(path)


def test_markdown_table(tiny_index):
    report = evaluate(tiny_index, [EvalQuestion("a", "cats fish", (Target("cats"),))], ks=(1,))
    table = format_markdown([report])
    assert table.splitlines()[0].startswith("| mode | embedder | recall@1 | MRR")
    assert "| hybrid | hashing-256-c4 | 1.000 | 1.000 | 0/0 | 0/1 |" in table


def test_cli_eval_json_on_bundled_corpus(capsys):
    code = main(
        [
            "eval",
            "--corpus",
            str(CORPUS),
            "--dataset",
            str(DATASET),
            "--modes",
            "bm25,hybrid",
            "-k",
            "1,5",
            "--json",
        ]
    )
    assert code == 0
    reports = json.loads(capsys.readouterr().out)
    assert [r["mode"] for r in reports] == ["bm25", "hybrid"]
    assert set(reports[0]["recall"]) == {"@1", "@5"}


def test_cli_ingest_search_docs_delete(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDER", "hashing:256")
    index = str(tmp_path / "index.sqlite")
    assert main(["--index", index, "ingest", str(CORPUS / "retries.md")]) == 0
    assert "added" in capsys.readouterr().out
    assert main(["--index", index, "search", "dead letter queue", "-k", "1"]) == 0
    assert "[retries:" in capsys.readouterr().out
    assert main(["--index", index, "docs"]) == 0
    assert "retries.md" in capsys.readouterr().out
    assert main(["--index", index, "delete", "retries"]) == 0
    assert main(["--index", index, "delete", "retries"]) == 1


def test_cli_ingest_reports_bad_files(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDER", "hashing:256")
    empty = tmp_path / "empty.md"
    empty.write_text("   ")
    assert main(["--index", str(tmp_path / "i.sqlite"), "ingest", str(empty)]) == 1
    assert "no extractable text" in capsys.readouterr().err
