from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_rag_api.config import Settings
from hybrid_rag_api.embeddings import HashingEmbedder
from hybrid_rag_api.service import RagService
from tests.fakes import FakeAnswerModel

CORPUS = Path(__file__).resolve().parents[1] / "eval" / "corpus"


@pytest.fixture
def settings() -> Settings:
    return Settings(index_path=Path(":memory:"), chunk_words=80, overlap_words=15)


@pytest.fixture
def fake_llm() -> FakeAnswerModel:
    return FakeAnswerModel()


@pytest.fixture
def service(settings: Settings, fake_llm: FakeAnswerModel) -> RagService:
    svc = RagService.from_settings(settings, embedder=HashingEmbedder(dim=512), llm=fake_llm)
    for path in sorted(CORPUS.iterdir()):
        svc.ingest_path(path)
    return svc
