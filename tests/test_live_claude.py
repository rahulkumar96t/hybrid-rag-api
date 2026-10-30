"""Opt-in smoke test against the real Claude API.

Skipped unless both ANTHROPIC_API_KEY and RUN_LIVE_TESTS=1 are set, so CI and
offline runs never make network calls.
"""

import asyncio
import os

import pytest

from hybrid_rag_api.config import Settings
from hybrid_rag_api.embeddings import HashingEmbedder
from hybrid_rag_api.generation import ClaudeAnswerModel
from hybrid_rag_api.service import RagService
from tests.conftest import CORPUS

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("RUN_LIVE_TESTS") == "1"),
        reason="set ANTHROPIC_API_KEY and RUN_LIVE_TESTS=1 to call the real API",
    ),
]


def test_live_answer_cites_and_refuses():
    settings = Settings.from_env()
    model = ClaudeAnswerModel(
        settings.model, effort=settings.effort, fallbacks=settings.fallbacks, max_tokens=4000
    )
    service = RagService.from_settings(
        Settings(index_path=":memory:"), embedder=HashingEmbedder(), llm=model
    )
    for path in sorted(CORPUS.iterdir()):
        service.ingest_path(path)

    answered = asyncio.run(service.answer("How many times is a failing job attempted by default?"))
    assert not answered.refused, answered.raw_answer
    assert any(c.doc_id == "retries" for c in answered.citations)

    refused = asyncio.run(service.answer("What is the maximum payload size for a single job?"))
    assert refused.refused
