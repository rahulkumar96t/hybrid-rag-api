import asyncio
from dataclasses import replace

import pytest

from hybrid_rag_api.generation import REFUSAL_MARKER, SYSTEM_PROMPT
from hybrid_rag_api.service import REFUSAL_MESSAGES, LLMUnavailableError, RagService
from tests.fakes import FakeAnswerModel

QUESTION = "How many times is a failing job attempted by default?"


def top_id(service: RagService, question: str = QUESTION) -> str:
    return service.search(question)[0].chunk.id


def collect(service: RagService, question: str) -> list[tuple[str, object]]:
    async def run():
        return [event async for event in service.answer_stream(question)]

    return asyncio.run(run())


def test_answer_with_valid_citations(service, fake_llm):
    cid = top_id(service)
    fake_llm.replies = [f"A job is attempted at most 5 times by default [{cid}]."]
    answer = asyncio.run(service.answer(QUESTION))
    assert not answer.refused
    assert [c.id for c in answer.citations] == [cid]
    assert answer.citations[0].doc_id == "retries"
    assert answer.model == "fake-model"
    system, user = fake_llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert f'<passage id="{cid}"' in user and user.endswith(QUESTION)


def test_retrieval_gate_refuses_without_calling_the_model(service, fake_llm):
    answer = asyncio.run(service.answer("Explain photosynthesis in green plants"))
    assert answer.refused and answer.refusal_reason == "no_relevant_context"
    assert answer.answer == REFUSAL_MESSAGES["no_relevant_context"]
    assert fake_llm.calls == []


def test_model_saying_insufficient_context_is_a_refusal(service, fake_llm):
    fake_llm.replies = [f"{REFUSAL_MARKER}\nThe passages do not mention payload limits."]
    answer = asyncio.run(service.answer("What is the maximum payload size for a job?"))
    assert answer.refused and answer.refusal_reason == "insufficient_context"
    assert answer.raw_answer.startswith(REFUSAL_MARKER)
    assert answer.citations == []


def test_uncited_answer_is_refused(service, fake_llm):
    fake_llm.replies = ["Five attempts, I believe."]
    answer = asyncio.run(service.answer(QUESTION))
    assert answer.refused and answer.refusal_reason == "uncited_answer"
    assert answer.raw_answer == "Five attempts, I believe."


def test_answer_citing_only_unknown_ids_is_refused(service, fake_llm):
    fake_llm.replies = ["Five attempts [retries:999]."]
    answer = asyncio.run(service.answer(QUESTION))
    assert answer.refusal_reason == "uncited_answer"
    assert answer.invalid_citations == ["retries:999"]


def test_mixed_citations_keep_the_answer_and_report_invalid_ones(service, fake_llm):
    cid = top_id(service)
    fake_llm.replies = [f"Five attempts [{cid}][ghost:1]."]
    answer = asyncio.run(service.answer(QUESTION))
    assert not answer.refused
    assert answer.invalid_citations == ["ghost:1"]


def test_citations_can_be_optional(service, fake_llm):
    service.settings = replace(service.settings, require_citations=False)
    fake_llm.replies = ["Five attempts."]
    assert not asyncio.run(service.answer(QUESTION)).refused


def test_api_refusal_stop_reason(service):
    service.llm = FakeAnswerModel(replies=[""], stop_reason="refusal")
    answer = asyncio.run(service.answer(QUESTION))
    assert answer.refused and answer.refusal_reason == "model_refusal"


def test_missing_model_raises(service):
    service.llm = None
    with pytest.raises(LLMUnavailableError):
        asyncio.run(service.answer(QUESTION))


def test_stream_emits_retrieval_tokens_and_done(service, fake_llm):
    cid = top_id(service)
    reply = f"By default a job is attempted 5 times [{cid}]."
    fake_llm.replies = [reply]
    events = collect(service, QUESTION)
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "retrieval" and kinds[-1] == "done"
    assert "".join(payload for kind, payload in events if kind == "token") == reply
    assert events[0][1][0]["id"] == cid
    done = events[-1][1]
    assert not done.refused and [c.id for c in done.citations] == [cid]


def test_stream_never_leaks_the_refusal_marker(service, fake_llm):
    fake_llm.chunk_size = 3
    fake_llm.replies = [f"  {REFUSAL_MARKER}\nNot covered."]
    events = collect(service, "What is the maximum payload size for a job?")
    assert [kind for kind, _ in events] == ["retrieval", "done"]
    assert events[-1][1].refusal_reason == "insufficient_context"


def test_stream_flushes_answers_shorter_than_the_marker(service):
    service.llm = FakeAnswerModel(replies=["INSUFF"])
    service.settings = replace(service.settings, require_citations=False)
    events = collect(service, QUESTION)
    assert ("token", "INSUFF") in events


def test_stream_gate_refusal_has_no_tokens(service, fake_llm):
    events = collect(service, "Explain photosynthesis in green plants")
    assert [kind for kind, _ in events] == ["retrieval", "done"]
    assert events[-1][1].refusal_reason == "no_relevant_context"
    assert fake_llm.calls == []


def test_reingest_is_idempotent_and_updates_on_change(service):
    first = service.ingest_bytes("extra.md", b"# Extra\n\nZebras are striped.")
    again = service.ingest_bytes("extra.md", b"# Extra\n\nZebras are striped.")
    changed = service.ingest_bytes("extra.md", b"# Extra\n\nZebras are black and white.")
    assert (first.status, again.status, changed.status) == ("added", "unchanged", "updated")
    hit = service.search("zebras black white", k=1)[0]
    assert hit.chunk.doc_id == "extra" and "black and white" in hit.chunk.text


def test_delete_removes_document_from_search(service):
    service.ingest_bytes("extra.md", b"# Extra\n\nZebras are striped.")
    assert service.delete_document("extra")
    assert all(r.chunk.doc_id != "extra" for r in service.search("zebras striped"))
    assert not service.delete_document("extra")
