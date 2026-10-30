import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from hybrid_rag_api.api import create_app
from hybrid_rag_api.generation import REFUSAL_MARKER
from tests.fakes import make_pdf

QUESTION = "Which endpoint should a load balancer use as a readiness probe?"


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def parse_sse(body: str) -> list[tuple[str, object]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((fields.get("event"), json.loads(fields["data"])))
    return events


def test_health_reports_index_and_model(client, service):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 9
    assert body["chunks"] == len(service.index)
    assert body["model"] == "fake-model"


def test_upload_list_and_delete_documents(client):
    files = [
        ("files", ("guide.md", b"# Guide\n\nThe answer is forty-two.", "text/markdown")),
        ("files", ("scan.pdf", make_pdf(["Page one mentions otters."]), "application/pdf")),
    ]
    response = client.post("/documents", files=files)
    assert response.status_code == 201
    assert [(r["doc_id"], r["status"]) for r in response.json()] == [
        ("guide", "added"),
        ("scan", "added"),
    ]
    ids = [d["id"] for d in client.get("/documents").json()]
    assert {"guide", "scan"} <= set(ids)

    hits = client.post("/search", json={"query": "otters", "k": 1}).json()
    assert hits[0]["doc_id"] == "scan" and hits[0]["section"] == "page 1"

    assert client.delete("/documents/scan").status_code == 204
    assert client.delete("/documents/scan").status_code == 404


def test_upload_rejects_unsupported_and_oversized_files(client, service):
    files = [
        ("files", ("ok.md", b"# Fine", "text/markdown")),
        ("files", ("x.png", b"\x89PNG", "image/png")),
    ]
    assert client.post("/documents", files=files).status_code == 415
    assert "ok" not in {d["id"] for d in client.get("/documents").json()}  # all or nothing
    empty = client.post("/documents", files=[("files", ("e.md", b"  ", "text/markdown"))])
    assert empty.status_code == 422
    service.settings = replace(service.settings, max_upload_bytes=10)
    big = client.post("/documents", files=[("files", ("a.txt", b"x" * 11, "text/plain"))])
    assert big.status_code == 413


def test_search_validates_input_and_supports_modes(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    hits = client.post("/search", json={"query": "readyz probe", "k": 2, "mode": "bm25"}).json()
    assert len(hits) == 2 and hits[0]["bm25_rank"] == 1 and hits[0]["dense_rank"] is None


def test_query_returns_a_cited_answer(client, service, fake_llm):
    cid = service.search(QUESTION)[0].chunk.id
    fake_llm.replies = [f"Use /readyz [{cid}]."]
    body = client.post("/query", json={"question": QUESTION}).json()
    assert body["refused"] is False
    assert body["citations"][0]["id"] == cid
    assert body["retrieved"][0]["id"] == cid


def test_query_refusal_is_a_normal_response(client, fake_llm):
    fake_llm.replies = [REFUSAL_MARKER]
    body = client.post("/query", json={"question": "What is the payload size limit?"}).json()
    assert body["refused"] is True and body["refusal_reason"] == "insufficient_context"


def test_query_stream_sends_sse_events(client, service, fake_llm):
    cid = service.search(QUESTION)[0].chunk.id
    fake_llm.replies = [f"Use the /readyz endpoint [{cid}]."]
    with client.stream("POST", "/query/stream", json={"question": QUESTION}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.read().decode())
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "retrieval" and kinds[-1] == "done" and "token" in kinds
    tokens = "".join(data for kind, data in events if kind == "token")
    assert tokens == f"Use the /readyz endpoint [{cid}]."
    assert events[-1][1]["citations"][0]["id"] == cid


def test_stream_reports_errors_in_band(client, service):
    class Exploding:
        name = "boom"

        async def stream(self, system, user):
            raise RuntimeError("upstream failure")
            yield  # pragma: no cover

    service.llm = Exploding()
    with client.stream("POST", "/query/stream", json={"question": QUESTION}) as response:
        events = parse_sse(response.read().decode())
    assert events[-1] == ("error", {"type": "RuntimeError"})


def test_generation_endpoints_return_503_without_a_model(client, service):
    service.llm = None
    assert client.post("/query", json={"question": QUESTION}).status_code == 503
    assert client.post("/query/stream", json={"question": QUESTION}).status_code == 503
    assert client.post("/search", json={"query": "readyz"}).status_code == 200


def test_model_api_errors_become_bad_gateway(client, service):
    import anthropic
    import httpx2

    class Unreachable:
        name = "unreachable"

        async def generate(self, system, user):
            request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.APIConnectionError(request=request)

    service.llm = Unreachable()
    response = client.post("/query", json={"question": QUESTION})
    assert response.status_code == 502
    assert response.json() == {"detail": "model API error (connection)"}
