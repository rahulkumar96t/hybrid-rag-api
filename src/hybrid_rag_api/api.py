"""HTTP API (FastAPI)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable
from dataclasses import asdict
from typing import Annotated, Literal

import anthropic
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from . import __version__
from .loaders import DocumentError, UnsupportedFileType, kind_for
from .service import Answer, RagService

log = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=50)
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)


def create_app(service: RagService) -> FastAPI:
    app = FastAPI(
        title="hybrid-rag-api",
        version=__version__,
        description="Hybrid BM25 + dense retrieval with cited answers from Claude.",
    )
    app.state.service = service

    @app.exception_handler(anthropic.APIError)
    async def upstream_error(_: Request, exc: anthropic.APIError) -> JSONResponse:
        # Rate limits, overloads and auth problems at the model provider are not
        # this service's fault: report them as a bad gateway, without internals.
        log.warning("model API error: %s", exc)
        status = getattr(exc, "status_code", None)
        return JSONResponse({"detail": f"model API error ({status or 'connection'})"}, 502)

    def require_llm() -> None:
        # A dependency, so it runs before a streaming response has started.
        if service.llm is None:
            raise HTTPException(503, "answer generation is not configured (no API credentials)")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "documents": len(service.documents()),
            "chunks": len(service.index),
            "embedder": service.index.embedder.name,
            "model": service.llm.name if service.llm else None,
        }

    @app.get("/documents")
    def list_documents() -> list[dict]:
        return [asdict(d) for d in service.documents()]

    @app.post("/documents", status_code=201)
    async def upload_documents(files: Annotated[list[UploadFile], File()]) -> list[dict]:
        """Ingest Markdown, text or PDF files. Re-uploading a file replaces it."""
        limit = service.settings.max_upload_bytes
        batch: list[tuple[str, bytes]] = []
        # Check every file before ingesting any, so a bad file rejects the whole batch.
        for upload in files:
            name = upload.filename or "upload.txt"
            try:
                kind_for(name)
            except UnsupportedFileType as exc:
                raise HTTPException(415, str(exc)) from exc
            data = await upload.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(413, f"{name} exceeds the {limit}-byte upload limit")
            batch.append((name, data))
        results = []
        for name, data in batch:
            try:
                result = await run_in_threadpool(service.ingest_bytes, name, data)
            except DocumentError as exc:
                raise HTTPException(422, str(exc)) from exc
            results.append(asdict(result))
        return results

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str) -> None:
        if not service.delete_document(doc_id):
            raise HTTPException(404, f"document {doc_id!r} not found")

    @app.post("/search")
    def search(request: SearchRequest) -> list[dict]:
        hits = service.search(request.query, request.k, request.mode)
        return [hit.to_dict() for hit in hits]

    @app.post("/query", dependencies=[Depends(require_llm)])
    async def query(request: QueryRequest) -> dict:
        answer = await service.answer(request.question, request.k)
        return answer.to_dict()

    @app.post(
        "/query/stream",
        response_class=EventSourceResponse,
        dependencies=[Depends(require_llm)],
    )
    async def query_stream(request: QueryRequest) -> AsyncIterable[ServerSentEvent]:
        """Server-sent events: ``retrieval``, any number of ``token``, then ``done``.

        ``done`` carries the full answer object, including whether it was refused
        after the fact (for example because it cited nothing); clients should
        treat it as authoritative over the streamed tokens.
        """
        try:
            async for kind, payload in service.answer_stream(request.question, request.k):
                data = payload.to_dict() if isinstance(payload, Answer) else payload
                yield ServerSentEvent(event=kind, data=data)
        except Exception as exc:  # the 200 status is already sent, so report in-band
            log.exception("streaming answer failed")
            yield ServerSentEvent(event="error", data={"type": type(exc).__name__})

    return app
