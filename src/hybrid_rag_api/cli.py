"""Command-line interface: ingest, search, ask, serve and evaluate."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

import anthropic

from .config import Settings
from .embeddings import make_embedder
from .evaluation import build_index, evaluate, format_markdown, load_dataset
from .generation import AnswerModel, ClaudeAnswerModel, MissingCredentialsError
from .loaders import DocumentError, discover
from .service import RagService


def _service(args: argparse.Namespace, *, with_llm: bool) -> RagService:
    settings = Settings.from_env()
    if args.index:
        settings = replace(settings, index_path=Path(args.index))
    llm: AnswerModel | None = None
    if with_llm:
        try:
            llm = ClaudeAnswerModel(
                settings.model,
                effort=settings.effort,
                fallbacks=settings.fallbacks,
                max_tokens=settings.max_tokens,
            )
        except (anthropic.AnthropicError, MissingCredentialsError) as exc:
            print(f"warning: answer generation disabled ({exc})", file=sys.stderr)
    return RagService.from_settings(settings, llm=llm, with_llm=False)


def cmd_ingest(args: argparse.Namespace) -> int:
    service = _service(args, with_llm=False)
    failures = 0
    for path in discover(args.paths):
        try:
            result = service.ingest_path(path)
        except DocumentError as exc:
            failures += 1
            print(f"skip  {path}: {exc}", file=sys.stderr)
            continue
        print(f"{result.status:<9} {result.doc_id:<40} {result.chunks:>4} chunks")
    print(f"index: {service.settings.index_path} ({len(service.index)} chunks)")
    return 1 if failures else 0


def cmd_search(args: argparse.Namespace) -> int:
    service = _service(args, with_llm=False)
    for rank, hit in enumerate(service.search(args.query, args.k, args.mode), 1):
        c = hit.chunk
        where = f"{c.source} > {c.section}" if c.section else c.source
        print(f"{rank}. [{c.id}] rrf={hit.score:.4f} bm25={hit.bm25_rank} dense={hit.dense_rank}")
        print(f"   {where}")
        print(f"   {c.text[:160].replace(chr(10), ' ')}...")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    service = _service(args, with_llm=True)
    if service.llm is None:
        print("error: set ANTHROPIC_API_KEY to generate answers", file=sys.stderr)
        return 2
    answer = asyncio.run(service.answer(args.question, args.k))
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return 0
    print(answer.answer)
    if answer.citations:
        print("\nSources:")
        for c in answer.citations:
            print(f"  [{c.id}] {c.source}" + (f" > {c.section}" if c.section else ""))
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    service = _service(args, with_llm=False)
    for doc in service.documents():
        print(f"{doc.id:<40} {doc.n_chunks:>4} chunks  {doc.kind:<8} {doc.source}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    service = _service(args, with_llm=False)
    if service.delete_document(args.doc_id):
        print(f"deleted {args.doc_id}")
        return 0
    print(f"not found: {args.doc_id}", file=sys.stderr)
    return 1


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - starts a server
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(_service(args, with_llm=True)), host=args.host, port=args.port)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    questions = load_dataset(args.dataset)
    reports = []
    for spec in args.embedder:
        embedder = make_embedder(spec)
        index = build_index(args.corpus, embedder, args.chunk_words, args.overlap_words)
        for mode in args.modes:
            reports.append(evaluate(index, questions, args.k, mode, args.candidates))
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        print(f"{len(questions)} questions, corpus {args.corpus}, chunk {args.chunk_words} words")
        print(format_markdown(reports))
        for r in reports:
            if r.misses:
                print(f"\n{r.mode}/{r.embedder} missed in top {max(r.ks)}: {', '.join(r.misses)}")
    return 0


def _csv(cast):
    return lambda raw: [cast(part) for part in raw.split(",") if part]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hybrid-rag-api", description=__doc__)
    parser.add_argument(
        "--index", help="index file (default: $RAG_INDEX_PATH or data/index.sqlite)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index Markdown, text and PDF files or directories")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="retrieve chunks without calling the model")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a question with citations (needs ANTHROPIC_API_KEY)")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("docs", help="list indexed documents")
    p.set_defaults(func=cmd_docs)

    p = sub.add_parser("delete", help="remove a document and its chunks")
    p.add_argument("doc_id")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="offline retrieval evaluation (recall@k, MRR)")
    p.add_argument("--corpus", default="eval/corpus")
    p.add_argument("--dataset", default="eval/qa.jsonl")
    p.add_argument(
        "--embedder",
        type=_csv(str),
        default=["hashing"],
        help="comma-separated embedder specs, e.g. hashing,fastembed",
    )
    p.add_argument("--modes", type=_csv(str), default=["bm25", "dense", "hybrid"])
    p.add_argument("-k", type=_csv(int), default=[1, 3, 5])
    p.add_argument("--candidates", type=int, default=30)
    p.add_argument("--chunk-words", type=int, default=200)
    p.add_argument("--overlap-words", type=int, default=40)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
