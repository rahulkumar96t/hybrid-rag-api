"""Hybrid BM25 + dense retrieval-augmented question answering with cited answers."""

__version__ = "0.1.0"


def main() -> None:
    from .cli import main as cli_main

    raise SystemExit(cli_main())
