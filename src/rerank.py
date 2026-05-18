"""Rerank a list of (query, passages) candidates with the cross-encoder."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .model import DEFAULT_HF_HUB_ID, load

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    passages: list[str],
    *,
    checkpoint: str | Path = DEFAULT_HF_HUB_ID,
    top_n: int | None = None,
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """Rerank passages for a query. Returns list of ``(passage, score)`` sorted descending."""
    model = load(checkpoint)
    scores = model.predict(
        [(query, p) for p in passages],
        batch_size=batch_size,
        show_progress_bar=False,
    )
    ranked = sorted(zip(passages, scores.tolist(), strict=True), key=lambda x: -x[1])
    return ranked[:top_n] if top_n else ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank passages for a query")
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--passages-file",
        type=Path,
        help="Text file with one passage per line. If omitted, reads from stdin.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_HF_HUB_ID)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--json", action="store_true", help="Output JSON instead of plain text")
    args = parser.parse_args()

    if args.passages_file:
        passages = [
            line.strip() for line in args.passages_file.read_text().splitlines() if line.strip()
        ]
    else:
        passages = [line.strip() for line in sys.stdin if line.strip()]

    if not passages:
        sys.exit("No passages provided.")

    ranked = rerank(
        args.query,
        passages,
        checkpoint=args.checkpoint,
        top_n=args.top_n,
        batch_size=args.batch_size,
    )

    if args.json:
        print(
            json.dumps(
                [{"passage": p, "score": s} for p, s in ranked], ensure_ascii=False, indent=2
            )
        )
    else:
        for passage, score in ranked:
            print(f"{score:.4f}\t{passage}")


if __name__ == "__main__":
    main()
