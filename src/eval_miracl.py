"""Evaluation on MIRACL Portuguese subset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate(
    checkpoint: str | Path,
    *,
    split: str = "dev",
    top_k: int = 100,
    batch_size: int = 64,
) -> dict[str, float]:
    """Run MIRACL-PT evaluation. Returns dict with nDCG@10, Recall@100, MRR@10."""
    raise NotImplementedError("evaluate() to be implemented in Phase 4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on MIRACL Portuguese")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_miracl.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    metrics = evaluate(
        args.checkpoint,
        split=args.split,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
