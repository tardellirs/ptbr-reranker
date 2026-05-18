"""Evaluation on mMARCO-PT dev split."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate(
    checkpoint: str | Path,
    *,
    candidates_source: str = "bm25",
    top_k: int = 1000,
    rerank_top_n: int = 100,
    batch_size: int = 64,
) -> dict[str, float]:
    """Run mMARCO-PT dev evaluation. Returns dict with MRR@10, nDCG@10, Recall@1000.

    Args:
        checkpoint: Local path or HF Hub id of the cross-encoder.
        candidates_source: ``"bm25"`` or ``"serafim-ir"``.
        top_k: Number of candidates retrieved upstream.
        rerank_top_n: Number of candidates actually rescored by the cross-encoder.
        batch_size: Inference batch size.
    """
    raise NotImplementedError("evaluate() to be implemented in Phase 4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on mMARCO-PT dev")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidates-source", choices=["bm25", "serafim-ir"], default="bm25")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--rerank-top-n", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_mmarco.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    metrics = evaluate(
        args.checkpoint,
        candidates_source=args.candidates_source,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        batch_size=args.batch_size,
    )
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
