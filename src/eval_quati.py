"""Cross-domain evaluation on Quati (native PT-BR web retrieval benchmark).

Quati (Bonifacio et al., 2024) is the first PT-BR-native passage retrieval
benchmark not derived from MS MARCO: queries and passages come from
ClueWeb22-pt, with 50 topics densely judged (GPT-4 + TREC-style 4-point scale)
over either 1M or 10M passages.

Reference: https://arxiv.org/abs/2404.06976
Dataset:   https://huggingface.co/datasets/unicamp-dl/quati
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

QUATI_REPO = "unicamp-dl/quati"
QUATI_DEFAULT_CONFIG = "1M"  # smaller pool; use "10M" for the harder setting


def evaluate(
    checkpoint: str | Path,
    *,
    config: str = QUATI_DEFAULT_CONFIG,
    first_stage: str = "bm25",
    top_k: int = 1000,
    rerank_top_n: int = 100,
    batch_size: int = 64,
) -> dict[str, float]:
    """Rerank Quati first-stage candidates and return standard IR metrics.

    Args:
        checkpoint: Local path or HF Hub id of the cross-encoder.
        config: ``"1M"`` (~50 queries, 1M passages) or ``"10M"`` (10M passages).
        first_stage: Source of the initial top-K candidates (``"bm25"`` or ``"serafim-ir"``).
        top_k: Number of candidates retrieved upstream.
        rerank_top_n: Number of candidates the cross-encoder actually rescores.
        batch_size: Cross-encoder inference batch size.

    Returns:
        Dict with at least ``nDCG@10``, ``Recall@100``, ``MRR@10``. Per-query
        scores are also serialized to ``outputs/eval_quati_per_query.parquet``
        for bootstrap confidence intervals via ``src.stats``.
    """
    raise NotImplementedError("evaluate() to be implemented in Phase 4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on Quati (PT-BR retrieval)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=QUATI_DEFAULT_CONFIG, choices=["1M", "10M"])
    parser.add_argument("--first-stage", default="bm25", choices=["bm25", "serafim-ir"])
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--rerank-top-n", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_quati.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    metrics = evaluate(
        args.checkpoint,
        config=args.config,
        first_stage=args.first_stage,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        batch_size=args.batch_size,
    )
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
