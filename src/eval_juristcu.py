"""Domain-shift evaluation on JurisTCU (Brazilian legal jurisprudence retrieval).

JurisTCU (Ribeiro et al., 2025) is a native PT-BR retrieval benchmark drawn
from Brazilian Federal Court of Accounts (TCU) jurisprudence: 16,045
documents, 150 queries (across three styles: real keyword, synthetic
keyword, synthetic question), 2,250 LLM-scored + expert-validated relevance
judgments.

Provides a hard out-of-distribution probe for models trained on mMARCO
(web text), since legal Portuguese has substantially different vocabulary
and structure.

Reference: https://arxiv.org/abs/2503.08379
Dataset:   https://huggingface.co/datasets/LeandroRibeiro/JurisTCU
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

JURISTCU_REPO = "LeandroRibeiro/JurisTCU"
JURISTCU_QUERY_STYLES = ("real_keyword", "synthetic_keyword", "synthetic_question")


def evaluate(
    checkpoint: str | Path,
    *,
    query_style: str = "real_keyword",
    first_stage: str = "bm25",
    top_k: int = 1000,
    rerank_top_n: int = 100,
    batch_size: int = 64,
) -> dict[str, float]:
    """Rerank JurisTCU first-stage candidates and return standard IR metrics.

    Returns:
        Dict with nDCG@10, MRR@10, Recall@100, stratified by query style.
    """
    raise NotImplementedError("evaluate() to be implemented in Phase 4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on JurisTCU (Brazilian legal)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query-style", default="real_keyword", choices=JURISTCU_QUERY_STYLES)
    parser.add_argument("--first-stage", default="bm25", choices=["bm25", "serafim-ir"])
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--rerank-top-n", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_juristcu.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    metrics = evaluate(
        args.checkpoint,
        query_style=args.query_style,
        first_stage=args.first_stage,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        batch_size=args.batch_size,
    )
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
