"""Build the final (query, positive, negative) training triples.

Combines the qrels-derived positives with mined hard negatives, optionally
mixing in BM25 negatives for diversity. The output is consumed by ``src/train.py``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build(
    queries_path: Path,
    collection_path: Path,
    qrels_path: Path,
    hard_negatives_path: Path,
    output_path: Path,
    *,
    bm25_negatives_path: Path | None = None,
    mix_ratio_bm25: float = 0.0,
    seed: int = 42,
) -> None:
    """Build training triples and save to parquet."""
    raise NotImplementedError("build() to be implemented in Phase 2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training triples")
    parser.add_argument("--queries", type=Path, default=Path("data/raw/mmarco/queries.train.tsv"))
    parser.add_argument(
        "--collection", type=Path, default=Path("data/raw/mmarco/collection.tsv")
    )
    parser.add_argument("--qrels", type=Path, default=Path("data/raw/mmarco/qrels.train.tsv"))
    parser.add_argument(
        "--hard-negatives", type=Path, default=Path("data/processed/hard_negatives.parquet")
    )
    parser.add_argument("--bm25-negatives", type=Path, default=None)
    parser.add_argument("--mix-ratio-bm25", type=float, default=0.0)
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/triples.parquet")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(
        args.queries,
        args.collection,
        args.qrels,
        args.hard_negatives,
        args.output,
        bm25_negatives_path=args.bm25_negatives,
        mix_ratio_bm25=args.mix_ratio_bm25,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
