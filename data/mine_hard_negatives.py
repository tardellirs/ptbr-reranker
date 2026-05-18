"""Mine hard negatives for training using the Serafim-IR bi-encoder.

Pipeline:
1. Encode the mMARCO-PT collection with Serafim-100m-IR.
2. Index with FAISS (HNSW for speed; flat for exactness in a smaller subset).
3. For each training query, retrieve top-K candidates.
4. Remove positives (from qrels).
5. Sample N hard negatives from ranks 10-100 (avoids false negatives at top ranks).
6. Save (query_id, positive_id, [negative_id_1, ..., negative_id_N]) triples.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_BIENCODER = "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir"
DEFAULT_TOP_K = 200
DEFAULT_NUM_NEGATIVES = 7
DEFAULT_RANK_MIN = 10
DEFAULT_RANK_MAX = 100


def encode_collection(
    biencoder: str,
    collection_path: Path,
    output_path: Path,
    *,
    batch_size: int = 256,
) -> None:
    """Encode passages with the bi-encoder and save embeddings."""
    raise NotImplementedError("encode_collection() to be implemented in Phase 2")


def build_index(embeddings_path: Path, index_path: Path, *, index_type: str = "hnsw") -> None:
    """Build a FAISS index from embeddings."""
    raise NotImplementedError("build_index() to be implemented in Phase 2")


def mine(
    queries_path: Path,
    qrels_path: Path,
    index_path: Path,
    output_path: Path,
    *,
    top_k: int = DEFAULT_TOP_K,
    num_negatives: int = DEFAULT_NUM_NEGATIVES,
    rank_min: int = DEFAULT_RANK_MIN,
    rank_max: int = DEFAULT_RANK_MAX,
    seed: int = 42,
) -> None:
    """Mine hard negatives and write the resulting triples to parquet."""
    raise NotImplementedError("mine() to be implemented in Phase 2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives with Serafim-IR")
    parser.add_argument("--biencoder", default=DEFAULT_BIENCODER)
    parser.add_argument("--collection", type=Path, default=Path("data/raw/mmarco/collection.tsv"))
    parser.add_argument("--queries", type=Path, default=Path("data/raw/mmarco/queries.train.tsv"))
    parser.add_argument("--qrels", type=Path, default=Path("data/raw/mmarco/qrels.train.tsv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/hard_negatives.parquet")
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--num-negatives", type=int, default=DEFAULT_NUM_NEGATIVES)
    parser.add_argument("--rank-min", type=int, default=DEFAULT_RANK_MIN)
    parser.add_argument("--rank-max", type=int, default=DEFAULT_RANK_MAX)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Implementation: encode -> index -> mine.
    raise NotImplementedError("main() to be implemented in Phase 2")


if __name__ == "__main__":
    main()
