"""Mine hard negatives for training using the Serafim-IR bi-encoder.

Pipeline (canonical ANCE-style hard negative mining, adapted for PT-BR):

1. Encode the mMARCO-PT collection (8.8M passages) with
   ``PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir`` in batches.
2. Index the embeddings with FAISS (HNSW for speed on CPU; flat for exactness
   on a smaller subset).
3. Encode each training query and retrieve the top-K candidates.
4. Remove known positives via the qrels file.
5. Sample ``num_negatives`` hard negatives uniformly from ranks
   ``rank_min`` .. ``rank_max``. The lower bound avoids false negatives at the
   very top; the upper bound keeps the negatives reasonably hard.
6. Write ``(query_id, positive_id, [neg_id_1, ..., neg_id_N])`` rows to parquet.

Designed to run on a single mid-tier GPU. Estimated cost on Runpod A40
($0.49/h): ~6 hours for the full mMARCO-PT (~$3); ~30 minutes for
``--small`` mode (used to validate the pipeline on Kaggle T4x2 free tier).
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import faiss

logger = logging.getLogger(__name__)

DEFAULT_BIENCODER = "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir"
DEFAULT_TOP_K = 200
DEFAULT_NUM_NEGATIVES = 7
DEFAULT_RANK_MIN = 10
DEFAULT_RANK_MAX = 100
DEFAULT_ENCODE_BATCH_SIZE = 256


@dataclass
class MiningConfig:
    biencoder: str = DEFAULT_BIENCODER
    top_k: int = DEFAULT_TOP_K
    num_negatives: int = DEFAULT_NUM_NEGATIVES
    rank_min: int = DEFAULT_RANK_MIN
    rank_max: int = DEFAULT_RANK_MAX
    encode_batch_size: int = DEFAULT_ENCODE_BATCH_SIZE
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0 <= self.rank_min < self.rank_max <= self.top_k:
            raise ValueError(
                f"Require 0 <= rank_min ({self.rank_min}) < rank_max "
                f"({self.rank_max}) <= top_k ({self.top_k})."
            )
        if self.num_negatives > (self.rank_max - self.rank_min):
            raise ValueError(
                f"num_negatives ({self.num_negatives}) must fit in "
                f"[rank_min, rank_max) of size {self.rank_max - self.rank_min}."
            )


def resolve_device(requested: str | None) -> str:
    """Pick a torch device that is actually usable by the current install.

    Some hosted environments (e.g. Kaggle's P100 instances paired with newer
    PyTorch builds compiled only for sm_70+) advertise CUDA as available but
    will then raise ``CUDA error: no kernel image is available for execution``
    on the first kernel launch. We do a tiny probe to detect that situation
    and fall back to CPU.
    """
    import torch

    if requested == "cpu":
        return "cpu"
    if requested and requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _ = (torch.zeros(1, device="cuda") + 1).cpu()
        return "cuda"
    except Exception as exc:  # pragma: no cover - environment-specific
        logger.warning("CUDA advertised but unusable (%s); falling back to CPU.", exc)
        return "cpu"


def _load_collection(collection_path: Path) -> tuple[np.ndarray, list[str]]:
    """Return (passage_ids, passage_texts) preserving file order.

    The order matters: row index in the FAISS index maps back to the passage_id
    via the returned arrays.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(collection_path)  # type: ignore[no-untyped-call]
    ids = table["id"].to_numpy()
    texts = table["text"].to_pylist()
    logger.info("Loaded %d passages from %s", len(ids), collection_path)
    return ids, texts


def _load_queries(queries_path: Path) -> tuple[np.ndarray, list[str]]:
    """Return (query_ids, query_texts) preserving file order."""
    import pyarrow.parquet as pq

    table = pq.read_table(queries_path)  # type: ignore[no-untyped-call]
    ids = table["id"].to_numpy()
    texts = table["text"].to_pylist()
    logger.info("Loaded %d queries from %s", len(ids), queries_path)
    return ids, texts


def _load_qrels(qrels_path: Path) -> dict[int, set[int]]:
    """Parse a MS MARCO-style qrels TSV file: ``query_id 0 passage_id 1``.

    Returns a dict mapping query_id -> set of positive passage_ids.
    """
    qrels: dict[int, set[int]] = {}
    with qrels_path.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, pid, _ = parts
            qrels.setdefault(int(qid), set()).add(int(pid))
    logger.info("Loaded qrels for %d queries from %s", len(qrels), qrels_path)
    return qrels


def encode_collection(
    biencoder: str,
    texts: list[str],
    *,
    batch_size: int = DEFAULT_ENCODE_BATCH_SIZE,
    device: str | None = None,
) -> np.ndarray:
    """Encode passages with the bi-encoder. Returns ``(N, dim)`` float32 array."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(biencoder, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_index(embeddings: np.ndarray, *, index_type: str = "hnsw") -> faiss.Index:
    """Build a FAISS index over normalized embeddings (inner product = cosine)."""
    import faiss

    dim = embeddings.shape[1]
    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 64, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
    elif index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError(f"Unknown index_type: {index_type}")
    index.add(embeddings)
    logger.info("Built %s index with %d vectors (dim=%d)", index_type, index.ntotal, dim)
    return index


def mine(
    config: MiningConfig,
    queries_path: Path,
    qrels_path: Path,
    collection_path: Path,
    output_path: Path,
    *,
    index_type: str = "hnsw",
    device: str | None = None,
) -> int:
    """Mine hard negatives and write triples to parquet.

    Returns the number of (query, positive, negatives) rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer

    rng = random.Random(config.seed)

    resolved_device = resolve_device(device)
    logger.info("Resolved device: %s (requested=%s)", resolved_device, device)

    pid_by_row, passage_texts = _load_collection(collection_path)
    qid_by_row, query_texts = _load_queries(queries_path)
    qrels = _load_qrels(qrels_path)

    logger.info("Encoding collection with %s", config.biencoder)
    passage_emb = encode_collection(
        config.biencoder,
        passage_texts,
        batch_size=config.encode_batch_size,
        device=resolved_device,
    )
    index = build_index(passage_emb, index_type=index_type)

    logger.info("Encoding queries")
    model = SentenceTransformer(config.biencoder, device=resolved_device)
    query_emb = model.encode(
        query_texts,
        batch_size=config.encode_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    logger.info("Searching top-%d for %d queries", config.top_k, len(query_texts))
    _, neighbor_rows = index.search(query_emb, config.top_k)

    # Positives must be present in our collection — otherwise downstream
    # build_triples cannot resolve the positive_id to text. In a full-mode
    # run the collection covers all passages and this is a no-op; in
    # --small mode where collection is a slice, this filter excludes
    # queries whose only positives sit outside the slice.
    collection_pids_set: set[int] = {int(p) for p in pid_by_row.tolist()}

    rows_out: list[dict[str, object]] = []
    skipped_no_qrel = 0
    skipped_insufficient_pool = 0
    skipped_positive_not_in_collection = 0

    for q_idx, qid in enumerate(qid_by_row.tolist()):
        positives = qrels.get(int(qid))
        if not positives:
            skipped_no_qrel += 1
            continue

        present_positives = [p for p in positives if p in collection_pids_set]
        if not present_positives:
            skipped_positive_not_in_collection += 1
            continue

        candidate_rows = neighbor_rows[q_idx]
        # Filter out positives from the candidate ranks.
        pool: list[int] = []
        for cand_row in candidate_rows.tolist():
            cand_pid = int(pid_by_row[cand_row])
            if cand_pid in positives:
                continue
            pool.append(cand_pid)
            if len(pool) >= config.rank_max:
                break
        # Restrict to the configured rank window inside the filtered pool.
        windowed = pool[config.rank_min : config.rank_max]
        if len(windowed) < config.num_negatives:
            skipped_insufficient_pool += 1
            continue
        negatives = rng.sample(windowed, config.num_negatives)

        # Emit one row per known positive — the same negatives are reused.
        for pos_pid in present_positives:
            rows_out.append(
                {
                    "query_id": int(qid),
                    "positive_id": int(pos_pid),
                    "negative_ids": negatives,
                }
            )

        if (q_idx + 1) % 10_000 == 0:
            logger.info(
                "Mined %d/%d queries "
                "(skipped_no_qrel=%d, skipped_pool=%d, skipped_pos_off_collection=%d)",
                q_idx + 1,
                len(qid_by_row),
                skipped_no_qrel,
                skipped_insufficient_pool,
                skipped_positive_not_in_collection,
            )

    if not rows_out:
        raise RuntimeError(
            f"No triples mined. Stats: skipped_no_qrel={skipped_no_qrel}, "
            f"skipped_pool={skipped_insufficient_pool}, "
            f"skipped_pos_off_collection={skipped_positive_not_in_collection}. "
            "In --small mode the slice of the collection rarely intersects the "
            "qrels positives — this is expected; full-mode mining is needed for "
            "end-to-end validation."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows_out)
    pq.write_table(table, output_path)  # type: ignore[no-untyped-call]
    logger.info(
        "Wrote %d triples to %s (skipped_no_qrel=%d, skipped_pool=%d)",
        len(rows_out),
        output_path,
        skipped_no_qrel,
        skipped_insufficient_pool,
    )
    return len(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives with Serafim-IR")
    parser.add_argument("--biencoder", default=DEFAULT_BIENCODER)
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path("data/raw/mmarco/collection_portuguese.parquet"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/raw/mmarco/queries_train_portuguese.parquet"),
    )
    parser.add_argument("--qrels", type=Path, default=Path("data/raw/mmarco/qrels.dev.small.tsv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/hard_negatives.parquet")
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--num-negatives", type=int, default=DEFAULT_NUM_NEGATIVES)
    parser.add_argument("--rank-min", type=int, default=DEFAULT_RANK_MIN)
    parser.add_argument("--rank-max", type=int, default=DEFAULT_RANK_MAX)
    parser.add_argument("--encode-batch-size", type=int, default=DEFAULT_ENCODE_BATCH_SIZE)
    parser.add_argument("--index-type", default="hnsw", choices=["hnsw", "flat"])
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device. 'auto' probes CUDA and falls back to CPU on incompatibility.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = MiningConfig(
        biencoder=args.biencoder,
        top_k=args.top_k,
        num_negatives=args.num_negatives,
        rank_min=args.rank_min,
        rank_max=args.rank_max,
        encode_batch_size=args.encode_batch_size,
        seed=args.seed,
    )
    mine(
        config,
        args.queries,
        args.qrels,
        args.collection,
        args.output,
        index_type=args.index_type,
        device=args.device,
    )


if __name__ == "__main__":
    main()
