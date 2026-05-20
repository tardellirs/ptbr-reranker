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


MIN_CUDA_CAPABILITY = (7, 0)


def resolve_device(requested: str | None) -> str:
    """Pick a torch device that is actually usable by the current install.

    Some hosted environments (e.g. Kaggle's P100 instances paired with newer
    PyTorch builds compiled only for sm_70+) advertise CUDA as available but
    will then raise ``CUDA error: no kernel image is available for execution``
    on the first kernel launch. A basic ``zeros + 1`` op passes on sm_60, so
    additionally check compute capability against the supported floor.
    """
    import torch

    if requested == "cpu":
        return "cpu"
    if requested and requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        capability = torch.cuda.get_device_capability(0)
    except Exception as exc:  # pragma: no cover - environment-specific
        logger.warning("Could not query CUDA capability (%s); falling back to CPU.", exc)
        return "cpu"
    if capability < MIN_CUDA_CAPABILITY:
        logger.warning(
            "Detected CUDA device with capability sm_%d%d which is below the "
            "PyTorch-supported floor sm_%d%d; falling back to CPU.",
            capability[0],
            capability[1],
            MIN_CUDA_CAPABILITY[0],
            MIN_CUDA_CAPABILITY[1],
        )
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


def _load_positives_from_triples(triples_tsv: Path) -> dict[int, set[int]]:
    """Extract ``query_id -> {positive_id}`` from a 3-col MS MARCO triples TSV.

    mMARCO ships only ``triples.train.ids.small.tsv`` (``qid pos_pid neg_pid``)
    for the training split, not a TREC-style qrels.train file. This helper
    rebuilds the positives set from those triples so the mining pipeline
    can run on the training queries without an extra format-conversion step.
    """
    positives: dict[int, set[int]] = {}
    with triples_tsv.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            qid, pos_pid, _ = parts
            positives.setdefault(int(qid), set()).add(int(pos_pid))
    logger.info(
        "Loaded positives for %d queries from %s (extracted from triples)",
        len(positives),
        triples_tsv,
    )
    return positives


def encode_collection(
    biencoder: str,
    texts: list[str],
    *,
    batch_size: int = DEFAULT_ENCODE_BATCH_SIZE,
    device: str | None = None,
) -> np.ndarray:
    """Encode passages with the bi-encoder. Returns ``(N, dim)`` float32 array.

    For small inputs this just calls ``model.encode`` once. Prefer
    ``encode_collection_streaming`` for the full mMARCO-PT collection
    (8.8M passages would otherwise need ~27 GB of RAM at the final stack).
    """
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


def encode_collection_streaming(
    biencoder: str,
    texts: list[str],
    output_path: Path,
    *,
    batch_size: int = DEFAULT_ENCODE_BATCH_SIZE,
    chunk_size: int = 500_000,
    device: str | None = None,
) -> np.memmap:
    """Encode passages chunk-by-chunk to a memmap'd .npy on disk.

    ``model.encode()`` accumulates every batch into a single in-memory list
    before stacking, so calling it on 8.8 M passages allocates ~27 GB of
    Python objects + one final 27 GB ndarray simultaneously. On a 32 GB
    container that overshoots the cgroup limit and the OOM killer fires
    (seen on Salad node, 2026-05-20). Encoding in ``chunk_size`` slices
    bounds peak RAM at one chunk (~1.5 GB) and writes each chunk straight
    into a numpy memmap so the FAISS index step can mmap-read it back.
    """
    import struct

    from sentence_transformers import SentenceTransformer

    n = len(texts)
    model = SentenceTransformer(biencoder, device=device)
    # Probe dimensionality with a single tiny encode.
    probe = np.asarray(
        model.encode(
            texts[: min(2, n)],
            batch_size=2,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    dim = probe.shape[1]
    logger.info(
        "Streaming encode: n=%d chunks of %d, dim=%d -> %s (%.1f GB on disk)",
        n,
        chunk_size,
        dim,
        output_path,
        n * dim * 4 / 1024**3,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(output_path, dtype=np.float32, mode="w+", shape=(n, dim))
    arr[: probe.shape[0]] = probe
    written = probe.shape[0]
    for start in range(written, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_emb = np.asarray(
            model.encode(
                texts[start:end],
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        arr[start:end] = chunk_emb
        arr.flush()
        del chunk_emb
        logger.info("Streaming encode: %d / %d passages done", end, n)
    # Persist a tiny shape sidecar so the consumer can read without
    # re-running the dim probe.
    shape_file = output_path.with_suffix(output_path.suffix + ".shape")
    shape_file.write_bytes(struct.pack("<qq", n, dim))
    return arr


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
    qrels_path: Path | None,
    collection_path: Path,
    output_path: Path,
    *,
    index_type: str = "hnsw",
    device: str | None = None,
    positives_from_triples: Path | None = None,
) -> int:
    """Mine hard negatives and write triples to parquet.

    Exactly one of ``qrels_path`` (TREC qrels file) or
    ``positives_from_triples`` (a 3-col MS MARCO triples TSV — used for
    the training split since mMARCO ships no qrels.train) must be set.

    Returns the number of (query, positive, negatives) rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer

    if (qrels_path is None) == (positives_from_triples is None):
        raise ValueError(
            "Pass exactly one of --qrels (TREC qrels file) or "
            "--positives-from-triples (MS MARCO triples TSV)."
        )

    rng = random.Random(config.seed)

    resolved_device = resolve_device(device)
    logger.info("Resolved device: %s (requested=%s)", resolved_device, device)

    pid_by_row, passage_texts = _load_collection(collection_path)
    qid_by_row, query_texts = _load_queries(queries_path)
    qrels = (
        _load_qrels(qrels_path)
        if qrels_path is not None
        else _load_positives_from_triples(positives_from_triples)  # type: ignore[arg-type]
    )

    # Disk-backed caches for each expensive phase. On retry after a pod
    # eviction these files let us skip the 3+ hour collection encode and
    # 10+ min FAISS build entirely. Passage embeddings use a raw memmap
    # binary (.bin) instead of .npy so they never have to be fully loaded
    # into RAM — the FAISS step also reads them via memmap.
    cache_passage = output_path.with_suffix(".cache.passage_emb.bin")
    cache_passage_shape = output_path.with_suffix(".cache.passage_emb.bin.shape")
    cache_query = output_path.with_suffix(".cache.query_emb.npy")
    cache_index = output_path.with_suffix(".cache.faiss.bin")

    if cache_passage.exists() and cache_passage_shape.exists():
        logger.info("Loading cached passage embeddings (memmap) from %s", cache_passage)
        import struct as _struct

        n_p, dim_p = _struct.unpack("<qq", cache_passage_shape.read_bytes())
        passage_emb = np.memmap(cache_passage, dtype=np.float32, mode="r", shape=(n_p, dim_p))
    else:
        logger.info("Encoding collection (streaming) with %s", config.biencoder)
        passage_emb = encode_collection_streaming(
            config.biencoder,
            passage_texts,
            cache_passage,
            batch_size=config.encode_batch_size,
            chunk_size=500_000,
            device=resolved_device,
        )
        logger.info(
            "Cached passage embeddings to %s (%.1f GB)",
            cache_passage,
            cache_passage.stat().st_size / 1024**3,
        )

    import faiss as _faiss

    if cache_index.exists():
        logger.info("Loading cached FAISS index from %s", cache_index)
        index = _faiss.read_index(str(cache_index))
    else:
        index = build_index(passage_emb, index_type=index_type)
        tmp = cache_index.with_suffix(cache_index.suffix + ".tmp")
        _faiss.write_index(index, str(tmp))
        tmp.replace(cache_index)
        logger.info("Cached FAISS index to %s", cache_index)

    if cache_query.exists():
        logger.info("Loading cached query embeddings from %s", cache_query)
        query_emb = np.load(cache_query)
    else:
        logger.info("Encoding queries")
        model = SentenceTransformer(config.biencoder, device=resolved_device)
        query_emb = model.encode(
            query_texts,
            batch_size=config.encode_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        tmp = cache_query.with_suffix(cache_query.suffix + ".tmp")
        np.save(tmp, query_emb)
        tmp.replace(cache_query)
        logger.info("Cached query embeddings to %s", cache_query)

    logger.info("Searching top-%d for %d queries", config.top_k, len(query_texts))
    _, neighbor_rows = index.search(query_emb, config.top_k)

    # Positives must be present in our collection — otherwise downstream
    # build_triples cannot resolve the positive_id to text. In a full-mode
    # run the collection covers all passages and this is a no-op; in
    # --small mode where collection is a slice, this filter excludes
    # queries whose only positives sit outside the slice.
    collection_pids_set: set[int] = {int(p) for p in pid_by_row.tolist()}

    # Resume support: load any partial parquet from a prior interrupted run.
    rows_out: list[dict[str, object]] = []
    done_qids: set[int] = set()
    if output_path.exists():
        prior = pq.read_table(output_path)  # type: ignore[no-untyped-call]
        for batch in prior.to_batches():
            qids_b = batch["query_id"].to_pylist()
            pos_b = batch["positive_id"].to_pylist()
            neg_b = batch["negative_ids"].to_pylist()
            for q, p, n in zip(qids_b, pos_b, neg_b, strict=True):
                rows_out.append(
                    {"query_id": int(q), "positive_id": int(p), "negative_ids": list(n)}
                )
                done_qids.add(int(q))
        logger.info(
            "Resumed: %d rows / %d unique qids already on disk", len(rows_out), len(done_qids)
        )

    skipped_no_qrel = 0
    skipped_insufficient_pool = 0
    skipped_positive_not_in_collection = 0

    def _flush_rows(rows: list[dict[str, object]]) -> None:
        """Atomic write of the full current rows_out to output_path."""
        if not rows:
            return
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        pq.write_table(pa.Table.from_pylist(rows), tmp)  # type: ignore[no-untyped-call]
        tmp.replace(output_path)

    for q_idx, qid in enumerate(qid_by_row.tolist()):
        qid_int = int(qid)
        if qid_int in done_qids:
            continue
        positives = qrels.get(qid_int)
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
                    "query_id": qid_int,
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
        # Periodic flush of rows_out to disk so a crash mid-sampling loses
        # at most the last ~50k queries' worth of work.
        if (q_idx + 1) % 50_000 == 0:
            _flush_rows(rows_out)
            logger.info("Checkpointed rows_out to %s (%d rows)", output_path, len(rows_out))

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
    _flush_rows(rows_out)
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
    parser.add_argument(
        "--qrels",
        type=Path,
        default=None,
        help="TREC qrels TSV (qid 0 pid rel). Mutually exclusive with --positives-from-triples.",
    )
    parser.add_argument(
        "--positives-from-triples",
        type=Path,
        default=None,
        help="3-col MS MARCO triples TSV (qid pos_pid neg_pid) — used for "
        "the training split since mMARCO ships no qrels.train file.",
    )
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
        positives_from_triples=args.positives_from_triples,
    )


if __name__ == "__main__":
    main()
