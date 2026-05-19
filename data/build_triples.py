"""Build the final training triples consumed by ``src/train.py``.

Two sources can be combined:

1. The **official** MS MARCO triples (``triples.train.ids.small.tsv``):
   ``(query_id, positive_id, negative_id)`` per line. Negatives here are
   sampled by the MS MARCO authors using BM25; they are the standard
   baseline distribution and require no GPU to consume. This is the
   *baseline* recipe in ``configs/train_baseline.yaml``.

2. **Mined** hard negatives produced by ``data/mine_hard_negatives.py``:
   ``(query_id, positive_id, [neg_id_1, ..., neg_id_N])`` per row, where the
   negatives come from the Serafim-IR bi-encoder's top-K minus the qrels
   positives. Used by ``configs/train_hardneg.yaml``.

The output joins to the mMARCO-PT collection and queries so ``src/train.py``
can stream ``(query_text, positive_text, negative_text)`` triples directly.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_id_to_text(parquet_path: Path) -> dict[int, str]:
    """Read a parquet with columns ``id, text`` into a dict."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["id", "text"])  # type: ignore[no-untyped-call]
    ids = table["id"].to_pylist()
    texts = table["text"].to_pylist()
    return dict(zip(ids, texts, strict=True))


def _iter_official_triples(
    triples_tsv: Path,
) -> Iterator[tuple[int, int, int]]:
    """Yield (query_id, positive_id, negative_id) from the MS MARCO triples TSV."""
    with triples_tsv.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            yield int(parts[0]), int(parts[1]), int(parts[2])


def _iter_mined_triples(
    mined_parquet: Path,
) -> Iterator[tuple[int, int, int]]:
    """Yield (query_id, positive_id, negative_id) by exploding mined negatives."""
    import pyarrow.parquet as pq

    table = pq.read_table(mined_parquet)  # type: ignore[no-untyped-call]
    for batch in table.to_batches():
        qids = batch["query_id"].to_pylist()
        pos_ids = batch["positive_id"].to_pylist()
        neg_lists = batch["negative_ids"].to_pylist()
        for qid, pos_id, negs in zip(qids, pos_ids, neg_lists, strict=True):
            for neg_id in negs:
                yield int(qid), int(pos_id), int(neg_id)


def build(
    queries_path: Path,
    collection_path: Path,
    output_path: Path,
    *,
    official_triples_tsv: Path | None = None,
    mined_triples_parquet: Path | None = None,
    mix_ratio_official: float = 0.0,
    max_triples: int | None = None,
    seed: int = 42,
) -> int:
    """Materialize ``(query_text, positive_text, negative_text)`` triples.

    At least one of ``official_triples_tsv`` or ``mined_triples_parquet`` must
    be provided. When both are set, ``mix_ratio_official`` controls the
    sampling proportion of official-vs-mined rows (0.0 = mined only,
    1.0 = official only, 0.3 = 30% official + 70% mined).

    Returns the number of triples written.
    """
    if not official_triples_tsv and not mined_triples_parquet:
        raise ValueError("Provide official_triples_tsv and/or mined_triples_parquet")

    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    logger.info("Loading queries from %s", queries_path)
    query_text = _load_id_to_text(queries_path)
    logger.info("Loading collection from %s", collection_path)
    passage_text = _load_id_to_text(collection_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    written = 0
    dropped_unknown_ids = 0
    schema = pa.schema(
        [
            ("query_id", pa.int64()),
            ("query_text", pa.string()),
            ("positive_text", pa.string()),
            ("negative_text", pa.string()),
        ]
    )
    buffer: list[dict[str, object]] = []
    BUFFER_SIZE = 10_000

    def _flush() -> None:
        nonlocal writer
        if not buffer:
            return
        batch = pa.RecordBatch.from_pylist(buffer, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema)  # type: ignore[no-untyped-call]
        writer.write_batch(batch)  # type: ignore[no-untyped-call]
        buffer.clear()

    def _emit(qid: int, pos_id: int, neg_id: int) -> bool:
        nonlocal written, dropped_unknown_ids
        if qid not in query_text or pos_id not in passage_text or neg_id not in passage_text:
            dropped_unknown_ids += 1
            return False
        buffer.append(
            {
                "query_id": qid,
                "query_text": query_text[qid],
                "positive_text": passage_text[pos_id],
                "negative_text": passage_text[neg_id],
            }
        )
        written += 1
        if len(buffer) >= BUFFER_SIZE:
            _flush()
        return True

    sources: list[tuple[str, float]] = []
    iters: dict[str, object] = {}
    if official_triples_tsv:
        iters["official"] = iter(_iter_official_triples(official_triples_tsv))
        sources.append(("official", mix_ratio_official if mined_triples_parquet else 1.0))
    if mined_triples_parquet:
        iters["mined"] = iter(_iter_mined_triples(mined_triples_parquet))
        weight_mined = (1.0 - mix_ratio_official) if official_triples_tsv else 1.0
        sources.append(("mined", weight_mined))

    if not (0.0 <= mix_ratio_official <= 1.0):
        raise ValueError(f"mix_ratio_official must be in [0, 1], got {mix_ratio_official}")

    names = [name for name, _ in sources]
    weights = [weight for _, weight in sources]
    exhausted: set[str] = set()

    try:
        while max_triples is None or written < max_triples:
            if len(exhausted) == len(names):
                break
            available = [n for n in names if n not in exhausted]
            available_weights = [weights[names.index(n)] for n in available]
            choice = rng.choices(available, weights=available_weights, k=1)[0]
            try:
                qid, pos_id, neg_id = next(iters[choice])  # type: ignore[call-overload]
            except StopIteration:
                exhausted.add(choice)
                continue
            _emit(qid, pos_id, neg_id)
            if written % 100_000 == 0 and written > 0:
                logger.info("Wrote %d triples so far", written)
    finally:
        _flush()
        if writer is not None:
            writer.close()  # type: ignore[no-untyped-call]

    logger.info(
        "Wrote %d triples to %s (dropped %d rows whose ids were not in the "
        "loaded queries/collection)",
        written,
        output_path,
        dropped_unknown_ids,
    )
    if written == 0:
        raise RuntimeError(
            f"build_triples produced 0 rows (dropped {dropped_unknown_ids} due to "
            "unknown ids). Check that --queries / --collection parquets cover the "
            "ids referenced by the triples source. For --small mode + official "
            "MS MARCO triples this is expected because the random ID sample rarely "
            "intersects; mine triples on the small slice instead."
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training triples")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/raw/mmarco/queries_train_portuguese.parquet"),
    )
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path("data/raw/mmarco/collection_portuguese.parquet"),
    )
    parser.add_argument(
        "--official-triples",
        type=Path,
        default=None,
        help="MS MARCO official triples TSV (data/raw/mmarco/triples.train.ids.small.tsv).",
    )
    parser.add_argument(
        "--mined-triples",
        type=Path,
        default=None,
        help="Parquet output of mine_hard_negatives.py.",
    )
    parser.add_argument(
        "--mix-ratio-official",
        type=float,
        default=0.0,
        help="When both sources provided: 0.0 = mined only, 1.0 = official only.",
    )
    parser.add_argument(
        "--max-triples",
        type=int,
        default=None,
        help="Cap total triples written (useful for --small dev runs).",
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/triples.parquet"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(
        args.queries,
        args.collection,
        args.output,
        official_triples_tsv=args.official_triples,
        mined_triples_parquet=args.mined_triples,
        mix_ratio_official=args.mix_ratio_official,
        max_triples=args.max_triples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
