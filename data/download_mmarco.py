"""Download mMARCO-PT directly via Hugging Face Hub parquet snapshots.

The ``unicamp-dl/mmarco`` dataset ships with a legacy loader script
(``mmarco.py``), which was removed from upstream ``datasets`` at v4.0 for
security reasons. We therefore bypass ``load_dataset`` and download the
auto-converted parquet shards at ``refs/convert/parquet`` directly:

- ``collection-portuguese/mmarco-collection-{00000..00006}-of-00007.parquet``
- ``queries-portuguese/train/0000.parquet``
- ``queries-portuguese/dev/0000.parquet``

Each downloaded split is recorded in ``data/raw/mmarco/manifest.json`` along
with the resolved revision SHA. The manifest is the artefact consumed by the
ACL/NeurIPS-style reproducibility checklist (``docs/reproducibility.md``).

Cross-domain evaluation: MIRACL has no Portuguese subset. We use
**Quati** (``unicamp-dl/quati``, native PT-BR web from ClueWeb22-pt) as the
primary out-of-domain benchmark and **JurisTCU** (``LeandroRibeiro/JurisTCU``,
Brazilian legal jurisprudence) as a hard domain-shift probe. See
``src/eval_quati.py`` and ``src/eval_juristcu.py``.

For development, use ``--small`` to materialize the first ~10k passages from
the first collection shard plus a sample of train/dev queries.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MMARCO_REPO = "unicamp-dl/mmarco"
MMARCO_PARQUET_REVISION = "refs/convert/parquet"
COLLECTION_SHARDS = 7
COLLECTION_PATH_TEMPLATE = (
    "collection-portuguese/mmarco-collection-{shard:05d}-of-{total:05d}.parquet"
)
QUERIES_TRAIN_PATH = "queries-portuguese/train/0000.parquet"
QUERIES_DEV_PATH = "queries-portuguese/dev/0000.parquet"

# qrels and triples live on main (TSV, language-agnostic). mMARCO inherits
# qrels from MS MARCO since passage/query IDs are shared across translations.
QRELS_DEV_PATH = "data/qrels.dev.tsv"
QRELS_DEV_SMALL_PATH = "data/qrels.dev.small.tsv"
TRIPLES_TRAIN_PATH = "data/triples.train.ids.small.tsv"
BM25_RUN_PT_PATH = "data/google/runs/run.bm25_portuguese-msmarco.txt"

DEFAULT_RAW_DIR = Path("data/raw")
MANIFEST_NAME = "manifest.json"

EXPECTED_COUNTS_FULL = {
    "collection": 8_841_823,
    "queries_train": 808_731,
    "queries_dev": 101_093,
}
EXPECTED_COUNTS_SMALL = {
    "collection": 10_000,
    "queries_train": 1_000,
    "queries_dev": 100,
}
# qrels.dev.small has ~7437 (query_id, 0, passage_id, 1) lines. Used for eval.
EXPECTED_QRELS_DEV_SMALL_LINES = 7437
# triples.train.ids.small has ~39M (query_id, pos_id, neg_id) lines. Used for training.
TRIPLES_TRAIN_SMALL_HEAD = 100_000  # small mode caps to this many triples


@dataclass
class DatasetSnapshot:
    """Manifest entry recording exactly which slice of a dataset was downloaded."""

    repo: str
    config: str
    split: str
    revision: str | None
    num_rows: int
    output_path: str


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _resolve_revision(repo_id: str, revision: str) -> str | None:
    """Resolve a ref like ``refs/convert/parquet`` to its commit SHA."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
        return info.sha
    except Exception as exc:  # pragma: no cover - network failure path
        logger.warning("Could not resolve revision for %s@%s: %s", repo_id, revision, exc)
        return None


def _download_file(remote_path: str, revision: str) -> Path:
    """Fetch any file from HF Hub into the local cache."""
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=MMARCO_REPO,
        filename=remote_path,
        repo_type="dataset",
        revision=revision,
    )
    return Path(cached)


# Backwards-compatible alias.
_download_parquet = _download_file


def _materialize(
    remote_paths: list[str],
    output_path: Path,
    *,
    revision: str,
    row_limit: int | None,
) -> int:
    """Read one or more remote parquet shards, optionally truncate, and rewrite locally."""
    import pyarrow.parquet as pq

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    writer: pq.ParquetWriter | None = None
    try:
        for remote_path in remote_paths:
            local = _download_parquet(remote_path, revision)
            parquet_file = pq.ParquetFile(local)  # type: ignore[no-untyped-call]
            for batch in parquet_file.iter_batches(batch_size=2048):  # type: ignore[no-untyped-call]
                if row_limit is not None and rows_written >= row_limit:
                    break
                if row_limit is not None and rows_written + batch.num_rows > row_limit:
                    batch = batch.slice(0, row_limit - rows_written)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, batch.schema)  # type: ignore[no-untyped-call]
                writer.write_batch(batch)  # type: ignore[no-untyped-call]
                rows_written += batch.num_rows
            if row_limit is not None and rows_written >= row_limit:
                break
    finally:
        if writer is not None:
            writer.close()  # type: ignore[no-untyped-call]
    return rows_written


def _materialize_tsv_head(
    remote_path: str,
    output_path: Path,
    *,
    revision: str = "main",
    row_limit: int | None,
) -> int:
    """Stream a TSV file from HF Hub into local storage, optionally truncating."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cached = _download_file(remote_path, revision)
    rows_written = 0
    with cached.open("rb") as src, output_path.open("wb") as dst:
        for line in src:
            if row_limit is not None and rows_written >= row_limit:
                break
            dst.write(line)
            rows_written += 1
    return rows_written


def download_qrels_and_triples(
    target_dir: Path,
    *,
    small: bool = False,
) -> list[DatasetSnapshot]:
    """Download qrels (dev split) and the official MS MARCO training triples.

    These files live on the mMARCO main branch as TSV (not in the parquet
    revision). They are language-agnostic: qrels and triples reference
    (query_id, passage_id) integers shared across all mMARCO translations.

    Returns the list of snapshots written.
    """
    mmarco_dir = target_dir / "mmarco"
    mmarco_dir.mkdir(parents=True, exist_ok=True)
    snapshots: list[DatasetSnapshot] = []

    # qrels.dev.small.tsv — small, always download in full.
    qrels_path = mmarco_dir / "qrels.dev.small.tsv"
    logger.info("Materializing %s/%s", MMARCO_REPO, QRELS_DEV_SMALL_PATH)
    qrels_rows = _materialize_tsv_head(QRELS_DEV_SMALL_PATH, qrels_path, row_limit=None)
    logger.info("  → %s (%d lines)", qrels_path, qrels_rows)
    snapshots.append(
        DatasetSnapshot(
            repo=MMARCO_REPO,
            config="qrels",
            split="dev.small",
            revision="main",
            num_rows=qrels_rows,
            output_path=str(qrels_path.relative_to(target_dir.parent)),
        )
    )

    # triples.train.ids.small.tsv — large (~5GB); cap in small mode.
    triples_path = mmarco_dir / "triples.train.ids.small.tsv"
    limit = TRIPLES_TRAIN_SMALL_HEAD if small else None
    logger.info("Materializing %s/%s (limit=%s)", MMARCO_REPO, TRIPLES_TRAIN_PATH, limit)
    triples_rows = _materialize_tsv_head(TRIPLES_TRAIN_PATH, triples_path, row_limit=limit)
    logger.info("  → %s (%d lines)", triples_path, triples_rows)
    snapshots.append(
        DatasetSnapshot(
            repo=MMARCO_REPO,
            config="triples",
            split="train.small",
            revision="main",
            num_rows=triples_rows,
            output_path=str(triples_path.relative_to(target_dir.parent)),
        )
    )

    return snapshots


def download_mmarco(
    target_dir: Path,
    *,
    small: bool = False,
    revision: str | None = None,
) -> list[DatasetSnapshot]:
    """Download the mMARCO Portuguese collection, train queries, and dev queries.

    Args:
        target_dir: Directory where parquet snapshots land under ``mmarco/``.
        small: If True, truncate each split to a development sample.
        revision: Pin to a specific commit SHA. Defaults to the current
            ``refs/convert/parquet`` head.
    """
    mmarco_dir = target_dir / "mmarco"
    mmarco_dir.mkdir(parents=True, exist_ok=True)

    resolved_revision = revision or _resolve_revision(MMARCO_REPO, MMARCO_PARQUET_REVISION)
    if resolved_revision is None:
        resolved_revision = MMARCO_PARQUET_REVISION

    counts = EXPECTED_COUNTS_SMALL if small else EXPECTED_COUNTS_FULL
    snapshots: list[DatasetSnapshot] = []

    plan = [
        (
            "collection",
            "collection",
            "portuguese",
            [
                COLLECTION_PATH_TEMPLATE.format(shard=i, total=COLLECTION_SHARDS)
                for i in range(COLLECTION_SHARDS)
            ],
            counts["collection"] if small else None,
            mmarco_dir / "collection_portuguese.parquet",
        ),
        (
            "queries",
            "queries",
            "train.portuguese",
            [QUERIES_TRAIN_PATH],
            counts["queries_train"] if small else None,
            mmarco_dir / "queries_train_portuguese.parquet",
        ),
        (
            "queries",
            "queries",
            "dev.portuguese",
            [QUERIES_DEV_PATH],
            counts["queries_dev"] if small else None,
            mmarco_dir / "queries_dev_portuguese.parquet",
        ),
    ]

    for _label, config, split, remote_paths, row_limit, output_path in plan:
        logger.info(
            "Materializing %s/%s (split=%s, small=%s, limit=%s)",
            MMARCO_REPO,
            config,
            split,
            small,
            row_limit,
        )
        num_rows = _materialize(
            remote_paths,
            output_path,
            revision=resolved_revision,
            row_limit=row_limit,
        )
        logger.info("  → %s (%d rows)", output_path, num_rows)
        snapshots.append(
            DatasetSnapshot(
                repo=MMARCO_REPO,
                config=config,
                split=split,
                revision=resolved_revision,
                num_rows=num_rows,
                output_path=str(output_path.relative_to(target_dir.parent)),
            )
        )

    return snapshots


def write_manifest(
    target_dir: Path,
    snapshots: list[DatasetSnapshot],
    *,
    small: bool,
) -> Path:
    """Write or update the manifest with the downloaded snapshots."""
    manifest_path = target_dir / MANIFEST_NAME
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    manifest.setdefault("downloads", []).append(
        {
            "timestamp_utc": _now_iso(),
            "mode": "small" if small else "full",
            "snapshots": [asdict(s) for s in snapshots],
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Wrote manifest %s", manifest_path)
    return manifest_path


def check(target_dir: Path, *, small: bool = False) -> bool:
    """Validate that expected files exist and have plausible row counts."""
    expected = EXPECTED_COUNTS_SMALL if small else EXPECTED_COUNTS_FULL
    manifest_path = target_dir / MANIFEST_NAME
    if not manifest_path.exists():
        logger.error("Missing manifest at %s", manifest_path)
        return False

    manifest = json.loads(manifest_path.read_text())
    latest = manifest["downloads"][-1] if manifest.get("downloads") else None
    if latest is None:
        logger.error("Manifest has no downloads recorded")
        return False

    counts: dict[str, int] = {}
    for snap in latest["snapshots"]:
        path = target_dir.parent / snap["output_path"]
        if not path.exists():
            logger.error("Missing snapshot file: %s", path)
            return False
        # Only the parquet snapshots from refs/convert/parquet count toward
        # the row-count budget. qrels and triples (TSVs on main) are tracked
        # in the manifest for reproducibility but use their own size schema.
        if snap["config"] == "collection":
            counts["collection"] = snap["num_rows"]
        elif snap["config"] == "queries" and "train" in snap["split"]:
            counts["queries_train"] = snap["num_rows"]
        elif snap["config"] == "queries" and "dev" in snap["split"]:
            counts["queries_dev"] = snap["num_rows"]

    ok = True
    for key, expected_count in expected.items():
        actual = counts.get(key)
        if actual is None:
            logger.error("Missing count for %s", key)
            ok = False
            continue
        if small:
            if actual > expected_count:
                logger.error("%s has %d rows, expected ≤ %d (small)", key, actual, expected_count)
                ok = False
        else:
            if actual != expected_count:
                logger.error("%s has %d rows, expected %d", key, actual, expected_count)
                ok = False

    if ok:
        logger.info("All counts validated (%s mode).", "small" if small else "full")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Download mMARCO-PT")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--small",
        action="store_true",
        help="Download a small development slice (≈10k passages, 1k+100 queries).",
    )
    parser.add_argument(
        "--mmarco-revision",
        default=None,
        help="Pin mMARCO-PT to a specific revision SHA (default: latest refs/convert/parquet).",
    )
    parser.add_argument(
        "--skip-qrels-triples",
        action="store_true",
        help="Skip qrels and training triples download (useful for re-running collection only).",
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate existing data without downloading."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args.target_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        ok = check(args.target_dir, small=args.small)
        sys.exit(0 if ok else 1)

    snapshots = download_mmarco(args.target_dir, small=args.small, revision=args.mmarco_revision)
    if not args.skip_qrels_triples:
        snapshots.extend(download_qrels_and_triples(args.target_dir, small=args.small))
    write_manifest(args.target_dir, snapshots, small=args.small)
    check(args.target_dir, small=args.small)


if __name__ == "__main__":
    main()
