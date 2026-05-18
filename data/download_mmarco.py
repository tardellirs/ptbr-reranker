"""Download mMARCO-PT and MIRACL-PT from Hugging Face Datasets.

The mMARCO Portuguese subset comprises:
- ``collections``: 8.8M passages translated to PT-BR.
- ``queries``: 502,939 train queries + 6,980 dev queries, also translated.
- ``qrels``: relevance judgements (binary) inherited from MS MARCO.
- BM25 runs: top-1000 per query (used as candidates and as easy negatives).

We snapshot each split to a local parquet under ``data/raw/mmarco/`` and record
the Hugging Face revision SHA in ``data/raw/mmarco/manifest.json`` for the
ACL/NeurIPS-style reproducibility checklist.

For development on a laptop or in CI, use ``--small`` to materialize a
~10k-passage / 100-query slice; full download is gated behind explicit flags.
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
MMARCO_LANG = "portuguese"
MIRACL_REPO = "miracl/miracl"
MIRACL_LANG = "pt"

DEFAULT_RAW_DIR = Path("data/raw")
MANIFEST_NAME = "manifest.json"

EXPECTED_COUNTS_FULL = {
    "collection": 8_841_823,
    "queries_train": 502_939,
    "queries_dev": 6_980,
}
EXPECTED_COUNTS_SMALL = {
    "collection": 10_000,
    "queries_train": 1_000,
    "queries_dev": 100,
}


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


def _resolve_revision(repo_id: str) -> str | None:
    """Resolve the current ``main`` commit SHA of an HF dataset (best-effort)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(repo_id=repo_id)
        return info.sha
    except Exception as exc:  # pragma: no cover - network failure path
        logger.warning("Could not resolve revision for %s: %s", repo_id, exc)
        return None


def _save_parquet(rows: Any, output_path: Path) -> int:
    """Persist a `datasets.Dataset` to parquet and return the row count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(str(output_path))
    return int(rows.num_rows)


def download_mmarco(
    target_dir: Path,
    *,
    small: bool = False,
    revision: str | None = None,
) -> list[DatasetSnapshot]:
    """Download the mMARCO Portuguese collection, train queries, and dev queries.

    Args:
        target_dir: Directory where ``mmarco/{collection,queries,qrels}.parquet`` land.
        small: If True, slice each split to a development-sized sample.
        revision: Pin to a specific HF revision SHA. ``None`` resolves to ``main``.
    """
    from datasets import load_dataset

    mmarco_dir = target_dir / "mmarco"
    mmarco_dir.mkdir(parents=True, exist_ok=True)

    resolved_revision = revision or _resolve_revision(MMARCO_REPO)
    snapshots: list[DatasetSnapshot] = []

    splits = [
        ("collection", "collection", MMARCO_LANG, EXPECTED_COUNTS_SMALL["collection"]),
        ("queries", "queries", f"train.{MMARCO_LANG}", EXPECTED_COUNTS_SMALL["queries_train"]),
        ("queries", "queries", f"dev.{MMARCO_LANG}", EXPECTED_COUNTS_SMALL["queries_dev"]),
    ]

    for label, config, split, small_n in splits:
        logger.info("Loading %s/%s split=%s (small=%s)", MMARCO_REPO, config, split, small)
        ds = load_dataset(MMARCO_REPO, config, split=split, revision=resolved_revision)
        if small:
            n = min(small_n, len(ds))
            ds = ds.select(range(n))

        filename = f"{label}_{split.replace('.', '_')}.parquet"
        output_path = mmarco_dir / filename
        num_rows = _save_parquet(ds, output_path)
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


def download_miracl(
    target_dir: Path,
    *,
    revision: str | None = None,
) -> list[DatasetSnapshot]:
    """Download MIRACL Portuguese dev split for cross-domain evaluation."""
    from datasets import load_dataset

    miracl_dir = target_dir / "miracl"
    miracl_dir.mkdir(parents=True, exist_ok=True)
    resolved_revision = revision or _resolve_revision(MIRACL_REPO)
    snapshots: list[DatasetSnapshot] = []

    for split in ("dev",):
        logger.info("Loading %s/%s split=%s", MIRACL_REPO, MIRACL_LANG, split)
        ds = load_dataset(MIRACL_REPO, MIRACL_LANG, split=split, revision=resolved_revision)
        output_path = miracl_dir / f"queries_{split}.parquet"
        num_rows = _save_parquet(ds, output_path)
        logger.info("  → %s (%d rows)", output_path, num_rows)
        snapshots.append(
            DatasetSnapshot(
                repo=MIRACL_REPO,
                config=MIRACL_LANG,
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
    """Validate that expected files exist and have plausible row counts.

    Returns True if all checks pass; logs failures otherwise.
    """
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
        if snap["config"] == "collection":
            counts["collection"] = snap["num_rows"]
        elif "train" in snap["split"]:
            counts["queries_train"] = snap["num_rows"]
        elif "dev" in snap["split"]:
            counts["queries_dev"] = snap["num_rows"]

    ok = True
    for key, expected_count in expected.items():
        actual = counts.get(key)
        if actual is None:
            logger.error("Missing count for %s", key)
            ok = False
            continue
        # In full mode, expect exact counts; in small mode, expect <= the sample target.
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
    parser = argparse.ArgumentParser(description="Download mMARCO-PT and MIRACL-PT")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--small",
        action="store_true",
        help="Download a small development slice (≈10k passages, 1k+100 queries).",
    )
    parser.add_argument(
        "--mmarco-revision", default=None, help="Pin mMARCO to a specific revision SHA."
    )
    parser.add_argument(
        "--miracl-revision", default=None, help="Pin MIRACL to a specific revision SHA."
    )
    parser.add_argument("--skip-miracl", action="store_true")
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

    snapshots: list[DatasetSnapshot] = []
    snapshots.extend(
        download_mmarco(args.target_dir, small=args.small, revision=args.mmarco_revision)
    )
    if not args.skip_miracl:
        snapshots.extend(download_miracl(args.target_dir, revision=args.miracl_revision))

    write_manifest(args.target_dir, snapshots, small=args.small)
    check(args.target_dir, small=args.small)


if __name__ == "__main__":
    main()
