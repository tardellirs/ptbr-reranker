"""Download mMARCO-PT and MIRACL-PT datasets from Hugging Face Datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MMARCO_REPO = "unicamp-dl/mmarco"
MMARCO_LANG = "portuguese"
MIRACL_REPO = "miracl/miracl"
MIRACL_LANG = "pt"

DEFAULT_RAW_DIR = Path("data/raw")


def download_mmarco(target_dir: Path) -> None:
    """Download mMARCO Portuguese subset: collection, queries, qrels, BM25 runs."""
    raise NotImplementedError("download_mmarco() to be implemented in Phase 1")


def download_miracl(target_dir: Path) -> None:
    """Download MIRACL Portuguese subset."""
    raise NotImplementedError("download_miracl() to be implemented in Phase 1")


def check(target_dir: Path) -> None:
    """Validate expected file counts and basic shape of downloaded data."""
    raise NotImplementedError("check() to be implemented in Phase 1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download mMARCO-PT and MIRACL-PT")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--skip-miracl", action="store_true")
    parser.add_argument("--check", action="store_true", help="Only validate existing files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.target_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        check(args.target_dir)
        return

    download_mmarco(args.target_dir)
    if not args.skip_miracl:
        download_miracl(args.target_dir)


if __name__ == "__main__":
    main()
