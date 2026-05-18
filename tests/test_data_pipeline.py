"""Tests for the data pipeline scripts (CPU-only, no network unless marked)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.download_mmarco import (
    EXPECTED_COUNTS_SMALL,
    DatasetSnapshot,
    check,
    write_manifest,
)


def _build_snapshots(target_dir: Path, *, small: bool) -> list[DatasetSnapshot]:
    counts = (
        EXPECTED_COUNTS_SMALL
        if small
        else {
            "collection": 8_841_823,
            "queries_train": 808_731,
            "queries_dev": 101_093,
        }
    )
    mmarco = target_dir / "mmarco"
    mmarco.mkdir(parents=True, exist_ok=True)

    splits = [
        ("collection", "collection", "portuguese", counts["collection"]),
        ("queries", "queries", "train.portuguese", counts["queries_train"]),
        ("queries", "queries", "dev.portuguese", counts["queries_dev"]),
    ]
    snapshots: list[DatasetSnapshot] = []
    for label, config, split, num_rows in splits:
        if label == "collection":
            filename = "collection_portuguese.parquet"
        elif "train" in split:
            filename = "queries_train_portuguese.parquet"
        else:
            filename = "queries_dev_portuguese.parquet"
        path = mmarco / filename
        path.write_bytes(b"")
        snapshots.append(
            DatasetSnapshot(
                repo="unicamp-dl/mmarco",
                config=config,
                split=split,
                revision="abc123",
                num_rows=num_rows,
                output_path=str(path.relative_to(target_dir.parent)),
            )
        )
    return snapshots


def test_check_succeeds_with_valid_small_manifest(tmp_path: Path) -> None:
    target = tmp_path / "raw"
    target.mkdir()
    snapshots = _build_snapshots(target, small=True)
    write_manifest(target, snapshots, small=True)

    assert check(target, small=True) is True


def test_check_fails_when_manifest_missing(tmp_path: Path) -> None:
    target = tmp_path / "raw"
    target.mkdir()
    assert check(target, small=True) is False


def test_check_fails_when_snapshot_file_is_missing(tmp_path: Path) -> None:
    target = tmp_path / "raw"
    target.mkdir()
    snapshots = _build_snapshots(target, small=True)
    write_manifest(target, snapshots, small=True)
    (target / "mmarco" / "collection_portuguese.parquet").unlink()

    assert check(target, small=True) is False


def test_check_fails_when_small_count_exceeds_target(tmp_path: Path) -> None:
    target = tmp_path / "raw"
    target.mkdir()
    snapshots = _build_snapshots(target, small=True)
    snapshots[0] = DatasetSnapshot(
        repo=snapshots[0].repo,
        config=snapshots[0].config,
        split=snapshots[0].split,
        revision=snapshots[0].revision,
        num_rows=EXPECTED_COUNTS_SMALL["collection"] + 1,
        output_path=snapshots[0].output_path,
    )
    write_manifest(target, snapshots, small=True)

    assert check(target, small=True) is False


def test_check_fails_when_full_count_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "raw"
    target.mkdir()
    snapshots = _build_snapshots(target, small=False)
    snapshots[2] = DatasetSnapshot(
        repo=snapshots[2].repo,
        config=snapshots[2].config,
        split=snapshots[2].split,
        revision=snapshots[2].revision,
        num_rows=snapshots[2].num_rows + 1,
        output_path=snapshots[2].output_path,
    )
    write_manifest(target, snapshots, small=False)

    assert check(target, small=False) is False


def test_write_manifest_appends_multiple_downloads(tmp_path: Path) -> None:
    target = tmp_path / "raw"
    target.mkdir()
    snaps = _build_snapshots(target, small=True)

    write_manifest(target, snaps, small=True)
    write_manifest(target, snaps, small=True)

    manifest = json.loads((target / "manifest.json").read_text())
    assert len(manifest["downloads"]) == 2
    assert manifest["downloads"][0]["mode"] == "small"


@pytest.mark.slow
def test_albertina_loads_and_predicts() -> None:
    """Smoke test: load Albertina-100m cross-encoder and score a (query, passage) pair.

    Requires network (downloads ~400MB on first run). Marked slow so CI skips it
    by default; run locally with ``pytest -m slow`` to validate the model path.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        "PORTULAN/albertina-100m-portuguese-ptbr-encoder",
        num_labels=1,
        max_length=64,
    )
    scores = model.predict(
        [
            ("qual a capital do Brasil?", "Brasília é a capital do Brasil."),
            ("qual a capital do Brasil?", "Receita de bolo de chocolate."),
        ]
    )
    assert len(scores) == 2
    assert all(isinstance(float(s), float) for s in scores)
