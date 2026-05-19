"""Tests for the training pipeline (CPU-only; full training is GPU-only)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _write_synthetic_triples(path: Path, n: int = 5) -> None:
    queries = [f"qual é a definição de termo {i}?" for i in range(n)]
    positives = [f"O termo {i} é definido como uma entidade exemplo número {i}." for i in range(n)]
    negatives = [f"Tópico não relacionado: pixel arte e jogos retrô (linha {i})." for i in range(n)]
    table = pa.table(
        {
            "query_id": list(range(n)),
            "query_text": queries,
            "positive_text": positives,
            "negative_text": negatives,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def test_triples_to_pairs_doubles_rows(tmp_path: Path) -> None:
    from src.train import triples_to_pairs

    triples_path = tmp_path / "triples.parquet"
    _write_synthetic_triples(triples_path, n=5)

    ds = triples_to_pairs(triples_path)
    assert len(ds) == 10  # 5 triples x (1 positive pair + 1 negative pair)
    assert set(ds.column_names) == {"sentence_A", "sentence_B", "label"}


def test_triples_to_pairs_labels_alternate_one_zero(tmp_path: Path) -> None:
    from src.train import triples_to_pairs

    triples_path = tmp_path / "triples.parquet"
    _write_synthetic_triples(triples_path, n=3)

    ds = triples_to_pairs(triples_path)
    labels = ds["label"]
    expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert labels == expected


def test_triples_to_pairs_query_repeats_for_pair(tmp_path: Path) -> None:
    from src.train import triples_to_pairs

    triples_path = tmp_path / "triples.parquet"
    _write_synthetic_triples(triples_path, n=2)

    ds = triples_to_pairs(triples_path)
    # Row 0 and row 1 share the same query (positive/negative pair for triple 0)
    assert ds["sentence_A"][0] == ds["sentence_A"][1]
    # Row 2 and row 3 share the same query (positive/negative pair for triple 1)
    assert ds["sentence_A"][2] == ds["sentence_A"][3]
    # And the first query differs from the second.
    assert ds["sentence_A"][0] != ds["sentence_A"][2]


def test_triples_to_pairs_positive_then_negative(tmp_path: Path) -> None:
    from src.train import triples_to_pairs

    triples_path = tmp_path / "triples.parquet"
    _write_synthetic_triples(triples_path, n=1)

    ds = triples_to_pairs(triples_path)
    # Positive comes first, negative second.
    assert ds["sentence_B"][0].startswith("O termo")
    assert ds["sentence_B"][1].startswith("Tópico não relacionado")


def test_set_seed_is_deterministic() -> None:
    import numpy as np
    import torch

    from src.train import set_seed

    set_seed(123)
    a_py = (np.random.random(), torch.rand(1).item())
    set_seed(123)
    b_py = (np.random.random(), torch.rand(1).item())
    assert a_py == b_py


def test_resolve_device_respects_cpu_override() -> None:
    from src.train import resolve_device

    assert resolve_device("cpu") == "cpu"


def test_train_config_from_yaml(tmp_path: Path) -> None:
    from src.train import TrainConfig

    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(
        """base_model: foo/bar
train_triples: data/processed/triples.parquet
output_dir: runs/test
num_epochs: 2
batch_size: 16
learning_rate: 1e-5
wandb_project: ptbr-reranker
wandb_tags: [test, sample]
"""
    )
    cfg = TrainConfig.from_yaml(yaml_path)
    assert cfg.base_model == "foo/bar"
    assert cfg.num_epochs == 2
    assert cfg.batch_size == 16
    assert cfg.learning_rate == pytest.approx(1e-5)
    assert cfg.wandb_project == "ptbr-reranker"
    assert cfg.wandb_tags == ["test", "sample"]
    # Defaults preserved
    assert cfg.max_length == 256
    assert cfg.seed == 42


@pytest.mark.slow
def test_train_smoke_runs_on_cpu(tmp_path: Path) -> None:
    """End-to-end CPU smoke test on a tiny synthetic dataset (no GPU required).

    Slow-marked because Albertina-100m download is ~400MB and CPU forward passes
    take seconds. Use ``pytest -m slow`` before any production training run.
    """
    from src.train import TrainConfig, train

    triples_path = tmp_path / "triples.parquet"
    _write_synthetic_triples(triples_path, n=4)

    cfg = TrainConfig(
        base_model="PORTULAN/albertina-100m-portuguese-ptbr-encoder",
        train_triples=triples_path,
        output_dir=tmp_path / "runs",
        num_epochs=1,
        batch_size=2,
        grad_accum_steps=1,
        max_length=32,
        seed=0,
        eval_every_steps=10_000,
        save_every_steps=10_000,
        log_every_steps=1,
        mixed_precision="no",
        wandb_project=None,
        codecarbon_enabled=False,
        device="cpu",
    )
    best = train(cfg, debug=True, max_steps=2)
    assert best.exists()
    assert (cfg.output_dir / "training_config.json").exists()
