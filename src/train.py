"""Training loop for the cross-encoder reranker."""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    base_model: str
    train_triples: Path
    output_dir: Path
    num_epochs: int = 1
    batch_size: int = 32
    grad_accum_steps: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_length: int = 256
    seed: int = 42
    eval_every_steps: int = 10_000
    wandb_project: str = "ptbr-reranker"
    wandb_run_name: str | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        with path.open() as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls(
            base_model=raw["base_model"],
            train_triples=Path(raw["train_triples"]),
            output_dir=Path(raw["output_dir"]),
            num_epochs=raw.get("num_epochs", 1),
            batch_size=raw.get("batch_size", 32),
            grad_accum_steps=raw.get("grad_accum_steps", 2),
            learning_rate=raw.get("learning_rate", 2e-5),
            warmup_ratio=raw.get("warmup_ratio", 0.1),
            max_length=raw.get("max_length", 256),
            seed=raw.get("seed", 42),
            eval_every_steps=raw.get("eval_every_steps", 10_000),
            wandb_project=raw.get("wandb_project", "ptbr-reranker"),
            wandb_run_name=raw.get("wandb_run_name"),
        )


def set_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config: TrainConfig, *, debug: bool = False, max_steps: int | None = None) -> Path:
    """Train the cross-encoder. Returns path to best checkpoint."""
    set_seed(config.seed)
    raise NotImplementedError("train() to be implemented in Phase 3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PT-BR cross-encoder reranker")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (tiny data)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = TrainConfig.from_yaml(args.config)
    train(config, debug=args.debug, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
