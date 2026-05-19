"""Training loop for the PT-BR cross-encoder reranker.

Uses the sentence-transformers v5 ``CrossEncoderTrainer`` (a HuggingFace
``Trainer``-style API) with ``BinaryCrossEntropyLoss``: each
``(query, positive, negative)`` triple from ``data/build_triples.py`` is
expanded into two pair examples with binary labels and the model is fit
to score (query, positive) high and (query, negative) low.

Optional integrations:
- W&B logging (set ``wandb_project`` in the config; otherwise disabled).
- ``codecarbon`` emissions tracker (best-effort; logged into the run
  directory for the paper's Environmental Impact section).

Reproducibility: every config knob is documented in
``configs/train_*.yaml`` and the resolved values are serialized to
``<output_dir>/training_config.json`` alongside the seed and resolved
device, so a paper reviewer can reconstruct the exact run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

if TYPE_CHECKING:
    import datasets

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
    save_every_steps: int = 10_000
    log_every_steps: int = 50
    keep_top_k_checkpoints: int = 2
    weight_decay: float = 0.01
    mixed_precision: str = "bf16"  # one of "no", "fp16", "bf16"
    gradient_checkpointing: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    codecarbon_enabled: bool = True
    device: str = "auto"

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
            # PyYAML parses "1e-5" as string and "1.0e-5" as float; coerce defensively.
            learning_rate=float(raw.get("learning_rate", 2e-5)),
            warmup_ratio=float(raw.get("warmup_ratio", 0.1)),
            max_length=raw.get("max_length", 256),
            seed=raw.get("seed", 42),
            eval_every_steps=raw.get("eval_every_steps", 10_000),
            save_every_steps=raw.get("save_every_steps", 10_000),
            log_every_steps=raw.get("log_every_steps", 50),
            keep_top_k_checkpoints=raw.get("keep_top_k_checkpoints", 2),
            weight_decay=float(raw.get("weight_decay", 0.01)),
            mixed_precision=raw.get("mixed_precision", "bf16"),
            gradient_checkpointing=raw.get("gradient_checkpointing", False),
            wandb_project=raw.get("wandb_project"),
            wandb_run_name=raw.get("wandb_run_name"),
            wandb_tags=list(raw.get("wandb_tags", [])),
            codecarbon_enabled=raw.get("codecarbon_enabled", True),
            device=raw.get("device", "auto"),
        )


def set_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    """Pick a usable torch device. Mirrors ``data.mine_hard_negatives.resolve_device``."""
    if requested == "cpu":
        return "cpu"
    if requested not in ("auto", "cuda"):
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _ = (torch.zeros(1, device="cuda") + 1).cpu()
        return "cuda"
    except Exception as exc:  # pragma: no cover - environment-specific
        logger.warning("CUDA advertised but unusable (%s); falling back to CPU.", exc)
        return "cpu"


def triples_to_pairs(triples_path: Path) -> datasets.Dataset:
    """Read ``(query, positive, negative)`` triples and expand to ``(text_a, text_b, label)`` pairs.

    Each triple becomes two pair examples with labels 1.0 (positive) and
    0.0 (negative). The HF datasets ``Dataset`` returned is consumed by
    ``CrossEncoderTrainer`` and ``BinaryCrossEntropyLoss``.
    """
    from datasets import Dataset

    table = pq.read_table(triples_path)  # type: ignore[no-untyped-call]
    queries = table["query_text"].to_pylist()
    positives = table["positive_text"].to_pylist()
    negatives = table["negative_text"].to_pylist()

    sentence_a: list[str] = []
    sentence_b: list[str] = []
    labels: list[float] = []
    for q, p, n in zip(queries, positives, negatives, strict=True):
        sentence_a.append(q)
        sentence_b.append(p)
        labels.append(1.0)
        sentence_a.append(q)
        sentence_b.append(n)
        labels.append(0.0)

    return Dataset.from_dict({"sentence_A": sentence_a, "sentence_B": sentence_b, "label": labels})


def _maybe_start_codecarbon(output_dir: Path, enabled: bool) -> object | None:
    """Best-effort start of a codecarbon emissions tracker. Returns the tracker or None."""
    if not enabled:
        return None
    try:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(
            project_name="ptbr-reranker",
            output_dir=str(output_dir),
            log_level="warning",
        )
        tracker.start()
        return tracker
    except Exception as exc:  # pragma: no cover - codecarbon is optional
        logger.warning("Could not start codecarbon: %s", exc)
        return None


def _maybe_init_wandb(config: TrainConfig, resolved_device: str) -> bool:
    """Initialize wandb if a project is configured. Returns True on success."""
    if not config.wandb_project:
        return False
    try:
        import wandb
    except ImportError:  # pragma: no cover - wandb is optional
        logger.warning("wandb_project set but wandb is not installed; skipping.")
        return False
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        tags=config.wandb_tags or None,
        config={**asdict(config), "resolved_device": resolved_device},
    )
    return True


def _persist_config(config: TrainConfig, resolved_device: str) -> None:
    """Serialize the resolved config so the run can be reconstructed later."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(config), "resolved_device": resolved_device}
    # Path objects → strings for JSON
    for k, v in list(payload.items()):
        if isinstance(v, Path):
            payload[k] = str(v)
    out = config.output_dir / "training_config.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", out)


def train(
    config: TrainConfig,
    *,
    debug: bool = False,
    max_steps: int | None = None,
) -> Path:
    """Train the cross-encoder. Returns the path to the best checkpoint."""
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import (
        CrossEncoderTrainingArguments,
    )

    set_seed(config.seed)
    resolved_device = resolve_device(config.device)
    logger.info("Resolved device: %s (requested=%s)", resolved_device, config.device)

    _persist_config(config, resolved_device)
    wandb_active = _maybe_init_wandb(config, resolved_device)
    tracker = _maybe_start_codecarbon(config.output_dir, config.codecarbon_enabled)

    logger.info("Loading triples from %s", config.train_triples)
    train_dataset = triples_to_pairs(config.train_triples)
    if debug:
        n = min(64, len(train_dataset))
        train_dataset = train_dataset.select(range(n))
        logger.info("Debug mode: truncated to %d pairs", n)

    logger.info("Loading base model %s", config.base_model)
    model_kwargs: dict[str, Any] = {}
    if resolved_device == "cpu" or config.mixed_precision == "no":
        model_kwargs["torch_dtype"] = torch.float32
    model = CrossEncoder(
        config.base_model,
        num_labels=1,
        max_length=config.max_length,
        device=resolved_device,
        model_kwargs=model_kwargs,
    )
    if resolved_device == "cpu":
        model.model.float()

    loss = BinaryCrossEntropyLoss(model)

    args_kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "num_train_epochs": config.num_epochs,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.grad_accum_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "save_strategy": "steps",
        "save_steps": config.save_every_steps,
        "save_total_limit": config.keep_top_k_checkpoints,
        "logging_steps": config.log_every_steps,
        "seed": config.seed,
        "gradient_checkpointing": config.gradient_checkpointing,
        "report_to": ["wandb"] if wandb_active else "none",
    }
    if config.mixed_precision == "bf16" and resolved_device == "cuda":
        args_kwargs["bf16"] = True
    elif config.mixed_precision == "fp16" and resolved_device == "cuda":
        args_kwargs["fp16"] = True
    if max_steps is not None:
        args_kwargs["max_steps"] = max_steps

    training_args = CrossEncoderTrainingArguments(**args_kwargs)
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    try:
        trainer.train()
    finally:
        if tracker is not None:
            tracker.stop()  # type: ignore[attr-defined]
        if wandb_active:
            import wandb

            wandb.finish()

    best_dir = config.output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    logger.info("Saved best checkpoint to %s", best_dir)
    return best_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PT-BR cross-encoder reranker")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (tiny data)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # Avoid TOKENIZERS_PARALLELISM warning when forking.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = TrainConfig.from_yaml(args.config)
    train(config, debug=args.debug, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
