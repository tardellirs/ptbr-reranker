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


def patch_deberta_attention_dtype() -> bool:
    """In-place patch transformers DeBERTa attention so mixed precision works.

    Upstream uses ``torch.finfo(query_layer.dtype).min`` in the attention mask
    fill, which overflows the destination dtype under autocast (the query
    stays fp32 while attention_scores moves to bf16/fp16). The robust upstream
    fix would be to use ``finfo(attention_scores.dtype).min`` — that's exactly
    what this helper does on disk, idempotently, before training imports
    transformers. Returns True if the patch was applied or already in place.
    """
    try:
        import transformers.models.deberta.modeling_deberta as _deberta_mod
    except Exception as exc:  # pragma: no cover - only fires if transformers absent
        logger.warning("Could not import transformers DeBERTa to patch: %s", exc)
        return False

    fpath = Path(_deberta_mod.__file__)
    try:
        content = fpath.read_text()
    except OSError as exc:  # pragma: no cover - file not writable
        logger.warning("Could not read %s for patching: %s", fpath, exc)
        return False

    needle = "finfo(query_layer.dtype).min"
    replacement = "finfo(attention_scores.dtype).min"

    if needle not in content:
        if replacement in content:
            logger.info("DeBERTa attention already patched for mixed precision.")
            return True
        logger.info("DeBERTa attention does not require the dtype patch.")
        return False

    patched = content.replace(needle, replacement)
    try:
        fpath.write_text(patched)
    except OSError as exc:  # pragma: no cover - file not writable
        logger.warning("Could not write patched DeBERTa at %s: %s", fpath, exc)
        return False

    # Invalidate any already-imported submodules so the next import re-reads disk.
    import importlib
    import sys

    for mod_name in list(sys.modules.keys()):
        if "transformers.models.deberta" in mod_name:
            sys.modules.pop(mod_name, None)
    importlib.invalidate_caches()
    logger.info("Patched DeBERTa attention at %s for mixed precision.", fpath)
    return True


MIN_CUDA_CAPABILITY = (7, 0)


def resolve_device(requested: str) -> str:
    """Pick a usable torch device. Mirrors ``data.mine_hard_negatives.resolve_device``.

    A basic ``zeros + 1`` op passes on Kaggle's P100 (sm_60) even though the
    PyTorch build there only ships kernels for sm_70+, so the model forward
    eventually crashes inside DeBERTa attention. Check compute capability
    explicitly and fall back to CPU when below the supported floor.
    """
    if requested == "cpu":
        return "cpu"
    if requested not in ("auto", "cuda"):
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


def triples_to_pairs(triples_path: Path) -> datasets.Dataset:
    """Read ``(query, positive, negative)`` triples and expand to ``(text_a, text_b, label)`` pairs.

    Each triple becomes two pair examples with labels 1.0 (positive) and
    0.0 (negative). For 10M+ triples we cannot materialise the whole
    expansion in Python lists — ``.to_pylist()`` over a 3GB parquet blows
    up to ~30GB of string objects and gets cgroup-OOM-killed silently.
    Stream batches through pyarrow, write an on-disk pair parquet next to
    the input, and return a memory-mapped ``Dataset.from_parquet``.

    The pair parquet is cached at ``<input>_pairs.parquet``; subsequent
    calls reuse it. Delete the cache file to force regeneration.
    """
    import pyarrow as pa
    from datasets import Dataset

    pairs_path = triples_path.with_name(triples_path.stem + "_pairs.parquet")
    schema = pa.schema(
        [
            ("sentence_A", pa.string()),
            ("sentence_B", pa.string()),
            ("label", pa.float32()),
        ]
    )

    if not pairs_path.exists():
        logger.info("Expanding triples → pairs parquet at %s", pairs_path)
        pf = pq.ParquetFile(triples_path)  # type: ignore[no-untyped-call]
        writer = pq.ParquetWriter(pairs_path, schema)  # type: ignore[no-untyped-call]
        written = 0
        try:
            for batch in pf.iter_batches(  # type: ignore[no-untyped-call]
                batch_size=50_000,
                columns=["query_text", "positive_text", "negative_text"],
            ):
                queries = batch["query_text"].to_pylist()
                positives = batch["positive_text"].to_pylist()
                negatives = batch["negative_text"].to_pylist()
                sa: list[str] = []
                sb: list[str] = []
                lab: list[float] = []
                for q, p, n in zip(queries, positives, negatives, strict=True):
                    sa.append(q)
                    sb.append(p)
                    lab.append(1.0)
                    sa.append(q)
                    sb.append(n)
                    lab.append(0.0)
                out_batch = pa.RecordBatch.from_arrays(
                    [pa.array(sa), pa.array(sb), pa.array(lab, type=pa.float32())],
                    schema=schema,
                )
                writer.write_batch(out_batch)  # type: ignore[no-untyped-call]
                written += len(sa)
                if written % 1_000_000 == 0:
                    logger.info("Expanded %d pair rows so far", written)
        finally:
            writer.close()  # type: ignore[no-untyped-call]
        logger.info("Wrote %d pair rows to %s", written, pairs_path)

    # ``Dataset.from_parquet`` caches a copy of the parquet to
    # ``$HF_DATASETS_CACHE`` (default ``~/.cache/huggingface/datasets``).
    # On hosted pods the home overlay is small (~8GB) while the workspace
    # volume has more room; respect HF_DATASETS_CACHE if the caller set it
    # to a roomier path, otherwise leave HF's default. ``keep_in_memory``
    # is False so big datasets remain memory-mapped.
    return Dataset.from_parquet(str(pairs_path), keep_in_memory=False)


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
    """Initialize wandb if a project is configured AND credentials are available.

    Returns True on success. Returns False (with a logged warning) when the
    YAML sets ``wandb_project`` but the environment lacks ``WANDB_API_KEY`` or
    a cached netrc login — common in CI and hosted notebooks where we don't
    want a missing token to crash an otherwise valid training run.
    """
    if not config.wandb_project:
        return False
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        logger.info("WANDB_DISABLED is set; skipping W&B init.")
        return False
    if not os.environ.get("WANDB_API_KEY") and not Path("~/.netrc").expanduser().exists():
        logger.warning(
            "wandb_project=%r is set but no WANDB_API_KEY env var or ~/.netrc found; "
            "skipping W&B logging.",
            config.wandb_project,
        )
        return False
    try:
        import wandb
    except ImportError:  # pragma: no cover - wandb is optional
        logger.warning("wandb_project set but wandb is not installed; skipping.")
        return False
    try:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            tags=config.wandb_tags or None,
            config={**asdict(config), "resolved_device": resolved_device},
        )
    except Exception as exc:  # pragma: no cover - network / auth failures
        logger.warning("wandb.init failed (%s); proceeding without W&B logging.", exc)
        return False
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
    patch_deberta_attention_dtype()
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
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Override HF_DATASETS_CACHE (useful on pods where the home "
        "overlay is small). Defaults to <output_dir>/.hf_cache if unset.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # Avoid TOKENIZERS_PARALLELISM warning when forking.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = TrainConfig.from_yaml(args.config)
    cache_dir = args.hf_cache_dir or (config.output_dir / ".hf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))
    os.environ.setdefault("HF_HOME", str(cache_dir))
    train(config, debug=args.debug, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
