"""Cross-encoder model wrapper around sentence-transformers."""

from __future__ import annotations

from pathlib import Path

from sentence_transformers import CrossEncoder

DEFAULT_BASE_MODEL = "PORTULAN/albertina-100m-portuguese-ptbr-encoder"
DEFAULT_HF_HUB_ID = "tardellirs/cross-encoder-albertina-ptbr-mmarco"


def load(
    checkpoint: str | Path = DEFAULT_HF_HUB_ID,
    *,
    max_length: int = 256,
    device: str | None = None,
) -> CrossEncoder:
    """Load a cross-encoder from local checkpoint or Hugging Face Hub.

    Args:
        checkpoint: Path to a local directory or a HF Hub repo id.
        max_length: Max sequence length for (query, passage) pairs.
        device: Torch device string (e.g. ``"cuda"``, ``"cpu"``). ``None`` auto-selects.

    Returns:
        A configured ``CrossEncoder`` ready for ``predict``.
    """
    model: CrossEncoder = CrossEncoder(str(checkpoint), max_length=max_length, device=device)
    return model


def load_base(
    base_model: str = DEFAULT_BASE_MODEL,
    *,
    num_labels: int = 1,
    max_length: int = 256,
) -> CrossEncoder:
    """Initialize a fresh cross-encoder from a foundation encoder (for training)."""
    model: CrossEncoder = CrossEncoder(base_model, num_labels=num_labels, max_length=max_length)
    return model
