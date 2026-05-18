"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def checkpoint_path() -> str:
    """Checkpoint to test against.

    Defaults to the production HF Hub repo. Override with
    ``PTBR_RERANKER_CHECKPOINT`` (e.g. a local ``runs/best`` directory) for
    pre-release evaluation.
    """
    return os.environ.get(
        "PTBR_RERANKER_CHECKPOINT",
        "tardellirs/cross-encoder-albertina-ptbr-mmarco",
    )


@pytest.fixture(scope="session")
def small_corpus() -> list[str]:
    """A tiny in-memory corpus used by fast CPU tests."""
    return [
        "Brasília é a capital federal do Brasil desde 1960.",
        "São Paulo é a maior cidade do Brasil em população.",
        "O Rio de Janeiro foi capital do Brasil até 1960.",
        "O Brasil tem 26 estados e um distrito federal.",
        "A culinária brasileira tem influências indígenas, africanas e europeias.",
    ]
