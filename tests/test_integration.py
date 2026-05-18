"""Phase 5.5 — End-to-end pipeline integration tests.

Validates that bi-encoder retrieval + cross-encoder reranking produces a coherent,
deterministic, and performant ranking on a small in-memory corpus.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]


@pytest.mark.gpu
def test_reranker_improves_over_biencoder(small_corpus: list[str]) -> None:
    """Aggregated MRR with reranker should exceed bi-encoder alone."""
    pytest.skip("Integration — implement after Phase 3 training is complete")


@pytest.mark.gpu
def test_reranker_is_deterministic() -> None:
    """Same seed and inputs produce the same ranking."""
    pytest.skip("Integration — implement after Phase 3 training is complete")


@pytest.mark.gpu
def test_pipeline_latency_under_threshold() -> None:
    """Retrieval + rerank of top-100 should complete in < 200 ms on A100."""
    pytest.skip("Integration — implement after Phase 3 training is complete")


def test_stats_bootstrap_basic() -> None:
    """Sanity-check the bootstrap utility (CPU-only, fast)."""
    import numpy as np

    from src.stats import bootstrap_metric

    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=0.1, size=200).clip(0, 1)
    result = bootstrap_metric(values, n_resamples=200, confidence=0.95, seed=0)

    assert result.ci_low <= result.mean <= result.ci_high
    assert 0.0 <= result.ci_low <= result.ci_high <= 1.0
    assert result.n_resamples == 200
