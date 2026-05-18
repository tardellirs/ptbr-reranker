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


def test_stats_bootstrap_rejects_empty() -> None:
    import pytest

    from src.stats import bootstrap_metric

    with pytest.raises(ValueError, match="empty"):
        bootstrap_metric([], n_resamples=10)


def test_paired_bootstrap_detects_better_system() -> None:
    """When A is clearly better than B per-query, p-value should be small."""
    import numpy as np

    from src.stats import paired_bootstrap_pvalue

    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(loc=0.5, scale=0.1, size=n)
    system_b = base.clip(0, 1)
    system_a = (base + 0.05).clip(0, 1)

    p_value = paired_bootstrap_pvalue(system_a, system_b, n_resamples=500, seed=0)
    assert p_value < 0.05


def test_paired_bootstrap_no_difference_when_b_better() -> None:
    """If A is not better than B, the test should not claim significance."""
    from src.stats import paired_bootstrap_pvalue

    system_a = [0.4] * 50
    system_b = [0.6] * 50
    p_value = paired_bootstrap_pvalue(system_a, system_b, n_resamples=200, seed=0)
    assert p_value == 1.0


def test_paired_bootstrap_rejects_mismatched_lengths() -> None:
    import pytest

    from src.stats import paired_bootstrap_pvalue

    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_pvalue([0.1, 0.2], [0.3])


def test_bootstrap_result_as_str_formatting() -> None:
    from src.stats import BootstrapResult

    result = BootstrapResult(
        mean=0.5234, ci_low=0.4123, ci_high=0.6356, n_resamples=1000, confidence=0.95
    )
    formatted = result.as_str(decimals=3)
    assert "0.523" in formatted
    assert "0.412" in formatted
    assert "0.636" in formatted
    assert "CI 95%" in formatted
