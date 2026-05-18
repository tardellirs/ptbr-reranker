"""Phase 5.2 — Robustness to noise and orthographic perturbations.

Tests how the reranker behaves under common PT-BR informal-writing patterns:
typos, missing accents, case variation, and abbreviations.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.quality]


@pytest.mark.slow
def test_robustness_missing_accents() -> None:
    """MRR drop when removing accents should be < 10%."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")


@pytest.mark.slow
def test_robustness_typos() -> None:
    """MRR drop with 1-3 char typos should be < 15%."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")


@pytest.mark.slow
def test_robustness_case_variation() -> None:
    """MRR drop with mixed case should be < 5%."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")


@pytest.mark.slow
def test_robustness_common_abbreviations() -> None:
    """Common PT-BR abbreviations (vc, pq, tb, td) should not destroy rankings."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")
