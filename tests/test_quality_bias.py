"""Phase 5.3 — Biases and generalization across variants and demographics.

Covers:
- PT-BR vs PT-PT (vocabulary, spelling differences)
- Gender / demographic perturbations on neutral content
- Per-domain MRR stratification
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.quality]


@pytest.mark.slow
def test_pt_br_vs_pt_pt_generalization() -> None:
    """Gap between PT-BR and PT-PT phrasing should be small (< 5pp MRR)."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")


@pytest.mark.slow
def test_neutral_content_invariance() -> None:
    """Score should not change significantly for neutral queries under demographic swaps."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")


@pytest.mark.slow
def test_per_domain_metric_stratification() -> None:
    """No domain (news / scientific / technical) should collapse far below average."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")
