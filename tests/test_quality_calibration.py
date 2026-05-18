"""Phase 5.1 — Calibration of cross-encoder scores.

Goals:
- Score(query, relevant) > Score(query, related-but-irrelevant) > Score(query, random)
- Reliability diagram: precision per score bucket should track the score
- Report Expected Calibration Error (ECE)
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.quality]


@pytest.mark.slow
def test_score_ordering_holds() -> None:
    """Relevant > related > random on a curated mini-set."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")


@pytest.mark.slow
def test_expected_calibration_error_below_threshold() -> None:
    """ECE on mMARCO-PT dev should be < 0.10 for release."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")


@pytest.mark.slow
def test_score_distribution_separation() -> None:
    """Positive and negative score distributions should be clearly separated."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")
