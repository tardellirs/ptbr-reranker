"""Phase 5.4 — Qualitative evaluation on curated PT-BR cases.

A hand-curated set of 50 cases covering domains where translated mMARCO is weakest:
Brazilian legal text, clinical PT-BR, slang/colloquialisms, polysemy, and negation.

Each case is rated on a binary rubric (top-3 contains a relevant passage or not).
Release threshold: >= 80% of curated cases pass.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.quality]


@pytest.mark.slow
def test_qualitative_brazilian_legal() -> None:
    """Brazilian legal queries (jurisprudência STF/STJ/TST) match correct passages."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")


@pytest.mark.slow
def test_qualitative_clinical_ptbr() -> None:
    """Clinical PT-BR terminology (CID-10 PT, drug names) is handled correctly."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")


@pytest.mark.slow
def test_qualitative_slang_and_colloquialisms() -> None:
    """Informal PT-BR queries (gírias) match formal content."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")


@pytest.mark.slow
def test_qualitative_polysemy() -> None:
    """Polysemous terms (banco, manga, sede) are disambiguated by query context."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")


@pytest.mark.slow
def test_qualitative_negation() -> None:
    """Queries with explicit negation rank passages with the negated content lower."""
    pytest.skip("Phase 5 — implement after Phase 3 training is complete")
