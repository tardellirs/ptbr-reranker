"""Phase 5.1 — Calibration of cross-encoder scores on mMARCO-PT dev.

Reads the eval artifacts produced by ``src.eval_mmarco`` (the rerank score
parquet and the qrels file) and computes calibration diagnostics:

- score histogram for positives vs negatives
- 10-bin reliability diagram
- Expected Calibration Error (ECE)
- Brier score

These tests work offline — no model load — so they run in CI / on a laptop.

Skip-friendly if the eval artifacts are missing locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.quality]

EVAL_RERANK = Path("outputs/v0.1_top1000/eval_mmarco_v0.1_top1000_per_query_rerank.parquet")
QRELS = Path("data/raw/mmarco/qrels.dev.small.tsv")
ECE_THRESHOLD = 0.10  # release criterion from the project plan


def _load_pairs() -> pd.DataFrame:
    """Return DataFrame with columns (qid, docid, score, is_pos)."""
    if not EVAL_RERANK.exists():
        pytest.skip(f"missing {EVAL_RERANK} — run eval_mmarco first")
    if not QRELS.exists():
        pytest.skip(f"missing {QRELS}")
    rerank = pd.read_parquet(EVAL_RERANK)
    qrels = pd.read_csv(QRELS, sep="\t", names=["qid", "iter", "docid", "rel"])
    pos = set(map(tuple, qrels[qrels["rel"] > 0][["qid", "docid"]].astype(int).values))
    rerank["is_pos"] = rerank.apply(
        lambda r: (int(r["qid"]), int(r["docid"])) in pos, axis=1
    )
    return rerank


def _expected_calibration_error(
    scores: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict]]:
    """ECE with equal-width bins on [0,1]. Returns (ECE, bin diagnostics)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(scores)
    ece = 0.0
    diag: list[dict] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (
            (scores >= lo) & (scores < hi)
            if i < n_bins - 1
            else (scores >= lo) & (scores <= hi)
        )
        if mask.sum() == 0:
            diag.append({"bin": (lo, hi), "n": 0, "mean_score": None, "pos_rate": None, "gap": None})
            continue
        mean_score = float(scores[mask].mean())
        pos_rate = float(labels[mask].mean())
        gap = abs(mean_score - pos_rate)
        ece += (mask.sum() / n) * gap
        diag.append(
            {
                "bin": (round(lo, 2), round(hi, 2)),
                "n": int(mask.sum()),
                "mean_score": mean_score,
                "pos_rate": pos_rate,
                "gap": gap,
            }
        )
    return ece, diag


def _brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(((scores - labels) ** 2).mean())


@pytest.fixture(scope="module")
def pairs() -> pd.DataFrame:
    return _load_pairs()


def test_positives_score_higher_than_negatives(pairs: pd.DataFrame) -> None:
    """Mean score of relevant pairs must exceed mean of non-relevant pairs."""
    pos_mean = pairs.loc[pairs["is_pos"], "score"].mean()
    neg_mean = pairs.loc[~pairs["is_pos"], "score"].mean()
    print(f"\n  pos mean score: {pos_mean:.4f}")
    print(f"  neg mean score: {neg_mean:.4f}")
    print(f"  gap:            {pos_mean - neg_mean:.4f}")
    assert pos_mean > neg_mean, (
        f"Positive mean {pos_mean:.4f} not above negative mean {neg_mean:.4f}"
    )


def test_expected_calibration_error_within_threshold(pairs: pd.DataFrame) -> None:
    """ECE on mMARCO-PT dev should be < ECE_THRESHOLD for release."""
    scores = pairs["score"].to_numpy()
    labels = pairs["is_pos"].to_numpy().astype(np.float64)
    ece, diag = _expected_calibration_error(scores, labels)
    brier = _brier_score(scores, labels)
    print(f"\n  ECE   = {ece:.4f} (threshold {ECE_THRESHOLD})")
    print(f"  Brier = {brier:.4f}")
    print(f"  reliability diagram (bin, n, mean_score, pos_rate, gap):")
    for row in diag:
        if row["n"] == 0:
            print(f"    {row['bin']}: empty")
        else:
            print(
                f"    {row['bin']}: n={row['n']:>7} "
                f"score={row['mean_score']:.3f} pos_rate={row['pos_rate']:.4f} gap={row['gap']:.3f}"
            )
    assert ece < ECE_THRESHOLD, f"ECE={ece:.4f} above threshold {ECE_THRESHOLD}"


def test_score_distribution_has_separation(pairs: pd.DataFrame) -> None:
    """Fraction of positives above the negative 90th-percentile score must exceed 30 %."""
    pos_scores = pairs.loc[pairs["is_pos"], "score"].to_numpy()
    neg_scores = pairs.loc[~pairs["is_pos"], "score"].to_numpy()
    threshold = np.quantile(neg_scores, 0.90)
    pos_above_p90 = float((pos_scores > threshold).mean())
    print(f"\n  neg score 90th percentile: {threshold:.4f}")
    print(f"  fraction of positives above neg-p90: {pos_above_p90:.3f}")
    assert pos_above_p90 > 0.30, (
        f"Only {pos_above_p90:.3f} of positives exceed the negative 90th percentile"
    )
