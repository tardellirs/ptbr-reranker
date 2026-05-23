"""Phase 5.2 — Robustness to PT-BR orthographic perturbations.

Test pattern: take a sample of dev queries, pair each with its top-1 reranked
passage from v0.1's saved eval, perturb the query (no accent, case mix, typos,
abbreviations), and measure the score delta. A robust reranker keeps the score
within ~10 % of the original.

Runs offline on CPU — uses v0.1 from the local HF cache + saved eval scores
to bound the candidate set.

The test is parametrised by perturbation kind so failures point to the
specific weakness.
"""

from __future__ import annotations

import random
import unicodedata
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.quality, pytest.mark.slow]

EVAL_RERANK = Path("outputs/v0.1_top1000/eval_mmarco_v0.1_top1000_per_query_rerank.parquet")
QUERIES = Path("data/raw/mmarco/queries_dev_portuguese.parquet")
COLLECTION = Path("data/raw/mmarco/collection_portuguese.parquet")
MODEL_REPO = "tardellirs/ptbr-reranker-v0.1"

SAMPLE_SIZE = 50  # queries
SCORE_DROP_THRESHOLD = {
    "no_accent": 0.10,        # PT-BR informal often drops accents
    "case_lower": 0.05,       # very common (all lowercase)
    "case_upper": 0.10,       # less common but plausible
    "typo_2chars": 0.15,      # 2 swap-typos
    "abbreviations": 0.10,    # vc, pq, tb, td, etc.
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _case_lower(s: str) -> str:
    return s.lower()


def _case_upper(s: str) -> str:
    return s.upper()


def _typo_2chars(s: str, rng: random.Random) -> str:
    """Swap two random adjacent characters, twice."""
    chars = list(s)
    for _ in range(2):
        if len(chars) < 4:
            break
        i = rng.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


_ABBR_MAP = {
    "você": "vc",
    "voce": "vc",
    "por que": "pq",
    "porque": "pq",
    "também": "tb",
    "tambem": "tb",
    "tudo": "td",
    "que": "q",
    "para": "pra",
    "está": "ta",
    "esta": "ta",
    "porra": "porra",  # unchanged sentinel
    "não": "naum",
}


def _abbreviations(s: str) -> str:
    out = s
    for full, abbr in _ABBR_MAP.items():
        out = out.replace(" " + full + " ", " " + abbr + " ")
        out = out.replace(full + " ", abbr + " ")
        out = out.replace(" " + full, " " + abbr)
    return out


PERTURBATIONS: dict[str, Callable[[str, random.Random], str]] = {
    "no_accent": lambda s, rng: _strip_accents(s),
    "case_lower": lambda s, rng: _case_lower(s),
    "case_upper": lambda s, rng: _case_upper(s),
    "typo_2chars": _typo_2chars,
    "abbreviations": lambda s, rng: _abbreviations(s),
}


@pytest.fixture(scope="module")
def model():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        pytest.skip("sentence-transformers not installed")
    try:
        return CrossEncoder(MODEL_REPO, device="cpu", max_length=256)
    except Exception as exc:
        pytest.skip(f"could not load model {MODEL_REPO}: {exc}")


@pytest.fixture(scope="module")
def sample_pairs() -> pd.DataFrame:
    """50 (qid, top-1 docid) pairs from v0.1's saved rerank."""
    if not EVAL_RERANK.exists() or not QUERIES.exists() or not COLLECTION.exists():
        pytest.skip("missing eval artifacts or raw mmarco — run eval_mmarco first")
    rng = random.Random(42)

    rerank = pd.read_parquet(EVAL_RERANK)
    # top-1 per query by score
    top1 = (
        rerank.sort_values(["qid", "score"], ascending=[True, False])
        .groupby("qid", as_index=False)
        .first()
    )
    sampled_qids = sorted(rng.sample(top1["qid"].tolist(), min(SAMPLE_SIZE, len(top1))))
    top1 = top1[top1["qid"].isin(sampled_qids)]

    queries = pd.read_parquet(QUERIES)
    qmap = dict(zip(queries["id"].astype(int), queries["text"]))
    # only load the passages we need
    needed = set(top1["docid"].astype(int).tolist())
    coll = pd.read_parquet(COLLECTION, columns=["id", "text"])
    coll = coll[coll["id"].isin(needed)]
    cmap = dict(zip(coll["id"].astype(int), coll["text"]))

    rows: list[dict] = []
    for _, r in top1.iterrows():
        qid, did, score = int(r["qid"]), int(r["docid"]), float(r["score"])
        q = qmap.get(qid)
        p = cmap.get(did)
        if q is None or p is None:
            continue
        rows.append({"qid": qid, "docid": did, "query": q, "passage": p, "score_saved": score})
    return pd.DataFrame(rows)


def _re_score(model, queries: list[str], passages: list[str]) -> np.ndarray:
    pairs = list(zip(queries, passages))
    scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
    return np.asarray(scores).astype(np.float64)


@pytest.mark.parametrize("perturbation_name", list(PERTURBATIONS.keys()))
def test_score_stable_under_perturbation(model, sample_pairs: pd.DataFrame, perturbation_name: str) -> None:
    """Score on perturbed query should stay within the threshold for that perturbation."""
    rng = random.Random(hash(perturbation_name) & 0xFFFFFFFF)
    fn = PERTURBATIONS[perturbation_name]
    passages = sample_pairs["passage"].tolist()

    # 1. score the ORIGINAL pair to anchor (saved score is from eval but recompute for fairness)
    orig_queries = sample_pairs["query"].tolist()
    orig_scores = _re_score(model, orig_queries, passages)

    # 2. score the PERTURBED pair
    pert_queries = [fn(q, rng) for q in orig_queries]
    pert_scores = _re_score(model, pert_queries, passages)

    # 3. average absolute drop normalised by original score (skip orig==0 to avoid div by zero)
    drops = []
    for o, p in zip(orig_scores, pert_scores):
        if o > 1e-3:
            drops.append((o - p) / o)
        else:
            drops.append(0.0)
    mean_rel_drop = float(np.mean(drops))
    median_rel_drop = float(np.median(drops))
    abs_drop = float(np.mean(orig_scores - pert_scores))

    threshold = SCORE_DROP_THRESHOLD[perturbation_name]
    print(
        f"\n  [{perturbation_name}] mean rel drop={mean_rel_drop:.3f} median={median_rel_drop:.3f} "
        f"abs={abs_drop:.4f} threshold={threshold}"
    )
    # show 3 worst cases for inspection
    pert_df = pd.DataFrame({
        "qid": sample_pairs["qid"],
        "orig_q": orig_queries,
        "pert_q": pert_queries,
        "orig_score": orig_scores,
        "pert_score": pert_scores,
        "rel_drop": drops,
    })
    worst = pert_df.nlargest(3, "rel_drop")[["qid", "orig_q", "pert_q", "orig_score", "pert_score", "rel_drop"]]
    print("  worst-3 cases:")
    for _, w in worst.iterrows():
        print(f"    qid={w['qid']}  '{w['orig_q'][:50]}' -> '{w['pert_q'][:50]}'  "
              f"{w['orig_score']:.3f} -> {w['pert_score']:.3f}  drop={w['rel_drop']:.3f}")

    assert mean_rel_drop < threshold, (
        f"[{perturbation_name}] mean relative drop {mean_rel_drop:.3f} exceeds {threshold}"
    )
