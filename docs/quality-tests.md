# Quality Testing Protocol

The PTBR-Reranker undergoes a structured quality battery before any public release. The protocol covers five dimensions, each with explicit release thresholds.

## Release criteria

A new version is published to the Hugging Face Hub **only** if it passes:

| Criterion | Threshold |
|---|---|
| mMARCO-PT MRR@10 over Serafim-IR alone | ≥ +3 pp |
| MIRACL-PT nDCG@10 over BGE-reranker-v2-m3 | strictly higher |
| Expected Calibration Error (ECE) | < 0.10 |
| Robustness drop (no accents) | < 10% |
| Robustness drop (typos 1-3 chars) | < 15% |
| Robustness drop (case variation) | < 5% |
| Curated qualitative cases passing | ≥ 80% |
| End-to-end pipeline latency (A100, top-100) | < 200 ms |

## Dimensions

### 5.1 Score calibration

- Verify ordering: relevant > related-but-not-relevant > random.
- Plot score distribution histograms (positives vs. negatives).
- Reliability diagram with 10 buckets; precision per bucket should track the score.
- Report ECE.

### 5.2 Robustness to noise

200 dev queries perturbed:
- Typos at 1, 2, 3 character positions.
- Accent removal (very common in informal PT-BR).
- Mixed case.
- Common abbreviations (vc, pq, tb, td, qd).

Metric: relative MRR@10 drop versus original queries.

### 5.3 Bias and generalization

- **PT-BR vs PT-PT**: 100 queries rewritten in European variant.
- **Demographic invariance**: paired queries with neutral content, varying gender/region terms.
- **Per-domain stratification**: MRR@10 split by approximate domain.

### 5.4 Qualitative PT-BR cases (curated)

50 hand-curated cases:
- Brazilian legal text (jurisprudência STF/STJ/TST).
- Clinical PT-BR (CID-10 PT, drug nomenclature).
- Slang and colloquialisms → formal content.
- Polysemy (banco, manga, sede).
- Explicit negation in queries.

Rubric: top-3 contains a relevant passage (binary) plus qualitative comment.

### 5.5 End-to-end integration

Pytest fixtures with a 10k-passage in-memory corpus:
- Reranker improves MRR vs bi-encoder alone (aggregate).
- Deterministic with same seed.
- Latency under 200 ms on A100.

## Where the code lives

- `tests/test_quality_calibration.py`
- `tests/test_quality_robustness.py`
- `tests/test_quality_bias.py`
- `tests/test_quality_qualitative.py`
- `tests/test_integration.py`

Run the full battery with:

```bash
pytest tests/ -v -m quality   # quality tests only
pytest tests/ -v              # everything
```
