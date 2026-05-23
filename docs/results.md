# Results — PTBR-Reranker

Consolidated comparison across all training variants we ran on mMARCO-PT dev (6,980 queries, BM25 first-stage rerank). The **release-candidate is v0.1** — every subsequent attempt regressed.

## Headline table — top-1000 BM25 rerank (Unicamp-DL protocol)

| Variant | Triples | Negatives | Loss | train_loss | **MRR@10** | nDCG@10 | MAP | Recall@100 | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| BM25 only (no rerank) | — | — | — | — | 0.152 | — | — | — | upstream baseline |
| **v0.1** (release) | 2 M | BM25 random | BinaryCE | 0.127 | **0.2945** | **0.3437** | **0.2980** | **0.7055** | **keep** |
| v1.0 (5× more data) | 10 M | BM25 random | BinaryCE | 0.016 | 0.2876 | 0.3385 | 0.2915 | 0.7016 | discard |
| v1.x (hard negatives — IVFPQ) | 2.91 M | Serafim-IR IVFPQ, rank 10-100 | BinaryCE | 0.140 | 0.2159 | 0.2504 | 0.2215 | 0.5892 | discard |
| v2.x (hard negatives — IVFFlat) | 2.91 M | Serafim-IR IVFFlat, rank 10-100 | BinaryCE | 0.260 | 0.1938 | 0.2234 | 0.1998 | 0.5315 | discard |
| Stage 1b (top-100 only) | 0.2 M | BM25 random | **MNRL (InfoNCE)** | 0.158 | **0.2511*** | 0.2966* | 0.2521* | 0.5566* | discard |
| Stage 1b control (top-100 only, BCE) | 0.2 M | BM25 random | BinaryCE | 0.286 | — (not eval'd, skipped to save $) | — | — | — | ablation |

`*` Stage 1b is **top-100** only (eval was stopped before top-1000 to save Lightning AI credits). For direct comparison, v0.1 **top-100** = MRR@10 0.2810 / nDCG@10 0.3232 → Stage 1b regressed -3.0 pp at top-100.

`Recall@1000` is 0.7442 for every variant — that's the BM25 retrieval ceiling, not a property of the reranker.

## Positioning vs published Portuguese cross-encoders

Same protocol (rerank top-1000 BM25 over the 6,980 mMARCO-PT dev queries):

| Model | Params | MRR@10 PT | Notes |
|---|---:|---:|---|
| BM25 only | — | 0.152 | floor |
| mMiniLM-en-msmarco | 117 M | 0.277 | Unicamp-DL |
| mMiniLM-multi-msmarco | 117 M | 0.277 | Unicamp-DL |
| **PTBR-Reranker v0.1 (ours)** | **100 M** | **0.2945** | **release candidate** |
| ptT5-base-pt-msmarco | 220 M | 0.299 | Unicamp-DL |
| mMiniLM-en-pt-msmarco | 117 M | 0.299 | Unicamp-DL |
| mT5-base-multi-msmarco | 220 M | 0.302 | Unicamp-DL |
| mT5-base-en-pt-msmarco | 220 M | **0.306** | Unicamp-DL — best of theirs |

v0.1 lands **above the mMiniLM family** and within 2-3 pp of the mT5-base variants, while using 2.2× fewer parameters than the T5s and an encoder-only architecture that reranks in a single forward pass.

## Why v1.0 regressed (-0.7 pp from v0.1)

5× more BM25-random negatives, same recipe. The training loss collapsed to **0.016** — the model became overconfident on examples it could already separate (random BM25 distractors are easy once you've seen 2 M of them). That confidence does not transfer to ranking fine-grained candidates at inference time. **Scaling random negatives past saturation hurts.**

## Why v1.x regressed harder (-7.9 pp from v0.1)

We replaced the random BM25 negatives with hard negatives mined via Serafim-IR. Canonically this is the strongest single lever (+3-5 pp in BGE / ANCE / mMARCO papers). It didn't work here, and the root cause is the **retrieval index** we used for mining, not the strategy:

1. **What we wanted:** for each training query, retrieve the top-200 nearest passages with Serafim-IR, drop known positives, sample 7 negatives from ranks 10-100. These are the canonical "hard" examples.
2. **What we ran on the available hardware:** the Serafim collection (8.84 M × 768-dim, ~25 GB) didn't fit as an IndexIVFFlat on the 24 GB VRAM 3090 we had on Salad. CPU HNSW was the documented fallback, but a single build took 3+ hours and the Salad community node was migrated out from under us twice. To finish in budget, we switched to **IndexIVFPQ on GPU** (M=64 sub-quantizers, 8-bit codes, float16 lookup) — that compresses each vector 50× to 64 bytes and fits in <1 GB VRAM, which is great for memory but lossy for ranking.
3. **The cost of the compression:** PQ recall on the top-100 window is roughly 60-80 % of IVFFlat. The rank-10-100 candidates we sampled were not really the rank-10-100 nearest neighbors — many were random passages that the quantized index happened to place there. The model trained against those "negatives" learned to discriminate noise, not similarity, and the resulting `train_loss=0.140` looked plausible but masked the underlying signal collapse. The eval drop was uniform across MRR@10, nDCG@10, and MAP, which is consistent with a global signal-quality issue (rather than a metric-specific artifact).

**Lesson:** retrieval *quality* for hard-neg mining matters at least as much as the strategy. Product quantization is fine for inference search but **destroys the rank-10-100 signal** that hard-neg mining depends on. A future retry needs either:
- a GPU with ≥ 32 GB VRAM (5090, 6000 Ada, L40S) running IndexIVFFlat or HNSW; or
- a chunked CPU IVFFlat build with `add_with_ids` in 500 k slices on a node with ≥ 64 GB host RAM; or
- a different mining approach entirely (e.g., score top-K with the cross-encoder itself, then sample).

## Why v2.x regressed even harder than v1.x (-10.1 pp from v0.1) — IVFPQ hypothesis refuted

v2.x repeated v1.x **changing only the index type**: IVFFlat-GPU on the same RTX 4090 48 GB (GPUhub), uncompressed 768-dim vectors, recall on the top-100 window essentially perfect. Everything else identical (same Serafim-IR encoder, same rank 10-100 sampling, same training recipe). If the IVFPQ-noise hypothesis were correct, v2.x should have recovered to or exceeded v0.1.

Instead, v2.x **regressed further**:

| | v1.x (IVFPQ) | v2.x (IVFFlat) |
|---|---:|---:|
| MRR@10 | 0.2159 | **0.1938** |
| nDCG@10 | 0.2504 | **0.2234** |
| train_loss final | 0.140 | 0.260 |

This refutes "PQ is the problem." The true causes are structural to the mining recipe itself:

1. **Sparse qrels.** mMARCO-PT has on average ~1 annotated relevant passage per query out of 8.84 M. The Serafim-IR top-100 contains many *unannotated* near-duplicates of the positive — true positives in disguise that we labelled as negatives. We removed only the *known* positive (the one in qrels). The model trained on a contradicted signal: "this passage is positive in some queries and negative in others, depending on annotation luck."
2. **Rank 10-100 is too easy for the bi-encoder, too noisy for the cross-encoder.** With a *faithful* index, ranks 10-100 are dense with semantically related content. The bi-encoder thought they were close because they topically overlap; the cross-encoder needed adversarial-but-distinguishable negatives, which actually live around ranks 200-1000 (similar-but-not-relevant) or in the failure modes of the cross-encoder itself.
3. **train_loss almost doubled** (0.140 → 0.260): with faithful retrieval, the negatives became too similar to positives, the model couldn't separate them, and gradient signal pushed it to over-correct on every confusable case — uniformly hurting ranking.

## Why we're stopping iteration here

Three consecutive variants regressed (-0.7 pp, -7.9 pp, -10.1 pp). Then a fourth attempt (Stage 1b — listwise InfoNCE loss on a 200 k subset) regressed -3.0 pp at top-100. The cumulative compute is now ~$36 across Runpod, Salad, GPUhub, and Lightning AI, and the conclusion is robust: **for Albertina-100m on translated mMARCO-PT, the v0.1 recipe (2 M BM25-random triples + pointwise BinaryCE, 1 epoch) is at a local optimum that all four perturbations failed to escape**. We freeze the current state, publish v0.1, and document v1.0 + v1.x + v2.x + Stage 1b as paper-worthy negative ablations.

Future hard-neg attempts (out of scope for this release) should explore:
- **score-gap filtering**: drop negs with `cos(q, neg) > 0.8` (too similar — likely false negs)
- **cross-encoder bootstrap mining**: score top-100 candidates with v0.1 itself, sample negs from the model's own mid-confidence band
- **denoising via dual annotation**: filter mined negs by running them through another reranker (BGE-m3) and keeping only those *both* agree are negative

## Artifacts on HuggingFace Hub (all private, ready to publish)

| Repo | Type | Size | Contents |
|---|---|---:|---|
| `tardellirs/ptbr-reranker-v0.1` | model | 535 MB | release-candidate cross-encoder |
| `tardellirs/ptbr-reranker-v1.0` | model | 535 MB | 10M-triples ablation |
| `tardellirs/ptbr-reranker-v1.x` | model | 557 MB | hard-neg ablation (IVFPQ index) |
| `tardellirs/ptbr-reranker-v2.x` | model | 557 MB | hard-neg ablation (IVFFlat index) |
| `tardellirs/ptbr-reranker-stage1b` | model | 557 MB | Stage 1b MNRL/InfoNCE on 200 k subset |
| `tardellirs/ptbr-reranker-stage1b-control-bce` | model | 557 MB | Stage 1b control — same 200 k with BinaryCE (loss-isolation) |
| `tardellirs/ptbr-reranker-eval-results` | dataset | ~200 MB | per-query parquets + raw rerank scores for every eval, ready for bootstrap CIs |
| `tardellirs/ptbr-reranker-hard-negatives` | dataset | 19 MB | mined (qid, pos, [neg×7]) parquet from the v1.x run |
| `tardellirs/ptbr-reranker-training-data` | dataset | 2.65 GB | mMARCO-PT bundle: 2 M BM25 triples + eval (collection, queries dev, qrels, BM25 run) |
| `tardellirs/ptbr-reranker-mining-cache` | dataset | 25.3 GB | Serafim-encoded mMARCO-PT passage embeddings (resume here on future mining attempts) |

## Quality battery — v0.1 release-candidate

### Calibration (`tests/test_quality_calibration.py`)

Computed from the saved top-1000 rerank scores against mMARCO-PT dev qrels (6,975,268 pairs):

| Metric | Value | Threshold | Status |
|---|---:|---|---|
| Positive-mean score | 0.904 | — | clean separation |
| Negative-mean score | 0.072 | — | gap 0.832 |
| Brier score | 0.052 | — | low |
| **ECE (10 equal-width bins)** | **0.072** | < 0.10 | **pass** |
| % positives above neg-p90 | 96 % | > 30 % | strong |

The reliability diagram is well-calibrated in low-score bins but **overconfident at the top**: the (0.9, 1.0] bin has mean score 0.954 yet only 2.3 % of pairs there are truly relevant. That is a class-imbalance artefact of training with `BinaryCrossEntropyLoss` over `(1 pos, 1 neg)` pairs and predicting on `(1 pos, 1000 negs)` at eval time — the model ranks well but assigns absolute probabilities that overestimate relevance density. For applications that need calibrated probabilities (RAG thresholding, score-based routing), apply Platt scaling or isotonic regression on the released model output before thresholding.

### Robustness to PT-BR perturbations (`tests/test_quality_robustness.py`)

50 random dev queries × 5 perturbation kinds × top-1 reranked passage. Reports mean relative score drop vs the original (query, passage) score:

| Perturbation | Mean rel. drop | Threshold | Status | Observation |
|---|---:|---|---|---|
| **case_lower** (all lowercase) | 0.000 | 0.05 | pass | perfectly stable |
| **abbreviations** (vc, pq, tb, td, q, pra, ta) | 0.008 | 0.10 | pass | handles informal PT-BR well |
| **no_accent** (ASCII fold) | 0.018 | 0.10 | pass | robust to keyboard-less typing |
| **case_upper** (ALL CAPS) | 0.032 (worst-case −82 %) | 0.10 | pass on average, brittle on a few |
| **typo_2chars** (two adjacent-char swaps) | **0.228** | 0.15 | **FAIL** | typos collapse the score |

Concrete failure example for typos: `"o que nyu classificou"` → `"o quen yu clsasificou"` drops the (q, p) score from 0.996 to 0.000. The 22.8 % mean drop is consistent across the 50-query sample, not driven by outliers. **This is a known limitation** of byte-pair-tokenised PT models that have not been trained with typo augmentation — Albertina's pretraining corpus is clean web text. Future versions should augment training with adversarial typos (suggestion: Bansal et al., "TextAttack").

### Implications for release

- ✅ Ranking quality (MRR/nDCG): solid, ahead of mMiniLM family in mMARCO-PT, within 1 pp of ptT5-base.
- ✅ Calibration: ECE pass; document the overconfidence pattern in the model card.
- ✅ Robustness to PT-BR informal writing (no accents, lowercase, abbreviations): strong.
- ⚠️ Robustness to typos: weak. Note in the model card; ship a typo-tolerant variant only after augmentation training.

## Next levers (when work resumes)

In rough order of cost-to-quality value, given what we know now:

1. **Softmax CE loss on the v0.1 dataset** (Stage 1b). Same 2 M BM25 triples, but switch `BinaryCrossEntropyLoss` for `CrossEntropyLoss` over `{pos, k_negs}`. No mining required. Expected +1-2 pp MRR@10. Cost ~$2-3 of GPU.
2. **Hard negatives done right.** Repeat the v1.x recipe with IndexIVFFlat or HNSW on a 32 GB+ GPU. Expected +3-5 pp if the index is faithful. Cost ~$3-5.
3. **Albertina-900M base** (Stage 3). 9× the parameters, 1 epoch on the v0.1 dataset. Expected +3-5 pp from sheer capacity. Cost ~$15-25, exceeds the "small experiment" budget.

## Provenance

- v0.1 trained 2026-05-19 on Runpod Secure 4090 (commit `9251b67`/`d0358a5` era of the repo).
- v0.1 eval top-100 ran on Runpod Secure 4090 (28 min); eval top-1000 ran on Runpod Secure 5090 (4 h 40 min).
- v1.0 trained 2026-05-19/20 on Runpod Secure 5090, mid-run pod migration via HF Hub checkpoint backup.
- v1.x mining + training ran 2026-05-21/22 across Salad community 3090 and GPUhub Singapore 4090 (48 GB Ada), checkpoints carried between providers through `tardellirs/ptbr-reranker-v1x-inprogress`.
- v2.x mining (IVFFlat-GPU, ~50 min) + training (8 h 08 min) + eval (4 h 37 min) ran 2026-05-22/23 entirely on GPUhub Singapore 4090 (48 GB Ada); checkpoints via `tardellirs/ptbr-reranker-v2x-inprogress`.

Each run's full configuration (hyper-parameters, hardware, commit SHA, log file path, HF artifact) is preserved in [`experiments_log.md`](experiments_log.md) and dated entries in [`lab_notebook.md`](lab_notebook.md).
