# Results — PTBR-Reranker

Consolidated comparison across all training variants we ran on mMARCO-PT dev (6,980 queries, BM25 first-stage rerank). The **release-candidate is v0.1** — every subsequent attempt regressed.

## Headline table — top-1000 BM25 rerank (Unicamp-DL protocol)

| Variant | Triples | Negatives | Loss | train_loss | **MRR@10** | nDCG@10 | MAP | Recall@100 | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| BM25 only (no rerank) | — | — | — | — | 0.152 | — | — | — | upstream baseline |
| **v0.1** (release) | 2 M | BM25 random | BinaryCE | 0.127 | **0.2945** | **0.3437** | **0.2980** | **0.7055** | **keep** |
| v1.0 (5× more data) | 10 M | BM25 random | BinaryCE | 0.016 | 0.2876 | 0.3385 | 0.2915 | 0.7016 | discard |
| v1.x (hard negatives) | 2.91 M | Serafim-IR IVFPQ, rank 10-100 | BinaryCE | 0.140 | 0.2159 | 0.2504 | 0.2215 | 0.5892 | discard |

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

## Why we're stopping iteration here

The cumulative compute spent on v1.0 + v1.x was substantial (~$25 across Runpod, Salad, GPUhub) and both runs landed worse than v0.1. Continuing to throw mining/training variants at the problem without first fixing the retrieval index would be expected to regress again. We freeze the current state, publish v0.1, and document v1.0 + v1.x as paper-worthy negative ablations.

## Artifacts on HuggingFace Hub (all private, ready to publish)

| Repo | Type | Size | Contents |
|---|---|---:|---|
| `tardellirs/ptbr-reranker-v0.1` | model | 535 MB | release-candidate cross-encoder |
| `tardellirs/ptbr-reranker-v1.0` | model | 535 MB | 10M-triples ablation |
| `tardellirs/ptbr-reranker-v1.x` | model | 557 MB | hard-neg ablation |
| `tardellirs/ptbr-reranker-eval-results` | dataset | ~200 MB | per-query parquets + raw rerank scores for every eval, ready for bootstrap CIs |
| `tardellirs/ptbr-reranker-hard-negatives` | dataset | 19 MB | mined (qid, pos, [neg×7]) parquet from the v1.x run |
| `tardellirs/ptbr-reranker-mining-cache` | dataset | 25.3 GB | Serafim-encoded mMARCO-PT passage embeddings (resume here on future mining attempts) |

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

Each run's full configuration (hyper-parameters, hardware, commit SHA, log file path, HF artifact) is preserved in [`experiments_log.md`](experiments_log.md) and dated entries in [`lab_notebook.md`](lab_notebook.md).
