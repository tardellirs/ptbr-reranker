---
language:
  - pt
license: mit
library_name: sentence-transformers
pipeline_tag: text-ranking
tags:
  - sentence-transformers
  - cross-encoder
  - reranker
  - information-retrieval
  - portuguese
  - brazilian-portuguese
  - pt-br
  - mteb
datasets:
  - unicamp-dl/mmarco
  - unicamp-dl/quati
  - LeandroRibeiro/JurisTCU
base_model: PORTULAN/albertina-100m-portuguese-ptbr-encoder
model-index:
  - name: cross-encoder-albertina-ptbr-mmarco
    results:
      - task:
          type: text-ranking
          name: Passage Reranking
        dataset:
          name: mMARCO-PT (dev)
          type: unicamp-dl/mmarco
          config: portuguese
          split: dev
        metrics:
          - type: mrr_at_10
            value: 0.2945
          - type: ndcg_at_10
            value: 0.3437
          - type: map
            value: 0.2980
          - type: recall_at_100
            value: 0.7055
          - type: recall_at_1000
            value: 0.7442
      - task:
          type: text-ranking
          name: Passage Reranking
        dataset:
          name: Quati 1M
          type: unicamp-dl/quati
          config: "1M"
        metrics:
          - type: ndcg_at_10
            value: TBD
          - type: recall_at_100
            value: TBD
      - task:
          type: text-ranking
          name: Passage Reranking (Legal)
        dataset:
          name: JurisTCU
          type: LeandroRibeiro/JurisTCU
        metrics:
          - type: ndcg_at_10
            value: TBD
          - type: mrr_at_10
            value: TBD
---

# PTBR-Reranker (Albertina-100m / mMARCO-PT)

A cross-encoder reranker for Brazilian Portuguese, designed for the second stage of a retrieval pipeline (after a bi-encoder retriever such as Serafim-IR).

## Model description

This is a cross-encoder built on top of [PORTULAN/albertina-100m-portuguese-ptbr-encoder](https://huggingface.co/PORTULAN/albertina-100m-portuguese-ptbr-encoder), fine-tuned on mMARCO-PT with hard negatives mined using [PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir](https://huggingface.co/PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir).

It takes a `(query, passage)` pair and returns a single relevance score.

## Intended uses

- Reranking the top-K passages from a first-stage retriever in a Portuguese-language RAG or search system.
- Reranking BM25 outputs for Portuguese passage retrieval.

## Limitations and bias

- Training data (mMARCO-PT) is **machine-translated** from English MS MARCO. Quality on natively-Brazilian content (legal, clinical, governmental) is assessed via a curated qualitative battery, but residual bias from translation may exist.
- Maximum input length is 256 tokens. For longer documents, consider chunking.
- The base model (Albertina) was trained primarily on Brazilian Portuguese text; performance on European Portuguese (PT-PT) may be lower.

## Training data

- [`unicamp-dl/mmarco`](https://huggingface.co/datasets/unicamp-dl/mmarco), subset `portuguese`.
- Hard negatives mined from the mMARCO-PT collection using Serafim-100m-IR, sampling 7 negatives per query from ranks 10–100.

## Training procedure

- **Optimizer**: AdamW, lr 2e-5, weight decay 0.01.
- **Schedule**: 10% warmup, linear decay.
- **Batch size**: 64 effective (32 per device, gradient accumulation 2).
- **Loss**: cross-entropy softmax over the set `{positive, 7 hard negatives}`.
- **Max length**: 256.
- **Precision**: bf16.
- **Hardware**: 1× NVIDIA A100 SXM 80GB.
- **Compute**: TBD GPU-hours.
- **Seed**: 42.

## Evaluation

| Model | mMARCO-PT MRR@10 | mMARCO-PT nDCG@10 | Quati nDCG@10 | JurisTCU nDCG@10 |
|---|---|---|---|---|
| BM25 | TBD | TBD | TBD | TBD |
| Serafim-IR (bi-encoder) | TBD | TBD | TBD | TBD |
| mMiniLM-L12 (multilingual) | TBD | TBD | TBD | TBD |
| BGE-reranker-v2-m3 | TBD | TBD | TBD | TBD |
| **PTBR-Reranker (this)** | **TBD** | **TBD** | **TBD** | **TBD** |

Statistical significance reported via paired bootstrap (n=1000) with 95% CIs.

## Quality testing

This model has been evaluated on a structured battery covering calibration (ECE), robustness (accents, typos, case, abbreviations), bias (PT-BR vs PT-PT, demographic invariance), and curated qualitative PT-BR cases (legal, clinical, slang, polysemy, negation). See [`docs/quality-tests.md`](https://github.com/tardellirs/ptbr-reranker/blob/main/docs/quality-tests.md) for the full protocol.

## Environmental impact

Trained on Runpod Community A100 SXM. Carbon emissions tracked with [`codecarbon`](https://codecarbon.io/):
- Total energy: TBD kWh
- Estimated CO2eq: TBD kg

## Usage

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("tardellirs/cross-encoder-albertina-ptbr-mmarco")

query = "qual é a capital do Brasil?"
passages = [
    "Brasília é a capital federal do Brasil desde 1960.",
    "São Paulo é a maior cidade do Brasil.",
]
scores = model.predict([(query, p) for p in passages])
```

## Citation

```bibtex
@software{ptbr_reranker_2026,
  author = {Stekel},
  title = {PTBR-Reranker: A Brazilian Portuguese Cross-Encoder for Passage Reranking},
  year = {2026},
  url = {https://github.com/tardellirs/ptbr-reranker},
  publisher = {Hugging Face}
}
```

## References

- **Albertina**: Rodrigues et al., "Advancing Neural Encoding of Portuguese with Transformer Albertina PT-\*". [arXiv:2403.01897](https://arxiv.org/abs/2403.01897)
- **Serafim**: Rodrigues et al., "Open Sentence Embeddings for Portuguese with the Serafim PT* encoders family". [arXiv:2407.19527](https://arxiv.org/abs/2407.19527)
- **mMARCO**: Bonifacio et al., "mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset". [arXiv:2108.13897](https://arxiv.org/abs/2108.13897)
- **Quati**: Bonifacio et al., "Quati: A Brazilian Portuguese Information Retrieval Dataset". [arXiv:2404.06976](https://arxiv.org/abs/2404.06976)
- **JurisTCU**: Ribeiro et al., "JurisTCU: A Brazilian Legal Information Retrieval Benchmark". [arXiv:2503.08379](https://arxiv.org/abs/2503.08379)
