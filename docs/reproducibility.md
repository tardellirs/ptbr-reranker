# Reproducibility — ACL/NeurIPS-style checklist

Documento vivo. Atualizar durante o desenvolvimento, não no final.

## Checklist

### Code and resources
- [ ] Source code is publicly available (GitHub repo link in README).
- [ ] License is specified (MIT).
- [ ] Trained model is available on Hugging Face Hub.
- [ ] Mined hard-negatives dataset is available as a separate HF Datasets repo.

### Hyperparameters
- [ ] All hyperparameters reported in `configs/train_*.yaml` (committed).
- [ ] Exact final hyperparameters reproduced from a single config (no manual overrides).
- [ ] Range of hyperparameters explored documented in `docs/experiments_log.md`.

### Seeds and determinism
- [ ] Python / NumPy / PyTorch / CUDA seeds set explicitly (via `src.train.set_seed`).
- [ ] Number of runs and aggregated variance reported (not best-of-N alone).
- [ ] Mining seed for hard negatives also reported.

### Compute
- [ ] Hardware exact: GPU model, driver, CUDA, PyTorch versions.
- [ ] Total training time and cost reported.
- [ ] Carbon footprint reported (via `codecarbon`).

### Data
- [ ] Dataset versions pinned: mMARCO-PT commit SHA, MIRACL-PT commit SHA.
- [ ] All preprocessing steps in `data/*.py` and reproducible from raw downloads.
- [ ] Train/dev/test splits unchanged from upstream sources.
- [ ] No overlap between training and evaluation queries (validated by `data/download_mmarco.py --check`).

### Evaluation
- [ ] Standard metrics (MRR@10, nDCG@10, Recall@1000) computed with `pytrec_eval`.
- [ ] At least 2 strong baselines compared (Serafim-IR bi-encoder + BGE-reranker-v2-m3).
- [ ] Statistical significance via paired bootstrap (n=1000), p-values reported.
- [ ] Per-domain stratification (not just aggregate metric).

### Documentation
- [ ] README provides 5-line quickstart that works on a clean environment.
- [ ] Model card on HF Hub follows the official template, including limitations.
- [ ] Quality testing protocol documented in `docs/quality-tests.md`.

### Reproducibility script
- [ ] `scripts/reproduce.sh` runs the full pipeline end-to-end.
- [ ] Validated on a clean Runpod instance against the reported metrics (±0.5pp).

## Exact versions used in published model

To be filled at v1.0.0 release:

- `transformers`: TBD
- `sentence-transformers`: TBD
- `torch`: TBD
- CUDA: TBD
- GPU: TBD
- mMARCO-PT commit SHA (refs/convert/parquet): `d2da87d4433168219522a69ef38c30de16bbce80` _(resolved 2026-05-18 in Phase 1 small-mode validation; will re-resolve before full-mode training)_
- MIRACL commit SHA: N/A — MIRACL has no Portuguese subset. Cross-domain evaluation venue TBD.
- Base model (Albertina-100m) commit SHA: TBD
