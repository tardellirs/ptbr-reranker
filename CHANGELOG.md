# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffolding: pyproject.toml, ruff/mypy/pytest config, pre-commit hooks.
- Repository governance: LICENSE (MIT), CITATION.cff, CONTRIBUTING.md, CODE_OF_CONDUCT.md.
- GitHub Actions workflows (CI, model card publishing, release).
- Module skeletons: `src/model.py`, `src/train.py`, `src/eval_mmarco.py`, `src/eval_miracl.py`, `src/rerank.py`, `src/stats.py`.
- Statistical utilities: `bootstrap_metric` and `paired_bootstrap_pvalue` in `src/stats.py` (full implementation + 96% test coverage).
- Data pipeline skeletons: `data/download_mmarco.py`, `data/mine_hard_negatives.py`, `data/build_triples.py`.
- Training configs: `configs/train_baseline.yaml`, `configs/train_hardneg.yaml`.
- Quality testing battery skeletons in `tests/`.
- Scientific cataloging infrastructure: `docs/lab_notebook.md`, `docs/experiments_log.md`, `docs/reproducibility.md`, `docs/model-card.md`.
- Paper LaTeX skeleton in `paper/`.
- Gradio demo skeleton in `space/` and usage examples in `examples/`.

### Fixed
- Removed extraneous f-string prefix in `examples/rag_pipeline.py`.
- `src/model.py` type annotations updated so `mypy --strict` passes.
- Applied `ruff format` repo-wide.

### Changed
- `data/download_mmarco.py` implemented end-to-end: pulls mMARCO Portuguese splits and MIRACL-PT from Hugging Face Datasets, persists parquet snapshots, records the HF revision SHA in a per-target manifest, and validates row counts in `--full` and `--small` modes.
- Added `tests/test_data_pipeline.py` covering the manifest/validation logic (no network) and a slow-marked smoke test that loads Albertina-100m through `sentence_transformers.CrossEncoder` to confirm the model path before any training run.
- Renamed all GitHub URLs and Hugging Face repo IDs from `stekel/*` to `tardellirs/*` to match the maintainer's actual handles. The institutional email `stekel@ifsp.edu.br` is unchanged.
- Added `notebooks/kaggle_phase1_validation.ipynb` and `notebooks/README.md` — a free-tier Kaggle Kernel that clones the repo, runs the small-mode mMARCO-PT download, validates the manifest, and executes the slow-marked Albertina smoke test in CPU. Validates the entire Phase 1 pipeline without paid GPU. **Validated end-to-end on 2026-05-18** — kernel public at https://www.kaggle.com/code/tardellistekel/ptbr-reranker-phase-1-validation.
- Refactored `data/download_mmarco.py` to fetch parquet shards directly from the `refs/convert/parquet` HF revision, bypassing the legacy mmarco.py loader script (removed in datasets>=4.0).
- Smoke test `test_albertina_loads_and_predicts` now pins `torch_dtype=float32` to work around Kaggle CPU image's bfloat16 default that mixes badly with float32 inputs in DeBERTa attention.
- Removed MIRACL from the pipeline: `miracl/miracl` has no Portuguese subset. Cross-domain evaluation venue captured as a TODO in `docs/lab_notebook.md`.
- Recorded resolved mMARCO-PT revision SHA in `docs/reproducibility.md`.
- Pivoted cross-domain evaluation: MIRACL has no PT subset, replaced with Quati (`unicamp-dl/quati`, native PT-BR web from ClueWeb22-pt, primary) and JurisTCU (`LeandroRibeiro/JurisTCU`, Brazilian legal jurisprudence, domain-shift probe). Renamed `src/eval_miracl.py` → `src/eval_quati.py` (git mv preserves history) and added `src/eval_juristcu.py`.
- Phase 2 implementation (Tier 2): `data/mine_hard_negatives.py` now functional — Serafim-IR encoding, FAISS HNSW index, top-K search, qrels filtering, rank-windowed sampling, parquet output.
- `data/build_triples.py` functional — consumes (a) the official MS MARCO `triples.train.ids.small.tsv` (39M BM25-mined triples, our baseline) and/or (b) mined hard negatives from Phase 2, weighted via `mix_ratio_official`. Outputs full text triples ready for `src/train.py`.
- `data/download_mmarco.py` extended to pull qrels (`data/qrels.dev.small.tsv`) and the official training triples from the mMARCO main branch (TSV, language-agnostic, IDs shared across translations).
- Bibliography updated with Quati (arXiv:2404.06976) and JurisTCU (arXiv:2503.08379) entries.

## [0.1.0] - 2026-05-18

Initial repository scaffold (no trained model yet).
