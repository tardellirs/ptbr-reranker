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
- Added `notebooks/kaggle_phase1_validation.ipynb` and `notebooks/README.md` — a free-tier Kaggle Kernel that clones the repo, runs the small-mode mMARCO-PT download, validates the manifest, and executes the slow-marked Albertina smoke test in CPU. Validates the entire Phase 1 pipeline without paid GPU.

## [0.1.0] - 2026-05-18

Initial repository scaffold (no trained model yet).
