#!/usr/bin/env bash
# Full training pipeline on a Runpod instance.
# Assumes: A100 80GB SXM, Python 3.10+, CUDA 12.x, persistent volume at /workspace.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/ptbr-reranker}"
CONFIG="${CONFIG:-configs/train_hardneg.yaml}"
WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set}"

cd "$REPO_DIR"

# 1) Environment
python -m pip install --upgrade pip
pip install -e ".[dev,gpu]"

# 2) Data
python data/download_mmarco.py

# 3) Mining (~6h on A40; ~2-3h on A100)
python data/mine_hard_negatives.py --config "$CONFIG"

# 4) Build training triples
python data/build_triples.py \
    --hard-negatives data/processed/hard_negatives.parquet \
    --output data/processed/triples.parquet

# 5) Train (~30-40h on A100 SXM)
python src/train.py --config "$CONFIG"

# 6) Evaluate
python src/eval_mmarco.py --checkpoint runs/hardneg/best
python src/eval_miracl.py --checkpoint runs/hardneg/best

# 7) Quality battery
pytest tests/ -v -m quality --tb=short

# 8) (Optional) push to HF Hub if criteria pass
# python scripts/push_to_hub.py \
#     --checkpoint runs/hardneg/best \
#     --metrics outputs/eval_mmarco.json
