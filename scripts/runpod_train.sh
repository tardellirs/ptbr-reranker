#!/usr/bin/env bash
# Full training pipeline on a Runpod instance — thin wrapper around
# scripts/reproduce.sh that asserts the GPU/secrets environment.
#
# Assumes: A100 80GB SXM, Python 3.10+, CUDA 12.x, persistent volume at /workspace.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/ptbr-reranker}"
CONFIG="${CONFIG:-configs/train_hardneg.yaml}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set}"

cd "$REPO_DIR"

# Install dev + gpu extras once.
python -m pip install --upgrade pip
pip install -e ".[dev,gpu]"

# Delegate to the canonical pipeline.
exec scripts/reproduce.sh --config "$CONFIG"
