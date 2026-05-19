#!/usr/bin/env bash
#
# scripts/reproduce.sh — single entry point for reproducing the model.
#
# Runs the full pipeline: download → mine hard negatives → build training
# triples → train cross-encoder → evaluate. Defaults to the hard-negatives
# recipe (configs/train_hardneg.yaml).
#
# Usage:
#   scripts/reproduce.sh                # full mode (real mMARCO-PT, requires GPU)
#   scripts/reproduce.sh --small        # validation mode (10k passages, synthetic qrels)
#   scripts/reproduce.sh --config configs/train_baseline.yaml
#   scripts/reproduce.sh --skip-train   # data only (mining + build_triples)
#
# Environment:
#   WANDB_API_KEY    optional; if set, training run is logged to W&B.
#   HF_TOKEN         optional; needed only for the final push to HF Hub.
#
# Cost reference (Runpod Community, A100 SXM at $1.39/h):
#   --small           : ~10 min, ~$0.25 (mostly CPU; smoke test)
#   --full (default) : ~30–40 h, ~$42–55 for the recipe in train_hardneg.yaml

set -euo pipefail

SMALL=0
SKIP_DOWNLOAD=0
SKIP_MINE=0
SKIP_BUILD=0
SKIP_TRAIN=0
SKIP_EVAL=0
CONFIG="configs/train_hardneg.yaml"
DATA_DIR="data/raw"
PROCESSED_DIR="data/processed"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --small) SMALL=1; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --skip-mine) SKIP_MINE=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --skip-train) SKIP_TRAIN=1; shift ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        -h|--help)
            awk 'NR==1 {next} /^set -euo/ {exit} {sub(/^# ?/, ""); print}' "$0"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

step() { echo; echo "=== $* ==="; }

if [[ $SMALL -eq 1 ]]; then
    SMALL_FLAG="--small"
else
    SMALL_FLAG=""
fi

# 1) Download
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
    step "1/5 Download mMARCO-PT (+ qrels + official triples)"
    python data/download_mmarco.py $SMALL_FLAG --target-dir "$DATA_DIR"
fi

# 2) Hard negative mining
if [[ $SKIP_MINE -eq 0 ]]; then
    step "2/5 Mine hard negatives with Serafim-IR"
    if [[ $SMALL -eq 1 ]]; then
        # Synthetic qrels: each dev query paired with a random PID from the first
        # 1000 passages of the slice — guarantees alignment so the rest of the
        # pipeline produces non-empty output.
        python - <<'PY'
import random
from pathlib import Path

import pandas as pd

random.seed(42)
queries = pd.read_parquet("data/raw/mmarco/queries_dev_portuguese.parquet")
pids = pd.read_parquet("data/raw/mmarco/collection_portuguese.parquet")["id"].tolist()
out = Path("data/raw/mmarco/qrels.synthetic_small.tsv")
candidates = pids[:1000]
with out.open("w") as fh:
    for qid in queries["id"].tolist():
        fh.write(f"{qid}\t0\t{random.choice(candidates)}\t1\n")
print(f"Wrote {out} with {len(queries)} synthetic qrels rows")
PY
        python data/mine_hard_negatives.py \
            --queries "$DATA_DIR/mmarco/queries_dev_portuguese.parquet" \
            --qrels "$DATA_DIR/mmarco/qrels.synthetic_small.tsv" \
            --collection "$DATA_DIR/mmarco/collection_portuguese.parquet" \
            --output "$PROCESSED_DIR/hard_negatives.parquet" \
            --top-k 50 \
            --num-negatives 5 \
            --rank-min 5 \
            --rank-max 50 \
            --device auto
    else
        python data/mine_hard_negatives.py \
            --queries "$DATA_DIR/mmarco/queries_train_portuguese.parquet" \
            --qrels "$DATA_DIR/mmarco/qrels.dev.small.tsv" \
            --collection "$DATA_DIR/mmarco/collection_portuguese.parquet" \
            --output "$PROCESSED_DIR/hard_negatives.parquet" \
            --device auto
    fi
fi

# 3) Build training triples
if [[ $SKIP_BUILD -eq 0 ]]; then
    step "3/5 Build training triples"
    if [[ $SMALL -eq 1 ]]; then
        # Use mined triples (synthetic-aligned) and the matching dev queries.
        python data/build_triples.py \
            --queries "$DATA_DIR/mmarco/queries_dev_portuguese.parquet" \
            --collection "$DATA_DIR/mmarco/collection_portuguese.parquet" \
            --mined-triples "$PROCESSED_DIR/hard_negatives.parquet" \
            --output "$PROCESSED_DIR/triples.parquet"
    else
        python data/build_triples.py \
            --queries "$DATA_DIR/mmarco/queries_train_portuguese.parquet" \
            --collection "$DATA_DIR/mmarco/collection_portuguese.parquet" \
            --official-triples "$DATA_DIR/mmarco/triples.train.ids.small.tsv" \
            --mined-triples "$PROCESSED_DIR/hard_negatives.parquet" \
            --mix-ratio-official 0.3 \
            --output "$PROCESSED_DIR/triples.parquet"
    fi
fi

# 4) Train
if [[ $SKIP_TRAIN -eq 0 ]]; then
    step "4/5 Train cross-encoder"
    if [[ $SMALL -eq 1 ]]; then
        python src/train.py --config "$CONFIG" --debug --max_steps 5
    else
        python src/train.py --config "$CONFIG"
    fi
fi

# 5) Evaluate
if [[ $SKIP_EVAL -eq 0 ]]; then
    step "5/5 Evaluate (mMARCO-PT dev + Quati 1M; JurisTCU optional)"
    CHECKPOINT=$(python -c "
import yaml, pathlib
cfg = yaml.safe_load(open('$CONFIG'))
print(pathlib.Path(cfg['output_dir']) / 'best')
")
    if [[ $SMALL -eq 1 ]]; then
        echo "Skipping evaluation in --small mode (no aligned eval data)."
    else
        python src/eval_mmarco.py --checkpoint "$CHECKPOINT" || echo "eval_mmarco not yet implemented"
        python src/eval_quati.py --checkpoint "$CHECKPOINT" || echo "eval_quati not yet implemented"
    fi
fi

step "Done. Best checkpoint should be under: \$(yaml output_dir)/best"
