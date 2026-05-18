# Data

Scripts to download, prepare, and mine training data.

## Datasets

| Dataset | Source | Used for |
|---|---|---|
| mMARCO-PT | [`unicamp-dl/mmarco`](https://huggingface.co/datasets/unicamp-dl/mmarco) (subset `portuguese`) | Training + dev evaluation |
| MIRACL-PT | [`miracl/miracl`](https://huggingface.co/datasets/miracl/miracl) (lang `pt`) | Cross-domain evaluation |

## Pipeline

```bash
# 1. Download raw data (queries, collection, qrels, BM25 top-1000)
python data/download_mmarco.py

# 2. Mine hard negatives with Serafim-IR (requires GPU; ~6h on A40)
python data/mine_hard_negatives.py --config configs/train_hardneg.yaml

# 3. Build final (query, positive, [negatives]) triples
python data/build_triples.py --output data/processed/triples.parquet
```

Files in `data/raw/`, `data/processed/`, and `data/cache/` are gitignored.
Refer to `docs/reproducibility.md` for exact dataset versions (commit SHAs).
