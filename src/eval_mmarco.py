"""Rerank evaluation on mMARCO-PT dev split.

Pipeline:
1. Load the cross-encoder checkpoint.
2. Load mMARCO-PT dev queries (6,980 by default), the qrels.dev.small TREC
   relevance file (7,437 judgements), the collection parquet, and the BM25
   first-stage run published by Unicamp-DL.
3. For each dev query, take the top ``rerank_top_n`` BM25 candidates,
   score every (query, passage) pair with the cross-encoder, and rewrite
   the ranking by descending score.
4. Compute MRR@10, nDCG@10, Recall@100, Recall@1000 with pytrec_eval, and
   serialize per-query metrics to a parquet so ``src.stats`` can produce
   bootstrap confidence intervals for the paper.

Reference: Bonifacio et al., 'mMARCO: A Multilingual Version of the MS MARCO
Passage Ranking Dataset', arXiv:2108.13897.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_QUERIES = Path("data/raw/mmarco/queries_dev_portuguese.parquet")
DEFAULT_COLLECTION = Path("data/raw/mmarco/collection_portuguese.parquet")
DEFAULT_QRELS = Path("data/raw/mmarco/qrels.dev.small.tsv")
DEFAULT_BM25 = Path("data/raw/mmarco/run.bm25_portuguese-msmarco.txt")


def _load_qrels(qrels_path: Path) -> dict[str, dict[str, int]]:
    """Parse a TREC qrels file: ``qid 0 docid rel``. Returns ``{qid: {docid: rel}}``."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with qrels_path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, did, rel = parts[0], parts[1], parts[2], parts[3]
            qrels[qid][did] = int(rel)
    logger.info("Loaded qrels for %d queries from %s", len(qrels), qrels_path)
    return dict(qrels)


def _load_bm25_run(
    run_path: Path,
    *,
    query_ids: set[str],
    top_k: int,
) -> dict[str, list[str]]:
    """Parse a TREC run: ``qid Q0 docid rank score tag``.

    Returns ``{qid: [docid sorted by rank]}`` filtered to ``query_ids`` and
    capped at ``top_k`` candidates each.
    """
    by_qid: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with run_path.open() as fh:
        for line in fh:
            parts = line.split()
            # Two formats observed in the wild:
            # - TREC 6-col: ``qid Q0 docid rank score tag``
            # - Anserini/pyserini 3-col: ``qid docid rank`` (used by mMARCO's
            #   data/google/runs/run.bm25_portuguese-msmarco.txt)
            if len(parts) >= 6:
                qid, did, rank = parts[0], parts[2], int(parts[3])
            elif len(parts) == 3:
                qid, did, rank = parts[0], parts[1], int(parts[2])
            else:
                continue
            if qid not in query_ids:
                continue
            by_qid[qid].append((rank, did))
    out: dict[str, list[str]] = {}
    for qid, pairs in by_qid.items():
        pairs.sort()
        out[qid] = [d for _, d in pairs[:top_k]]
    logger.info(
        "Loaded BM25 candidates for %d queries (top_k=%d) from %s",
        len(out),
        top_k,
        run_path,
    )
    return out


def _load_id_text_dict(parquet_path: Path, needed_ids: set[str]) -> dict[str, str]:
    """Load a parquet with ``id, text`` columns, filtered to ``needed_ids`` for memory.

    The mMARCO-PT collection has 8.8M passages; we typically only need the
    union of BM25 candidates across the dev queries (≈700k unique passage IDs
    for rerank_top_n=100), so filtering avoids materialising the whole map.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)  # type: ignore[no-untyped-call]
    out: dict[str, str] = {}
    remaining = set(needed_ids)
    for batch in pf.iter_batches(batch_size=100_000, columns=["id", "text"]):  # type: ignore[no-untyped-call]
        ids = batch["id"].to_pylist()
        texts = batch["text"].to_pylist()
        for i, t in zip(ids, texts, strict=True):
            key = str(i)
            if key in remaining:
                out[key] = t
                remaining.discard(key)
        if not remaining:
            break
    logger.info(
        "Loaded text for %d / %d ids from %s",
        len(out),
        len(needed_ids),
        parquet_path,
    )
    return out


def evaluate(
    checkpoint: str | Path,
    *,
    queries_path: Path = DEFAULT_QUERIES,
    collection_path: Path = DEFAULT_COLLECTION,
    qrels_path: Path = DEFAULT_QRELS,
    bm25_path: Path = DEFAULT_BM25,
    rerank_top_n: int = 100,
    batch_size: int = 64,
    max_length: int = 256,
    max_queries: int | None = None,
    device: str | None = None,
    per_query_output: Path | None = None,
) -> dict[str, float]:
    """Rerank BM25 candidates and compute IR metrics. Returns aggregate dict."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytrec_eval
    from sentence_transformers import CrossEncoder

    from .train import patch_deberta_attention_dtype, resolve_device

    patch_deberta_attention_dtype()
    resolved_device = resolve_device(device or "auto")
    logger.info("Resolved device: %s", resolved_device)

    qrels = _load_qrels(qrels_path)
    queries_table = pq.read_table(queries_path)  # type: ignore[no-untyped-call]
    query_text = {
        str(i): t
        for i, t in zip(
            queries_table["id"].to_pylist(),
            queries_table["text"].to_pylist(),
            strict=True,
        )
    }
    qids = set(query_text.keys()) & set(qrels.keys())
    if max_queries is not None:
        qids = set(sorted(qids, key=int)[:max_queries])
    logger.info("Evaluating %d queries", len(qids))

    bm25 = _load_bm25_run(bm25_path, query_ids=qids, top_k=rerank_top_n)
    needed_doc_ids: set[str] = set()
    for cands in bm25.values():
        needed_doc_ids.update(cands)
    passage_text = _load_id_text_dict(collection_path, needed_doc_ids)

    model_kwargs: dict[str, Any] = {}
    if resolved_device == "cpu":
        import torch

        model_kwargs["torch_dtype"] = torch.float32
    model = CrossEncoder(
        str(checkpoint),
        max_length=max_length,
        num_labels=1,
        device=resolved_device,
        model_kwargs=model_kwargs,
    )
    if resolved_device == "cpu":
        model.model.float()

    rerank_run: dict[str, dict[str, float]] = {}
    for idx, qid in enumerate(sorted(qids, key=int), start=1):
        cands = bm25.get(qid, [])
        if not cands:
            continue
        pairs = [(query_text[qid], passage_text.get(d, "")) for d in cands]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        rerank_run[qid] = dict(zip(cands, scores.tolist(), strict=True))
        if idx % 500 == 0:
            logger.info("Reranked %d / %d queries", idx, len(qids))

    # pytrec_eval normaliza nomes: ``ndcg_cut.10`` -> ``ndcg_cut_10``.
    # Para MRR@10 a métrica correta é ``recip_rank_cut.10`` (vira
    # ``recip_rank_cut_10`` no dict). ``recip_rank.10`` simplesmente não
    # existe — ``recip_rank`` sozinho não aceita cutoff.
    metrics_set = {
        "map",
        "ndcg_cut.10",
        "recip_rank_cut.10",
        "recall.100",
        "recall.1000",
    }
    qrels_for_eval = {qid: qrels[qid] for qid in qids if qid in qrels}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_for_eval, metrics_set)
    per_query = evaluator.evaluate(rerank_run)

    aggregate: dict[str, float] = {
        "map": float(np.mean([m["map"] for m in per_query.values()])),
        "ndcg_at_10": float(np.mean([m["ndcg_cut_10"] for m in per_query.values()])),
        "mrr_at_10": float(np.mean([m["recip_rank_cut_10"] for m in per_query.values()])),
        "recall_at_100": float(np.mean([m["recall_100"] for m in per_query.values()])),
        "recall_at_1000": float(np.mean([m["recall_1000"] for m in per_query.values()])),
        "num_queries": len(per_query),
    }
    logger.info("Aggregate metrics: %s", aggregate)

    if per_query_output is not None:
        per_query_output.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "qid": qid,
                "map": m["map"],
                "ndcg_at_10": m["ndcg_cut_10"],
                "mrr_at_10": m["recip_rank_cut_10"],
                "recall_at_100": m["recall_100"],
                "recall_at_1000": m["recall_1000"],
            }
            for qid, m in per_query.items()
        ]
        pq.write_table(pa.Table.from_pylist(rows), per_query_output)  # type: ignore[no-untyped-call]
        logger.info("Wrote per-query metrics to %s", per_query_output)

    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on mMARCO-PT dev")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--bm25", type=Path, default=DEFAULT_BM25)
    parser.add_argument("--rerank-top-n", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Cap the number of dev queries (useful for smoke tests).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_mmarco.json"))
    parser.add_argument(
        "--per-query-output",
        type=Path,
        default=Path("outputs/eval_mmarco_per_query.parquet"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = evaluate(
        args.checkpoint,
        queries_path=args.queries,
        collection_path=args.collection,
        qrels_path=args.qrels,
        bm25_path=args.bm25,
        rerank_top_n=args.rerank_top_n,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_queries=args.max_queries,
        device=args.device,
        per_query_output=args.per_query_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote aggregate metrics to %s", args.output)


if __name__ == "__main__":
    main()
