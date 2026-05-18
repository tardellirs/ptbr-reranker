"""Two-stage retrieval pipeline: Serafim-IR bi-encoder + PTBR-Reranker cross-encoder.

Bi-encoder fetches top-K candidates from a corpus; the cross-encoder reorders
the top-N for higher precision. This is the standard pattern in RAG systems.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

BI_ENCODER_ID = "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir"
CROSS_ENCODER_ID = "stekel/cross-encoder-albertina-ptbr-mmarco"

CORPUS = [
    "Brasília é a capital federal do Brasil desde 1960.",
    "São Paulo é a maior cidade do Brasil em população.",
    "O Rio de Janeiro foi capital do Brasil até 1960.",
    "O Brasil tem 26 estados e um distrito federal.",
    "A culinária brasileira tem influências indígenas, africanas e europeias.",
    "O futebol é o esporte mais popular no Brasil.",
    "A Amazônia abrange aproximadamente 60% do território brasileiro.",
    "O português é o idioma oficial do Brasil.",
]


def main() -> None:
    bi = SentenceTransformer(BI_ENCODER_ID)
    cross = CrossEncoder(CROSS_ENCODER_ID)

    corpus_emb = bi.encode(CORPUS, normalize_embeddings=True, show_progress_bar=False)

    query = "qual é a capital do Brasil?"
    query_emb = bi.encode([query], normalize_embeddings=True, show_progress_bar=False)
    sims = (query_emb @ corpus_emb.T)[0]

    top_k = 4
    top_idx = np.argsort(-sims)[:top_k]
    candidates = [CORPUS[i] for i in top_idx]
    print(f"=== Top-{top_k} from bi-encoder (Serafim-IR) ===")
    for rank, (passage, score) in enumerate(
        zip(candidates, sims[top_idx].tolist(), strict=True), start=1
    ):
        print(f"  {rank}. [{score:+.3f}] {passage}")

    reranked_scores = cross.predict([(query, p) for p in candidates])
    reranked = sorted(
        zip(candidates, reranked_scores.tolist(), strict=True), key=lambda x: -x[1]
    )
    print(f"\n=== Reranked by PTBR-Reranker ===")
    for rank, (passage, score) in enumerate(reranked, start=1):
        print(f"  {rank}. [{score:+.3f}] {passage}")


if __name__ == "__main__":
    main()
