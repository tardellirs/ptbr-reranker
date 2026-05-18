"""Minimal example: load the reranker and score (query, passage) pairs."""

from __future__ import annotations

from sentence_transformers import CrossEncoder


def main() -> None:
    model = CrossEncoder("stekel/cross-encoder-albertina-ptbr-mmarco")

    query = "qual é a capital do Brasil?"
    passages = [
        "Brasília é a capital federal do Brasil desde 1960.",
        "São Paulo é a maior cidade do Brasil.",
        "O Rio de Janeiro foi capital do Brasil até 1960.",
        "O Brasil tem 26 estados e um distrito federal.",
    ]

    scores = model.predict([(query, p) for p in passages])
    for passage, score in sorted(zip(passages, scores.tolist(), strict=True), key=lambda x: -x[1]):
        print(f"{score:+.4f}  {passage}")


if __name__ == "__main__":
    main()
