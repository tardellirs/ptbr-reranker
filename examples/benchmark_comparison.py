"""Head-to-head: PTBR-Reranker vs multilingual rerankers on a small PT-BR set.

Quick demonstration on a curated mini-set. For full mMARCO-PT evaluation,
use ``python src/eval_mmarco.py`` instead.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

MODELS = {
    "PTBR-Reranker (ours)": "stekel/cross-encoder-albertina-ptbr-mmarco",
    "BGE-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "mMiniLM-L12-v2-msmarco": "cross-encoder/msmarco-MiniLM-L12-en-de-v1",
}

CASES = [
    {
        "query": "como protocolar uma reclamação no PROCON?",
        "passages": [
            "Para registrar uma reclamação no PROCON, o consumidor deve apresentar documentos de identificação, comprovante da relação de consumo e descrição detalhada do problema.",
            "O PROCON é um órgão de defesa do consumidor presente em diversas cidades brasileiras.",
            "Reclamações trabalhistas são processadas pela Justiça do Trabalho.",
        ],
        "relevant_index": 0,
    },
    {
        "query": "sintomas de dengue em crianças",
        "passages": [
            "Em crianças, os sintomas da dengue incluem febre alta, dor abdominal, vômitos persistentes e manchas vermelhas pelo corpo.",
            "A dengue é transmitida pelo mosquito Aedes aegypti.",
            "Vacina contra a dengue está disponível no SUS para grupos prioritários.",
        ],
        "relevant_index": 0,
    },
]


def main() -> None:
    for name, model_id in MODELS.items():
        print(f"\n=== {name} ===")
        model = CrossEncoder(model_id)
        for case in CASES:
            scores = model.predict([(case["query"], p) for p in case["passages"]])
            ranking = sorted(enumerate(scores.tolist()), key=lambda x: -x[1])
            top_idx = ranking[0][0]
            hit = "✓" if top_idx == case["relevant_index"] else "✗"
            print(f"  [{hit}] {case['query'][:60]}")
            print(f"      top-1: {case['passages'][top_idx][:80]}")


if __name__ == "__main__":
    main()
