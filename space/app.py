"""Gradio demo Space for the PTBR-Reranker.

Hosted at: https://huggingface.co/spaces/stekel/ptbr-reranker-demo
"""

from __future__ import annotations

import gradio as gr
from sentence_transformers import CrossEncoder

MODEL_ID = "stekel/cross-encoder-albertina-ptbr-mmarco"
model = CrossEncoder(MODEL_ID)

EXAMPLES: list[list[str]] = [
    [
        "qual é a capital do Brasil?",
        "Brasília é a capital federal do Brasil desde 1960.\n"
        "São Paulo é a maior cidade do Brasil.\n"
        "O Rio de Janeiro foi capital do Brasil até 1960.\n"
        "O Brasil tem 26 estados e um distrito federal.",
    ],
    [
        "como protocolar reclamação no PROCON?",
        "O PROCON é um órgão de defesa do consumidor presente em diversas cidades.\n"
        "Para registrar uma reclamação no PROCON, o consumidor deve apresentar documentos de identificação, comprovante da relação de consumo e descrição detalhada do problema.\n"
        "Reclamações trabalhistas são processadas pela Justiça do Trabalho.",
    ],
    [
        "sintomas de dengue em crianças",
        "A dengue é transmitida pelo mosquito Aedes aegypti.\n"
        "Em crianças, os sintomas da dengue incluem febre alta, dor abdominal, vômitos persistentes e manchas vermelhas pelo corpo.\n"
        "Vacina contra a dengue está disponível no SUS para grupos prioritários.",
    ],
]


def rerank(query: str, passages_text: str) -> list[list[str | float]]:
    passages = [p.strip() for p in passages_text.splitlines() if p.strip()]
    if not query or not passages:
        return []
    scores = model.predict([(query, p) for p in passages])
    ranked = sorted(zip(passages, scores.tolist(), strict=True), key=lambda x: -x[1])
    return [[i + 1, round(s, 4), p] for i, (p, s) in enumerate(ranked)]


with gr.Blocks(title="PTBR-Reranker") as demo:
    gr.Markdown(
        """
        # PTBR-Reranker — Demo

        Cross-encoder reranker para **português brasileiro**, baseado em
        [Albertina-100m](https://huggingface.co/PORTULAN/albertina-100m-portuguese-ptbr-encoder)
        e treinado em mMARCO-PT.

        Cole uma query e várias passagens (uma por linha). O modelo reordena por relevância.
        """
    )

    with gr.Row():
        with gr.Column():
            query_box = gr.Textbox(label="Query", placeholder="Sua pergunta em português…")
            passages_box = gr.Textbox(
                label="Passagens (uma por linha)",
                lines=8,
                placeholder="Passagem 1\nPassagem 2\nPassagem 3",
            )
            submit = gr.Button("Rerank", variant="primary")
        with gr.Column():
            output = gr.Dataframe(
                headers=["#", "Score", "Passagem"],
                datatype=["number", "number", "str"],
                wrap=True,
            )

    submit.click(rerank, inputs=[query_box, passages_box], outputs=output)
    gr.Examples(EXAMPLES, inputs=[query_box, passages_box])

if __name__ == "__main__":
    demo.launch()
