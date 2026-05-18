<div align="center">

# PTBR-Reranker

**Cross-encoder reranker para português brasileiro, baseado em Albertina-100m e treinado em mMARCO-PT com hard negatives minerados.**

[![CI](https://github.com/stekel/ptbr-reranker/actions/workflows/ci.yml/badge.svg)](https://github.com/stekel/ptbr-reranker/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/stekel/cross-encoder-albertina-ptbr-mmarco)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/stekel/ptbr-reranker-demo)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stekel/ptbr-reranker/blob/main/examples/notebooks/quickstart.ipynb)

</div>

---

## Por quê?

O ecossistema de IR para português brasileiro tem bi-encoders sólidos (Serafim, da PORTULAN) mas **não tem um cross-encoder/reranker PT-BR de qualidade publicamente disponível**. Em inglês, a pipeline padrão é em duas etapas: bi-encoder recupera top-K candidatos, cross-encoder reranqueia top-N. Em PT-BR a segunda etapa está incompleta. Este projeto preenche a lacuna.

## Highlights

| Métrica | BM25 | Serafim-IR (bi-encoder) | mMiniLM-L12 reranker | BGE-reranker-v2-m3 | **PTBR-Reranker (este)** |
|---|---|---|---|---|---|
| mMARCO-PT MRR@10 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | **_TBD_** |
| mMARCO-PT nDCG@10 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | **_TBD_** |
| MIRACL-PT nDCG@10 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | **_TBD_** |

> _Tabela será atualizada após a Fase 3 (treino) e Fase 4 (avaliação)._

## Quickstart

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("stekel/cross-encoder-albertina-ptbr-mmarco")

query = "qual é a capital do Brasil?"
passages = [
    "Brasília é a capital federal do Brasil desde 1960.",
    "São Paulo é a maior cidade do Brasil.",
    "O Brasil tem 26 estados e um distrito federal.",
]

scores = model.predict([(query, p) for p in passages])
ranked = sorted(zip(passages, scores), key=lambda x: -x[1])
for passage, score in ranked:
    print(f"{score:.3f}  {passage}")
```

Veja `examples/` para mais snippets:
- `basic_usage.py` — uso mínimo
- `rag_pipeline.py` — pipeline Serafim-IR (retriever) + PTBR-Reranker
- `benchmark_comparison.py` — comparação head-to-head com baselines

## Instalação

```bash
pip install ptbr-reranker
# ou para desenvolvimento:
git clone https://github.com/stekel/ptbr-reranker.git
cd ptbr-reranker
pip install -e ".[dev]"
pre-commit install
```

## Reprodução

Pipeline completo de treino reproduzível em ~40h de A100 (custo estimado ~$55 em Runpod Community):

```bash
# 1. Baixar dados
python data/download_mmarco.py

# 2. Hard negative mining (~6h em A40)
python data/mine_hard_negatives.py --config configs/train_hardneg.yaml

# 3. Treino (~30-40h em A100)
python src/train.py --config configs/train_hardneg.yaml

# 4. Avaliação
python src/eval_mmarco.py --checkpoint runs/best
python src/eval_miracl.py --checkpoint runs/best

# 5. Testes de qualidade
pytest tests/ -v
```

Detalhes completos em [`docs/reproducibility.md`](docs/reproducibility.md).

## Qualidade do modelo

Além das métricas padrão, este modelo passa por uma **bateria de testes de qualidade** específica para PT-BR antes de qualquer release:

- **Calibração de scores** (ECE)
- **Robustez ortográfica** (acentos, erros de digitação, abreviações comuns em PT-BR)
- **Vieses** (PT-BR vs PT-PT, gênero, domínio)
- **Casos qualitativos PT-BR** (jurídico, médico, gírias, ambiguidade, negação)
- **Pipeline end-to-end** (retriever + reranker)

Detalhes em [`docs/quality-tests.md`](docs/quality-tests.md).

## Estrutura do repositório

```
.
├── src/                # código principal (modelo, treino, avaliação, rerank)
├── data/               # scripts de download e preparação
├── configs/            # configs YAML de treino
├── tests/              # pytest (incluindo bateria de qualidade)
├── examples/           # snippets de uso
├── docs/               # documentação, lab notebook, model card
├── paper/              # LaTeX do artigo
├── space/              # demo Gradio (HuggingFace Space)
└── scripts/            # utilitários (push to hub, build modelcard)
```

## Modelo base e dados

- **Modelo base**: [`PORTULAN/albertina-100m-portuguese-ptbr-encoder`](https://huggingface.co/PORTULAN/albertina-100m-portuguese-ptbr-encoder) — DeBERTa-v3 adaptado para PT-BR.
- **Dados de treino**: [`unicamp-dl/mmarco`](https://huggingface.co/datasets/unicamp-dl/mmarco) (subset português) + hard negatives minerados com `PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir`.
- **Avaliação**: mMARCO-PT dev split + MIRACL-PT.

## Citação

```bibtex
@software{ptbr_reranker_2026,
  author = {Stekel},
  title = {PTBR-Reranker: A Brazilian Portuguese Cross-Encoder for Passage Reranking},
  year = {2026},
  url = {https://github.com/stekel/ptbr-reranker},
  publisher = {Hugging Face}
}
```

Se você usar este modelo em pesquisa, por favor cite também os trabalhos sobre os quais ele se apoia:

- **Albertina**: [arXiv:2403.01897](https://arxiv.org/abs/2403.01897)
- **Serafim**: [arXiv:2407.19527](https://arxiv.org/abs/2407.19527)
- **mMARCO**: [arXiv:2108.13897](https://arxiv.org/abs/2108.13897)
- **MIRACL**: [arXiv:2210.09984](https://arxiv.org/abs/2210.09984)

## Contribuindo

Pull requests são bem-vindos. Veja [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Licença

[MIT](LICENSE) — você pode usar comercialmente, modificar, e redistribuir.

## Agradecimentos

- **PORTULAN/LIACC** pelos foundation models Albertina e Serafim.
- **UNICAMP DL** pelo mMARCO-PT.
- **Instituto Federal de São Paulo (IFSP)** pelo suporte institucional.
