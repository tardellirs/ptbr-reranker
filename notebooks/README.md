# Notebooks

Notebooks de validação e análise. Mantidos no repositório como código (não como artefato exploratório descartável).

Cada subdiretório é um Kaggle Kernel independente (cada um com seu `kernel-metadata.json` para o `kaggle kernels push`).

## `phase1_validation/kaggle_phase1_validation.ipynb`

Valida o pipeline de Phase 1 sem custo de GPU paga, no free tier do Kaggle Kernels.

**O que ele faz:**
1. Clona `tardellirs/ptbr-reranker` direto do GitHub.
2. Instala extras de dev (`pip install -e ".[dev]"`).
3. Roda `python data/download_mmarco.py --small` — baixa ~10k passagens + 1.1k queries do mMARCO-PT e o dev de MIRACL-PT.
4. Inspeciona o `manifest.json` (SHA da revisão HF — artefato para `docs/reproducibility.md`).
5. Roda `pytest -m slow tests/test_data_pipeline.py::test_albertina_loads_and_predicts` — confirma que Albertina-100m carrega via `CrossEncoder` em CPU.
6. Roda toda a suite não-slow para regressão.
7. Faz inferência qualitativa em 3 passagens com 1 query.

**Como subir no Kaggle:**

### Via Kaggle CLI (recomendado para reprodutibilidade)

```bash
# 1) Autenticar (uma vez, token vem de https://www.kaggle.com/settings/account)
mkdir -p ~/.kaggle && echo "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

# 2) Push do kernel
kaggle kernels push -p notebooks/phase1_validation/

# 3) Acompanhar status até completar
until kaggle kernels status tardellistekel/ptbr-reranker-phase-1-validation 2>&1 | grep -qE 'COMPLETE|ERROR'; do sleep 20; done

# 4) Baixar output
kaggle kernels output tardellistekel/ptbr-reranker-phase-1-validation -p /tmp/kaggle-output
```

`notebooks/kernel-metadata.json` configura:
- Privado (`is_private: true`)
- Internet habilitado (`enable_internet: true`)
- CPU-only (`enable_gpu: false`)

### Via interface web (alternativa)

1. Acesse [kaggle.com/code](https://www.kaggle.com/code) e clique em **New Notebook**.
2. No editor, **File → Import Notebook → Upload** e selecione `notebooks/kaggle_phase1_validation.ipynb`.
3. Em **Settings** (painel direito):
   - **Accelerator:** None (CPU). Tudo aqui roda em CPU; não gaste cota de GPU.
   - **Internet:** On (necessário para `git clone` e download do HuggingFace).
   - **Language:** Python 3.
   - **Persistence:** Variables and files (opcional, mas acelera reruns).
4. Clique em **Save Version → Save & Run All (Commit)**.

**Tempo esperado:** 5–10 min.

**O que esperar como output:**
- Manifesto com `revision` resolvida (SHA do mMARCO e MIRACL).
- Tabela de splits com row counts (≈10k collection, ≈1k train queries, 100 dev queries).
- Pytest `test_albertina_loads_and_predicts PASSED`.
- 3 passagens scoreadas (scores ainda não refletem relevância — modelo ainda não foi fine-tuned).

**Após rodar:**
- Atualizar `docs/lab_notebook.md` com entrada datada (resultado, SHAs, observações).
- Atualizar a seção "Exact versions used in published model" em `docs/reproducibility.md` com as SHAs resolvidas.
- Linkar o notebook executado (versão pública no Kaggle) no lab notebook.

## `phase2_mining/kaggle_phase2_mining.ipynb`

Valida o pipeline de Phase 2 (mineração de hard negatives + construção de triples de treino) no Kaggle Kernels com **GPU T4 free tier**.

**O que ele faz:**
1. Clona o repo e instala extras de dev.
2. Roda `data/download_mmarco.py --small` (já validado em Phase 1).
3. Confere cobertura de qrels nas 100 dev queries (sanity).
4. Roda `data/mine_hard_negatives.py` em GPU T4 — encoda 10k passagens + 100 queries com `PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir`, indexa com FAISS HNSW, samplea 5 hard negatives por query.
5. Inspeção qualitativa: mostra query + positivo + 5 negativos para conferir que o mining produz negativos plausíveis-mas-irrelevantes.
6. Roda `data/build_triples.py` em dois modos:
   - **Baseline**: triples oficiais MS MARCO (BM25 negs, sem custo de GPU) — recipe `train_baseline.yaml`.
   - **Hardneg**: triples a partir dos negativos minerados — recipe `train_hardneg.yaml`.
7. Sanity check do schema final `(query_id, query_text, positive_text, negative_text)`.

**Push e execução:**

```bash
kaggle kernels push -p notebooks/phase2_mining/

until kaggle kernels status tardellistekel/ptbr-reranker-phase-2-mining 2>&1 | grep -qE 'COMPLETE|ERROR'; do sleep 30; done

kaggle kernels output tardellistekel/ptbr-reranker-phase-2-mining -p /tmp/kaggle-phase2
```

**Settings do Kaggle:**
- Accelerator: **GPU T4 x2** (ou single T4; suficiente para 10k passagens).
- Internet: **On**.
- Persistence: **Variables and files**.

**Tempo esperado:** 8–15 min.

## Notebooks futuros (planejados)

- `phase3_training/` — treino real do cross-encoder em A100 (Runpod) ou T4 (Kaggle slice).
- `analysis/` — análise de erros e exemplos qualitativos para a seção Discussion do paper.
- `wins_showcase/` — 30 casos onde o modelo ganha do baseline, para a seção Qualitative Analysis.
