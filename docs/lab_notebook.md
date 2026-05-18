# Lab Notebook — PTBR-Reranker

Diário datado, atualizado **a cada sessão de trabalho** (não a posteriori). Decisões, hipóteses, debugging, mudanças de rumo.

Princípio: se não está aqui, não aconteceu. O paper sai daqui.

---

## 2026-05-18 — Kickoff

**Hipótese:** A lacuna mais clara no ecossistema PT-BR é a ausência de um cross-encoder/reranker de qualidade pública. Bi-encoder existe (Serafim) mas pipeline de duas etapas está incompleta.

**Decisões:**
- Modelo base: `PORTULAN/albertina-100m-portuguese-ptbr-encoder` (melhor custo/qualidade entre encoders PT-BR).
- Dados: mMARCO-PT + hard negatives minerados com `PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir` (faixa rank 10-100, 7 negs por query).
- Avaliação: mMARCO-PT dev + MIRACL-PT.
- Provedor GPU primário: Runpod Community A100 SXM ($1.39/h). Prototipagem: Kaggle (free) + Modal ($30 créditos).
- Tracking: W&B obrigatório em todos os runs; lab_notebook.md datado; experiments_log.md como tabela mestra.

**O que fiz:**
- Pesquisa de estado da arte em modelos PT-BR (documentado em `future-projects.md`).
- Plano de execução em 6 fases (em `~/.claude/plans/criar-o-plano-para-functional-puddle.md`).
- Setup completo do repositório: pyproject.toml, LICENSE, CITATION.cff, CONTRIBUTING.md, CODE_OF_CONDUCT.md, CHANGELOG.md, GitHub Actions (CI/publish-modelcard/release), pre-commit, README profissional com badges.
- Esqueletos de `src/` (model, train, eval_mmarco, eval_miracl, rerank, stats), `data/` (download_mmarco, mine_hard_negatives, build_triples), `tests/` (bateria de qualidade em 5 dimensões), configs YAML (baseline e hardneg).

**TODO próxima sessão:**
- Criar conta W&B e projeto `ptbr-reranker`.
- Subir Kaggle Kernel para validar Phase 1 (download de dados, smoke test do Albertina).
- Esqueleto LaTeX do paper.
- Primeiro commit no GitHub (privado inicialmente).

---

## 2026-05-18 — Validação local do esqueleto (continuação)

**Hipótese:** Antes de implementar Phase 1 de fato, vale validar o esqueleto local: lint, type-check, testes mínimos. Se algo está quebrado, descobrir agora é mais barato.

**O que fiz:**
- `uv run --with ruff ruff check .` — pegou 1 erro (`F541` f-string sem placeholder em `examples/rag_pipeline.py`). Corrigido.
- `ruff format .` — 5 arquivos reformatados (formatação inicial era inconsistente).
- `uv pip install -e ".[dev]"` — instalou pytest, ruff, mypy, pre-commit.
- `mypy src/` — 2 erros em `src/model.py` (return Any em CrossEncoder). Corrigido com anotação explícita do tipo local.
- Adicionei 5 testes para `paired_bootstrap_pvalue` e `BootstrapResult.as_str` em `tests/test_integration.py`.
- Encontrei detalhe: banker's rounding em Python — `round(0.6345, 3) == 0.634` (não 0.635). Ajustei valor de teste para 0.6356 → 0.636 para evitar a borda.

**Resultado:**
- Lint: ✅ All checks passed
- Format: ✅
- Mypy: ✅ Success, no issues in 7 source files
- Pytest: ✅ 6 passed, 18 deselected (gpu/slow markers)
- Coverage de `src/stats.py`: 96%

**Decisão:**
- Esqueleto está saudável. CI no GitHub Actions deve passar sem alterações na primeira execução.
- Manter `bootstrap_metric` e `paired_bootstrap_pvalue` como utilitários de primeira classe — vão ser usados em toda comparação com baseline.

**TODO próxima sessão:**
- Implementar `data/download_mmarco.py` para valer (datasets do HF).
- Smoke test: carregar Albertina-100m e fazer inferência num par (query, passage) em CPU.
- Criar conta W&B.

---

## 2026-05-18 — `download_mmarco.py` implementado + smoke test do Albertina

**Hipótese:** Para validar Phase 1 sem queimar tempo de GPU paga, preciso de (a) um downloader real do mMARCO-PT que suporte modo `--small` para iteração local/CI e (b) um smoke test que garanta que o caminho `Albertina-100m → CrossEncoder → predict` funciona em CPU.

**O que fiz:**
- Implementei `data/download_mmarco.py` completo:
  - `download_mmarco()` carrega `unicamp-dl/mmarco` via `datasets.load_dataset` para 3 splits (collection, queries-train, queries-dev), serializa para parquet em `data/raw/mmarco/`, registra revisão (SHA HF) em `manifest.json`.
  - `download_miracl()` faz o mesmo para `miracl/miracl` (config `pt`).
  - `check()` valida contagens esperadas (com semântica diferente para `--small` vs `--full`).
  - `write_manifest()` é append-only para preservar histórico de downloads.
  - Flag `--small` materializa slice de ~10k passagens + 1k train + 100 dev (cabe em laptop e em Kaggle Kernel free).
  - Constantes `EXPECTED_COUNTS_FULL = {8_841_823, 502_939, 6_980}` e `EXPECTED_COUNTS_SMALL` documentadas no topo do módulo.
- Adicionei 6 testes em `tests/test_data_pipeline.py` cobrindo manifest válido, manifest ausente, arquivo de snapshot ausente, contagem inflada em small mode, mismatch em full mode, e append múltiplo. Tudo com `tmp_path`, zero rede.
- Adicionei smoke test `test_albertina_loads_and_predicts` em `tests/test_data_pipeline.py`, marcado `@pytest.mark.slow`. Carrega `PORTULAN/albertina-100m-portuguese-ptbr-encoder` via CrossEncoder e scoreia 2 pares — confirma que o modelo base é carregável e produz floats antes de qualquer fine-tuning. CI básico (sem `-m slow`) pula automaticamente.

**Resultado:**
- 12 testes passando (6 novos), 0 falhas.
- Cobertura `data/download_mmarco.py`: 58% (toda a lógica sem rede coberta).
- mypy `--strict` continua limpo em 11 arquivos fonte.
- ruff lint + format limpos.

**Decisão:**
- Não rodar `download_mmarco` em CI nem agora — espera-se o primeiro download real no Kaggle Kernel da próxima sessão (free tier, sem custo).
- Manter o smoke test `slow` no repositório como gate manual antes do primeiro treino: `pytest -m slow tests/test_data_pipeline.py::test_albertina_loads_and_predicts` deve passar antes de subir ao Runpod.
- Manifest com SHA HF embutido é o artifact que vai para `docs/reproducibility.md` "Exact versions used in published model" no momento do release.

**TODO próxima sessão:**
- Em Kaggle Kernel free: rodar `python data/download_mmarco.py --small` (CPU, free) → confirmar que o caminho funciona end-to-end com dados reais.
- Implementar `data/mine_hard_negatives.py` (encoding com Serafim-IR + FAISS HNSW).
- Criar conta W&B e projeto `ptbr-reranker`.
- Decidir se hospedo o dataset de hard negatives como repo HF separado (`tardellirs/mmarco-ptbr-hardnegatives`).

---

## 2026-05-18 — Notebook Kaggle de validação Phase 1 + renomeação de handles

**Hipótese:** Para validar Phase 1 sem rodar nada localmente nem pagar GPU, vale criar um notebook Kaggle auto-suficiente que clona o repo, baixa dados (small), valida manifest, e roda o smoke test do Albertina em CPU.

**O que fiz:**
- Renomeei todas as URLs GitHub e repo IDs HF de `stekel/*` → `tardellirs/*` em 15 arquivos. Email institucional `stekel@ifsp.edu.br` mantido. Assumi também HF user = `tardellirs` (confirmar antes do primeiro push para HF Hub).
- Criei `notebooks/kaggle_phase1_validation.ipynb` (8 células markdown + 7 código) com:
  - Clone via `git clone --depth 1 https://github.com/tardellirs/ptbr-reranker.git`
  - `pip install -e ".[dev]"`
  - `python data/download_mmarco.py --small`
  - Pretty-print do manifest.json
  - `python data/download_mmarco.py --check --small` (exit code 0)
  - Sanity dos parquets via pandas
  - `pytest -v -m slow tests/test_data_pipeline.py::test_albertina_loads_and_predicts`
  - Suite rápida (`pytest -m "not gpu and not slow"`) para regressão
  - Inferência qualitativa com 3 passagens
  - Checklist final + próximos passos
- Criei `notebooks/README.md` com instruções de upload (Settings: Accelerator None, Internet On, Persistence Variables+files), tempo esperado (5–10min), output esperado, e o que fazer com os resultados.
- Validei: notebook é JSON parseável; ruff/mypy/pytest tudo limpo (12 testes).
- Pego: B905 (zip sem `strict=`) também aplica dentro de notebooks. Tive que usar `NotebookEdit` em vez de `Edit` para corrigir.

**Resultado:**
- Lint, format, mypy, pytest: ✅
- 12 testes passando.

**Decisão:**
- O notebook só executa de fato após o push para GitHub público. Vou pedir autorização explícita do usuário antes do push, com comando `gh repo create` + `git push` documentado.
- Confiar no Kaggle base image para ter torch/transformers/datasets já presentes — `pip install -e ".[dev]"` só adiciona pytest/ruff/mypy/codecarbon (rápido).

**TODO próxima sessão:**
- Push para GitHub público.
- Subir o notebook no Kaggle, executar, registrar SHAs resolvidas em `docs/reproducibility.md`.
- Confirmar HF username (`tardellirs`?) antes de criar repo no HF Hub.

---

<!-- Template para próximas entradas:

## YYYY-MM-DD — Título curto

**Hipótese:**

**O que fiz:**
- comandos exatos, run_id W&B, configs alteradas

**Resultado:**
- métricas, observações qualitativas, screenshots se aplicável

**Decisão:**
- seguir com X / mudar para Y porque Z

**TODO:**
- próximos passos derivados

-->
