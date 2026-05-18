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
