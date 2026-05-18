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
