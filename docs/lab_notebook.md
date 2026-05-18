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

## 2026-05-18 — Push para GitHub + ajustes de CI

**Hipótese:** Pushar para `tardellirs/ptbr-reranker` público e deixar a primeira execução de CI verde.

**O que fiz:**
- `gh repo create tardellirs/ptbr-reranker --public --source=. --remote=origin --push` — repositório criado em https://github.com/tardellirs/ptbr-reranker com a descrição correta.
- `gh repo edit` adicionou topics: `portuguese-nlp`, `brazilian-portuguese`, `cross-encoder`, `reranker`, `information-retrieval`, `sentence-transformers`, `huggingface`, `pt-br`, `nlp`, `albertina`. Homepage apontando para o futuro repo HF.
- CI da primeira run falhou em **dois** pontos:
  1. **Publish model card** falhou no step de upload porque `HF_TOKEN` secret ainda não existe. Movei o `env: HF_TOKEN: ${{ secrets.HF_TOKEN }}` do nível do step para o nível do job — antes o `if: env.HF_TOKEN != ''` não conseguia acessar o env definido no próprio step (chicken-and-egg). Agora o workflow emite notice e sai limpo quando falta token, em vez de falhar.
  2. **mypy em Python 3.10** falhou: `src/stats.py:30: error: Missing type arguments for generic type "ndarray" [type-arg]`. Localmente passei porque uso 3.14, onde a regra é relaxada. Substituí `np.ndarray` bare por `FloatArray = npt.NDArray[np.float64]` (numpy.typing). Portável entre 3.10/3.11/3.12.
- Pushei dois commits de fix; segunda CI run em andamento.

**Resultado:**
- Repositório público em https://github.com/tardellirs/ptbr-reranker.
- 4 commits no histórico: scaffold inicial, validação local, downloader mMARCO, notebook Kaggle + renames.
- Mais 2 commits de fix: `c0a8a9e` (CI publish) e `f0de6fe` (mypy 3.10).

**Pegadas didáticas:**
- `env:` em workflow definido no nível do step **não é** visível no `if:` do mesmo step. Tem que estar no nível do job ou workflow.
- mypy em Python 3.10 ainda exige type args explícitos em `np.ndarray`. Em 3.12+ é relaxado. Usar `numpy.typing.NDArray[dtype]` resolve.
- Matrix testing no CI (3.10 + 3.11 + 3.12) é o que pega esses problemas — se tivesse só uma versão local não veria.

**TODO próxima sessão:**
- Conferir que a segunda CI run passou nas 3 versões de Python (run id 26063071111).
- Subir o notebook no Kaggle, rodar, atualizar `docs/reproducibility.md` com SHAs resolvidas.
- Confirmar HF username e criar conta W&B.

---

## 2026-05-18 — Validação Phase 1 no Kaggle (4 tentativas)

**Hipótese:** O notebook Kaggle clona o repo, baixa mMARCO-PT em modo `--small`, valida o manifest, roda o smoke test do Albertina, e completa em ~10min em CPU free.

**O que fiz / quatro versões do kernel:**

| Versão | Resultado | Erro |
|---|---|---|
| v1 | ERROR | `Could not resolve host: github.com` — telefone não verificado, internet bloqueada silenciosamente apesar de `enable_internet=true` na metadata. |
| v2 | ERROR | Após verificar telefone: `RuntimeError: Dataset scripts are no longer supported, but found mmarco.py` — `datasets>=4.0` no Kaggle removeu suporte a loader scripts. |
| v3 | ERROR | Após refatorar `download_mmarco.py` para usar `hf_hub_download` direto na revisão `refs/convert/parquet` (e descobrir que MIRACL não tem PT): download passou, mas smoke test do Albertina falhou com `expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16` no DeBERTa attention — Kaggle CPU image carrega pesos em bfloat16 mas inputs vêm em float32. |
| v4 | **COMPLETE** | Forçei `torch_dtype=torch.float32` + `.float()` no smoke test. Tudo passou. |

**Resultado final (v4):**
- mMARCO-PT `--small` baixado: 10000 + 1000 + 100 rows. SHA da revisão `d2da87d4433168219522a69ef38c30de16bbce80` (data.frame: `unicamp-dl/mmarco@refs/convert/parquet`).
- Manifest validado (`All counts validated (small mode)`).
- `pytest -m slow` (Albertina smoke): **1 passed** em 49.34s.
- `pytest -m "not gpu and not slow"`: **12 passed**.
- Inferência qualitativa: scores ~0.53 em todas as 3 passagens (esperado — modelo não fine-tuned, todos os pares retornam score similar; é o smoke test do pipeline, não validação de relevância).

**Decisões importantes:**
- MIRACL sai da lista de benchmarks de avaliação cross-domain. **Não existe MIRACL-PT** (idiomas: ar/bn/es/fa/fi/hi/id/ja/ko/sw/te/th/yo/zh). Preciso decidir alternativa em sessão futura.
- Em ambiente Kaggle / Colab / qualquer GPU/CPU shared, **forçar float32 explicitamente** ao carregar modelos para evitar mixed-dtype issues. Vale também para o treino real (Phase 3) se rodar em ambientes onde o dtype default não é controlado.
- SHA do mMARCO-PT registrada em `docs/reproducibility.md`. Quando rodar `--full`, re-resolver e atualizar.

**Pegadas didáticas:**
- Kaggle silenciosamente desabilita Internet em kernels de contas sem telefone verificado, mesmo com `enable_internet=true`. O sintoma é DNS failure (não erro 403).
- `datasets>=4.0` removeu loader scripts. Datasets HF antigos (como mmarco) só funcionam via `refs/convert/parquet` agora.
- Estratégia "bypassar load_dataset e ir direto no parquet auto-gerado" é mais robusta e fica imune a futuras quebras do `datasets` library.

**Kernel público (para reprodução futura):**
https://www.kaggle.com/code/tardellistekel/ptbr-reranker-phase-1-validation

**TODO próxima sessão:**
- Decidir avaliação cross-domain (alternativas a MIRACL: Mr.TyDi se tiver PT, ou criar BEIR-PT mini, ou usar BM25-vs-reranker em outro slice).
- Phase 2: implementar `data/mine_hard_negatives.py` em GPU Kaggle T4×2 free.
- Criar conta W&B e projeto `ptbr-reranker`.
- Rotar token Kaggle (foi exposto na conversa).

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
