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

## 2026-05-18 — Pivot da avaliação cross-domain + Phase 2 implementada

**Hipótese:** Substituir MIRACL (sem PT) por benchmarks PT-BR nativos e implementar a pipeline de mineração + construção de triples para o treino.

**Pesquisa cross-domain (agente em background):**
- **Mr.TyDi**: não tem PT (mesmo gap do MIRACL — ambos derivam do TyDi QA).
- **NeuCLIR / MultiCPR / HC4 / TyDi QA / XQuAD / MLQA**: nenhum tem PT.
- **Mintaka PT** existe mas é traduzido (mesmo problema do mMARCO-PT).
- **Quati** (`unicamp-dl/quati`, arXiv:2404.06976): **MELHOR opção**. Native PT-BR de ClueWeb22-pt, 50 topics com dense judgments TREC-style 4-point, 1M ou 10M passages, CC-BY-4.0. **Não derivado de mMARCO** — true generalization test.
- **JurisTCU** (`LeandroRibeiro/JurisTCU`, arXiv:2503.08379): native PT-BR de jurisprudência do TCU, 16k docs + 150 queries + 2250 judgments (3 query styles). Probe de domínio jurídico hard.

**Decisão:**
- Primário cross-domain: **Quati 1M** (web nativo, generalização legítima).
- Secundário cross-domain: **JurisTCU** (domain shift jurídico).
- Renomeei `src/eval_miracl.py` → `src/eval_quati.py` (mantendo histórico via `git mv`).
- Criei `src/eval_juristcu.py` para a segunda avaliação.

**Phase 2 implementada:**
- `data/mine_hard_negatives.py` agora funcional: encode collection com Serafim-IR (`SentenceTransformer`) → FAISS HNSW (`IndexHNSWFlat`, M=64, efConstruction=200) → search top-K (default 200) por query → filtrar positives via qrels → samplear N negs (default 7) da faixa rank [10, 100) → escrever parquet com schema `(query_id, positive_id, negative_ids[List[int]])`.
- `data/build_triples.py` funcional: consome (a) `triples.train.ids.small.tsv` oficial do MS MARCO (BM25 negs, baseline grátis) e/ou (b) `hard_negatives.parquet` minerado. Sample weighted via `mix_ratio_official` ∈ [0,1]. Output: parquet com `(query_id, query_text, positive_text, negative_text)` pronto para o `src/train.py`.
- `data/download_mmarco.py` estendido para puxar qrels e triples oficiais do mMARCO main branch (TSV, language-agnostic, IDs compartilhados entre traduções). Grande descoberta: **não precisamos minerar negativos para o baseline** — o MS MARCO já tem 39M triples (qid, pos, neg) prontos. A mineração Serafim vira upgrade opcional para o modelo final, permitindo ablation no paper.

**Pegadas didáticas:**
- mMARCO qrels e triples são *language-agnostic*: os IDs de query e passage são compartilhados entre todas as traduções, então `qrels.dev.small.tsv` e `triples.train.ids.small.tsv` no main branch servem para PT, EN, ES, etc.
- ruff `RUF002` reclama de `×` em docstrings (sinal de multiplicação Unicode confundido com `x`). Usar `x` ASCII em código; `×` só em markdown.
- pyarrow não tem stubs de tipo; mypy strict precisa de `# type: ignore[no-untyped-call]` nos calls. Alternativa: instalar `types-pyarrow` quando disponível (ainda não em maio/2026).

**Resultado:**
- 12 testes passando, lint e mypy limpos em 12 arquivos.
- Pipeline Phase 1 (download) → Phase 2 (mining + build_triples) → Phase 3 (train) está conectada.

**TODO próxima sessão:**
- Rodar `mine_hard_negatives.py --small` no Kaggle T4×2 free para validar end-to-end com GPU.
- Implementar `src/train.py` real (CrossEncoder fit loop).
- Criar conta W&B e projeto `ptbr-reranker`.
- Rotar token Kaggle (foi exposto na conversa anterior).

---

## 2026-05-18 — Validação Phase 2 no Kaggle (4 tentativas)

**Hipótese:** Validar mine_hard_negatives + build_triples end-to-end no Kaggle free tier.

**4 versões do kernel, 3 problemas diferentes descobertos:**

| Versão | Falha | Causa raiz | Fix |
|---|---|---|---|
| v1 | CUDA error sm_60 | Kaggle alocou Tesla P100 (sm_6.0), PyTorch só suporta sm_70+ | `resolve_device()` probe + CPU fallback |
| v2 | Mesmo problema | Kernel rodou com código v1 (sem probe) e bug em `check()` confundia qrels/triples com queries | (a) check() qualifica por config; (b) check pré-existente passou em CPU |
| v3 | build_triples 0 rows | Triples oficiais MS MARCO referenciam PIDs 0..8.8M, mas slice --small tem 0..9999 — intersecção estatística ≈ 0 | (a) build_triples agora `raise` em 0 rows; (b) notebook substitui baseline build por inspeção TSV |
| v4 | **COMPLETE** | — | mining filtra positives por collection + qrels sintéticos alinhados |

**Resultado final (v4):**
- Mining: **100 triples** (skipped_no_qrel=0, skipped_pool=0).
- Build_triples hardneg: **500 triples** (100 queries × 5 negs/query), schema correto `(query_id, query_text, positive_text, negative_text)`, dropped=0.
- Pipeline completo no Kaggle free tier (CPU): download → sintético qrels → Serafim encoding → FAISS HNSW → mining → build_triples.
- Tempo total: ~22min (~20min só no encoding CPU; T4 daria <1min).

**Decisões importantes registradas:**
- Em ambientes Kaggle/Colab/shared, **probe CUDA antes de usar** (alguns sm_xx não são suportados pelo PyTorch da imagem). Padrão: `resolve_device("auto")` em todos os scripts GPU.
- `--small` mode é validação de **code path**, não validação de **qualidade de dados**. Para mineração real (qrels.dev.small.tsv com PIDs reais), precisa `--full` ou alignment-aware download. Documentado em README do notebook.
- **Qrels sintéticos** são uma técnica válida para CI/free-tier testing: alinham IDs por construção, exercitam todo o pipeline, mas não geram triples semanticamente relevantes. Não usar para treinar modelo final.
- Invariante novo em mining: `mine_hard_negatives` só emite positives que estão na collection. Previne silent failures downstream.

**Pegadas didáticas:**
- Kaggle aloca GPU disponível (frequentemente P100, não T4) sem opção pela API. Para garantir T4 precisa configurar no UI manualmente.
- `build_triples` deve sempre erros em 0 rows com mensagem descritiva — silent failures são piores que crashes.
- Ordem de descoberta de bugs: device → check() → alignment de IDs. Cada fix expôs o próximo problema. CI helps, mas validação real só com dados reais.

**Kernel público (4 versões):**
https://www.kaggle.com/code/tardellistekel/ptbr-reranker-phase-2-hard-negative-mining

**TODO próxima sessão:**
- Phase 3: implementar `src/train.py` (CrossEncoder fit loop com W&B).
- Criar projeto W&B `ptbr-reranker`.
- Rotar token Kaggle (exposto em conversa).
- Considerar: refatorar `--small` para alignment-aware (download positives das qrels.dev.small) → mining real validado em free tier. Custo: tempo de refactor vs valor para o paper.

---

## 2026-05-18 — Phase 3 implementada: `src/train.py` funcional

**Hipótese:** Implementar treino real do cross-encoder usando sentence-transformers v5 (HF Trainer-style API) com W&B logging, codecarbon e checkpointing.

**O que fiz:**
- Substituí o skeleton de `src/train.py` por uma pipeline completa:
  - `TrainConfig` dataclass com defaults sensatos + `from_yaml()` parser (coerção defensiva de scalars YAML para float, p.ex. `1e-5` que YAML 1.2 parseia como string).
  - `triples_to_pairs()` expande o parquet `(query, positive, negative)` em `(sentence_A, sentence_B, label)` com label ∈ {1.0, 0.0}.
  - `train()` carrega Albertina, treina com `CrossEncoderTrainer` + `BinaryCrossEntropyLoss`, salva checkpoint `runs/<...>/best/`, persiste config em `training_config.json` para reprodutibilidade.
  - W&B opcional (`wandb_project` em yaml ativa) com tags e config completa serializada.
  - codecarbon opcional, best-effort. Loga emissões em `output_dir/emissions.csv` para a seção Environmental Impact do paper.
  - Device autodetect com fallback CPU (mesmo helper de `mine_hard_negatives.py`).
- Adicionei 8 testes em `tests/test_train_pipeline.py`:
  - 4 unit tests para `triples_to_pairs` (count, labels, query repeat, order)
  - 1 para `set_seed` (determinismo)
  - 1 para `resolve_device("cpu")`
  - 1 para `TrainConfig.from_yaml` (coerção de tipos)
  - 1 **slow smoke test** end-to-end em CPU com Albertina-100m real, 4 triples sintéticos, 2 steps, salva checkpoint

**Resultado:**
- 19 testes passando (não-slow), 1 passando com marker slow.
- Lint, format, mypy strict: todos limpos em 12 arquivos.
- Smoke test em CPU: **45s** (download de Albertina + treino + checkpoint).
- `runs/best/` é gravado, `training_config.json` também (artefato de reprodutibilidade).

**Decisões importantes:**
- Loss escolhida: `BinaryCrossEntropyLoss` (pairwise BCE). Comparado a alternativas:
  - `MarginMSELoss`: precisa de teacher scores (não temos).
  - `MultipleNegativesRankingLoss`: bi-encoder, não cross-encoder.
  - `ListMLELoss/LambdaLoss`: listwise, precisa de múltiplos negativos por query num único exemplo (poderia, mas BCE é mais simples).
  - BCE é a forma canônica em monoBERT / msmarco-MiniLM. Decisão registrada para o paper.
- Mixed precision automático: bf16 em CUDA (mais estável que fp16), float32 em CPU.
- `gradient_checkpointing` disponível via config (útil em A100 ↔ batch maior).

**TODO próxima sessão:**
- Criar projeto W&B `ptbr-reranker` e configurar API key.
- Notebook Kaggle Phase 3 (treino sintético em CPU/T4 — só para validar pipeline end-to-end de Phase 1+2+3).
- Rotar token Kaggle (exposto em conversa).
- Pensar em modo `--full` no Runpod (custo: ~$45 em A100 SXM Community para 30h).

---

## 2026-05-19 — Runpod RTX 4090 — primeira rodada paga

**Decisão de hardware:** RTX 4090 Community Cloud a $0.34/h venceu A100. ~2× mais compute por dólar (4090 0.6× A100 throughput a 0.34× preço de A100). Cabia até treino "full" em bf16 mas...

**4 bugs em cascata na primeira rodada paga (com fixes):**

| Bug | Causa | Fix |
|---|---|---|
| (1) Deploy "no resources" em COMMUNITY | Provisioner instável; tentei SECURE → cobrou $0.69/h em vez de $0.34 | Reduzi specs (vcpu 4→2, mem 16→8, disk 40→20). Retomou COMMUNITY ok. |
| (2) `transformers 5.8 + torch 2.4` incompat | runpod/pytorch:2.4 ships torch 2.4; transformers 5.x precisa torch>=2.5 | `pip install -U torch torchvision` → 2.6.0+cu124 |
| (3) DeBERTa + bf16 overflow | `attention_scores.masked_fill(~mask, torch.finfo(query_layer.dtype).min)` em modeling_deberta.py:276 — query é fp32 (autocast keeps inputs), attention_scores é bf16, finfo(fp32).min overflowa bf16 | Pin `transformers>=4.44,<5` + force fp32 para DeBERTa-family no train.py |
| (4) Mesma falha em fp16 | finfo(fp32).min também overflowa Half | Mudei "downgrade to fp16" → "force fp32" no train.py |

**Smoke test passou (4090, fp32):** 5 steps em 4.4s = 1.14 sps full pipeline. Treino 5 steps em 4.39s. Checkpoint salvo.

**Benchmark de throughput (bs=32, grad_accum=2 = effective 64, max_len=256, fp32):**
- **2.05 steps/sec** no 4090
- 1M triples → 31250 steps → **4.2h** ($1.43)
- 2M triples → **8.5h** ($2.90)
- 5M triples → 21h ($7.10)
- 40M (full mMARCO) → **169h ($58)** — não cabe nos $20

**fp32 é 50–60% mais lento que bf16 funcional teria sido.** Trade-off do bug DeBERTa upstream.

**Decisão de execução noturna:**
- Treino baseline: 2M triples (BM25 official triples), ~$3. Cabe em $20 com folga.
- Eval: opcional via Quati / mMARCO dev sample.
- Sem hardneg mining (mineração full requer FAISS 27GB índice, não cabe nos 20GB disk; e mining é cara). Hardneg fica para próxima rodada.

**Stack final que funcionou:**
- Runpod RTX 4090 Community Cloud sm_89
- runpod/pytorch:2.4 base → upgraded to torch 2.6.0+cu124 / torchvision 0.21
- transformers 4.57.6 (pinned <5)
- sentence-transformers 5.x trainer
- BinaryCrossEntropyLoss
- mixed_precision: fp32 forced for Albertina (DeBERTa-family)

**Verificação DeBERTa bf16 (durante a noite):** Confirmei que o bug do `finfo(query_layer.dtype).min` está em **TODAS** as versões de transformers (4.x e 5.x). É 1 linha, fix trivial: trocar para `attention_scores.dtype`. Apliquei via `patch_deberta_attention_dtype()` em src/train.py — modifica o arquivo on-disk de forma idempotente antes do import. bf16 passou no teste após patch. **Reverti o "force fp32 for DeBERTa"** — agora bf16 funciona corretamente.

**Bench bf16 real (4090, bs=64 effective, max_len=256):**
- **2.91 sps** com patch + bf16 (vs 2.05 sps fp32, +42%)
- 2M triples / 62500 steps → **5.96h** = $2.03
- 40M (full) → 119h = $40.5 (ainda over budget)

**Build_triples bug paralelo:** o emit one-row-per-batch criava 2M row groups → pyarrow falhava ao reler com "Exceeded size limit". Fix: buffer de 10k rows, flush em batches. Reduziu de 2M row groups → 200. Commit `6a67333`.

**Treino noturno iniciado 2026-05-19 02:40 UTC:**
- Config: `train_baseline_2M_runpod.yaml`
- 2M triples baseline (BM25 official negs), 1 epoch, bs=64 effective, max_len=256, lr 2e-5, bf16
- save_every_steps=2000 (≈11min checkpoints), log_every_steps=100
- Estimativa: 6h, $2 GPU
- PID 8197, log em `logs/train_2M.log`

**Treino concluído 2026-05-19 08:32 UTC:**

| Métrica | Valor |
|---|---|
| Duração | **5h 52min** (21,147s) |
| Steps | 62,500 (1 epoch completo) |
| Throughput | 2.96 sps |
| Loss inicial | 0.6928 |
| Loss média treino | 0.1922 |
| **Loss final** | **0.127** (5.4× redução) |
| GPU util | 91.3% |
| GPU power avg | 339W |
| **Energy** | **2.65 kWh** |
| **CO2eq** | **1.70 kg** (Taiwan, Taoyuan, PUE 1.0) |
| Cost | ~$2.00 ($0.34/h × 5.87h) |
| Output | runs/baseline_2M/best/ (~557MB safetensors) |

**Trajectory de loss:**
- step 0 → 0.69 (random init)
- step 5000 → 0.25
- step 15000 → 0.21
- step 25000 → 0.19
- step 35000 → 0.171
- step 50000 → 0.144
- step 62500 → 0.127 (final)

**Modelo v0.1 pronto** — primeiro checkpoint utilizável publicável. Tempo total da rodada Runpod (incluindo setup, fixes, build_triples, treino): ~6h 30min, custo ~$2.20.

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
