# Panorama de Embeddings para Português — Oportunidades de Contribuição

## Estado da arte atual

### Encoders de base (foundation models)
- **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) — baseline canônico desde 2020, ainda o mais baixado, mas defasado.
- **Albertina PT*** (PORTULAN) — DeBERTa-v3 adaptado, variantes 100M / 900M / 1.5B em PT-BR e PT-PT. Supera BERTimbau em quase todos os benchmarks. Paper: arXiv:2403.01897.
- **DeBERTinha** (40M) — surpreendentemente competitivo para o tamanho. arXiv:2309.16844.
- **BERTugues** — alternativa pretreinada, menos adotada que BERTimbau.
- **PTT5-v2** (UNICAMP, 2024) — encoder-decoder T5 para PT. arXiv:2406.10806.

### Sentence encoders (o que é de fato "embedding model")
- **Serafim PT*** (PORTULAN, EPIA 2024 — arXiv:2407.19527) — atual estado da arte para sentence embeddings em PT. Famílias 100M / 335M / 900M, com variantes STS (CoSENTLoss) e IR (GISTEmbedLoss). Treinado em OPUS + mMARCO 40M triplas.
  - **Limitações críticas:** janela de contexto de apenas 128 tokens; variante IR é PT-PT; sem inference provider support.
  - Serafim-900M em mMARCO MRR@10: 0.8539 vs. 0.7414 do melhor competidor em inglês.
- **rufimelo/bert-large-portuguese-cased-sts** — fine-tune comunitário do BERTimbau-Large em ASSIN/ASSIN2.
- **tcepi/sts_bertimbau** — outro fine-tune comunitário do BERTimbau para STS.
- **ricardo-filho/bert-base-portuguese-cased-nli-assin-2** — BERTimbau base em ASSIN2 NLI.

### Decoders generativos (não são embedding models)
Gervásio (8B/70B), Tucano, Sabiá / Sabiá-2 / Sabiá-3 (Maritaca), Cabrita, TeenyTinyLlama, Bode.
**Nenhum foi adaptado como embedding model** (via LLM2Vec, E5-mistral, GritLM, etc.).

## Lacunas reais do ecossistema

1. **Não existe embedding model PT baseado em LLM (classe 7B+).** Em inglês isso é mainstream desde 2024 (E5-Mistral-7B, GritLM-7B, NV-Embed-v2, LLM2Vec — consistentemente topam o MTEB). Em PT, **tudo é encoder-only sub-1B**.
2. **Não existe modelo PT com long-context.** Todos travam em 128–512 tokens. Em inglês temos BGE-M3 e Jina-v3 até 8192. Para RAG sério (jurídico, médico, acadêmico) isso é inviabilizante.
3. **Não existe BEIR-PT** consolidado. Só temos mMARCO (traduzido), Quati (50 queries, native web) e JurisTCU (150 queries, legal). Holandês tem BEIR-NL (2025), polonês tem BEIR-PL — PT-BR ainda é um espaço aberto para contribuição de benchmark.
4. **Não existe cross-encoder / reranker PT** de qualidade. A pipeline de duas etapas (retriever + reranker) está incompleta para PT.
5. **Não existe IR sentence encoder PT-BR dedicado.** Serafim-IR é PT-PT.
6. **MTEB-PT não está consolidado** — sem leaderboard de referência integrado para PT.
7. **Split PT-BR vs PT-PT.** Albertina/Serafim oferecem ambos, mas não há IR-optimized PT-BR. PT-BR tem vocabulário, ortografia e distribuição de domínios diferentes (pegada de internet maior).
8. **Domínios específicos sem embeddings com benchmark público:** jurídico brasileiro, clínico/biomédico, acadêmico/científico em PT.

## Benchmarks existentes para PT

| Benchmark | Tarefas | Status |
|---|---|---|
| ASSIN / ASSIN2 | STS (1–5), RTE | Estabelecido, nativo PT-BR |
| TweetSentBR | Sentimento (Twitter, 3 classes) | Nativo PT-BR |
| HateBR | Discurso de ódio (Instagram) | Nativo PT-BR |
| mMARCO PT | Passage retrieval | Traduzido do MS MARCO |
| ExtraGLUE (PORTULAN) | 14 tarefas (SST-2, MRPC, STS-B, RTE, MNLI…) | Traduzido do GLUE/SuperGLUE |
| CLARIN-PT-LDB | MMLU-PT, MuSR-PT, cultura/civilidade | Só LLMs generativas |
| Open PT LLM Leaderboard (eduagarcia) | 14 benchmarks generativos | Só LLMs generativas |
| PoETa v2 | Robustez de LLM | Só LLMs generativas |

**Observação crítica:** benchmarks para retrieval, clustering e busca semântica fora de STS estão essencialmente ausentes para PT. O único de retrieval (mMARCO) é traduzido, não nativo.

## Desenvolvimentos recentes (2024–2026)

- **Serafim PT*** (jul/2024, EPIA 2024) — primeiro family de sentence encoders sistemático e aberto para PT em múltiplos tamanhos.
- **Albertina 1.5B PT-BR** (2024) — maior encoder aberto para PT.
- **Tucano** (nov/2024) — LLM generativa from-scratch mais rigorosa para PT.
- **CLARIN-PT-LDB** (mar/2026, PROPOR 2026) — leaderboard de LLM para PT europeu, cultura e civilidade.
- **MMTEB** (fev/2025) — expansão do MTEB para 250+ línguas, mas cobertura PT permanece rasa.
- **AMALIA** (2025) — LLM PT europeu totalmente aberta.
- **Clinical NER PT** (mar/2026) — avaliação de BERT vs LLMs em NER clínico para PT.
- **Sem novo embedding focado em retrieval desde o Serafim (jul/2024)** — a lacuna persiste.

## Coleção PORTULAN/LIACC no HuggingFace

**Albertina** (encoder, fill-mask):
- `PORTULAN/albertina-100m-portuguese-ptpt-encoder`
- `PORTULAN/albertina-100m-portuguese-ptbr-encoder`
- `PORTULAN/albertina-900m-portuguese-ptpt-encoder`
- `PORTULAN/albertina-900m-portuguese-ptbr-encoder`
- `PORTULAN/albertina-1b5-portuguese-ptbr-encoder`

**Serafim** (sentence encoder / IR):
- `PORTULAN/serafim-100m-portuguese-pt-sentence-encoder`
- `PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir`
- `PORTULAN/serafim-335m-portuguese-pt-sentence-encoder`
- `PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir`
- `PORTULAN/serafim-900m-portuguese-pt-sentence-encoder`

**Gervásio** (decoder, text generation):
- `PORTULAN/gervasio-8b-portuguese-ptpt-decoder`
- `PORTULAN/gervasio-70b-portuguese-ptpt-decoder`
- Variantes 4-bit quantizadas

## Recomendações de contribuição (ordenadas por impacto)

### Tier 1 — máximo impacto, publicável em ACL / EMNLP / COLING

**(a) Embedding model PT-BR classe 7B com long-context**
Usar decoder PT 7B (Gervásio-8B, Sabiá-2 ou Tucano) ou multilingual Mistral, aplicando LLM2Vec / GritLM / E5-Mistral contrastive training em dados PT. Avaliar em ASSIN2-STS, mMARCO-PT e benchmarks novos.
- Primeiro embedding model em escala LLM para PT.
- Primeiro a suportar long-context (2048+ tokens) em PT.
- Diretamente comparável ao estado da arte inglês no MTEB.

**(b) BEIR-PT — benchmark de retrieval nativo**
Curar 6–10 datasets nativos PT para retrieval (jurídico brasileiro, biomédico, notícias, governo, acadêmico, FAQ). Preenche a lacuna de avaliação mais consequente e vira artefato citável permanente (mMARCO tem centenas de citações por essa razão).

Os dois juntos seriam a maior contribuição possível ao ecossistema PT.

### Tier 2 — contribuição sólida (PROPOR / STIL / EACL)

- **Cross-encoder / reranker PT-BR** treinado em mMARCO-PT com hard-negative mining. Completa a pipeline de duas etapas.
- **Serafim-PT-BR-IR** — variante PT-BR dedicada, treinada em BrWaC + notícias brasileiras + jurídico brasileiro.
- **Versão long-context do Serafim** (512 → 4096 tokens) via interpolação RoPE.

### Tier 3 — nichos comerciais valiosos

- **Legal-Serafim** — embedding para jurisprudência brasileira (alta demanda comercial).
- **Clinical-Serafim** — português clínico/biomédico.
- **STS benchmark PT nativo** além do ASSIN2 (pares de sentenças nativos, não traduzidos).

## Custos no Runpod (preços maio/2026)

### A100 80GB
| Tier | Variante | $/h |
|---|---|---|
| Community Cloud | A100 PCIe | $1.19 |
| Community Cloud | A100 SXM | $1.39 |
| Secure Cloud | A100 PCIe | $2.10 |
| Secure Cloud | A100 SXM | $2.45 |

### H100 80GB
| Tier | Variante | $/h |
|---|---|---|
| Community Cloud | H100 PCIe | $1.99 |
| Community Cloud | H100 SXM | $2.69 |
| Secure Cloud | H100 PCIe | $3.51 |
| Secure Cloud | H100 SXM | $4.76 |

### Estimativas de fine-tuning de modelo 7B

| Cenário | GPU | Duração | Custo |
|---|---|---|---|
| LoRA fine-tune (STS/IR, dataset pequeno) | A100 SXM | 5–10h | ~$7–14 |
| Cross-encoder em mMARCO-PT | A100 SXM | 24h | ~$33 |
| Full contrastive training | A100 SXM | 48h | ~$67 |
| Full contrastive training | A100 SXM | 72h | ~$100 |
| Contrastive 7B (LLM2Vec/GritLM) | 2× A100 SXM | 2–4 dias | ~$130–270 |
| Mesmo em H100 (mais rápido) | 2× H100 SXM | 1–2 dias | ~$130–260 |
| QLoRA em RTX 4090 (orçamento mínimo) | 1× 4090 ($0.74/h) | 1–3 dias | ~$18–55 |

### Notas práticas
- 1× A100 80GB cabe 7B em BF16 para full fine-tuning; para contrastive training usar 2–4 GPUs para batch sizes decentes.
- H100 é ~30–40% mais rápido em 7B mas 93% mais caro/hora em SXM Secure. Community Cloud aproxima os preços.
- Para contrastive em mMARCO (40M triplas): planejar 2–4 dias em 2× A100 ($55–160) ou 1–2 dias em 2× H100 ($130–260).
- QLoRA reduz VRAM para 12–24GB, viabilizando A40 ($0.49/h) ou 4090 ($0.74/h).

### Custo total esperado por tier
- **Tier 1 completo** (treino, ablations, retries): **$150–500**.
- **Tier 2**: menos de **$50–100**.
- **Tier 3 com QLoRA**: **$20–50**.

## Plano pragmático sugerido

Para virar referência com orçamento e tempo realistas:

1. **Começar pelo Tier 2 (cross-encoder/reranker PT-BR)** — valida pipeline, testa Runpod, gera artefato útil. ~$30, 2–3 semanas.
2. **Em paralelo, montar o BEIR-PT** — é trabalho de curadoria, não custa GPU. Mesmo um BEIR-PT com 4–5 datasets vira referência se for o primeiro.
3. **Atacar o Tier 1**: embedding LLM-based long-context, avaliado no próprio BEIR-PT.

Esse caminho dá artefatos publicáveis em três pontos (workshop PROPOR → STIL/EACL → ACL/EMNLP) com custo total de GPU abaixo de $500.

## Referências principais

- Serafim: arXiv:2407.19527 — https://arxiv.org/html/2407.19527v1
- Albertina: arXiv:2403.01897 — https://arxiv.org/html/2403.01897
- ExtraGLUE: arXiv:2404.05333 — https://arxiv.org/html/2404.05333
- DeBERTinha: arXiv:2309.16844 — https://arxiv.org/pdf/2309.16844
- TeenyTinyLlama: arXiv:2401.16640 — https://arxiv.org/html/2401.16640v1
- Tucano: https://www.sciencedirect.com/science/article/pii/S2666389925001734
- mMARCO: arXiv:2108.13897 — https://arxiv.org/pdf/2108.13897
- MMTEB: arXiv:2502.13595 — https://arxiv.org/html/2502.13595v3
- CLARIN-PT-LDB: arXiv:2603.12872 — https://arxiv.org/abs/2603.12872
- BEIR-NL: https://aclanthology.org/2025.bucc-1.5.pdf
- BERTimbau: https://huggingface.co/neuralmind/bert-base-portuguese-cased
- Portuguese NLP resources: https://github.com/ajdavidl/Portuguese-NLP
- Runpod pricing: https://deploybase.ai/articles/runpod-gpu-pricing
- Runpod fine-tuning guide: https://www.runpod.io/articles/guides/how-to-fine-tune-large-language-models-on-a-budget
