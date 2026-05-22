# Experiments Log — Tabela mestra de runs

Toda run de treino, incluindo failures e debugging, vai aqui. Esta tabela vira o apêndice "All experiments and hyperparameter exploration" do paper.

## Convenções

- **ID**: identificador da run no W&B.
- **Status**: `keep` (resultado válido a reportar) / `discard` (resultado descartado, mas registrado) / `retry` (a refazer).
- **Custo**: horas de GPU × preço/hora. Sempre preencher.
- **Métricas**: MRR@10 e nDCG@10 em mMARCO-PT dev (BM25 candidates, top-1000 → top-100 rerank). Adicionar MIRACL-PT nDCG@10 quando rodado.

## Runs

| ID | Data | Base | Dados | Loss | LR | Batch | Epochs | Max len | Seed | MRR@10 | nDCG@10 | MIRACL nDCG@10 | ECE | GPU·h | $ | Status | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_2M (v0.1) — top-100 | 2026-05-19 | albertina-100m | mmarco 2M BM25-only | BinaryCE | 2e-5 | 32×2=64 | 1 | 256 | 42 | 0.2810 | 0.3232 | – | – | 4090 ~4.5h | ~$3.2 train+eval | keep | rerank top-100, 28min na 4090 |
| baseline_2M (v0.1) — top-1000 | 2026-05-20 | albertina-100m | mmarco 2M BM25-only | BinaryCE | 2e-5 | 32×2=64 | 1 | 256 | 42 | **0.2945** | **0.3437** | – | – | 5090 ~4.7h | ~$4.7 eval | keep | mesmo checkpoint do top-100, eval top-1000 pra comparação direta com Unicamp-DL; **bate v1.0 head-to-head** |
| baseline_10M (v1.0) — top-1000 | 2026-05-20 | albertina-100m | mmarco 10M BM25-only | BinaryCE | 2e-5 | 32×2=64 | 1 | 256 | 42 | 0.2876 | 0.3385 | – | – | 5090 ~19h+4h | ~$15 train+eval (cross-account) | discard | 5× mais dados → -0.7pp vs v0.1; train_loss colapsou (0.0159) — saturou sem hard negatives; mantido como ablation no paper |
| hardneg_v1x (v1.x) — top-1000 | 2026-05-22 | albertina-100m | mmarco 2.91M Serafim-IVFPQ-HN rank 10-100 7-per-q | BinaryCE | 2e-5 | 32×2=64 | 1 | 256 | 42 | 0.2159 | 0.2504 | – | – | Salad 3090 + GPUhub 4090 ~13h | ~$6 mining+train+eval | discard | hard negs via IVFPQ retrieval **PIORARAM** -7.9pp MRR@10. Hipótese: PQ quantization (50× compress) deslocou ranking faz neg sampling em rank 10-100 não corresponder ao real top-100. Retry futuro precisa IVFFlat com VRAM ≥32GB. Loss final 0.140 indica modelo treinou em sinal de baixa qualidade |

## Ablations planejadas

- [ ] Base model: Albertina-100m vs BERTimbau-base vs DeBERTinha-40m
- [ ] Hard negatives: nenhum (só BM25) vs 3 negs vs 7 negs vs 15 negs
- [ ] Faixa de rank: (5-50) vs (10-100) vs (20-200)
- [ ] Loss: softmax CE vs pairwise hinge vs MultipleNegativesRankingLoss
- [ ] Max length: 128 vs 256 vs 384
- [ ] Epochs: 1 vs 2 vs 3
- [ ] Learning rate: 1e-5 vs 2e-5 vs 5e-5
- [ ] Effective batch size: 32 vs 64 vs 128
