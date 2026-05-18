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
| _placeholder_ | _2026-MM-DD_ | albertina-100m | mmarco+hardneg | softmax | 2e-5 | 64 | 1 | 256 | 42 | – | – | – | – | – | – | pending | first real run |

## Ablations planejadas

- [ ] Base model: Albertina-100m vs BERTimbau-base vs DeBERTinha-40m
- [ ] Hard negatives: nenhum (só BM25) vs 3 negs vs 7 negs vs 15 negs
- [ ] Faixa de rank: (5-50) vs (10-100) vs (20-200)
- [ ] Loss: softmax CE vs pairwise hinge vs MultipleNegativesRankingLoss
- [ ] Max length: 128 vs 256 vs 384
- [ ] Epochs: 1 vs 2 vs 3
- [ ] Learning rate: 1e-5 vs 2e-5 vs 5e-5
- [ ] Effective batch size: 32 vs 64 vs 128
