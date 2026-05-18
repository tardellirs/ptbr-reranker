# Contribuindo com o PTBR-Reranker

Obrigado pelo interesse em contribuir! Este projeto é open-source (MIT) e contribuições são bem-vindas — sejam elas correções de bugs, melhorias de documentação, novos testes de qualidade, ou propostas de novos experimentos.

## Como contribuir

### Reportar bugs

Use o template de issue "Bug report" em `.github/ISSUE_TEMPLATE/bug_report.md`. Inclua:
- Versão do `ptbr-reranker` (`pip show ptbr-reranker`)
- Python, PyTorch, e CUDA versions
- Comando exato que reproduz o problema
- Saída completa (stack trace)

### Sugerir melhorias

Use o template "Feature request". Descreva o caso de uso antes de propor a implementação.

### Pull Requests

1. Fork o repositório e crie uma branch a partir de `main`.
2. Instale as dependências de dev: `pip install -e ".[dev]"`
3. Configure os pre-commit hooks: `pre-commit install`
4. Faça suas alterações com testes correspondentes.
5. Rode `ruff check . && mypy src/ && pytest` antes de abrir o PR.
6. Atualize o `CHANGELOG.md` na seção `[Unreleased]`.
7. Abra o PR usando o template em `.github/PULL_REQUEST_TEMPLATE.md`.

### Padrões de código

- **Formatação**: ruff (automatizado via pre-commit).
- **Type hints**: obrigatórios em código novo (`mypy --strict` deve passar).
- **Testes**: cobertura mínima de 80% para módulos em `src/`.
- **Commits**: estilo [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- **Idioma**: docstrings e comentários em inglês; mensagens de commit em inglês; discussões podem ser em PT-BR ou inglês.

## Reproduzindo experimentos

Veja `docs/reproducibility.md` para o protocolo completo. Resumo:

```bash
# 1. Download de dados
python data/download_mmarco.py

# 2. Hard negative mining (precisa GPU)
python data/mine_hard_negatives.py --config configs/train_hardneg.yaml

# 3. Treino
python src/train.py --config configs/train_hardneg.yaml

# 4. Avaliação
python src/eval_mmarco.py --checkpoint runs/best
python src/eval_miracl.py --checkpoint runs/best

# 5. Testes de qualidade
pytest tests/ -v
```

## Catalogação científica

Este projeto tem ambição de publicação. Por isso:
- Todos os experimentos devem ser logados no W&B (projeto `ptbr-reranker`).
- Adicione cada run novo (mesmo failures) ao `docs/experiments_log.md`.
- Decisões de design ou debugging vão para `docs/lab_notebook.md` datadas.

## Código de Conduta

Este projeto adota o [Contributor Covenant 2.1](CODE_OF_CONDUCT.md). Espera-se que todos os contribuidores sigam essas diretrizes.

## Licença

Ao contribuir, você concorda que suas contribuições serão licenciadas sob a [MIT License](LICENSE).
