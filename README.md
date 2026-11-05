# Avaliação de consistência factual — Fase I

Ferramenta de pesquisa para verificar afirmações em resumos de documentos longos do GovReport e diagnosticar falhas de extração, recuperação de evidências, NLI e classificação de erros.

O projeto também oferece amostragem para revisão humana, interface bilíngue e salvamento automático das análises em disco. **Predições, traduções e rótulos do gerador não são referência gold.**

## Comece por aqui

| Quero… | Onde encontrar |
|---|---|
| Instalar e executar o pipeline | [Guia de execução](docs/usage.md) |
| Entender o método e os limites científicos | [Metodologia](docs/methodology.md) |
| Anotar na interface | [Guia da auditoria](docs/audit-interface.md) |
| Salvar minhas análises no computador | [Salvamento automático](docs/review-autosave.md) |
| Preparar traduções em português | [Auditoria bilíngue](docs/bilingual-audit.md) |
| Testar OPUS-MT, NLLB ou TranslateGemma | [Exemplos de tradução](examples/translation/README.md) |

A documentação completa está organizada em [docs/README.md](docs/README.md). Execute os comandos a partir da raiz, com o ambiente do projeto ativado.

## Abrir uma auditoria existente

Se já possui um `avaliacao.html`, substitua o caminho abaixo pelo arquivo gerado:

```bash
python scripts/serve_review.py \
  --html /caminho/avaliacao.html \
  --annotations outputs/annotations/minhas_analises.json \
  --open
```

Mantenha o terminal aberto e use a página local indicada. As alterações confirmadas como **Salvo em arquivo ✓** ficam no JSON informado, com histórico das versões anteriores. Para continuar outro dia, repita o comando com o mesmo arquivo de análises.

Para transferir anotações feitas no HTML antigo, exporte o JSON naquela janela e importe-o na nova interface uma vez. [Procedimento de recuperação](docs/review-autosave.md).

## Executar cinco documentos

Após instalar as dependências e os modelos spaCy conforme o [guia de execução](docs/usage.md):

```bash
python scripts/run_pipeline.py \
  --from-stage generate \
  --num-docs 5 \
  --device auto
```

O comando imprime o diretório da execução em `outputs/runs/`. Cada etapa pode ser inspecionada e reutilizada conforme o guia. GPU é usada quando disponível; em CPU, a execução pode ser demorada.

## Estrutura

```text
configs/              Configurações dos experimentos e da referência humana
src/factual_eval/     Código do pipeline, tradução e interface
scripts/              Comandos de execução e auditoria
examples/translation/ Testes manuais independentes de tradução
tests/               Testes automatizados e fixtures de regressão
docs/                Guias de uso, metodologia e histórico
outputs/             Resultados, análises e cache locais; fora do Git
```

Os módulos científicos e os comandos principais mantêm seus caminhos. Os antigos testes manuais `scripts/test_translation*.py` foram organizados em `examples/translation/`. O catálogo de experimentos está em [docs/experiments.md](docs/experiments.md), e as convenções de dados em [docs/data.md](docs/data.md).

## Dados e reprodução

- `outputs/runs/` mantém resultados e manifestos de execuções; um `--run-id` existente é recusado.
- `outputs/annotations/` mantém revisões humanas e histórico; preserve o mesmo JSON para continuar uma revisão.
- `outputs/translation-cache.sqlite3` permite reaproveitar traduções entre execuções.
- Resultados, análises e dados locais ficam fora do Git. Os [registros históricos](docs/history/README.md) e as fixtures de teste são preservados para rastreabilidade.

A exportação gold exige revisão humana concluída e validação pelo exportador. As traduções auxiliam a leitura; o inglês continua sendo a referência original.

## Testes

```bash
python -B -m unittest discover -s tests -v
```

A suíte padrão usa dados sintéticos e modelos simulados. Integrações reais com GPU, Ollama e navegador são opcionais: [instruções](tests/integration/README.md). Resultados de verificações anteriores estão em [docs/validation.md](docs/validation.md).

## Procedência e licença

O código deriva de uma implementação local modificada do FENICE e não calcula automaticamente o score agregado do FENICE original. A procedência está registrada no [histórico da migração](docs/history/migration.md). Consulte [LICENSE](LICENSE) e [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
