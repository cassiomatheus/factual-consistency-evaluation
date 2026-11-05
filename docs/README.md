# Documentação

## Usar o projeto

| Guia | Conteúdo |
|---|---|
| [Execução](usage.md) | Instalação, teste com cinco documentos, etapas, saídas, retomada, configuração e exportação gold |
| [Interface](audit-interface.md) | Os três passos da anotação, evidências, decisões e pendências |
| [Salvamento](review-autosave.md) | JSON em disco, migração das anotações, histórico e recuperação |
| [Tradução](bilingual-audit.md) | Ollama local, cache, qualidade, verificação humana e HTML bilíngue |
| [Dados](data.md) | Entradas próprias, dados derivados e referência humana |

## Entender e validar

| Documento | Conteúdo |
|---|---|
| [Metodologia](methodology.md) | Fronteiras científicas e interpretação das predições |
| [Contratos do pipeline](pipeline.md) | Etapas, formatos, IDs e validações |
| [Experimentos](experiments.md) | Catálogo E0–E6 e entradas necessárias |
| [Validações](validation.md) | Relatórios datados dos testes executados |

Os testes manuais de modelos estão em [examples/translation](../examples/translation/README.md).
Auditorias e inventários da migração ficam em [history/](history/README.md).

## O que deve permanecer separado

O código fica em `src/`, os pontos de entrada em `scripts/` e os testes
automatizados em `tests/`. As fixtures históricas são necessárias aos testes de
regressão e não são resultados descartáveis. Configurações em `configs/` têm
cópias em `src/factual_eval/resources/` para a instalação do pacote fora do checkout;
essa duplicação é intencional e verificada por testes.

`outputs/` contém dados locais e não é versionado. Sua presença e tamanho no
computador não representam arquivos incluídos no Git. Análises, traduções e
resultados já produzidos não foram movidos ou removidos na reorganização.
