# Dados e resultados locais

Os comandos gravam resultados em `outputs/`; não exigem uma pasta `data/` na raiz. Se precisar organizar entradas próprias, as pastas opcionais abaixo podem ser criadas localmente e ficam fora do Git.

## Entradas originais — `data/raw/`

Dados originais obtidos pelo gerador ou fornecidos em JSON/JSONL/Parquet com report e summary. Pesos e corpus completo não são versionados. Cada execução de geração preserva sua seleção original em generate/original.parquet, com hash no manifesto.

## Entradas preparadas — `data/processed/`

Entradas preparadas podem ser colocadas aqui pelo pesquisador. CSVs injetados, manifestos de alteração, claims e evidências são derivados; não são referência humana. Os comandos gravam por padrão em outputs/runs/<run_id>/, sem sobrescrever dados anteriores.

## Referências humanas — `data/reference/`

Referências independentes e anotações em andamento ficam separadas dos rótulos do gerador e das predições. Identifique dataset_id, selection_id, origem, revisão e hash. Não versione automaticamente dados pessoais ou anotações completas. Somente a exportação validada pode ser usada como referência humana; triagem não é gold.
