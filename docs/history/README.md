# Registros históricos

Estes arquivos documentam a inspeção e a migração de outubro de 2026. Foram
agrupados aqui para separar a procedência histórica dos guias de uso atuais.

| Registro | Finalidade |
|---|---|
| [migration.md](migration.md) | Origem, adaptações e limites da migração |
| [audit.md](audit.md) | Diagnóstico da implementação de origem |
| [items_to_review.md](items_to_review.md) | Pendências identificadas naquela inspeção |
| [environment_observed.json](environment_observed.json) | Ambiente observado na origem |
| [source_inventory.json](source_inventory.json) | Inventário de arquivos e hashes |
| [migration_sources.json](migration_sources.json) | Mapeamento de procedência |
| [scientific_implementation_audit.json](scientific_implementation_audit.json) | Comparações das implementações |
| [historical_regression.json](historical_regression.json) | Registro da regressão sobre probabilidades históricas |
| [validation_results.json](validation_results.json) | Resultados estruturados da validação histórica |

Os JSONs foram movidos sem modificar seus conteúdos. Caminhos e hashes dentro
desses registros descrevem o momento da inspeção, e não devem ser reescritos
retroativamente por causa da reorganização dos diretórios. Os testes e seus
snapshots continuam em `tests/`. Os relatórios de validação estão em
[../validation.md](../validation.md), com datas e escopo próprios.
