# Experimentos da Fase I

Este catálogo prepara a avaliação sem criar scripts vazios nem resultados simulados.

| Experimento | Objetivo | Referência/entrada necessária | Estado |
|---|---|---|---|
| E0 | Validade técnica e semântica das injeções | Originais, logs reversíveis, revisão independente das mudanças | Auditoria técnica implementada; validação semântica pendente |
| E1 | Qualidade e sobrevivência das proposições na extração | Claims e sentenças de referência anotadas | Pipeline/IDs preparados; avaliação pendente |
| E2 | Cobertura e ranking das evidências | Julgamentos humanos e conjuntos suficientes, inclusive fora do pool | Ferramenta de referência preservada; avaliação pendente |
| E3 | Decisão NLI isolada | Pares claim–evidência de referência, sem depender da recuperação automática | Modelo/agregação preparados; avaliação pendente |
| E4 | Categorias de inconsistência | Categorias humanas e distinção entre avaliação condicionada/ponta a ponta | Classificador preparado; avaliação pendente |
| E5 | Pipeline completo e diagnóstico de falhas | Referência independente com IDs rastreáveis por etapa | Orquestração preparada; experimento científico pendente |
| E6 | Comparação com baselines factuais | Versões, modelos, entradas e referências comuns | Baselines adicionais não implementados |

Cada experimento deve apontar para um run_id, configuração resolvida e referência identificada por hash. Rótulos do gerador são metadados automáticos, não gold. A/B/C devem ser reportados separadamente; generalização à população exige a amostra aleatória A adequadamente revisada. Não confundir o FENICE local modificado com o baseline upstream original.
