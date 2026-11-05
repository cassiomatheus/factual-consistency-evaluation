# Migração autorizada da Fase I

Origem: implementação local FENICE em `/home/cassiosilva/FENICE`, inspecionada em 22 de outubro de 2026. O diretório original tinha alterações não commitadas e exclusões preexistentes; não foi restaurado, limpo nem reescrito. O novo diretório é um repositório Git independente, sem remote ou publicação configurados pela migração.

O inventário com hashes é `source_inventory.json`. `migration_sources.json` associa os arquivos selecionados aos novos módulos e registra o commit histórico da origem, sem alegar que esse commit representa os arquivos modificados. `scientific_implementation_audit.json` compara as definições por AST; igualdade sintática é evidência limitada de preservação, não comprovação da validade científica.

## Código reaproveitado

| Origem | Destino | Adaptação |
|---|---|---|
| gerador_predicado_corref_entity_V5.py | data/injection.py | Funções científicas preservadas; seleção e saída em adaptador configurável |
| pipeline/01_extract_claims_v7_melhorado_novo.py | claims/extraction.py | Configuração e carregamento explícitos; IDs adicionais e checagem de contagens |
| 01_extract_claims_v7_melhorado.py da raiz | claims/enriched.py | Variante separada, spaCy md e regras SVO mantidas |
| metric/claim_extractor/claim_extractor.py | claims/model.py | Parâmetros de geração configuráveis com mesmos defaults; importação morta de utils retirada |
| pipeline/02_retrieve_evidence_v7_melhorado_novo.py | retrieval/pipeline.py | Caminhos/modelos/pesos configuráveis; raio e lambda passados explicitamente para respeitar configuração |
| metric/nli/nli_aligner.py | nli/model.py | chunks local mínimo; revisão explícita; checagem de ordem dos labels; sincronização CUDA guardada em CPU |
| pipeline/03_compute_nli_v7_melhorado_novo.py | nli/evaluation.py | Mesma aritmética e ordem de operações; literais de pesos externalizados |
| pipeline/04_classify_factual_status_melhorado_novo.py | taxonomy/classification.py | Mesmo filtro e prioridades; arquivo de matriz associado à execução |
| metric/utils/utils.py | utils/batching.py | Apenas chunks, cuja implementação foi preservada |
| Projeto de referência humana, preparação/amostragem/HTML/exportação | reference/ | Funções preservadas; checkpoint revisions e proveniência acrescentados |
| pipeline/06_audit_qwen.py | reference/qwen.py | Auditoria opcional de pares preservada; antigo main e caminhos fixos retirados |
| antonimos_govreport.json | resources/antonimos_govreport.json | Cópia idêntica, hash registrado |
| LICENSE.txt do histórico Git | LICENSE | Texto preservado com créditos em THIRD_PARTY_NOTICES.md |

A cópia de consulta do extrator dentro do projeto humano era idêntica em bytes ao extrator da raiz e não foi mantida como uma segunda implementação. FastCoref, API agregada FENICE, taxonomias alternativas, utilitários posicionais e diagnósticos antigos foram catalogados, não apagados da origem. A lista de revisão detalha as razões e dependências.

## Diferenças conhecidas e explícitas

- Cada comando cria uma nova execução e recusa sobrescrita. Diretórios, nomes de arquivos e mensagens operacionais mudam; decisões científicas baseline não devem mudar.
- Etapas downstream usam o lote produzido, eliminando o antigo caminho de taxonomia de 50 documentos após uma extração de cinco. A seleção default de extração permanece cinco; geração permanece 50.
- Modelos deixam de ser carregados como efeito de importar módulos. Downloads automáticos de modelos spaCy foram substituídos por instalação explícita.
- Os checkpoints HF baseline são fixados nas revisões observadas no cache da origem. Isso congela a reprodução futura, mas não prova qual revisão gerou cada arquivo antigo.
- `NLIAligner` verifica a ordem esperada entailment/neutral/contradiction. Checkpoints incompatíveis falham, sem reordenar logits silenciosamente.
- CPU é uma opção explícita de execução. O aquecimento preserva as chamadas originais, mas só sincroniza CUDA em dispositivo CUDA. A execução GPU baseline não foi alterada por essa guarda.
- Contratos falham em vez de omitir silenciosamente claims malformadas, documentos sem saída ou mudanças de contagem. Falhas ficam registradas, sem ser convertidas em sucesso científico.
- Identificadores são adicionados, não substituem os campos históricos. Na geração de subintervalos, source_index conserva a posição original e o prefixo do RNG é processado.
- O manifesto de alterações agrega offsets derivados dos marcadores e logs. O log original é preservado integralmente. Esse registro é técnico e não afirma contradição.
- A preparação humana aceita também sentence_source/sentence_index do extrator baseline como metadados de origem. Para entradas enriquecidas já usadas, a escolha dos campos anteriores permanece prioritária. Esse ajuste não modifica o texto da claim, ranking ou formação de IDs.
- Os adaptadores mantêm estado por módulo e são sequenciais por processo; não foi introduzida uma arquitetura concorrente.
- Foram mantidos o comportamento de batching/decodificação do T5, as heurísticas do gerador, o truncamento, a agregação, os limiares, as prioridades taxonômicas e a interpretação factual da implementação selecionada.

O auditor original de injeções estava ausente. O novo `audit_injections.py` implementa somente verificações técnicas necessárias aos contratos e testes pedidos; não se apresenta como reconstituição daquele método ausente.

## Equivalência e limites

Snapshots de funções originais e fixtures numéricas sem texto do corpus sustentam testes de regressão. O fluxo ponta a ponta rápido usa modelos simulados, identificados nos testes, para conferir formatos e identidades, não desempenho factual.

`historical_regression.json` registra a reaplicação do código migrado às probabilidades armazenadas: 457, 1.007 e 4.874 registros dos lotes de 5, 10 e 50 documentos mantêm escores e decisões exatamente. São 6.338 registros, sem repetir a inferência neural. O lote de 3 documentos preserva apenas 235/270 decisões ao aplicar a configuração atual e permanece histórico, sem sobrescrita.

Repetir inferência de T5, MPNet/CrossEncoder, DeBERTa e taxonomia com modelos reais continua necessário para uma afirmação de equivalência neural completa. Os testes GPU são opt-in. Consulte [as validações datadas](../validation.md) para os resultados efetivamente executados.
