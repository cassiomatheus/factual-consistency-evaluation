# Contratos e execução

`scripts/` contém pontos de entrada; `src/factual_eval/` contém implementações reutilizáveis. Configurações e modelos são inicializados somente na execução de uma etapa. Não há importação de `metric` nem acesso implícito ao repositório original.

| Etapa | Entrada | Saída principal | Unidade |
|---|---|---|---|
| generate | GovReport ou report/summary local | injected.csv + original.parquet + manifesto | documento |
| audit | CSV com resumos e logs | technical_audit.json | documento/variante |
| claims | CSV injetado | claims.parquet | documento/variante, claims aninhadas |
| retrieval | claims.parquet | retrieval.parquet | claim |
| nli | retrieval.parquet | nli.parquet | claim |
| taxonomy | nli.parquet | taxonomy.parquet | claim contradita |
| prepare-reference | Parquet de claims ou fixture JSON | JSONL e manifest.json | documento/sentença/claim/candidato |
| sample-reference | pasta preparada | plano e amostragem, com frame copiado | claim |
| review-html | pasta preparada/amostrada | avaliacao.html | interface |
| export-reference | pasta e anotações | gold_claims.jsonl, pendências e triagem | referência humana |

Os campos originais são conservados. A extração acrescenta `doc_id`, `claim_id` e `summary_sentence_id`; a recuperação acrescenta `sentence_id` e `window_id` aos itens; o NLI propaga a identidade da claim e registra `source_sentence_id`. O manifesto de injeções tem `change_id` e status explícito de validação pendente.

IDs seguem a origem e a ocorrência: `doc_<índice>_<sha256-da-fonte[:8]>:<variante>:C00001`. O índice da sentença de resumo não é um índice de sentença da fonte. Janelas têm início inclusivo/fim exclusivo, como na implementação anterior. O `source_sentence_id` do NLI é a âncora selecionada, não representa sozinho todo o conjunto de evidências de uma janela.

CSV legado usa `doc_idx` derivado da posição da linha. O CSV gerado pelo novo adaptador adiciona `source_index` para preservar a posição no split quando se seleciona um intervalo. Logs CSV com representação Python são lidos apenas com `ast.literal_eval`, nunca `eval`.

Validações identificam claims vazias/malformadas, perda ou duplicação de documentos/variantes, mudança de ordem/texto/contagem entre recuperação e NLI, IDs duplicados e divergência do subconjunto taxonômico. Entradas incompatíveis produzem falha explícita e execução marcada como failed, em lugar de exclusão silenciosa.

O framework de execução é a API pública para saídas protegidas. Funções de baixo nível herdadas da ferramenta humana preservam seu comportamento original; não as invoque diretamente sobre uma pasta com resultados anteriores. Use os comandos CLI para criação de diretórios exclusivos, logging e manifests.

Os adaptadores principais conservam funções com estado por módulo para reduzir a superfície de alteração científica. Chamadas sequenciais são suportadas. Concorra com processos separados, não threads compartilhando modelos e constantes. A retomada por `--from-stage` inicia outra execução; não edita a anterior.
