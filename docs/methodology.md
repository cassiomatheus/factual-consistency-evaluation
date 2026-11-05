# Protocolo e fronteiras metodológicas

Fase I: seleção de documentos/resumos humanos GovReport; alterações controladas; auditoria técnica e revisão independente; extração; recuperação; inferência NLI; classificação taxonômica; avaliação por componente e ponta a ponta.

Injeções, rótulos derivados, anotações humanas, predições e métricas têm funções distintas. Um log tecnicamente reversível não demonstra que a alteração contradiz a fonte. A categoria `none` do gerador não comprova verdade factual. Resumos humanos também podem conter erros. Resultados condicionados aos metadados do gerador devem ser nomeados como diagnósticos de perturbação, sem serem apresentados como avaliação contra referência independente.

O gerador preservado utiliza antônimo, negação e mudança temporal em ordem de prioridade. A correferência pode trocar pronome por pronome ou nome e não exige mais divergência de número. Entidades admitem ORG/GPE e PERSON/NORP compatíveis. Essas regras executadas prevalecem sobre comentários históricos incompatíveis. Alterá-las exige novo experimento identificado.

O extrator baseline preserva claims originais e hipóteses de antecedente; a variante enriquecida usa spaCy md, expansão e uma auditoria SVO heurística. Não são variantes equivalentes nem evidência gold. O recuperador principal usa MPNet, BM25 próprio, entidades e CrossEncoder; a ferramenta humana usa MiniLM, BM25/RRF e sentencizer. Os resultados desses recuperadores não são combinados como se viessem do mesmo protocolo.

NLI considera a claim como hipótese e a passagem como premissa. A ordem de logits no checkpoint principal observado é entailment, neutral, contradiction; o retorno de `NLIAligner.process_batch` mantém entailment, contradiction, neutral. Hipóteses contextualizadas são diagnóstico; não substituem a hipótese original na decisão. Conflitos estruturais não elevam escores NLI. Todos os pesos e limiares baseline estão em `configs/baseline.yaml`.

O status `NOT_SUPPORTED` reúne insuficiência e ambiguidade do procedimento; ausência na recuperação não prova ausência de suporte no documento. A taxonomia só opera sobre `CONTRADICTED`, na prioridade correferência, entidade, predicado e other. Relate seus resultados condicionados e também a cobertura do pipeline anterior.

Na referência humana, triagem rápida, sugestões de modelos e amostragem não criam gold. A exportação exige claim válida, revisão finalizada, documento conferido, busca concluída e conjuntos de evidências apropriados. A validade semântica dessas confirmações depende do trabalho humano; o exportador verifica consistência estrutural.

Não há geração BART/PRIMERA/LED nesta Fase I. Análise posicional não integra o fluxo obrigatório. Não foram implementadas novas métricas ou resultados científicos nesta reorganização.
