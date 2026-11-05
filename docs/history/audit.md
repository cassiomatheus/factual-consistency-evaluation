# Diagnóstico de origem

A Fase A foi realizada antes da autorização para migrar. Foram inspecionados 31 arquivos Python, importações, configurações, licenças e schemas dos dados/resultados presentes. Os caminhos, tamanhos e hashes estão em source_inventory.json; ausência do inventário não deve ser interpretada como autorização para apagar outros arquivos.

O fluxo compatível selecionado é gerador V5 → CSV v7 → extração novo → recuperação novo → NLI novo → taxonomia novo. O projeto humano usa outro extrator e outro recuperador. Não há demonstração de que nomes v7 identifiquem um único método.

Os CSVs têm id=N/A; doc_idx vem da posição. No lote de 50 documentos há quatro ocorrências excedentes de chaves textuais repetidas. Identidade precisa de ocorrência, não apenas texto. O extrator baseline produz 457 claims no lote de cinco; o enriquecido produz 459.

O recuperador atual difere do schema histórico de 10/50 documentos. A reavaliação da agregação atual sobre probabilidades históricas coincide nos lotes 5/10/50, mas diverge em 35 decisões no lote de três. Schemas não são reescritos nem parâmetros atribuídos retroativamente.

O gerador permite substituições mais amplas que seus comentários: correferência sem divergência obrigatória de número e substituição por nomes; entidades ORG/GPE e PERSON/NORP compatíveis. Essas regras foram preservadas, inclusive seed e ordem de sorteios. O dicionário de antônimos é dependência efetiva e sua ausência não pode virar um fallback silencioso.

ClaimExtractor e NLIAligner importavam metric/utils, que carregava BERT, MiniLM e spaCy globalmente. A extração não usa essas funções importadas; NLI usa somente chunks. A análise posicional não é dependência científica dessas etapas. A API FENICE agregada depende de FastCoref e pronouns.txt, mas não é chamada pelo fluxo selecionado.

O requirements original é incompleto e difere do ambiente observado. Não havia .gitignore. Muitos scripts científicos e resultados eram untracked. LICENSE.txt estava removido na árvore de trabalho, mas disponível no histórico. Nenhuma dessas condições foi corrigida por alteração do repositório original.

Na referência humana foram encontrados 5 documentos, 1.833 sentenças, 459 claims e 14.400 candidatos, plano A/B/C=80/80/40 e 23 anotações pendentes. Não havia gold exportado. Esses dados não foram promovidos a referência concluída nem copiados para Git.

Ausências relevantes: auditar_injecoes_govreport.py; testes de recuperação e validacao citados na documentação; requirements-auditoria.txt. Scripts BART/PRIMERA/LED não foram encontrados nesta árvore, apesar de menções em comentários. Itens de necessidade incerta constam em items_to_review.md.
