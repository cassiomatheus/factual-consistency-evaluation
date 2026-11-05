# Itens preservados para revisão

Nenhum dos arquivos abaixo foi removido da origem. O inventário completo com hashes e indicação de migração está em `source_inventory.json`.

- `auditar_injecoes_govreport.py`: ausente; auditoria semântica/protocolo original não foram reconstruídos por suposição. O novo `audit_injections.py` é somente técnico.
- `03_compute_nli_v7_melhorado.py` da raiz: checkpoint e thresholds diferentes; espera top_k_evidences ausente dos intermediários inspecionados. Fora do baseline.
- `pipeline/01_extract_claims.py`, `02_retrieve_evidence.py`, `03_compute_nli.py`: v6/histórico.
- `pipeline/05_classify_error_taxonomy.py`: sobrepõe decisões NLI e espera max_S/max_C. Não é equivalente ao classificador selecionado.
- `metric/taxonomy_classifier.py`: implementação alternativa não importada pelo fluxo selecionado.
- `metric/FENICE.py`, `metric/coreference_resolution/` e `data/pronouns.txt`: candidatos E6; sua incorporação exige identificar a versão original e os ajustes locais. FastCoref não é dependência do novo baseline.
- `pipeline/05_plot_positional_metrics.py` e trechos posicionais de metric/utils: fora do componente obrigatório.
- `pipeline/auditoria_por_etapas.py`, `diagnostico_mentiras_v2.py`, `04_classify_factual_status.py`: diagnósticos históricos, alguns com schemas incompatíveis.
- CSV v6: formato de marcadores incompatível com a auditoria técnica nova; não reportado como corrupção factual.
- CSV denominado dataset_govreport_50_docs_treino.csv: nome não comprova split ou configuração. Quantidades de alterações diferem da configuração atual.
- Resultados v7 de 3 documentos: 35/270 decisões divergem ao reaplicar a agregação atual. Não serão reescritos.
- Resultados de recuperação de 10 e 50 documentos: schemas anteriores ao recuperador atual; não atribuir retroativamente os novos parâmetros.
- Anotações existentes: 23 registros pendentes na inspeção; não foram copiadas para Git nem declaradas gold.
- Licença/procedência do dicionário de antônimos e condições de redistribuição do dataset: pendentes antes de publicação.
- Dataset revision histórica não comprovada; revisões locais de modelos identificam o cache observado, não todas as execuções passadas.
- A documentação antiga cita tests/, validacao/ e requirements-auditoria.txt que não estavam presentes.
