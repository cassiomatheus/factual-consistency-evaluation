# Créditos e licenças

Este projeto reorganiza código de pesquisa derivado do **FENICE — Factuality Evaluation of Summarization based on Natural Language Inference and Claim Extraction**, de Alessandro Scirè, Karim Ghonim e Roberto Navigli, Findings of ACL 2024, pp. 14148–14161.

- Código original: https://github.com/Babelscape/FENICE
- Artigo: https://aclanthology.org/2024.findings-acl.841/
- Licença original: CC BY-NC-SA 4.0; texto integral preservado em `LICENSE`.

Os módulos `claims/model.py` e `nli/model.py` são adaptações dos componentes locais `metric/claim_extractor/claim_extractor.py` e `metric/nli/nli_aligner.py`. `utils/batching.py` preserva a função `chunks`. O código de origem já continha modificações locais; esta distribuição não o apresenta como uma cópia imutável do upstream. As adaptações de empacotamento, configuração, execução e IDs estão descritas em `docs/migration.md`; hashes e caminhos constam em `docs/history/migration_sources.json`.

O T5 `Babelscape/t5-base-summarization-claim-extractor` declara CC BY-NC-SA 4.0. O checkpoint `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` declara MIT. Os modelos Sentence Transformers MiniLM/MPNet e MS MARCO MiniLM declaram Apache 2.0 nas respectivas model cards. Os pesos não são distribuídos neste repositório. As licenças dos modelos não substituem a licença do código derivado do FENICE.

GovReport é obtido de https://huggingface.co/datasets/ccdv/govreport-summarization . Datasets, documentos completos e anotações pessoais não foram redistribuídos. Confirme as condições dos dados antes de publicar uma distribuição derivada.

`resources/antonimos_govreport.json` é uma cópia idêntica do recurso de pesquisa fornecido no projeto local. Sua procedência/licença específica não estava documentada na origem inspecionada e permanece pendente antes de publicação.

```bibtex
@inproceedings{scire-etal-2024-fenice,
  title = {{FENICE}: Factuality Evaluation of summarization based on Natural language Inference and Claim Extraction},
  author = {Scirè, Alessandro and Ghonim, Karim and Navigli, Roberto},
  booktitle = {Findings of the Association for Computational Linguistics ACL 2024},
  year = {2024},
  pages = {14148--14161},
  url = {https://aclanthology.org/2024.findings-acl.841/}
}
```
