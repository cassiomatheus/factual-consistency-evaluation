# Guia completo de execução

Instalação, pipeline, referência humana, tradução e testes. Para escolher um assunto, consulte o [índice da documentação](README.md). Execute os exemplos a partir da raiz do projeto.

## Instalação

Ambiente observado: Python 3.10.20, PyTorch 2.5.1+cu121, Transformers 4.38.2, Sentence Transformers 2.7.0, spaCy 3.8.16. As versões estão declaradas no `pyproject.toml`; o inventário completo do ambiente de origem está em `docs/history/environment_observed.json`. Ele é uma observação histórica, não um lockfile portátil.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[injection,reference]'
```

Se já utiliza o ambiente Conda do projeto, ative-o com `conda activate nlp_mestrado` e execute os exemplos abaixo na raiz do repositório. A criação de um novo ambiente é necessária somente para uma instalação separada.

Instale a distribuição de PyTorch compatível com sua GPU antes, quando necessário. Não troque automaticamente a versão de Transformers ou os checkpoints.

Os modelos spaCy do ambiente observado são `en_core_web_sm==3.7.0` e `en_core_web_md==3.8.0`. A combinação de `sm` 3.7.0 com spaCy 3.8.16 gera aviso de compatibilidade: foi preservada e deve ser testada, não atualizada silenciosamente.

```bash
python -m pip install --no-deps https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
```

Nenhum modelo é baixado ao importar o pacote. A execução de uma etapa carrega os modelos necessários; os checkpoints HF são obtidos no primeiro uso, salvo cache prévio. GPU é o padrão de extração/NLI; `--device cpu` constitui uma execução explicitamente identificada em CPU. A auditoria técnica e os testes rápidos não precisam de GPU.

Você também pode executar os arquivos em `scripts/` sem instalar o pacote, usando um Python que já contenha as dependências. Todos os caminhos relativos de entrada, saída e configuração são resolvidos contra a raiz deste novo projeto. Caminhos absolutos são aceitos.

Na instalação de um wheel fora do checkout, a raiz de trabalho passa a ser o diretório atual; `FACTUAL_EVAL_ROOT` permite defini-la explicitamente. Os defaults são distribuídos dentro do pacote. Relatórios datados de migração, testes e auditoria bilíngue estão em [docs/validation.md](validation.md); resultados históricos não substituem a validação de uma nova execução.

## Teste inicial com cinco documentos

Com as dependências e os modelos spaCy instalados, este comando obtém os cinco primeiros documentos do split `test` e executa o pipeline completo:

```bash
PIPELINE_RUN="govreport5-$(date -u +%Y%m%dT%H%M%SZ)"

python scripts/run_pipeline.py \
  --from-stage generate \
  --num-docs 5 \
  --device auto \
  --run-id "$PIPELINE_RUN"
```

`--device auto` seleciona CUDA quando disponível e CPU nos demais casos; a execução em CPU pode ser demorada. A configuração padrão, sem essa opção, mantém GPU para extração/NLI. A seleção é por posição no dataset. Os cinco documentos geram versões de resumo e múltiplas claims: `--num-docs 5` não limita a cinco afirmações.

Os comandos imprimem `Run:` e `Output:`. Use o diretório indicado na próxima etapa. Nesta sessão de terminal, os resultados do exemplo ficam em `outputs/runs/$PIPELINE_RUN/`:

| Caminho dentro da execução | Conteúdo |
|---|---|
| `generate/injected.csv` | Fonte, resumo original, versões alteradas e logs de alterações |
| `generate/original.parquet` | Registros originais selecionados |
| `generate/technical_audit.json` | Auditoria técnica das alterações |
| `claims/claims.parquet` | Claims aninhadas por documento/variante, com IDs |
| `retrieval/retrieval.parquet` | Evidências e scores por claim |
| `nli/nli.parquet` | Decisão factual automática, passagem, escores e motivo |
| `taxonomy/taxonomy.parquet` | Categoria e diagnóstico das claims contraditas |
| `taxonomy/taxonomy.confusion.csv` | Comparação com os metadados do gerador |
| `manifest.json`, `execution.log` | Estado, proveniência e registro da execução |

As etapas de claims, recuperação, NLI e taxonomia também salvam `stage_counts.json`. Os Parquets podem ser consultados com `pandas.read_parquet`; o CSV da geração pode ser aberto em uma planilha. Para um resumo pelo terminal, após a conclusão do exemplo:

```bash
python -m json.tool "outputs/runs/$PIPELINE_RUN/nli/stage_counts.json"
```

## Dados e geração controlada

O gerador original seleciona o split `test` de `ccdv/govreport-summarization`, usa até cinco alterações por categoria e seed 42. A seleção é por posição, não aleatória. Dataset, split, seed e demais parâmetros estão em `configs/baseline.yaml`.

```bash
python scripts/generate_dataset.py --num-docs 50
```

A execução grava `generate/original.parquet`, `generate/injected.csv`, `generate/injection_manifest.jsonl` e `generate/technical_audit.json` dentro de uma pasta única. O CSV mantém resumos, logs e campos legados; o manifesto identifica cada alteração como **AUTOMATIC_CHANGE_UNVALIDATED**, nunca gold. Também aceita `--input` com JSON/JSONL/Parquet contendo `report` e `summary`, para reproduzir uma seleção local sem novo download.

Para auditar tecnicamente um CSV existente:

```bash
python scripts/audit_injections.py --input /caminho/injected.csv
```

Esta ferramenta verifica contagens, marcadores e reconstrução exata. Não substitui revisão semântica nem o auditor `auditar_injecoes_govreport.py`, ausente da origem inspecionada. Formatos antigos de marcadores são reportados como incompatíveis, sem adaptação silenciosa.

Ao usar `--start`, a geração processa o prefixo anterior antes de selecionar a saída, preservando a sequência global de sorteios do gerador. Isso pode custar mais tempo; mudar para seeds por documento alteraria o método.

## Pipeline e etapas individuais

```bash
python scripts/run_pipeline.py \
  --input /caminho/injected.csv \
  --num-docs 5 \
  --device cuda:0
```

O padrão de extração continua sendo os cinco primeiros documentos (`selection` no YAML). O pipeline alinha as etapas seguintes à saída efetivamente produzida. Para gerar os dados e executar tudo em uma chamada, use `--from-stage generate --num-docs 50` e omita `--input`. Para todos os documentos de um CSV já pronto, use `--all-docs`.

Cada etapa pode ser executada separadamente:

```bash
python scripts/extract_claims.py --input /caminho/injected.csv --start 0 --num-docs 5
python scripts/retrieve_evidence.py --input /caminho/claims.parquet
python scripts/evaluate_nli.py --input /caminho/retrieval.parquet
python scripts/classify_errors.py --input /caminho/nli.parquet
```

As etapas de recuperação, NLI e taxonomia processam toda a entrada fornecida, salvo seleção explícita por documentos. `--num-docs` nunca conta linhas de claim ou combinações documento/variante.

Configurações parciais YAML são combinadas com o baseline. Chaves desconhecidas são recusadas. Exemplo de experimento em CPU:

```yaml
name: baseline_cpu
runtime:
  device: cpu
selection:
  num_docs: 2
```

```bash
python scripts/run_pipeline.py --config /caminho/experimento.yaml --input /caminho/injected.csv
```

As funções científicas podem ser importadas de `factual_eval.claims`, `retrieval`, `nli` e `taxonomy`. Os adaptadores `run()` mantêm estado de modelos/configuração por módulo e devem ser utilizados sequencialmente por processo. Para experimentos concorrentes, use processos separados.

## Saídas, retomada e reutilização

Os comandos de pipeline e referência criam `outputs/runs/<UTC>_<identificador>/`. `--run-id` permite um nome explícito e **recusa um diretório existente**, inclusive se a tentativa anterior falhou. `--output-root` aceita outro local. Para repetir, omita `--run-id` ou escolha outro nome; preserve o diretório anterior para diagnóstico. Os testes manuais de tradução imprimem no terminal, e `verify_translation.py` atualiza somente o cache indicado.

`serve_review.py` abre uma interface existente com um arquivo de revisão persistente, sem criar outra execução em `outputs/runs/`. Seu arquivo de trabalho e histórico ficam em `outputs/annotations/` por padrão.

O diretório contém `config.resolved.yaml`, `manifest.json`, `execution.log` e uma pasta por etapa. O manifesto guarda hashes de entradas, saídas, recursos e código; versões das bibliotecas; revisões resolvidas dos modelos quando disponíveis; contagens; e estado de sucesso/falha. Erros deixam uma execução marcada como `failed`.

Uma terminação abrupta pode deixar `status: running` sem processo ativo. Esse estado sozinho não comprova que a execução continua: confira o processo, o log e os artefatos. Na tradução, `translation_manifest.json` é gravado ao final; aguarde sua geração antes de criar o HTML. O cache preserva os segmentos já confirmados mesmo se a execução for interrompida.

Reutilização explícita de intermediários:

```bash
python scripts/run_pipeline.py \
  --from-stage nli \
  --input /caminho/execucao-anterior/retrieval/retrieval.parquet
```

Para reaproveitar etapas verificadas de uma execução anterior, repita o comando original acrescentando `--reuse-from /caminho/execucao-anterior`. A reutilização exige coincidência de configuração, entradas, seleção, código e ambiente, além da conferência dos hashes dos artefatos. A geração remota sem entrada local não é reutilizada automaticamente. Artefatos reutilizados são copiados para a nova execução e identificados no manifesto.

`claim_id` usa ocorrência, não apenas texto. Os campos antigos são preservados e recebem IDs adicionais; isso permite manter claims repetidas. `generator_labels.jsonl` contém os rótulos derivados do gerador. O campo legado `gabarito_injetado` continua presente nos Parquets por compatibilidade, mas não é referência humana. A matriz produzida pela taxonomia é uma comparação com esses metadados, não uma estimativa de acurácia contra gold independente.

## Referência humana

A ferramenta humana possui seu próprio recuperador e sua própria configuração em `configs/reference.yaml`. Use o `claims.parquet` já produzido, baseline ou enriquecido, sem repetir a extração. No exemplo abaixo, substitua `/caminho/claims.parquet` pelo caminho real; para o teste inicial, ele é `outputs/runs/$PIPELINE_RUN/claims/claims.parquet`.

```bash
CLAIMS_INPUT="/caminho/claims.parquet"
REFERENCE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
PREPARO_RUN="referencia-preparo-$REFERENCE_TAG"
AMOSTRA_RUN="referencia-amostra-$REFERENCE_TAG"
HTML_RUN="referencia-html-$REFERENCE_TAG"
AMOSTRA_DIR="outputs/runs/$AMOSTRA_RUN/sample-reference"

python scripts/prepare_reference.py \
  --input "$CLAIMS_INPUT" --device auto --run-id "$PREPARO_RUN" &&
python scripts/sample_reference.py \
  --input "outputs/runs/$PREPARO_RUN/prepare-reference" \
  --sample-size 200 --seed 42 --run-id "$AMOSTRA_RUN" &&
python scripts/generate_review_html.py \
  --input "$AMOSTRA_DIR" --run-id "$HTML_RUN" &&
xdg-open "outputs/runs/$HTML_RUN/review-html/avaliacao.html"
```

Os comandos estão encadeados com `&&`: cada etapa começa somente se a anterior terminar com sucesso. O HTML pode ser aberto diretamente no navegador, sem servidor. Para uma amostra existente, use sua pasta `sample-reference` como entrada do HTML e da exportação; ela inclui o frame original e o plano amostral. Grupos A/B/C são estratégias de seleção, não categorias factuais. Estimativas globais exigem revisão adequada de toda a amostra aleatória A; B/C são intencionais.

Na interface, preencha **Revisor** e utilize **Auditoria completa** para validar a claim, pesquisar o documento, julgar as evidências e registrar o rótulo humano. Para exportação gold, a claim precisa de `claim_valid: SIM`, status `FINALIZADO`, fonte conferida, busca concluída e rótulo `SUPPORTED`, `CONTRADICTED` ou `NOT_SUPPORTED`. Os dois primeiros exigem conjuntos de evidências; `NOT_SUPPORTED` exige conjuntos vazios. Uma decisão de triagem rápida não atende a esses requisitos.

A interface apresenta três passos — conferir a afirmação, selecionar evidências e registrar a decisão — com navegação direta e uma lista de pendências. **Concluir revisão e próxima** marca a revisão finalizada após o preenchimento; **Guardar pendente** permite continuar depois. As opções aparecem em português e preservam os valores do JSON. Scores e avaliações detalhadas ficam em seções expansíveis. Veja o [guia da interface](audit-interface.md).

### Salvar as análises automaticamente no disco

Na abertura direta do HTML, as anotações ficam no armazenamento do navegador. Para gravá-las automaticamente em um arquivo JSON do computador, abra uma interface já gerada pelo serviço local:

```bash
python scripts/serve_review.py \
  --html /caminho/avaliacao.html \
  --annotations outputs/annotations/minhas_analises.json \
  --port 8765 --open
```

Mantenha o terminal aberto e use `http://127.0.0.1:8765/`. O cabeçalho confirma **Salvo em arquivo ✓** após a gravação. O arquivo permanece após fechar o navegador ou reiniciar o serviço. O servidor utiliza o modelo atual de interface com os dados do HTML existente; não altera a execução original nem repete traduções ou etapas científicas.

Para transferir análises já feitas no HTML antigo, exporte a revisão JSON nele e use **Importar revisão** na interface local. Essa transferência é necessária uma vez, pois páginas de arquivo e páginas HTTP têm armazenamentos de navegador separados. Nas próximas aberturas, o serviço recupera o JSON salvo no disco.

Cada nova gravação preserva a anterior em `minhas_analises.historico/`, ao lado do JSON. A gravação é atômica, e alterações conflitantes entre abas são recusadas. Seleções de evidências ainda em montagem também são guardadas como rascunho; só viram conjuntos gold quando se clica em **Salvar conjunto selecionado**. O JSON continua compatível com `export_reference.py`. Detalhes de recuperação e testes em [docs/review-autosave.md](review-autosave.md).

Exporte a revisão pelo botão **Exportar revisão (.json)** e valide o arquivo com a mesma amostra usada na anotação:

```bash
python scripts/export_reference.py \
  --input "$AMOSTRA_DIR" \
  --annotations /caminho/revisao.json
```

A pasta `export-reference/` da nova execução contém `gold_claims.jsonl`, `julgamentos_evidencias.csv`, `triagem_provisoria.csv`, `pendencias.csv` e `resumo_validacao.json`. As pendências explicam exclusões; somente `gold_claims.jsonl` é a referência humana validada estruturalmente. A revisão semântica continua sendo responsabilidade do anotador.

Se precisar reproduzir especificamente o extrator enriquecido que produziu o esquema `01_B`, identifique essa variante em uma execução separada:

```bash
python scripts/extract_claims.py \
  --config configs/enriched_claims.yaml \
  --input /caminho/injected.csv --num-docs 5 --device auto
```

Anotações antigas podem continuar usando os diretórios originais como entrada. Não regenere claims, IDs ou planos amostrais para importar uma anotação existente. Os IDs e a validação de `dataset_id`/`selection_id` foram preservados.

Teste offline com dados sintéticos (somente demonstração):

```bash
python scripts/prepare_reference.py --input tests/fixtures/claims_sinteticas.json --backend tfidf --device cpu
```

Predições auxiliares por par podem ser geradas por `scripts/suggest_reference_nli.py`. A auditoria factual auxiliar com Qwen/Ollama está em `scripts/audit_qwen.py`, após `pip install -e '.[audit]'`, e tem uma finalidade diferente da tradução de leitura. Nenhuma dessas ferramentas cria referência humana automaticamente.

## Auditoria bilíngue com tradução local

A interface apresenta português primeiro e inglês original abaixo nas três colunas. O cabeçalho permite **Bilíngue**, **Somente português** ou **Somente inglês**, com busca nos dois idiomas. Traduções ausentes mantêm o original visível; possíveis diferenças recebem avisos para conferência.

O comando `translate_reference.py` usa `qwen3.5:9b` por padrão, instalado no Ollama local, com temperatura 0 e `think=False`. Não executa novamente extração, recuperação, NLI ou taxonomia. A tradução integrada usa a API HTTP local diretamente e não exige o pacote Python `ollama`; servidor e modelo precisam estar disponíveis. Confira com `ollama --version` e `ollama list`.

Para traduzir uma amostra já existente, substitua o caminho de `AMOSTRA_DIR` no bloco abaixo. Os nomes recebem data/hora para permitir novas execuções sem sobrescrever resultados:

```bash
AMOSTRA_DIR="/caminho/sample-reference"
TRADUCAO_RUN="traducao-$(date -u +%Y%m%dT%H%M%SZ)"
HTML_TRAD_RUN="$TRADUCAO_RUN-html"

python scripts/translate_reference.py \
  --input "$AMOSTRA_DIR" \
  --model qwen3.5:9b \
  --cache outputs/translation-cache.sqlite3 \
  --run-id "$TRADUCAO_RUN" &&
python scripts/generate_review_html.py \
  --input "$AMOSTRA_DIR" \
  --translations "outputs/runs/$TRADUCAO_RUN/translate-reference" \
  --run-id "$HTML_TRAD_RUN" &&
xdg-open "outputs/runs/$HTML_TRAD_RUN/review-html/avaliacao.html"
```

O cache SQLite é persistente e separado das execuções. Considera texto, contexto, modelo e prompt; a tradução é feita por segmentos, preservando IDs e ordem. `--max-calls 20` limita uma execução curta a 20 chamadas novas; `--scope all` inclui claims fora da amostra. `--batch-size` controla a frequência do log, e as chamadas continuam sequenciais.

Os artefatos `translate-reference/translations.jsonl` e `translation_manifest.json` registram traduções, estados e proveniência. Para retomar após interrupção, repita o bloco com a mesma amostra e cache, criando nomes novos. O mecanismo de retomada é o cache, independentemente de `--reuse-from`.

Estados `translated`, `needs_review`, `failed`, `pending` e `human_verified` descrevem a tradução, não a factualidade da claim. Avisos de qualidade são heurísticos; ausência de aviso não garante equivalência semântica. Após conferência humana de uma tradução, `scripts/verify_translation.py` permite protegê-la no cache. Glossário configurável, critérios de qualidade e comandos de verificação estão no [procedimento completo](bilingual-audit.md).

As anotações continuam ligadas ao inglês e mantêm o esquema JSON anterior. Antes de trocar de HTML, exporte a revisão no arquivo antigo e importe-a no novo. Use a amostra original em `export_reference.py`, preservando `dataset_id` e `selection_id`.

Para salvamento automático em arquivo, use `scripts/serve_review.py --html /caminho/avaliacao.html`, conforme o procedimento de referência humana acima. O serviço é exclusivamente local; traduções continuam incorporadas e não há chamadas ao Ollama durante a anotação.

## Exemplos independentes de tradução

Os testes manuais de OPUS-MT, NLLB e TranslateGemma estão documentados em [examples/translation/README.md](../examples/translation/README.md). Eles não alteram a auditoria nem geram gold.

## Testes e experimentos

```bash
python -B -m unittest discover -s tests -v
```

Os testes rápidos usam dados sintéticos, modelos simulados, snapshots de funções originais e probabilidades históricas sem texto do corpus. Os testes de tradução verificam também cache, retomada, qualidade heurística e preservação dos dados científicos, sem depender de GPU ou de Ollama ativo.

As integrações reais são opt-in. Para o Qwen local e a regressão da interface no navegador:

```bash
FACTUAL_EVAL_QWEN_TESTS=1 \
  python -B -m unittest discover -s tests -p test_translation_qwen.py -v
```

```bash
python -m pip install -e '.[browser-tests]'
FACTUAL_EVAL_BROWSER_TESTS=1 \
  python -B -m unittest discover -s tests -p test_bilingual_browser.py -v
```

O teste de navegador usa traduções simuladas e requer Firefox/geckodriver utilizáveis pelo Selenium. Para verificar também o serviço local de salvamento em disco, use `FACTUAL_EVAL_BROWSER_TESTS=1 python -B -m unittest discover -s tests -p test_disk_autosave_browser.py -v`. Os testes reais de extração/NLI com GPU têm configuração própria em [tests/integration/README.md](../tests/integration/README.md). Não se deve inferir equivalência neural completa a partir dos testes sem modelos; os testes manuais de tradução também não medem acurácia contra gold.

O catálogo [Catálogo de experimentos](experiments.md) organiza E0–E6 e define entradas de referência necessárias. Não foram inventadas métricas, resultados ou novas implementações de baselines.

## Limitações conhecidas

- Os artefatos antigos não possuem manifests suficientes para provar todas as versões usadas.
- O lote histórico de três documentos não coincide com a agregação NLI atual.
- O código do gerador permite regras mais amplas do que certos comentários antigos descrevem; elas foram preservadas.
- Ausência de evidência e `NOT_SUPPORTED` não comprovam contradição.
- A taxonomia só recebe contradições NLI e falha, como antes, se não houver nenhuma; não inventa um resultado vazio válido.
- O batch size configurado no extrator é preservado, mas o método legado processa todo o argumento recebido em uma chamada.
- Truncamento, versões spaCy e escolha de antecedentes continuam limitações científicas a avaliar.
- O split de origem deve ser congelado por revisão/hash antes do experimento final. A configuração mantém `dataset_revision: null` porque a revisão histórica não foi comprovada.
