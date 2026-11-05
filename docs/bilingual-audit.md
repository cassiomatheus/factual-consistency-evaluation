# Tradução local e auditoria bilíngue

A tradução auxilia a leitura. O inglês continua sendo a referência textual original.
Este fluxo não executa novamente extração, recuperação, NLI ou taxonomia e não cria
rótulos factuais, decisões de triagem ou gold. O esquema de revisão JSON permanece
na versão 1, com os mesmos IDs e campos de anotação.

## Preparar uma amostra existente

Use o ambiente do projeto e mantenha o Ollama local ativo. A integração utiliza
somente HTTP em loopback, sem proxy, sem serviços externos e sem downloads de
modelos. Foram verificados o servidor Ollama **0.35.0**, o cliente Python instalado
**0.6.3** e o modelo **qwen3.5:9b**. O novo módulo usa a API HTTP local diretamente,
portanto não exige o pacote Python Ollama. Não desativa `think=False` como fallback
se um servidor antigo rejeitar a opção: a chamada falha explicitamente.

```bash
conda activate nlp_mestrado
ollama --version
ollama list

python scripts/translate_reference.py \
  --input outputs/runs/gold-govreport5-amostra/sample-reference \
  --model qwen3.5:9b \
  --cache outputs/translation-cache.sqlite3 \
  --run-id gold-govreport5-traducao-completa
```

Não é necessário carregar outros modelos. As chamadas ao Qwen são sequenciais,
com temperatura 0, `think=False`, contexto de 4096 tokens e limite de geração de
2048 tokens por segmento. O modelo permanece disponível no Ollama por cinco
minutos após uso. O programa registra versão do servidor e digest dos pesos
instalados, sem alterar o modelo local.

Por padrão, são traduzidas as claims selecionadas na amostra, suas sentenças de
origem/variantes textuais e todas as sentenças dos documentos associados. Claims e
evidências são intercaladas na fila; os documentos completos vêm em seguida. Os
candidatos usam a mesma tradução da sentença-fonte. Sentenças não recuperadas
também ficam disponíveis para busca manual. Sem plano amostral, todas as claims
são consideradas. `--scope all` inclui também claims fora da amostra.

Para uma execução curta, acrescente `--max-calls 20`: esse limite conta chamadas
novas, não acertos no cache. Todas as unidades continuam no artefato, com as não
processadas marcadas `pending`. `--batch-size 20` controla o intervalo dos logs,
não carrega vários modelos nem envia várias sentenças em uma única tradução.

## Gerar e abrir o HTML

```bash
python scripts/generate_review_html.py \
  --input outputs/runs/gold-govreport5-amostra/sample-reference \
  --translations outputs/runs/gold-govreport5-traducao-completa/translate-reference \
  --run-id gold-govreport5-html-bilingue-completo

xdg-open outputs/runs/gold-govreport5-html-bilingue-completo/review-html/avaliacao.html
```

O HTML é portátil, funciona sem Ollama e não faz requisições de tradução. A opção
`--translations` aceita a pasta ou o arquivo `translations.jsonl`. Dataset,
seleção, hashes, IDs e correspondência com o inglês são verificados antes da
incorporação. Sem essa opção, a interface também funciona: mostra o inglês e
indica as traduções ausentes. Artefatos parciais concluídos com `--max-calls` são
aceitos; uma execução interrompida antes de gerar seu manifesto deve ser retomada.

No cabeçalho, **Visualização** permite alternar entre Bilíngue (padrão), Somente
português e Somente inglês. Em português, o inglês pode ser expandido; se faltar
tradução, o inglês permanece visível. A preferência usa `sessionStorage`, separado
das anotações em `localStorage`. A pesquisa consulta português e inglês em
qualquer modo. O ranking e os scores permanecem originais.

Claim, sentença do resumo, evidências, contextos e documento-fonte usam o mesmo
componente bilíngue. Se a sentença de origem coincidir exatamente com a claim, a
interface indica isso sem duplicar o texto; o campo e seu ID continuam nos dados.
O modo inglês recupera a leitura original. Os botões trabalham exclusivamente
com os IDs originais. Triagem rápida, auditoria completa, seleção de conjuntos,
revisor e importação/exportação de JSON seguem o fluxo anterior.

Se já anotou em um HTML anterior, exporte a revisão JSON nele e importe-a no novo
HTML. Não dependa do compartilhamento de armazenamento entre arquivos locais,
pois esse comportamento varia entre navegadores; dataset e seleção continuam os mesmos.

## Cache, interrupção e retomada

O cache SQLite está fora dos diretórios imutáveis de execução. Cada segmento
registra inglês, português, hashes do texto/contexto, modelo/digest, versão e hash
do prompt, hash do glossário, estado, data e avisos. Os artefatos mantêm também ID
da unidade, offsets das partes e metadados por segmento, inclusive a proveniência
original de uma tradução verificada reutilizada com outro modelo solicitado.

Textos e contextos idênticos podem reutilizar a tradução. Textos iguais em
contextos distintos recebem entradas diferentes. O contexto contém a sentença de
origem da claim ou as sentenças vizinhas da fonte; o prompt proíbe importar fatos
dele. O glossário padrão fica em
`src/factual_eval/resources/translation_glossary.json`; `--glossary arquivo.json`
permite outro glossário. As sugestões são contextuais, não substituições rígidas.

O cache é confirmado após cada segmento. Pode interromper com Ctrl+C e repetir o
comando com **outro `--run-id`**, a mesma entrada e o mesmo cache. Traduções prontas
são reutilizadas e falhas são tentadas novamente. A execução interrompida fica
registrada como `failed`; não apague nem reutilize seu diretório. O comando
`--reuse-from` não substitui esse mecanismo: a tradução consulta seu próprio cache.

Saídas principais em `translate-reference/`:

| Arquivo | Conteúdo |
|---|---|
| `translations.jsonl` | Uma linha por unidade original, incluindo partes, estados e alertas |
| `translation_manifest.json` | Dataset, seleção, modelo, prompt, configuração e contagens |

O `execution.log` da execução contém progresso e erros. `calls` conta chamadas ao
modelo; `reused` conta segmentos reaproveitados; os demais estados contam unidades
completas. Uma claim composta mantém o ID e é segmentada com offsets, traduzida
sentença por sentença e reconstruída na ordem original. Se qualquer parte falhar
ou estiver pendente, não é exibida uma tradução parcial como se fosse completa:
fica disponível o original, com sinalização. Limites de geração também geram aviso.

## Qualidade e verificação humana da tradução

Estados de tradução são distintos dos estados da auditoria factual:

| Estado | Significado |
|---|---|
| `translated` | Tradução gerada sem avisos das heurísticas |
| `needs_review` | Tradução gerada com possíveis diferenças que precisam de leitura humana |
| `failed` | Erro ou saída vazia |
| `pending` | Unidade ainda não processada no limite incremental |
| `human_verified` | Tradução conferida explicitamente por uma pessoa |

As verificações comparam números, datas, siglas, nomes identificáveis, negação,
modalidade, quantificadores, proporção de comprimento, saídas interrompidas e
possíveis acréscimos. Normalizam, por exemplo, `8`/`oito`, cardinais comuns,
percentuais e separadores numéricos inglês/português. Não corrigem a tradução nem
atestam equivalência semântica. A interface mostra **Tradução requer conferência.**
com motivos expansíveis. Verificações são reaplicadas a entradas não verificadas
reutilizadas, sem chamar novamente o modelo.

Após conferir integralmente uma unidade e suas partes, proteja a tradução no cache:

```bash
python scripts/verify_translation.py \
  --translations outputs/runs/gold-govreport5-traducao-completa/translate-reference/translations.jsonl \
  --cache outputs/translation-cache.sqlite3 \
  --unit-id 'ID_DA_CLAIM:claim' \
  --reviewer 'Seu nome'
```

O ID exato está em `unit_id` no JSONL (para uma sentença-fonte, é seu `sentence_id`).
É possível repetir `--unit-id`. Esse comando confirma apenas a tradução, não cria
gold e não modifica o artefato anterior. Uma nova execução de tradução refletirá
`human_verified`. Para o mesmo texto/contexto, a tradução verificada tem precedência
mesmo após mudar o prompt, o glossário ou o modelo. Nunca é substituída automaticamente.
O comando não é um editor de traduções; correções manuais de texto ainda não têm UI.

Para gerar a referência humana, continue exportando a revisão JSON pelo HTML e
use `scripts/export_reference.py` com a **pasta amostral original**, conforme o
README. As traduções e seus estados não são exportados como julgamentos humanos.

## Testes reproduzíveis

Suíte offline, com tradutor simulado, sem GPU ou Ollama:

```bash
python -B -m unittest discover -s tests -v
```

Integração real, somente com o Qwen local:

```bash
FACTUAL_EVAL_QWEN_TESTS=1 \
  python -B -m unittest discover -s tests -p test_translation_qwen.py -v
```

Regressão de navegador, usando textos e traduções simulados:

```bash
python -m pip install -e '.[browser-tests]'
FACTUAL_EVAL_BROWSER_TESTS=1 \
  python -B -m unittest discover -s tests -p test_bilingual_browser.py -v
```

Esse teste requer Firefox e geckodriver utilizáveis pelo Selenium. Se necessário,
defina `FACTUAL_EVAL_FIREFOX_BINARY=/caminho/firefox` e
`SE_GECKODRIVER=/caminho/geckodriver` para instalações fora do PATH. Cobre idiomas,
busca nas duas línguas, preferência de sessão, navegação de claims/documento,
evidências e scores, triagem, conjuntos, anotação completa, exportação/importação
com validação gold e fallback/avisos. Nenhuma dependência de navegador é necessária
para gerar ou usar o HTML.

## Limitações

- Tradução automática pode omitir detalhes ou alterar significado sem ser detectada.
  Ausência de aviso não significa fidelidade garantida; inglês permanece acessível.
- Nomes, cardinais complexos, datas por extenso e modalidade têm verificações
  heurísticas que podem gerar falsos positivos ou deixar erros passar.
- O sentencizador pode tratar abreviações ou texto extraído de PDF de forma
  imperfeita. Segmentos/contextos excessivamente longos falham explicitamente em
  vez de serem enviados sem controle de tamanho; exigem revisão/segmentação manual.
- Contexto ajuda na tradução, mas não elimina ambiguidades de correferência.
- A preparação completa pode ser demorada: a amostra de cinco documentos possui
  2.633 unidades auxiliares, incluindo variantes e todas as sentenças dos documentos.
  Elas não correspondem a 2.633 chamadas, pois há cache e claims compostas.
- O piloto gerado nesta implementação é **parcial** (oito chamadas reais), para
  demonstrar retomada e fallback. A tradução integral da amostra não foi executada.

## Arquivos da implementação

Criados:

- `src/factual_eval/reference/translation.py`: unidades, cliente HTTP local, SQLite e artefatos.
- `src/factual_eval/reference/translation_quality.py`: normalização e avisos heurísticos.
- `src/factual_eval/resources/translation_glossary.json`: sugestões terminológicas.
- `scripts/translate_reference.py` e `scripts/verify_translation.py`: execução e verificação explícita.
- `tests/reference/test_translation.py`: contratos e regressão sem modelos.
- `tests/integration/test_translation_qwen.py`: integração opt-in com o Qwen.
- `tests/integration/test_bilingual_browser.py`: operações da interface no navegador.
- `docs/bilingual-audit.md`: este procedimento.

Modificados:

- `src/factual_eval/cli.py`: novos argumentos/comando e proveniência dos artefatos auxiliares.
- `src/factual_eval/reference/html.py`: validação e incorporação das traduções.
- `src/factual_eval/reference/modelo.html`: apresentação bilíngue e busca, preservando as anotações.
- `pyproject.toml`: dependência opcional para testes de navegador.
- `README.md`, `docs/validation.md` e `tests/integration/README.md`: procedimento e evidências de validação.

Resultados e limites dos testes executados constam em [validation.md](validation.md).
