# Exemplos de tradução

Testes manuais de quatro textos do GovReport, independentes do pipeline e do
cache da auditoria. Execute os comandos na raiz do projeto, no ambiente instalado.

| Arquivo | Modelo |
|---|---|
| [opus_mt.py](opus_mt.py) | `Helsinki-NLP/opus-mt-tc-big-en-pt` |
| [nllb.py](nllb.py) | `facebook/nllb-200-distilled-600M` |
| [translategemma.py](translategemma.py) | `translategemma:12b` via Ollama |

## OPUS-MT e NLLB

```bash
python examples/translation/opus_mt.py
python examples/translation/nllb.py
```

Cada script traduz os quatro textos, um por vez. Os checkpoints Hugging Face são
obtidos no primeiro uso se não estiverem em cache. CPU/GPU são selecionadas
automaticamente. Execute um script por vez para limitar o uso de memória.

## TranslateGemma

O servidor Ollama deve estar ativo e o modelo local instalado. O cliente Python
está disponível no extra `audit`:

```bash
python -m pip install -e '.[audit]'
python examples/translation/translategemma.py --texto 1
```

Use `--texto 1 2 3` para os três primeiros, ou `--texto 1 2 3 4` para todos.
Cada texto usa uma chamada independente, com temperatura 0, contexto de 4096 e
limite de 512 tokens gerados.

## Caminhos anteriores

| Antes | Agora |
|---|---|
| `scripts/test_translation.py` | `examples/translation/opus_mt.py` |
| `scripts/test_translation_nllb.py` | `examples/translation/nllb.py` |
| `scripts/test_translation_translategemma.py` | `examples/translation/translategemma.py` |

Os arquivos foram movidos sem mudar a lógica de tradução. Estes exemplos não
geram referência gold nem escolhem o modelo usado pela auditoria. Para traduzir
uma amostra de referência, consulte o [guia bilíngue](../../docs/bilingual-audit.md).
