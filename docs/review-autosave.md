# Salvamento automático das análises no computador

`serve_review.py` abre uma auditoria já gerada e grava revisões em um JSON local.
A gravação fica independente dos dados do navegador. O serviço usa apenas a
biblioteca padrão do Python e escuta exclusivamente em `127.0.0.1`; não exige
Ollama, GPU, traduções novas ou uma nova execução científica. A implementação
usa bloqueio de arquivo POSIX (`fcntl`), apropriado ao ambiente Linux do projeto.

## Primeira abertura e migração

1. Se já anotou no HTML aberto como arquivo, clique em **Exportar revisão (.json)**
   nessa janela para baixar as análises existentes.
2. Na raiz do projeto, ative seu ambiente e inicie o serviço:

```bash
conda activate nlp_mestrado
python scripts/serve_review.py \
  --html /caminho/avaliacao.html \
  --annotations outputs/annotations/minhas_analises.json \
  --port 8765 --open
```

3. Use a página `http://127.0.0.1:8765/`. Clique em **Importar revisão**, selecione o
   JSON do passo 1 e confirme a substituição das anotações da interface.
4. Aguarde **Salvo em arquivo ✓**. O caminho exato aparece no rodapé e no terminal.

O serviço usa os mesmos dados e IDs do HTML, apresentando o modelo de interface
atual. Não reescreve o arquivo `avaliacao.html` nem os manifestos da execução
original. Sem `--annotations`, escolhe automaticamente um caminho em
`outputs/annotations/<hash-do-dataset-e-selecao>/revisao.json`.

Mantenha o terminal aberto enquanto anota. Para encerrar, aguarde a confirmação de
gravação e pressione Ctrl+C no terminal. Para continuar outro dia, repita o mesmo
comando, com o mesmo arquivo de revisão. Diferentemente de `--run-id`, esse caminho
é um arquivo de trabalho persistente; ele deve ser reutilizado para continuar a
mesma revisão. Use outro arquivo para uma revisão independente de outro anotador.

## O que fica salvo

O JSON preserva esquema 1, dataset, seleção, revisor, estados, notas, triagem,
julgamentos e conjuntos de evidências. As seleções ainda em montagem ficam no
campo auxiliar `draft_sets`, por `claim_id`. São recuperadas ao voltar à claim ou
reabrir a interface, mas não constituem conjuntos gold até a confirmação por
**Salvar conjunto selecionado**. O exportador ignora os rascunhos para fins de gold.

Mudanças são agrupadas após uma pequena pausa de digitação (400 ms). **Salvar
agora no arquivo** solicita gravação imediata. A confirmação em disco só aparece
após o servidor concluir a gravação; o armazenamento do navegador continua como
cópia auxiliar. Os dois indicadores têm significados diferentes.

Se o serviço parar ou não puder escrever no disco, a interface sinaliza que a
gravação falhou e mantém as alterações no navegador. Enquanto o serviço permanece
ativo, erros transitórios de escrita são tentados novamente. Ao reiniciar o
serviço, recarregue a página; se necessário, preserve primeiro a revisão com
**Exportar revisão**. O navegador avisa ao tentar sair com alterações ainda não
confirmadas em arquivo.

## Histórico e conflitos

Para cada atualização, a versão anterior é guardada em uma pasta ao lado do JSON,
por exemplo `minhas_analises.historico/`. O histórico não é removido automaticamente
e pode crescer com o uso. As gravações usam arquivo temporário no mesmo diretório,
`fsync` e substituição atômica. Uma falha não substitui deliberadamente uma revisão
válida por conteúdo incompleto.

Somente um processo do serviço pode trabalhar no mesmo arquivo de revisão. Em
abas distintas do mesmo serviço, versões são verificadas antes de cada gravação.
Se uma aba estiver desatualizada, a escrita é recusada em vez de substituir a
análise mais recente. Exporte o JSON da aba com conflito antes de recarregar.

Ao abrir, se o navegador e o arquivo contiverem revisões diferentes, a interface
pede qual continuar. Ao escolher o arquivo, a revisão alternativa do navegador
é preservada em disco no histórico. Ao escolher o navegador, a versão anterior do
arquivo é preservada pelo procedimento normal de salvamento.

Para recuperar uma versão anterior manualmente, encerre o serviço, preserve o
JSON atual e copie o JSON desejado do histórico para o caminho principal. Em
seguida, reinicie o serviço. Alternativamente, importe uma cópia do histórico
pela interface e aguarde a gravação.

## Exportação gold

O arquivo salvo automaticamente é uma revisão em andamento, não gold automático.
Continue usando os requisitos humanos da auditoria completa e valide o JSON com
a pasta amostral correspondente:

```bash
python scripts/export_reference.py \
  --input /caminho/sample-reference \
  --annotations outputs/annotations/minhas_analises.json
```

Os rótulos, as evidências originais e os critérios do exportador não foram alterados.
Exportar um JSON pelo navegador continua disponível como cópia adicional.

## Testes

Testes offline de escrita atômica, histórico, bloqueios, IDs, recuperação e
compatibilidade com o exportador:

```bash
python -B -m unittest discover -s tests -p test_review_storage.py -v
```

Teste real de navegador e servidor locais, com dados sintéticos e sem modelos:

```bash
FACTUAL_EVAL_BROWSER_TESTS=1 \
  python -B -m unittest discover -s tests -p test_disk_autosave_browser.py -v
```

Instale o extra `browser-tests` e configure Firefox/geckodriver conforme
[bilingual-audit.md](bilingual-audit.md#testes-reproduzíveis). O teste verifica a
gravação física do JSON, a recuperação após limpar o navegador e reiniciar o
serviço, a preservação de rascunhos e o bloqueio de sobrescrita entre abas.
