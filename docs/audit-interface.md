# Guia rápido da interface de auditoria

A interface inicia em **Auditoria completa**, com uma afirmação por vez. O modo
**Triagem rápida · opcional** continua disponível para decisões preliminares.

## Os três passos

1. **Confira a afirmação.** Leia a tradução e o inglês original. Compare com a
   sentença do resumo. Quando forem idênticos, a interface indica essa igualdade
   sem repetir o texto. Os IDs continuam nos dados e na fila.
2. **Selecione evidências.** Leia os trechos sugeridos ou busque no documento à
   direita. **Selecionar esta evidência** e **Selecionar evidência** montam uma
   seleção temporária. Confirme-a em **Salvar conjunto selecionado**. Trechos
   selecionados fora das sugestões são adicionados à análise. Use conjuntos
   separados para evidências alternativas e o mesmo conjunto para sentenças
   necessárias juntas.
3. **Registre a decisão.** Indique se a extração está correta e escolha uma
   conclusão: tem suporte, é contradita, evidência insuficiente ou ainda não sabe.
   Confirme a conferência da fonte e a conclusão da busca. A lista de pendências
   mostra o que falta; **Concluir revisão e próxima** marca a revisão finalizada
   e avança para outra afirmação selecionada.

Os botões numerados no alto da coluna central levam diretamente a cada passo.
**Guardar pendente e ir para próxima** permite continuar depois sem marcar a
análise como finalizada. Uma afirmação extraída incorretamente pode ser concluída
como inválida após registrar a justificativa; ela permanece fora da referência gold.

Os nomes dos rótulos aparecem em português, mas os valores exportados continuam
`SUPPORTED`, `CONTRADICTED`, `NOT_SUPPORTED`, `UNCERTAIN`, `SIM`, `NAO` etc. A
interface não gera julgamentos factuais por conta própria. Os critérios do
exportador em `reference/export.py` continuam sendo a validação final para gold.

## Ler, buscar e acompanhar

- **Dados da recuperação e identificador** expande scores e o ID integral do
  trecho. Nenhum ranking ou score é recalculado por essa apresentação.
- **Detalhar a avaliação deste trecho · opcional** expande os julgamentos
  individuais de relevância, relação e suficiência, além das notas do trecho.
- **Ver no documento** localiza a sentença na fonte. **Limpar busca e ver todo o
  documento** restaura a consulta integral, sem apagar as evidências selecionadas.
- **Abrir na análise** mostra um trecho do documento na coluna central.
- A fila mostra o número de revisões marcadas como concluídas e uma barra de
  progresso. “Revisada” não significa que a claim entrou automaticamente no gold.
- As colunas laterais têm rolagem própria em telas grandes; em telas menores,
  os painéis são empilhados. Português e inglês continuam disponíveis.

## Salvar não é concluir

As edições e os rascunhos continuam sendo salvos automaticamente. O modo com
servidor local grava no arquivo JSON e mantém histórico; confira o indicador do
cabeçalho. “Aguardando gravação” ou uma falha não são uma confirmação de gravação.
Concluir a revisão é uma ação separada do salvamento.

O serviço `serve_review.py` incorpora o modelo da interface ao iniciar. Se já
estiver aberto, aguarde **Salvo em arquivo ✓**, encerre com Ctrl+C e inicie o
mesmo comando novamente, mantendo o mesmo caminho `--annotations`. As revisões
serão recuperadas do arquivo. Não é necessário executar novamente modelos ou
preparar outra amostra. Veja [review-autosave.md](review-autosave.md).

Para HTMLs abertos diretamente como arquivo, gere uma nova interface com os dados
e traduções existentes. Antes de trocar de arquivo, exporte a revisão e importe-a
no novo HTML. Esse cuidado não é necessário para transferir entre versões servidas
que reutilizam o mesmo JSON em disco, salvo um conflito explicitamente indicado.

## Testes de usabilidade

```bash
FACTUAL_EVAL_BROWSER_TESTS=1 \
  python -B -m unittest discover -s tests -p test_audit_usability_browser.py -v
```

Os testes usam dados sintéticos e verificam pendências, confirmação de conjuntos,
conclusão e avanço, preservação dos rótulos exportados, exclusão de claims inválidas
do gold, limpeza da busca e ausência de rolagem horizontal em tela estreita.
