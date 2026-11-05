# Validações executadas na migração

## Usabilidade da auditoria — 5 de novembro de 2026

Fluxo e operação em [audit-interface.md](audit-interface.md).

| Verificação | Resultado |
|---|---|
| Suíte offline | 58 testes: 48 aprovados, 10 integrações opt-in ignoradas |
| Navegador | 7 testes aprovados no Firefox 159.0.1, incluindo 3 casos de usabilidade |
| Conclusão guiada | Sem conjunto salvo o botão não conclui; após os requisitos humanos preenchidos, finaliza e avança preservando IDs e rótulos |
| Afirmação inválida | Pode ser marcada como revisada com justificativa, sem fabricar um rótulo factual; permanece fora do gold |
| Navegação | Limpeza da busca, seleção de evidências e layout sem rolagem horizontal em largura de 390 px verificados |
| Regressão | Busca bilíngue, scores acessíveis, conjuntos, importação/exportação, gravação em disco e recuperação após reinício continuam funcionando |
| Interface gerada | `outputs/runs/auditoria-usabilidade-20261105/review-html/avaliacao.html`, com os dados e traduções existentes |

Nenhuma extração, recuperação, inferência NLI, taxonomia ou tradução foi repetida.
O exportador gold continua sendo a validação final da elegibilidade dos registros.

## Salvamento de análises em disco — 5 de novembro de 2026

Procedimento em [review-autosave.md](review-autosave.md).

| Verificação | Resultado |
|---|---|
| Suíte offline | 55 testes: 48 aprovados, 7 integrações opt-in ignoradas |
| Navegador real | 4 testes aprovados no Firefox 159.0.1: regressão bilíngue e salvamento local |
| Recuperação | Notas, revisor e seleção em montagem recuperados do JSON após limpar os dados do navegador e reiniciar o serviço |
| Histórico e falhas | Versões anteriores preservadas, escrita atômica e bloqueio de arquivos verificados |
| Conflitos e acesso | Escrita desatualizada de outra aba e requisições sem token/origem local recusadas |
| Compatibilidade | JSON salvo aceito pelo exportador gold; rascunhos não tratados como evidência gold |
| Integridade | HTML de entrada permanece inalterado; nenhuma etapa científica ou modelo foi executado para salvar anotações |

Os testes de navegador usaram dados sintéticos e arquivos temporários. As
anotações existentes no navegador do pesquisador devem ser migradas pelo JSON
exportado; não foram transferidas automaticamente para disco por esses testes.

## Auditoria bilíngue — 23 de outubro de 2026

Implementação e comandos em [bilingual-audit.md](bilingual-audit.md). Resultados
verificados com o Python do ambiente `nlp_mestrado`:

| Verificação | Resultado |
|---|---|
| Suíte offline atual | 47 testes: 42 aprovados, 5 integrações opt-in ignoradas (2 GPU, 1 Qwen, 2 navegador) |
| Tradução real | 1 teste aprovado com Ollama 0.35.0 e `qwen3.5:9b`, temperatura 0 e `think=False`; número 8 e negação preservados no exemplo |
| Navegador real isolado | 2 testes aprovados no Firefox 158.0.1/Selenium 4.51.0, com textos sintéticos e tradutor simulado |
| Operações da interface | Modos de idioma, preferência de sessão, busca EN/PT, navegação, paginação, scores, evidências manuais, triagem, conjuntos, exportação/importação e validação gold verificadas |
| Traduções ausentes/com aviso | Fallback para inglês e expansão do motivo verificados no navegador |
| Cache e retomada | Reutilização, isolamento por contexto, falha, limite incremental e proteção `human_verified` após mudar modelo/prompt verificados sem modelos |
| Integridade dos dados existentes | 13 arquivos conferidos por SHA-256 antes/depois do piloto: Parquets da execução de cinco documentos e arquivos da amostra original; nenhum alterado |
| Piloto real | `gold-govreport5-traducao-piloto`: 8 chamadas, 3 reutilizações de segmentos, 10 unidades `translated`, 1 `needs_review`, 2.622 `pending`, sem falhas |
| HTML real | `gold-govreport5-html-bilingue/review-html/avaliacao.html`, aberto e inspecionado visualmente; inglês, português e fallback presentes |

O digest do modelo foi
`56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`.
O piloto não traduziu integralmente as 2.633 unidades da amostra. As heurísticas e
o teste de uma frase não comprovam fidelidade semântica geral. Não foram criados
novos rótulos humanos ou resultados científicos a partir das traduções.

## Histórico da migração

Execução em 22 de outubro de 2026, usando o Python de `nlp_mestrado`. Nenhum modelo neural grande foi carregado para estes resultados. A disponibilidade CUDA observada foi falsa; a inferência real em GPU permanece pendente.

| Verificação | Resultado observado |
|---|---|
| Suíte unittest | 33 testes descobertos: 31 aprovados e 2 integrações GPU ignoradas por serem opt-in |
| Pipeline completo com modelos simulados | Extração → recuperação → NLI → taxonomia; contagens e IDs preservados; não mede factualidade |
| Métodos científicos vs snapshots originais | BM25, reranking contextual, janelas/MMR, geração/decodificação T5 simulada, agregação, limites NLI e prioridade taxonômica verificados |
| Probabilidades históricas, lotes 5/10/50 | 6.338 escores agregados e decisões idênticos, sem reinferência neural |
| Lote histórico de 3 documentos | 235/270 decisões coincidem com a configuração atual; mantido como histórico divergente |
| Auditoria técnica do CSV v7 real | 150 variantes verificadas; 150 reconstruções exatas; não comprova contradição factual |
| Preparação/amostragem/HTML reais com fixture sintética | 2 documentos, 4 claims, 22 candidatos; amostra de 3; HTML gerado sem modelos HF |
| Exclusividade e reutilização | Diretórios existentes recusados; mudança de configuração impede reutilização; adulteração de artefato detectada |
| Integridade da origem | 103 arquivos presentes no inventário continuam com hashes idênticos |
| Entradas de execução | 13 scripts respondem a --help a partir de /tmp |
| Empacotamento | Wheel construído e instalado sem dependências em diretório temporário; imports e CLI testados fora do projeto |
| Recursos/licenças no wheel | Configurações, antônimos, template HTML, LICENSE e THIRD_PARTY_NOTICES presentes |

Comando da suíte:

```bash
python -B -m unittest discover -s tests -v
```

Os testes da ferramenta humana original foram adaptados e incluídos. Foram acrescentados testes para reconstrução com expressões repetidas, offsets, esquemas Parquet, IDs de claims repetidas, subset taxonômico, manutenção do prefixo RNG na geração, ausência de carregamento de modelos por importação, configuração, execução e proteção de saídas.

As fixtures históricas versionadas contêm probabilidades e parâmetros, sem texto dos documentos. `historical_regression.json` registra hashes dos Parquets externos usados na comparação completa. Os resultados sintéticos estão em outputs/runs/verification-* e são ignorados pelo Git. Eles não são experimentos científicos do artigo.

Pendências: inferência neural real T5/NLI; reprodução de recuperação e taxonomia com modelos reais; instalação limpa de todas as dependências em outro ambiente; revisão da combinação spaCy/modelo sm observada; auditoria semântica das injeções; referência humana finalizada; implementação das avaliações científicas E0–E6. Nenhum desses itens é declarado aprovado por esta migração.

A igualdade de AST registrada em scientific_implementation_audit.json cobre 130 definições preservadas. Outras 12 foram adaptadas para execução/configuração/proveniência; 29 definições de componentes não necessários não foram incorporadas. Esses números se referem aos arquivos selecionados no mapeamento e não constituem cobertura de testes ou validação semântica.
