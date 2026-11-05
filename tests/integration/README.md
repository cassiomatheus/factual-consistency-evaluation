# Integração com modelos reais

Tradução local e regressão da interface bilíngue têm testes separados e opt-in:
veja [docs/bilingual-audit.md](../../docs/bilingual-audit.md#testes-reproduzíveis).
O teste de navegador usa traduções simuladas e não requer GPU nem Ollama.

O salvamento das análises em disco tem testes próprios em
`test_disk_autosave_browser.py`, habilitados por `FACTUAL_EVAL_BROWSER_TESTS=1`.
Eles utilizam um servidor em loopback e arquivos temporários; consulte
[review-autosave.md](../../docs/review-autosave.md).

Os testes rápidos do projeto usam modelos simulados. Estes testes adicionais são opt-in, exigem CUDA, checkpoints do baseline em cache e memória suficiente. Eles não foram aprovados por inferência em GPU durante a migração se o relatório final registrar CUDA indisponível.

```bash
HF_HUB_OFFLINE=1 FACTUAL_EVAL_GPU_TESTS=1 \
  python -B -m unittest discover -s tests -v
```

Os casos comparam os métodos migrados de geração T5 e inferência NLI com snapshots dos métodos originais, compartilhando os mesmos pesos/tokenizadores para controlar a entrada da comparação. Não substituem uma repetição completa de extração, recuperação e taxonomia sobre o corpus. Uma reprodução neural ponta a ponta continua sendo uma etapa separada, com run_id próprio.
