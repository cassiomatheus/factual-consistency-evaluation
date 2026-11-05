#!/usr/bin/env python3
"""NLI local opcional sobre evidência RANQUEADA (NÃO sobre a fonte integral).

Produz previsões por par claim-evidência, usadas apenas para PRIORIDADE.
Nunca cria gold, nem rotula a claim em relação ao documento completo.

Para discordância entre modelos: rode este script com dois checkpoints
DIFERENTES, preservando mesmo rank, e passe ambos os JSONL ao seletor.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def run(rawdir, output, model_id, rank=1, batch_size=8, max_claims=0, device='auto', max_length=384):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    root=Path(rawdir)
    claims=read_jsonl(root/'claims.jsonl')
    if max_claims:
        claims=claims[:max_claims]
    raw_candidates=read_jsonl(root/'candidatos.jsonl')
    by_id=defaultdict(dict)
    for cand in raw_candidates:
        by_id[cand['claim_id']][cand['rank_final']]=cand
    items=[]
    for c in claims:
        chosen=by_id[c['claim_id']].get(rank)
        if chosen:
            items.append((c,chosen))
    if not items:
        raise ValueError('Nenhum par para avaliar. Confira rank e candidatos.jsonl.')
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else 'cpu'
    print('Checkpoint:',model_id,'Dispositivo:',device,'Pares:',len(items),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    from factual_eval.provenance import record_model
    record_model('reference.suggestion',model_id,model)
    # A ordem difere entre modelos NLI. É obrigatório usar id2label real.
    raw_labels={int(k):str(v).upper() for k,v in model.config.id2label.items()}
    allowed={'ENTAILMENT','CONTRADICTION','NEUTRAL'}
    if set(raw_labels.values())!=allowed or len(raw_labels)!=3:
        raise ValueError(f'id2label inesperado: {raw_labels}. Use checkpoint treinado nas três classes NLI.')
    rows=[]
    for start in range(0,len(items),batch_size):
        batch=items[start:start+batch_size]
        premises=[cand['sentence_text'] for _,cand in batch]
        hypotheses=[claim['claim_text'] for claim,_ in batch]
        tokens=tokenizer(premises,hypotheses,padding=True,truncation=True,
                         max_length=max_length,return_tensors='pt')
        tokens={k:v.to(device) for k,v in tokens.items()}
        with torch.inference_mode():
            probs=torch.softmax(model(**tokens).logits.float(),dim=-1).cpu().tolist()
        for (claim,cand),p in zip(batch,probs):
            probability={raw_labels[i]:round(float(value),7) for i,value in enumerate(p)}
            label=max(probability,key=probability.get)
            rows.append({'claim_id':claim['claim_id'],'model':model_id,
                         'label':label,'probabilities':probability,
                         'evidence_sentence_id':cand['sentence_id'],'evidence_rank':rank,
                         'prediction_scope':'ONE_RETRIEVED_EVIDENCE_ONLY',
                         'warning':'NEUTRAL nesta evidência NÃO equivale a NOT_SUPPORTED no documento.'})
        print(f'Pares {min(start+batch_size,len(items))}/{len(items)}',flush=True)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print('Previsões opcionais:',output,'|',len(rows),'pares')
    return rows

