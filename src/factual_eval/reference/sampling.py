#!/usr/bin/env python3
"""Amostragem A/B/C e fila de prioridade para revisão humana.

A = amostra aleatória simples (válida para avaliação descritiva da população
    APENAS se todas as claims sorteadas forem adequadamente revisadas).
B = casos incertos segundo sinais de recuperação, opcionalmente modelos externos.
C = casos estruturalmente complexos, sem consultar rótulos de injeção.

Atribuições automáticas NUNCA são rótulos factuais nem gold.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def bound(x):
    return max(0.0, min(1.0, float(x)))


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a or b else 1.0


def structural_signals(claim):
    txt = claim.get('claim_text', '')
    original = claim.get('claim_original', '')
    resolved = claim.get('claim_resolved', '')
    tokens = re.findall(r"(?i)\b[a-z]+\b", txt)
    refs = {'it','they','them','their','he','she','his','her','this','these','those','its','which','who'}
    negatives = {'not','never','no','neither','nor','without','failed','deny','denied','rejected','refused'}
    signals = {
        'pronome_referencial': any(t.lower() in refs for t in tokens),
        'correferencia_expandida': bool(original and resolved and original != resolved),
        'numero_ou_data': bool(re.search(r'\b\d+(?:[.,]\d+)?\b', txt)),
        'negacao_ou_inversao': any(t.lower() in negatives for t in tokens) or bool(re.search(r"\b(?:didn't|wasn't|isn't|don't|doesn't|cannot|can't)\b", txt.lower())),
        'entidades_ou_siglas': bool(re.search(r'\b[A-Z]{2,}\b', txt)),
    }
    w = [.32, .38, .17, .25, .12]
    risk = bound(sum(weight for weight, hit in zip(w, signals.values()) if hit))
    return signals, risk


def normalize_label(label):
    label = str(label or '').upper().replace(' ', '_')
    if label in ('SUPPORTED','SUPPORTS','ENTAILMENT','ENTAILED','TRUE'):
        return 'SUPPORT'
    if label in ('CONTRADICTED','CONTRADICTION','CONTRADICTS','NOT_SUPPORTED','INSUFFICIENT_EVIDENCE','NEUTRAL','UNSUPPORTED','FALSE'):
        return 'NO_SUPPORT'
    return None


def load_predictions(paths, valid_claims):
    if not paths:
        return {}
    if isinstance(paths, (str, Path)):
        paths=[paths]
    predictions = defaultdict(dict)
    for path in paths:
      for row in read_jsonl(path):
        cid, model = row.get('claim_id'), row.get('model')
        if cid not in valid_claims:
            raise ValueError(f'Predição com claim_id desconhecido: {cid}')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('Toda predição deve indicar um identificador de modelo')
        if model in predictions[cid]:
            raise ValueError(f'Predição duplicada para claim {cid} e modelo {model}')
        predictions[cid][model] = row
    return predictions


def model_signal(rows):
    if not rows:
        return None, {'modelos_validos':0,'discordancia_modelos':None,'incerteza_probabilidade':None}
    binary_labels = [normalize_label(r.get('label')) for r in rows.values()]
    binary_labels = [r for r in binary_labels if r]
    disagreement = 1.0 if len(set(binary_labels)) > 1 else 0.0 if len(binary_labels) >= 2 else None
    entropies = []
    for r in rows.values():
        probs = r.get('probabilities')
        if not isinstance(probs, dict) or len(probs) < 2:
            continue
        values = list(probs.values())
        if any(not isinstance(x,(int,float)) or isinstance(x,bool) or x<0 or not math.isfinite(x) for x in values):
            continue
        total = sum(values)
        if total <= 0:
            continue
        p = [v/total for v in values if v>0]
        entropies.append(-sum(v*math.log(v) for v in p)/math.log(len(values)))
    entropy = sum(entropies)/len(entropies) if entropies else None
    score = max([s for s in (disagreement, entropy) if s is not None], default=None)
    details = {'modelos_validos':len(binary_labels),'discordancia_modelos':disagreement,
               'incerteza_probabilidade':round(entropy,4) if entropy is not None else None}
    return score, details


def rank_signals(candidates):
    if not candidates:
        return {'discordancia_ranking':1.0,'ambiguidade_margem':1.0,'top_cosine':None,'tamanho_pool':0}
    final = sorted(candidates, key=lambda x:x['rank_final'])
    top = [r['sentence_id'] for r in final[:3]]
    dense = [r['sentence_id'] for r in sorted(candidates,key=lambda x:(x['rank_dense'],x['sentence_id']))[:3]]
    bm = [r['sentence_id'] for r in sorted(candidates,key=lambda x:(x['rank_bm25'],x['sentence_id']))[:3]]
    disagreement = 1.0 - (jaccard(top,dense)+jaccard(top,bm)+jaccard(dense,bm))/3
    scores = [r.get('score_cross_encoder_logit') for r in final]
    scores = [float(x) for x in scores if isinstance(x,(int,float)) and math.isfinite(x)]
    if len(scores) >= 2:
        values = sorted(scores,reverse=True)
        gap = max(0,values[0]-values[1])
        span = max(0.01,values[0]-values[-1])
        uncertainty = 1-bound(gap/span)
    else:
        # Sem CE confiável, não tratar score RRF como confiança factual.
        uncertainty = 0.5
    cos = [r.get('score_cosine') for r in final[:1]]
    top_cos = cos[0] if cos and isinstance(cos[0],(int,float)) else None
    return {'discordancia_ranking':round(disagreement,5),
            'ambiguidade_margem':round(uncertainty,5),
            'top_cosine':top_cos,'tamanho_pool':len(final)}


def choose_diversified(items, amount, current_counts):
    """Escolhe altos escores com penalidade pequena por concentração documental."""
    selected=[]
    available=list(items)
    while available and len(selected)<amount:
        available.sort(key=lambda x:(-(x['_sort_score']-min(30,current_counts[x['doc_id']]*3)),x['claim_id']))
        item=available.pop(0)
        selected.append(item)
        current_counts[item['doc_id']]+=1
    return selected


def select(rawdir, sample_size=200, fractions=(.4,.4,.2), seed=42, predictions_file=None):
    root = Path(rawdir)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    claims=read_jsonl(root/'claims.jsonl')
    candidates=read_jsonl(root/'candidatos.jsonl')
    if not claims:
        raise ValueError('Não há claims para selecionar')
    if len({c['claim_id'] for c in claims})!=len(claims):
        raise ValueError('claim_id duplicado')
    if sample_size < 1 or any(f<0 for f in fractions) or abs(sum(fractions)-1)>1e-6:
        raise ValueError('Tamanho da amostra positivo e frações A+B+C somando 1 são obrigatórios')
    pred=load_predictions(predictions_file, {c['claim_id'] for c in claims})
    by_claim=defaultdict(list)
    for candidate in candidates:
        by_claim[candidate['claim_id']].append(candidate)
    # Os sinais usam EXCLUSIVAMENTE conteúdo da claim e ranking, jamais tipo de injeção.
    rows=[]
    for c in claims:
        signals,risk=structural_signals(c)
        signals_r=rank_signals(by_claim[c['claim_id']])
        auto,model_details=model_signal(pred.get(c['claim_id']))
        row={'claim_id':c['claim_id'],'doc_id':c['doc_id'],
             'sinais_estruturais':signals,'risco_estrutural':round(risk,5),
             **signals_r,**model_details,'incerteza_modelo':auto}
        rows.append(row)
    # Confiança do recuperador é relativa ao próprio documento, não factual.
    grouped=defaultdict(list)
    for r in rows:
        if r['top_cosine'] is not None:
            grouped[r['doc_id']].append(r['top_cosine'])
    for r in rows:
        cos = r['top_cosine']
        peers = grouped[r['doc_id']]
        cosine_shortfall = (1-sum(x<=cos for x in peers)/len(peers)) if cos is not None and len(peers)>=4 else 0.5
        r['cosine_relativa_baixa']=round(cosine_shortfall,5)
        terms=[(.40,r['discordancia_ranking']),(.25,r['ambiguidade_margem']),(.15,cosine_shortfall),(.20,r['risco_estrutural'])]
        if r['incerteza_modelo'] is not None:
            terms.append((.25,r['incerteza_modelo']))
        r['priority_score']=round(100*sum(w*v for w,v in terms)/sum(w for w,_ in terms),2)
        reasons=[]
        if r['discordancia_ranking']>=.5: reasons.append('rankings_divergentes')
        if r['ambiguidade_margem']>=.67: reasons.append('margem_pequena_entre_candidatos')
        if cosine_shortfall>=.7: reasons.append('cosseno_relativamente_baixo')
        if r['incerteza_modelo'] is not None and r['incerteza_modelo']>=.5: reasons.append('modelos_divergentes_ou_incertos')
        if r['risco_estrutural']>0: reasons.append('complexidade_linguistica')
        r['motivos_prioridade']=reasons
    rng=random.Random(seed)
    total=min(sample_size,len(rows))
    n_a=max(1,min(total,round(total*fractions[0]))) if fractions[0]>0 else 0
    n_c=min(total-n_a,round(total*fractions[2]))
    n_b=total-n_a-n_c
    all_sorted=sorted(rows,key=lambda r:r['claim_id'])
    a_ids={x['claim_id'] for x in rng.sample(all_sorted,n_a)}
    remaining=[r for r in all_sorted if r['claim_id'] not in a_ids]
    doc_counts=Counter(r['doc_id'] for r in all_sorted if r['claim_id'] in a_ids)
    structurals=[{**r,'_sort_score':100*r['risco_estrutural']+.2*r['priority_score']} for r in remaining if r['risco_estrutural']>0]
    c_selected=choose_diversified(structurals,n_c,doc_counts)
    c_ids={r['claim_id'] for r in c_selected}
    rest=[{**r,'_sort_score':r['priority_score']} for r in remaining if r['claim_id'] not in c_ids]
    b_selected=choose_diversified(rest,n_b+(n_c-len(c_selected)),doc_counts)
    b_ids={r['claim_id'] for r in b_selected}
    probability=round(n_a/len(rows),9)
    for r in rows:
        cid=r['claim_id']
        group='A_ALEATORIA' if cid in a_ids else 'B_INCERTA' if cid in b_ids else 'C_ESTRUTURAL' if cid in c_ids else 'NAO_SELECIONADA'
        r['selected']=group!='NAO_SELECIONADA'
        r['selection_group']=group
        r['random_inclusion_probability']=probability if group=='A_ALEATORIA' else None
        r['random_sampling_weight']=round(1/probability,7) if group=='A_ALEATORIA' and probability else None
    paths = [predictions_file] if isinstance(predictions_file,(str,Path)) else (predictions_file or [])
    prediction_hashes=[hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths]
    selection_hash=hashlib.sha256(json.dumps({
        'model_predictions_sha256':prediction_hashes,
        'selection_strategy_version':'v2.1',
        'dataset_id':manifest['dataset_id'],'seed':seed,'sample_size':total,'fractions':fractions,
        'assignment':[[r['claim_id'],r['selection_group']] for r in rows]},sort_keys=True).encode()).hexdigest()
    sample_meta={'schema_version':1,'dataset_id':manifest['dataset_id'],'selection_id':selection_hash,
                 'seed':seed,'population_claims':len(rows),'sample_requested':sample_size,
                 'sample_selected':sum(r['selected'] for r in rows),
                 'quota_a':n_a,'quota_b':n_b,'quota_c':n_c,
                 'counts':dict(Counter(r['selection_group'] for r in rows)),
                 'group_a_inclusion_probability':probability,
                 'external_model_predictions':bool(predictions_file),
                 'model_predictions_sha256':prediction_hashes,
                 'sampling_method':'A: SRS de claims no frame completo; C: estratificação intencional por sinais linguísticos; B: prioridade de incerteza no restante; B/C balanceamento documental heurístico',
                 'validity_warning':'Avaliação global não enviesada requer anotar TODAS as claims da amostra A. B e C NÃO são aleatórias; resultados separados. Scores não são probabilidade de factualidade.',
                 'label_warning':'Grupos B/C são seleção por heurística; nenhum rótulo factual é criado.'}
    write_jsonl(root/'amostragem.jsonl',rows)
    (root/'plano_amostragem.json').write_text(json.dumps(sample_meta,indent=2,ensure_ascii=False),encoding='utf-8')
    columns=['claim_id','doc_id','selected','selection_group','priority_score','discordancia_ranking',
             'ambiguidade_margem','cosine_relativa_baixa','risco_estrutural','incerteza_modelo','motivos_prioridade']
    with (root/'ranking_prioridade.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        for r in sorted(rows,key=lambda x:(-x['priority_score'],x['claim_id'])):
            writer.writerow({**r,'motivos_prioridade':';'.join(r['motivos_prioridade'])})
    print(json.dumps(sample_meta,indent=2,ensure_ascii=False))
    return sample_meta,rows


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dados',required=True,help='Pasta com JSONL produzidos por 01_preparar_dados.py')
    p.add_argument('--amostra',type=int,default=200)
    p.add_argument('--fracao-a',type=float,default=.4)
    p.add_argument('--fracao-b',type=float,default=.4)
    p.add_argument('--fracao-c',type=float,default=.2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--previsoes-factuais',nargs='+',default=None,help='Um ou mais arquivos JSONL opcionais (claim_id, model, label, probabilities)')
    return p


