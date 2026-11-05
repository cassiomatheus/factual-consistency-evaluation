#!/usr/bin/env python3
"""Exporta SOMENTE referência gold verificada por humano e arquivo silver de triagem.

Triagem rápida e scores de ranking nunca criam gold. Um plano amostral restringe
por padrão o gold às claims selecionadas; casos extras exigem flag explícita.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(x) for x in handle if x.strip()]


def write_jsonl(path,rows):
    with Path(path).open('w',encoding='utf-8') as h:
        for r in rows:
            h.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(path,rows,cols):
    with Path(path).open('w',encoding='utf-8',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=cols,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def validate(rawdir, annotations_path, output, include_outside_sample=False):
    root=Path(rawdir)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    claims=read_jsonl(root/'claims.jsonl')
    sentences=read_jsonl(root/'sentencas.jsonl')
    raw_candidates=read_jsonl(root/'candidatos.jsonl')
    annotation=json.loads(Path(annotations_path).read_text(encoding='utf-8'))
    if annotation.get('dataset_id')!=manifest['dataset_id']:
        raise ValueError('Arquivo de revisão de outro dataset. Confira dataset_id.')
    if annotation.get('schema_version')!=1:
        raise ValueError('Esquema de anotações não suportado.')
    selected={}
    sample_meta=None
    if (root/'plano_amostragem.json').exists():
        sample_meta=json.loads((root/'plano_amostragem.json').read_text(encoding='utf-8'))
        selected={r['claim_id']:r for r in read_jsonl(root/'amostragem.jsonl')}
        if sample_meta.get('dataset_id')!=manifest['dataset_id']:
            raise ValueError('Amostragem incompatível com dataset')
        revision_selection_id=annotation.get('selection_id')
        if revision_selection_id and revision_selection_id!=sample_meta['selection_id']:
            raise ValueError('Revisão de plano de amostragem diferente. Não combinar.')
        if set(selected)!={c['claim_id'] for c in claims}:
            raise ValueError('Plano amostral deve cobrir todas as claims')
    dictionary={s['sentence_id']:s for s in sentences}
    candidate_dict={(c['claim_id'],c['sentence_id']):c for c in raw_candidates}
    claim_ids={c['claim_id'] for c in claims}
    user=annotation.get('claims') or {}
    if not isinstance(user,dict):
        raise ValueError('Formato inválido de anotações')
    invalid_annotation_ids=set(user)-claim_ids
    if invalid_annotation_ids:
        raise ValueError(f'IDs de claims desconhecidos: {sorted(invalid_annotation_ids)[:3]}')
    gold=[]
    judgements=[]
    triage=[]
    skipped=[]
    for c in claims:
        cid=c['claim_id']
        a=user.get(cid) or {}
        if not isinstance(a,dict):
            raise ValueError(f'Anotação malformada: {cid}')
        sm=selected.get(cid)
        group=sm['selection_group'] if sm else ('SEM_PLANO' if not sample_meta else 'NAO_SELECIONADA')
        is_sampled=bool(sm['selected']) if sm else not bool(sample_meta)
        if a.get('triage_status'):
            triage.append({'claim_id':cid,'doc_id':c['doc_id'],
                           'selection_group':group,'triage_status':a['triage_status'],
                           'triage_sentence_id':a.get('triage_sentence_id',''),
                           'triage_utc':a.get('triage_utc',''),
                           'reviewed_by':a.get('reviewed_by') or annotation.get('reviewer',''),
                           'is_gold':False})
        if sample_meta and not is_sampled and not include_outside_sample:
            if a:
                skipped.append({'claim_id':cid,'motivos':['fora_da_amostra']})
            continue
        reason=[]
        if a.get('review_status')!='FINALIZADO': reason.append('status_nao_finalizado')
        if a.get('claim_valid')!='SIM': reason.append('claim_nao_validada')
        if a.get('full_document_checked') is not True: reason.append('fonte_nao_conferida')
        if a.get('search_complete') is not True: reason.append('busca_incompleta')
        label=a.get('factual_label')
        if label not in ('SUPPORTED','CONTRADICTED','NOT_SUPPORTED'):
            reason.append('sem_rotulo_factual_conclusivo')
        sets=a.get('gold_evidence_sets') or []
        if not isinstance(sets,list) or any(not isinstance(z,list) for z in sets):
            reason.append('formato_invalido_de_conjuntos')
            sets=[]
        clean_sets=[]
        for subset in sets:
            if not subset or len(set(subset))!=len(subset):
                reason.append('conjunto_vazio_ou_repetido')
                continue
            if any(sid not in dictionary or dictionary[sid]['doc_id']!=c['doc_id'] for sid in subset):
                reason.append('sentence_id_inexistente_ou_outro_documento')
                continue
            clean_sets.append(sorted(subset))
        if label in ('SUPPORTED','CONTRADICTED') and not clean_sets:
            reason.append('faltam_evidencias_suficientes')
        if label=='NOT_SUPPORTED' and sets:
            reason.append('not_supported_nao_pode_ter_conjunto_comprovante')
        if len({tuple(x) for x in clean_sets})!=len(clean_sets):
            reason.append('conjunto_repetido')
        if reason:
            skipped.append({'claim_id':cid,'motivos':sorted(set(reason))})
        else:
            gold.append({'claim_id':cid,'doc_id':c['doc_id'],'variant':c['variant'],
                         'claim_text':c['claim_text'],'claim_original':c['claim_original'],
                         'factual_label':label,'gold_evidence_sets':clean_sets,
                         'human_notes':a.get('notes',''),'reviewed_by':a.get('reviewed_by') or annotation.get('reviewer',''),
                         'reviewed_at':a.get('updated_utc',''),
                         'search_complete':True,'full_document_checked':True,
                         'selection_group':group,'in_random_control_sample':group=='A_ALEATORIA',
                         'random_inclusion_probability':sm.get('random_inclusion_probability') if sm else None,
                         'sampling_weight':sm.get('random_sampling_weight') if sm else None,
                         'selection_id':sample_meta['selection_id'] if sample_meta else None,
                         'dataset_id':manifest['dataset_id']})
        for sid,j in (a.get('candidate_annotations') or {}).items():
            if not isinstance(j,dict) or sid not in dictionary or dictionary[sid]['doc_id']!=c['doc_id']:
                skipped.append({'claim_id':cid,'motivos':[f'sentence_id_de_julgamento_invalido:{sid}']})
                continue
            cand=candidate_dict.get((cid,sid)) or {}
            judgements.append({'claim_id':cid,'doc_id':c['doc_id'],'sentence_id':sid,
                               'selection_group':group,'sentence_text':dictionary[sid]['text'],
                               'relevance':j.get('relevance',''),'relation':j.get('relation',''),
                               'sufficient_alone':j.get('sufficient',''),'notes':j.get('notes',''),
                               'candidate_rank':cand.get('rank_final'),
                               'cross_encoder_logit':cand.get('score_cross_encoder_logit'),
                               'cosine':cand.get('score_cosine'),
                               'is_manual':(cid,sid) not in candidate_dict,
                               'claim_gold_eligible':not bool(reason)})
    dest=Path(output)
    dest.mkdir(parents=True,exist_ok=True)
    write_jsonl(dest/'gold_claims.jsonl',gold)
    write_csv(dest/'julgamentos_evidencias.csv',judgements,
              ['claim_id','doc_id','sentence_id','selection_group','sentence_text','relevance','relation',
               'sufficient_alone','notes','candidate_rank','cross_encoder_logit','cosine','is_manual','claim_gold_eligible'])
    write_csv(dest/'triagem_provisoria.csv',triage,
              ['claim_id','doc_id','selection_group','triage_status','triage_sentence_id','triage_utc','reviewed_by','is_gold'])
    write_csv(dest/'pendencias.csv',
              [{'claim_id':x['claim_id'],'motivos':';'.join(x['motivos'])} for x in skipped],
              ['claim_id','motivos'])
    by_group=Counter(g['selection_group'] for g in gold)
    random_target=sample_meta['counts'].get('A_ALEATORIA',0) if sample_meta else 0
    report={'dataset_id':manifest['dataset_id'],'selection_id':sample_meta['selection_id'] if sample_meta else None,
            'claims_total':len(claims),'selected_claims':sum(r['selected'] for r in selected.values()) if sample_meta else len(claims),
            'claims_gold_validated':len(gold),'claims_pending':len(claims)-len(gold),
            'label_counts':dict(Counter(g['factual_label'] for g in gold)),
            'gold_by_group':dict(by_group),'random_group_selected':random_target,
            'random_group_gold':by_group.get('A_ALEATORIA',0),
            'random_group_complete':bool(sample_meta and by_group.get('A_ALEATORIA',0)==random_target),
            'triage_only_or_preliminary':len(triage),'evidence_judgements':len(judgements),
            'pending_events':len(skipped),
            'warning':'A/B/C são grupos de amostragem, não classes factuais. Somente gold_claims.jsonl é referência humana; triagem_provisoria.csv NÃO é gold. Estimativas globais requerem a amostra aleatória A completa.'}
    (dest/'resumo_validacao.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report


