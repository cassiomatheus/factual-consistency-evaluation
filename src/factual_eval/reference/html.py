#!/usr/bin/env python3
"""Parte 2: transforma os JSONL brutos em uma interface HTML portátil, sem Ollama."""
import argparse
import json
from pathlib import Path


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def run(rawdir,html,translations=None):
    folder=Path(rawdir)
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    documents=jsonl(folder/'documentos.jsonl')
    sentences=jsonl(folder/'sentencas.jsonl')
    claims=jsonl(folder/'claims.jsonl')
    candidates=jsonl(folder/'candidatos.jsonl')
    by_doc={d['doc_id']:[] for d in documents}
    for s in sentences:
        by_doc[s['doc_id']].append(s)
    for d in documents:
        d['sentences']=sorted(by_doc[d['doc_id']],key=lambda s:s['sentence_index'])
        # Não duplicar megabytes de texto: as sentenças já formam o documento integral.
        d.pop('source_text',None)
    # Amostragem opcional: mantém compatibilidade com datasets antigos sem fila seletiva.
    selection=[]
    sample_meta=None
    if (folder/'amostragem.jsonl').exists():
        selection=jsonl(folder/'amostragem.jsonl')
        sample_meta=json.loads((folder/'plano_amostragem.json').read_text(encoding='utf-8'))
        if sample_meta['dataset_id'] != manifest['dataset_id']:
            raise ValueError('A amostragem corresponde a um dataset diferente')
        if {r['claim_id'] for r in selection} != {c['claim_id'] for c in claims}:
            raise ValueError('A amostragem não cobre as claims deste dataset')
    payload={'manifest':manifest,'documents':documents,'claims':claims,
             'candidates':candidates,'selection':selection,'sample_meta':sample_meta}
    if translations:
        from .translation import collect_units, load_translations
        rows=load_translations(translations,manifest['dataset_id'],sample_meta['selection_id'] if sample_meta else None)
        originals={u['unit_id']:u['source_en'] for u in collect_units(folder,scope='all')}
        for uid,row in rows.items():
            if originals.get(uid)!=row['source_en']:
                raise ValueError(f'Tradução não corresponde ao original: {uid}')
        payload['translations']={uid:{k:r[k] for k in ('source_en','translation_pt','status','warnings')}
                                 for uid,r in rows.items()}
    here=Path(__file__).resolve().parent
    template=(here/'modelo.html').read_text(encoding='utf-8')
    # Impede fechamento acidental da tag script por texto do corpus.
    safe_json=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page=template.replace('__DATASET_JSON__',safe_json)
    destination=Path(html)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(page,encoding='utf-8')
    print('Interface HTML:',destination.resolve())
    print('Documentos:',len(documents),'Claims:',len(claims),'Candidatos:',len(candidates))
    return destination


