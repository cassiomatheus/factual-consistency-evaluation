#!/usr/bin/env python3
"""Parte 1: consome o Parquet original, segmenta o documento e recupera evidências.

Modos:
  --backend hf: cosine MiniLM + BM25 + CrossEncoder reranker (modo científico).
  --backend tfidf: TF-IDF cosine + BM25/RRF, somente smoke test offline.
Nenhum modelo é autorizado a criar gold labels.
"""
import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def clean_number(v):
    if v is None:
        return None
    try:
        x = float(v)
        return round(x, 7) if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def hash_string(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def clean_text(t):
    return re.sub(r'\s+', ' ', str(t)).strip()


def clean_markers(t):
    return re.sub(r'\[ERROR_(?:ENTITY|PREDICATE|COREFERENCE):\s*[^\]]*\]', '', str(t), flags=re.I)


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def load_records(path):
    p = Path(path)
    if p.suffix.lower() == '.parquet':
        import pandas as pd
        return pd.read_parquet(p).to_dict('records')
    if p.suffix.lower() in ('.jsonl', '.ndjson'):
        return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
    if p.suffix.lower() == '.json':
        obj = json.loads(p.read_text(encoding='utf-8'))
        return obj['records'] if isinstance(obj, dict) and 'records' in obj else obj
    raise ValueError('Entrada deve ser .parquet, .json ou .jsonl')


def unpack_claims(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f'Formato inesperado de claims: {type(value).__name__}')
    return value


def segment_document(source, docid):
    # Sentencizer, sem carregar dependências sintáticas custosas na fonte longa.
    import spacy
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    nlp.max_length = max(nlp.max_length, len(source) + 100)
    sentences = []
    for s in nlp(source).sents:
        txt = s.text.strip()
        if not txt:
            continue
        left = len(s.text) - len(s.text.lstrip())
        start = s.start_char + left
        end = start + len(txt)
        sentences.append({
            'sentence_id': f'{docid}:S{len(sentences)+1:05d}',
            'doc_id': docid,
            'sentence_index': len(sentences),
            'text': txt,
            'char_start': start,
            'char_end': end,
            'relative_position': round(start/max(1, len(source)), 6)
        })
    if not sentences:
        raise ValueError(f'Documento {docid} sem sentenças')
    return sentences


def prepare_inputs(records, variant_filter, field, max_records, max_claims):
    docs = {}
    claims = []
    n_records = 0
    for rec in records:
        variant = str(rec.get('tipo_resumo','original'))
        if variant_filter and variant not in variant_filter:
            continue
        if max_records and n_records >= max_records:
            break
        n_records += 1
        source = rec.get('documento_fonte')
        if not isinstance(source, str) or not source.strip():
            raise ValueError('documento_fonte ausente. Verifique a saída de seu extrator.')
        external_id = str(rec.get('doc_idx', n_records-1))
        docid = f'doc_{re.sub(r"[^a-zA-Z0-9_-]", "_", external_id)}_{hash_string(source)[:8]}'
        if docid not in docs:
            sentences = segment_document(source, docid)
            docs[docid] = {'doc_id': docid,'source_sha256': hash_string(source),
                           'source_text': source, 'source_length': len(source),
                           'sentences': sentences, 'external_doc_idx': external_id}
        for idx, cl in enumerate(unpack_claims(rec.get('claims'))):
            if isinstance(cl, str):
                cl = {'claim_original':cl, 'claim_resolvida':cl}
            if not isinstance(cl, dict):
                raise ValueError(f'Claim inválida: {type(cl).__name__}')
            text = clean_text(cl.get(field) or cl.get('claim_original') or cl.get('texto') or '')
            if not text:
                continue
            original = clean_text(cl.get('claim_original') or text)
            resolved = clean_text(cl.get('claim_resolvida') or cl.get('texto') or original)
            origin_sent = clean_markers(cl.get('sentenca_origem') or cl.get('sentence_source') or '')
            uid = f'{docid}:{variant}:C{idx+1:05d}'
            claims.append({
                'claim_id': uid, 'doc_id': docid, 'variant': variant,
                'claim_text': clean_markers(text), 'claim_original':clean_markers(original),
                'claim_resolved': clean_markers(resolved),
                'origin_sentence':clean_text(origin_sent),
                'origin_sentence_index':cl.get('indice_sentenca',cl.get('sentence_index')),
                'summary_text':clean_text(clean_markers(rec.get('resumo_texto') or '')),
                'claim_field_used':field
            })
            if max_claims and len(claims) >= max_claims:
                return docs, claims
    if not claims:
        raise ValueError('Nenhuma claim encontrada. Confira --variantes e --campo-claim.')
    return docs, claims


def tokenize(text):
    return re.findall(r'(?u)\b\w+\b', text.lower())


def simple_bm25(doc_sentences, query, k1=1.5, b=.75):
    """BM25 in-memory sem dependência externa, por documento."""
    toks = [tokenize(s) for s in doc_sentences]
    df = Counter()
    for arr in toks:
        df.update(set(arr))
    n = len(toks)
    avg = sum(map(len,toks))/max(1,n)
    result = []
    for arr in toks:
        tf = Counter(arr)
        score = 0.
        for term in set(tokenize(query)):
            if not tf[term]:
                continue
            idf = math.log(1+(n-df[term]+.5)/(df[term]+.5))
            score += idf * (tf[term]*(k1+1)/(tf[term]+k1*(1-b+b*len(arr)/max(avg,1))))
        result.append(score)
    return np.asarray(result, dtype=np.float32)


def ordered(scores, upper=None):
    ix = sorted(range(len(scores)), key=lambda i:(-float(scores[i]),i))
    return ix[:upper] if upper else ix


def rrf(rank_maps, index, k=60):
    return sum(1./(k+rank_map[index]) for rank_map in rank_maps if index in rank_map)


def get_dense_and_reranker(backend, device, embedding_model, reranker_model, embedding_revision=None, reranker_revision=None):
    if backend != 'hf':
        return None, None
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from sentence_transformers import SentenceTransformer, CrossEncoder
    print('Carregando encoder de sentenças:', embedding_model, flush=True)
    embedder = SentenceTransformer(embedding_model, device=device, revision=embedding_revision)
    print('Carregando cross-encoder:', reranker_model, flush=True)
    reranker = CrossEncoder(reranker_model, device=device, max_length=384, revision=reranker_revision)
    from factual_eval.provenance import record_model
    record_model('reference.embedding', embedding_model, embedder, embedding_revision)
    record_model('reference.reranker', reranker_model, reranker, reranker_revision)
    return embedder, reranker


def run(args):
    raw = load_records(args.entrada)
    variants = set(args.variantes.split(',')) if args.variantes else None
    docs, claims = prepare_inputs(raw, variants, args.campo_claim, args.max_registros, args.max_claims)
    output = Path(args.saida)
    output.mkdir(parents=True, exist_ok=True)
    embedder, reranker = get_dense_and_reranker(args.backend,args.device,args.embedding_model,args.reranker_model,
        getattr(args,'embedding_revision',None),getattr(args,'reranker_revision',None))
    claims_by_doc = defaultdict(list)
    for claim in claims:
        claims_by_doc[claim['doc_id']].append(claim)
    candidates = []
    print(f'{len(docs)} documentos; {len(claims)} claims; backend={args.backend}', flush=True)
    for docid, doc in docs.items():
        sentences = doc['sentences']
        texts = [s['text'] for s in sentences]
        batch_claims = claims_by_doc[docid]
        claim_texts = [c['claim_text'] for c in batch_claims]
        if args.backend == 'hf':
            sent_emb = embedder.encode(texts, batch_size=args.embedding_batch,
                                       convert_to_numpy=True, normalize_embeddings=True,
                                       show_progress_bar=False)
            cl_emb = embedder.encode(claim_texts, batch_size=args.embedding_batch,
                                     convert_to_numpy=True, normalize_embeddings=True,
                                     show_progress_bar=False)
            dense_matrix = np.asarray(cl_emb @ sent_emb.T, dtype=np.float32)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vectorizer = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
            all_mat = vectorizer.fit_transform(texts + claim_texts)
            dense_matrix = cosine_similarity(all_mat[len(texts):], all_mat[:len(texts)]).astype(np.float32)
        for k_claim, claim in enumerate(batch_claims):
            dense = dense_matrix[k_claim]
            lexical = simple_bm25(texts, claim['claim_text'])
            full_dense = ordered(dense)
            full_bm25 = ordered(lexical)
            dense_rank = {idx:r+1 for r,idx in enumerate(full_dense)}
            bm25_rank = {idx:r+1 for r,idx in enumerate(full_bm25)}
            pool = set(full_dense[:args.dense_pool]) | set(full_bm25[:args.bm25_pool])
            fused = sorted(pool, key=lambda i:(-rrf([dense_rank,bm25_rank],i),i))
            fused = fused[:args.rerank_pool]
            if reranker is not None:
                scores = reranker.predict([(claim['claim_text'],texts[i]) for i in fused],
                                          batch_size=args.reranker_batch,
                                          show_progress_bar=False)
                scored = {i:float(v) for i,v in zip(fused, np.asarray(scores).reshape(-1))}
                final = sorted(fused,key=lambda i:(-scored[i],i))
                rank_method = 'cross_encoder'
            else:
                scored = {i:None for i in fused}
                final = fused
                rank_method = 'rrf_tfidf_demo'
            for rank, i in enumerate(final, start=1):
                s = sentences[i]
                neighbors = sentences[max(0,i-1):min(len(sentences),i+2)]
                candidates.append({
                    'candidate_id':f'{claim["claim_id"]}__{s["sentence_id"]}',
                    'claim_id':claim['claim_id'], 'doc_id':docid,
                    'sentence_id':s['sentence_id'], 'sentence_index':i,
                    'sentence_text':s['text'],
                    'context_previous':sentences[i-1]['text'] if i>0 else '',
                    'context_next':sentences[i+1]['text'] if i+1<len(sentences) else '',
                    'rank_final':rank,'is_top_k':rank<=args.top_k,
                    'rank_dense':dense_rank[i], 'rank_bm25':bm25_rank[i],
                    'score_cosine':clean_number(dense[i]),
                    'score_bm25':clean_number(lexical[i]),
                    'score_rrf':clean_number(rrf([dense_rank,bm25_rank],i)),
                    'score_cross_encoder_logit':clean_number(scored[i]),
                    'ranking_method':rank_method,
                    'candidate_pool_methods':[
                        name for name, order in [('dense',full_dense[:args.dense_pool]),('bm25',full_bm25[:args.bm25_pool])]
                        if i in order
                    ]
                })
        print(f'  {docid}: {len(sentences)} sentenças, {len(batch_claims)} claims', flush=True)
    sentence_rows = [s for d in docs.values() for s in d['sentences']]
    document_rows = [{k:v for k,v in d.items() if k!='sentences'} for d in docs.values()]
    write_jsonl(output/'documentos.jsonl',document_rows)
    write_jsonl(output/'sentencas.jsonl',sentence_rows)
    write_jsonl(output/'claims.jsonl',claims)
    write_jsonl(output/'candidatos.jsonl',candidates)
    data_fingerprint = hash_string('\n'.join(
        f'{d["doc_id"]}:{d["source_sha256"]}' for d in document_rows
    )+'\n'+'\n'.join(f'{c["claim_id"]}:{c["claim_text"]}' for c in claims))
    manifest = {
        'schema_version':1, 'dataset_id':data_fingerprint, 'created_utc':datetime.now(timezone.utc).isoformat(),
        'input_file':str(Path(args.entrada).resolve()), 'top_k':args.top_k,
        'number_documents':len(document_rows),'number_claims':len(claims),
        'number_candidates':len(candidates),'retrieval_backend':args.backend,
        'embedding_model':args.embedding_model if embedder else 'tfidf_smoke_test_only',
        'reranker_model':args.reranker_model if reranker else 'none_smoke_test_only',
        'selection':'BM25 and cosine union > RRF shortlist > CrossEncoder rerank > top_k display',
        'score_note':'Cosine is bounded (-1..1). CE returns a logit, not a calibrated probability.',
        'gold_created':False
    }
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Arquivos brutos gravados em',output.resolve(), flush=True)
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entrada',required=True, help='Parquet do extrator V7 (ou fixture JSON/JSONL)')
    p.add_argument('--saida',default='saida/brutos')
    p.add_argument('--variantes',default='',help='Ex.: original,entity,predicate,coreference; vazio=todas')
    p.add_argument('--campo-claim',choices=['claim_original','claim_resolvida','texto'],default='claim_original')
    p.add_argument('--top-k',type=int,choices=[3,5],default=5)
    p.add_argument('--dense-pool',type=int,default=20)
    p.add_argument('--bm25-pool',type=int,default=20)
    p.add_argument('--rerank-pool',type=int,default=35)
    p.add_argument('--max-registros',type=int,default=0)
    p.add_argument('--max-claims',type=int,default=0)
    p.add_argument('--backend',choices=['hf','tfidf'],default='hf')
    p.add_argument('--embedding-model',default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--reranker-model',default='cross-encoder/ms-marco-MiniLM-L6-v2')
    p.add_argument('--device',default='cuda')
    p.add_argument('--embedding-batch',type=int,default=32)
    p.add_argument('--reranker-batch',type=int,default=16)
    return p


