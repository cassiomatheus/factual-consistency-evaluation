"""Adapted from pipeline/02_retrieve_evidence_v7_melhorado_novo.py; scientific rules retained. See docs/migration.md."""
import gc
import os
import re
import torch
import pandas as pd
import spacy
from tqdm import tqdm
import math
import numpy as np
from collections import Counter
from sentence_transformers import CrossEncoder, SentenceTransformer, util


SENT_TOP_K = 8

PAR_TOP_K = 4

SEMANTIC_POOL = 30

LEXICAL_POOL = 30

CONTEXT_RADIUS = 2

MAX_COREF_VARIANTS = 4

USE_MMR = False

MMR_LAMBDA = 0.85

MAX_WINDOW_OVERLAP = 0.8

CONTEXT_RERANK_WEIGHT = 0.30

W_SEMANTIC = 0.60

W_LEXICAL = 0.25

W_ENTITY = 0.15

ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "LAW", "DATE", "CARDINAL",
    "MONEY", "PERCENT", "QUANTITY",
}

STOP_WORDS = set("a an the of to in on at for from by with and or is are was were be been being this that these those it its as".split())

def clean_text(value):
    return value.strip() if isinstance(value, str) else ""

def normalize(text):
    return re.sub(r"\s+", " ", clean_text(text).lower()).strip()

def token_list(text):
    return [t for t in re.findall(r"\b\w+(?:[.,]\d+)*\b", normalize(text))
            if t not in STOP_WORDS]

def content_tokens(text):
    return set(token_list(text))

def lexical_overlap(query, passage):
    q = content_tokens(query)
    return len(q & content_tokens(passage)) / len(q) if q else 0.0

def entity_set(doc):
    return {normalize(e.text) for e in doc.ents if e.label_ in ENTITY_LABELS}

class BM25Index:
    """Índice lexical independente, construído uma vez por documento/unidade."""
    def __init__(self, passages, k1=1.5, b=0.75):
        self.counts = [Counter(token_list(p)) for p in passages]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_length = sum(self.lengths) / max(len(passages), 1) or 1.0
        self.df = Counter(t for c in self.counts for t in c)
        self.k1, self.b = k1, b

    def candidates(self, query, top_k):
        scores = []
        query_tokens = content_tokens(query)
        for idx, counts in enumerate(self.counts):
            score = 0.0
            for token in query_tokens:
                tf = counts.get(token, 0)
                if not tf:
                    continue
                idf = math.log(1 + (len(self.counts) - self.df[token] + 0.5) / (self.df[token] + 0.5))
                norm = self.k1 * (1 - self.b + self.b * self.lengths[idx] / self.avg_length)
                score += idf * tf * (self.k1 + 1) / (tf + norm)
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:top_k]

def build_window(sentences, idx, radius=CONTEXT_RADIUS):
    return " ".join(sentences[max(0, idx-radius):min(len(sentences), idx+radius+1)])

def contextual_query(claim_dict):
    pieces = [clean_text(claim_dict.get("claim_original")) or clean_text(claim_dict.get("texto"))]
    pieces += [clean_text(claim_dict.get(k)) for k in
               ("previous_sentence", "sentence_source", "next_sentence")]
    return " ".join(dict.fromkeys(p for p in pieces if p))

def entity_query(doc, claim):
    # Preserva a ordem original e torna a consulta reproduzível.
    tokens = [t.lemma_.lower() for t in doc
              if t.dep_ in {"nsubj", "nsubjpass", "dobj", "pobj"} or t.ent_type_]
    query = " ".join(dict.fromkeys(tokens))
    return query if len(query.split()) >= 2 else claim

def build_queries(claim_dict, claim_doc):
    claim = clean_text(claim_dict.get("claim_original")) or clean_text(claim_dict.get("texto"))
    queries = [("literal", claim), ("contextual", contextual_query(claim_dict)),
               ("entities", entity_query(claim_doc, claim))]
    variants = claim_dict.get("claim_variants")
    if variants is None:
        variants = []
    used = 0
    seen = {normalize(q) for _, q in queries}
    for variant in variants:
        if not isinstance(variant, dict) or variant.get("type") != "coref_resolved":
            continue
        text = clean_text(variant.get("text"))
        if text and normalize(text) not in seen:
            queries.append((f"coref_{used}", text))
            seen.add(normalize(text))
            used += 1
            if used >= MAX_COREF_VARIANTS:
                break
    return [(source, q) for source, q in queries if q]

def semantic_candidates(query, embeddings, top_k):
    if not len(embeddings["doc"]):
        return [], []
    q_emb = embeddings["model"].encode([query], convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False)
    scores = util.cos_sim(q_emb, embeddings["doc"])[0]
    # Ordem estável para empates, inclusive documentos curtos.
    indices = torch.argsort(scores, descending=True, stable=True)[:top_k]
    return indices.tolist(), scores[indices].detach().cpu().tolist()

def retrieve_candidates(queries, passages, embeddings, lexical_index, passage_entities, claim_entities):
    origins, semantic, lexical_raw = {}, {}, {}
    for source, query in queries:
        ids, values = semantic_candidates(query, embeddings, SEMANTIC_POOL)
        for idx, value in zip(ids, values):
            origins.setdefault(idx, set()).add(f"semantic:{source}")
            semantic[idx] = max(semantic.get(idx, -1.0), value)
        for idx, value in lexical_index.candidates(query, LEXICAL_POOL):
            origins.setdefault(idx, set()).add(f"lexical:{source}")
            lexical_raw[idx] = max(lexical_raw.get(idx, 0.0), value)
    ranked = []
    for idx in sorted(origins):
        overlap = max(lexical_overlap(q, passages[idx]) for _, q in queries)
        entity = len(claim_entities & passage_entities[idx]) / len(claim_entities) if claim_entities else 0.0
        sem = semantic.get(idx, 0.0)
        ranked.append({
            "idx": idx, "semantic_score": sem, "lexical_score": overlap,
            "bm25_score": lexical_raw.get(idx, 0.0), "entity_score": entity,
            "position_score": 0.0,
            "retrieval_score": W_SEMANTIC * max(0.0, sem) + W_LEXICAL * overlap + W_ENTITY * entity,
            "retrieval_sources": sorted(origins[idx]),
        })
    # Não descarta canais lexicais/entidades antes do cross-encoder.
    return sorted(ranked, key=lambda x: (-x["retrieval_score"], x["idx"]))

def rerank_cross(claim, candidates, passages, cross_model, context_passages=None, variants=None):
    if not candidates:
        return []
    queries = list(dict.fromkeys([claim] + list(variants or [])))
    pairs, owners = [], []
    for pos, item in enumerate(candidates):
        idx = item["idx"]
        views = [("exact", passages[idx])]
        if context_passages is not None and context_passages[idx] != passages[idx]:
            views.append(("context", context_passages[idx]))
        for query_idx, query in enumerate(queries):
            for view, passage in views:
                pairs.append([query, passage])
                owners.append((pos, query_idx, view))
    # Força logits para que a escala não dependa da versão da biblioteca.
    scores = cross_model.predict(pairs, batch_size=32, show_progress_bar=False,
                                 activation_fct=torch.nn.Identity())
    # Combina as vistas para cada hipótese separadamente. O contexto ajuda a
    # interpretar a âncora, mas não empresta sozinho relevância às vizinhas.
    view_scores = [{} for _ in candidates]
    for (pos, query_idx, view), score in zip(owners, scores):
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("Cross-encoder produziu score não finito")
        view_scores[pos].setdefault(query_idx, {})[view] = score
    reranked = []
    for pos, candidate in enumerate(candidates):
        hypotheses = []
        for query_idx, views in view_scores[pos].items():
            exact = views["exact"]
            context = views.get("context", exact)
            combined = (1 - CONTEXT_RERANK_WEIGHT) * exact + CONTEXT_RERANK_WEIGHT * context
            hypotheses.append((combined, query_idx, exact, context))
        logit, query_idx, exact, context = max(hypotheses, key=lambda x: x[0])
        entry = dict(candidate, cross_logit=logit, exact_cross_logit=exact,
                     context_cross_logit=context, rerank_query_index=query_idx)
        # Sigmoid é score de relevância limitado, não probabilidade factual.
        entry["cross_score"] = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
        reranked.append(entry)
    return sorted(reranked, key=lambda x: (-x["cross_logit"], x["idx"]))

def cosine_similarity_numpy(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)) @ (b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)).T

def mmr_select(ranked_candidates, sentence_embeddings, top_k, lambda_param=MMR_LAMBDA):
    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param deve estar entre 0 e 1")
    if not ranked_candidates or top_k <= 0:
        return []
    # Normaliza logits dentro do pool; não aplica um novo corte top-20.
    logits = np.array([x["cross_logit"] for x in ranked_candidates])
    relevance = (logits - logits.min()) / np.ptp(logits) if np.ptp(logits) else np.ones(len(logits))
    selected, remaining = [], list(range(len(ranked_candidates)))
    while remaining and len(selected) < top_k:
        def score(pos):
            if not selected:
                return relevance[pos]
            idx = ranked_candidates[pos]["idx"]
            chosen = [ranked_candidates[p]["idx"] for p in selected]
            redundancy = max(0.0, float(cosine_similarity_numpy(sentence_embeddings[idx:idx+1], sentence_embeddings[chosen]).max()))
            return lambda_param * relevance[pos] - (1 - lambda_param) * redundancy
        best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
    return [ranked_candidates[pos] for pos in selected]

def select_windows(ranked, bounds, top_k):
    if top_k <= 0:
        return []
    selected = []
    for item in ranked:
        start, end = bounds[item["idx"]]
        redundant = False
        for chosen in selected:
            a, b = bounds[chosen["idx"]]
            overlap = max(0, min(end, b) - max(start, a))
            if overlap / min(end-start, b-a) >= MAX_WINDOW_OVERLAP:
                redundant = True
                break
        if not redundant:
            selected.append(item)
        if len(selected) >= top_k:
            break
    return selected

def _execute():
    print(f"Lendo claims: {ARQUIVO_ENTRADA}")
    df = pd.read_parquet(ARQUIVO_ENTRADA)
    # Segmentação local: evita carregar modelos extras ao importar metric.utils.
    nlp = spacy.load(MODEL_OPTIONS["spacy_model"])
    device = MODEL_OPTIONS["device"]
    sbert_model = SentenceTransformer(MODEL_OPTIONS["embedding_model"], device=device, revision=MODEL_OPTIONS["embedding_revision"])
    cross_model = make_cross_encoder(MODEL_OPTIONS, device)
    record_model("retrieval.embedding", MODEL_OPTIONS["embedding_model"], sbert_model)
    record_model("retrieval.reranker", MODEL_OPTIONS["reranker_model"], cross_model)
    dados_recuperados = []
    for _, linha in tqdm(df.iterrows(), total=len(df), desc="Retrieval híbrido"):
        full_doc = clean_text(linha["documento_fonte"])
        claims = linha["claims"]
        if claims is None or (isinstance(claims, float) and math.isnan(claims)):
            claims = []
        source_doc = nlp(full_doc)
        spans = [s for s in source_doc.sents if s.text.strip()]
        sentences = [s.text for s in spans]
        if not sentences:
            continue
        sentence_entities = [entity_set(s) for s in spans]
        expanded = [build_window(sentences, i, radius=CONTEXT_RADIUS) for i in range(len(sentences))]
        # Janelas independentes, deduplicadas pelos limites na fonte.
        bounds = list(dict.fromkeys((max(0, i-CONTEXT_RADIUS), min(len(sentences), i+CONTEXT_RADIUS+1)) for i in range(len(sentences))))
        windows = [" ".join(sentences[a:b]) for a, b in bounds]
        window_entities = [set().union(*sentence_entities[a:b]) for a, b in bounds]
        sent_embs = sbert_model.encode(sentences, convert_to_tensor=True,
            normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        window_embs = sbert_model.encode(windows, convert_to_tensor=True,
            normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        sent_bm25, window_bm25 = BM25Index(sentences), BM25Index(windows)
        sent_numpy = sent_embs.detach().cpu().numpy() if USE_MMR else None
        for claim_dict in claims:
            if not isinstance(claim_dict, dict):
                continue
            claim = clean_text(claim_dict.get("claim_original")) or clean_text(claim_dict.get("texto"))
            if not claim:
                continue
            claim_doc = nlp(claim)
            queries = build_queries(claim_dict, claim_doc)
            variants = [q for source, q in queries if source.startswith("coref_")]
            ranked_sent = retrieve_candidates(queries, sentences,
                {"model": sbert_model, "doc": sent_embs}, sent_bm25, sentence_entities, entity_set(claim_doc))
            ranked_window = retrieve_candidates(queries, windows,
                {"model": sbert_model, "doc": window_embs}, window_bm25, window_entities, entity_set(claim_doc))
            reranked_sent = rerank_cross(claim, ranked_sent, sentences, cross_model, expanded, variants)
            reranked_window = rerank_cross(claim, ranked_window, windows, cross_model, variants=variants)
            selected = mmr_select(reranked_sent, sent_numpy, SENT_TOP_K, lambda_param=MMR_LAMBDA) if USE_MMR else reranked_sent[:SENT_TOP_K]
            selected_windows = select_windows(reranked_window, bounds, PAR_TOP_K)
            top_sentences = []
            for item in selected:
                idx = item["idx"]
                entry = {k: v for k, v in item.items() if k != "idx"}
                entry.update(text=sentences[idx], expanded_text=expanded[idx], sentence_index=idx,
                    start_char=spans[idx].start_char, end_char=spans[idx].end_char)
                top_sentences.append(entry)
            top_paragraphs = []
            for item in selected_windows:
                idx = item["idx"]
                start, end = bounds[idx]
                entry = {k: v for k, v in item.items() if k != "idx"}
                entry.update(text=windows[idx], anchor_sentence_index=(start+end-1)//2,
                    anchor_score=item["cross_score"], start_sentence_index=start,
                    end_sentence_index=end, start_char=spans[start].start_char,
                    end_char=spans[end-1].end_char)
                top_paragraphs.append(entry)
            dados_recuperados.append({
                "doc_idx": linha["doc_idx"], "tipo_resumo": linha["tipo_resumo"],
                "claim": claim, "claim_original": claim,
                "gabarito_injetado": claim_dict.get("gabarito_injetado", "none"),
                "claim_context": {key: claim_dict.get(key, default) for key, default in {
                    "sentence_index": -1, "sentence_source": "", "previous_sentence": "",
                    "next_sentence": "", "local_context": "", "has_possible_coreference": False,
                    "pronouns": [], "antecedent_candidates": [],
                }.items()},
                "retrieval_queries": [{"source": source, "text": q} for source, q in queries],
                "retrieval_diagnostics": {
                    "sentence_candidates": len(ranked_sent), "window_candidates": len(ranked_window),
                    "selected_sentences": len(selected), "selected_windows": len(selected_windows),
                    "mmr_enabled": USE_MMR,
                },
                "top_sentences": top_sentences, "top_paragraphs": top_paragraphs,
                "hyp_ents_cache": [e.text.lower() for e in claim_doc.ents],
                "documento_fonte": full_doc,
            })
    os.makedirs(os.path.dirname(ARQUIVO_SAIDA), exist_ok=True)
    pd.DataFrame(dados_recuperados).to_parquet(ARQUIVO_SAIDA, engine="pyarrow", index=False)
    print(f"Etapa 2 concluída: {ARQUIVO_SAIDA}")
    del sbert_model, cross_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


from factual_eval.provenance import record_model


def make_cross_encoder(options, device):
    return CrossEncoder(options['reranker_model'], device=device, revision=options['reranker_revision'])


def run(input_path, output_path, config):
    global ARQUIVO_ENTRADA, ARQUIVO_SAIDA, MODEL_OPTIONS
    from factual_eval.config import device_for
    from factual_eval.schemas import validate_extraction, add_retrieval_ids
    from factual_eval.schemas import stage_paths
    input_path,output_path=stage_paths(input_path,output_path)
    validate_extraction(input_path)
    options = config['retrieval']
    mapping = {'sentence_top_k':'SENT_TOP_K', 'window_top_k':'PAR_TOP_K',
        'semantic_pool':'SEMANTIC_POOL', 'lexical_pool':'LEXICAL_POOL',
        'context_radius':'CONTEXT_RADIUS', 'max_coref_variants':'MAX_COREF_VARIANTS',
        'use_mmr':'USE_MMR', 'mmr_lambda':'MMR_LAMBDA',
        'max_window_overlap':'MAX_WINDOW_OVERLAP', 'context_rerank_weight':'CONTEXT_RERANK_WEIGHT',
        'semantic_weight':'W_SEMANTIC', 'lexical_weight':'W_LEXICAL', 'entity_weight':'W_ENTITY'}
    globals().update({target: options[key] for key, target in mapping.items()})
    MODEL_OPTIONS = dict(options, device=device_for(config, 'retrieval'))
    ARQUIVO_ENTRADA, ARQUIVO_SAIDA = str(input_path), str(output_path)
    _execute()
    add_retrieval_ids(input_path, output_path)
    return output_path
