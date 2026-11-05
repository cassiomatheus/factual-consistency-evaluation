"""Adapted from pipeline/01_extract_claims_v7_melhorado_novo.py; scientific rules retained. See docs/migration.md."""
import gc
import os
import re
import torch
import pandas as pd
import spacy
from tqdm import tqdm
from .model import ClaimExtractor
from factual_eval.schemas import read_injected_csv


INICIO = 0

FIM = 5

TIPOS_AVALIACAO = {
    "original": "resumo_original",
    "entity": "resumo_entity",
    "predicate": "resumo_predicado",
    "coreference": "resumo_corref",
}

PRONOMES_COREFTIVOS = {
    "it", "he", "she", "they", "him", "her", "them",
    "this", "that", "these", "those",
}

ENTITY_LABELS = {
    "ORG", "PERSON", "GPE", "LOC", "FAC",
    "LAW", "NORP", "PRODUCT", "EVENT",
}

def normalize_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_candidate_antecedents(previous_sentences, max_candidates=5):
    """
    Gera candidatos de antecedente a partir das duas sentenças anteriores.

    Importante:
    esta função NÃO declara que o candidato é o antecedente correto.
    Ela apenas preserva hipóteses para as etapas posteriores.
    """
    candidates = []
    seen = set()

    for sentence in reversed(previous_sentences):
        if not sentence:
            continue

        doc = nlp(sentence)

        # Entidades nomeadas primeiro.
        for ent in doc.ents:
            if ent.label_ in ENTITY_LABELS:
                value = normalize_text(ent.text)
                key = value.lower()

                if key not in seen:
                    seen.add(key)
                    candidates.append({
                        "text": value,
                        "source": "named_entity",
                        "score_prior": 0.90 if not candidates else 0.75,
                    })

        # Depois noun chunks.
        for chunk in doc.noun_chunks:
            value = normalize_text(chunk.text)

            if not value:
                continue

            # Evita chunks puramente pronominais.
            if value.lower() in PRONOMES_COREFTIVOS:
                continue

            key = value.lower()

            if key not in seen:
                seen.add(key)
                candidates.append({
                    "text": value,
                    "source": "noun_chunk",
                    "score_prior": 0.55,
                })

        if len(candidates) >= max_candidates:
            break

    return candidates[:max_candidates]

def detect_pronoun_claim(claim):
    """
    Detecta pronomes potencialmente dependentes de contexto.
    Não tenta resolver a correferência sozinho.
    """
    doc = nlp(claim)

    found = []

    for token in doc:
        if token.text.lower() in PRONOMES_COREFTIVOS:
            if token.dep_ in {
                "nsubj", "nsubjpass", "dobj",
                "pobj", "attr", "appos"
            }:
                found.append(token.text.lower())

    return found

def build_context(sentences, idx, radius_before=2, radius_after=1):
    start = max(0, idx - radius_before)
    end = min(len(sentences), idx + radius_after + 1)

    return " ".join(sentences[start:end])

def remove_error_marker(sentence, pattern):
    match = re.search(pattern, sentence)

    if not match:
        return sentence, {
            "gabarito": "none",
            "palavra": "",
        }

    tipo_erro = f"{match.group(1).lower()}_error"
    palavra = normalize_text(match.group(2))

    clean = re.sub(pattern, "", sentence)
    clean = normalize_text(clean)

    return clean, {
        "gabarito": tipo_erro,
        "palavra": palavra,
    }

def _execute():

    print(f"Lendo base: {ARQUIVO_ENTRADA_CSV}")

    df_fonte = read_injected_csv(ARQUIVO_ENTRADA_CSV)
    df_lote = df_fonte.iloc[INICIO:FIM]

    print("Inicializando ClaimExtractor...")
    extrator = ClaimExtractor(**MODEL_OPTIONS)

    dados_processados = []

    padrao_mascara = r"\[ERROR_(ENTITY|PREDICATE|COREFERENCE):\s*(.*?)\]"

    total_claims = 0
    total_injected_claims = 0
    total_coref_candidates = 0

    for row_idx, linha in tqdm(
        df_lote.iterrows(),
        total=len(df_lote),
        desc=f"Extração estruturada ({INICIO}-{FIM})",
    ):

        fonte = str(linha["fonte"])

        for tipo_nome, coluna_resumo in TIPOS_AVALIACAO.items():

            valor_resumo = linha[coluna_resumo]

            if pd.isna(valor_resumo):
                continue

            resumo_texto = normalize_text(valor_resumo)

            if not resumo_texto:
                continue

            doc = nlp(resumo_texto)

            sentencas_brutas = [
                normalize_text(sent.text)
                for sent in doc.sents
                if len(normalize_text(sent.text)) > 10
            ]

            sentencas_limpas = []
            metadados = []

            for sent in sentencas_brutas:
                clean, metadata = remove_error_marker(
                    sent,
                    padrao_mascara,
                )

                sentencas_limpas.append(clean)
                metadados.append(metadata)

            claims_extraidos = []

            if sentencas_limpas:

                lista_claims = extrator.process_batch(
                    sentencas_limpas
                )

                for idx_sentenca, sublista in enumerate(lista_claims):

                    current = sentencas_limpas[idx_sentenca]

                    previous = sentencas_limpas[
                        max(0, idx_sentenca - 2):idx_sentenca
                    ]

                    next_sentence = (
                        sentencas_limpas[idx_sentenca + 1]
                        if idx_sentenca + 1 < len(sentencas_limpas)
                        else ""
                    )

                    local_context = build_context(
                        sentencas_limpas,
                        idx_sentenca,
                    )

                    antecedent_candidates = (
                        extract_candidate_antecedents(previous)
                    )

                    for claim in sublista:

                        claim = normalize_text(claim)

                        if not claim:
                            continue

                        gabarito = "none"

                        palavra_injetada = metadados[
                            idx_sentenca
                        ]["palavra"]

                        if (
                            metadados[idx_sentenca]["gabarito"] != "none"
                            and palavra_injetada
                            and palavra_injetada.lower() in claim.lower()
                        ):
                            gabarito = metadados[
                                idx_sentenca
                            ]["gabarito"]

                        pronouns = detect_pronoun_claim(claim)

                        has_coref = len(pronouns) > 0
                        
                        # Restaura os contadores que haviam sido apagados
                        if has_coref:
                            total_coref_candidates += len(antecedent_candidates)

                        if gabarito != "none":
                            total_injected_claims += 1

                        total_claims += 1
                        
                        # --- CÓDIGO NOVO PARA CORREFERÊNCIA ---
                        # Vamos gerar versões da claim substituindo os pronomes pelos candidatos
                        claim_variants = [{"text": claim, "type": "original"}]

                        if has_coref:
                            doc_claim = nlp(claim)
                            for token in doc_claim:
                                if token.text.lower() in PRONOMES_COREFTIVOS:
                                    for candidate in antecedent_candidates:
                                        # Substitui fisicamente o pronome pelo texto do candidato
                                        nova_claim = claim[:token.idx] + candidate["text"] + claim[token.idx + len(token.text):]
                                        claim_variants.append({
                                            "text": nova_claim,
                                            "type": "coref_resolved",
                                            "antecedent_used": candidate["text"]
                                        })
                        
                        claims_extraidos.append({
                            "texto": claim,
                            "claim_original": claim,
                            "sentence_index": idx_sentenca,
                            "sentence_source": current,
                            "previous_sentence": " ".join(previous),
                            "next_sentence": next_sentence,
                            "local_context": local_context,
                            "has_possible_coreference": has_coref,
                            "pronouns": pronouns,
                            "antecedent_candidates": antecedent_candidates,
                            "claim_variants": claim_variants,
                            "gabarito_injetado": gabarito,
                            "palavra_injetada": palavra_injetada if gabarito != "none" else "",
                        })

            dados_processados.append({
                "doc_idx": row_idx,
                "tipo_resumo": tipo_nome,
                "documento_fonte": fonte,
                "resumo_texto": resumo_texto,
                "claims": claims_extraidos,
            })

    df_saida = pd.DataFrame(dados_processados)

    os.makedirs(
        os.path.dirname(ARQUIVO_SAIDA_PARQUET),
        exist_ok=True,
    )

    df_saida.to_parquet(
        ARQUIVO_SAIDA_PARQUET,
        engine="pyarrow",
        index=False,
    )

    print("\n" + "=" * 80)
    print("ETAPA 1 CONCLUÍDA")
    print("=" * 80)
    print(f"Claims extraídas: {total_claims}")
    print(f"Claims injetadas identificadas: {total_injected_claims}")
    print(
        "Hipóteses de correferência preservadas: "
        f"{total_coref_candidates}"
    )
    print(f"Arquivo: {ARQUIVO_SAIDA_PARQUET}")

    del extrator
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(input_path, output_path, config, start=0, num_docs=None):
    """Execute one extraction run; module model state is process-local."""
    global ARQUIVO_ENTRADA_CSV, ARQUIVO_SAIDA_PARQUET, INICIO, FIM, MODEL_OPTIONS, nlp
    from factual_eval.config import device_for
    from factual_eval.schemas import read_injected_csv, add_extraction_ids
    from factual_eval.provenance import record_model
    from factual_eval.schemas import stage_paths
    input_path,output_path=stage_paths(input_path,output_path)
    settings = config['claims']
    ARQUIVO_ENTRADA_CSV, ARQUIVO_SAIDA_PARQUET = str(input_path), str(output_path)
    INICIO, FIM = start, None if num_docs is None else start + num_docs
    MODEL_OPTIONS = {key: settings[key] for key in ('model_name','revision','batch_size','max_input_length','max_output_length','num_beams','early_stopping')}
    MODEL_OPTIONS['device'] = device_for(config, 'claims')
    nlp = spacy.load(settings['spacy_model'])
    record_model('claims.segmenter', settings['spacy_model'], nlp)
    # The original range is positional; retain source_index for generated subsets.
    _execute()
    add_extraction_ids(input_path, output_path, start, num_docs)
    return output_path
