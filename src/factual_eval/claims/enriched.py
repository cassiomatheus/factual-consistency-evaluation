"""Adapted from 01_extract_claims_v7_melhorado.py; scientific rules retained. See docs/migration.md."""
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
    "coreference": "resumo_corref"
}

COREF_CONTEXT_SENTENCES = 3

PRONOMES_SUJEITO = {
    "it", "he", "she", "they", "this", "that", "these", "those"
}

COREF_ENTITY_LABELS = {
    "ORG", "PERSON", "GPE", "LOC", "LAW", "NORP", "FAC",
    "PRODUCT", "EVENT", "WORK_OF_ART"
}

def normalizar_texto(texto):
    texto = str(texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def unique_preservando_ordem(itens):
    vistos = set()
    saida = []
    for item in itens:
        chave = normalizar_texto(item).lower()
        if chave and chave not in vistos:
            vistos.add(chave)
            saida.append(item)
    return saida

def extrair_entidades_recentes(sentencas_contexto):
    """
    Extrai entidades em ordem de aparecimento.
    Mantém todas as candidatas para que a expansão não dependa
    exclusivamente da última entidade encontrada.
    """
    candidatos = []

    for sent in sentencas_contexto:
        doc_sent = nlp(sent)

        for ent in doc_sent.ents:
            if ent.label_ in COREF_ENTITY_LABELS:
                candidatos.append(ent.text.strip())

    return unique_preservando_ordem(candidatos)

def resolver_pronome_controladamente(
    claim,
    sentenca_origem,
    contexto_anterior,
    entidades_candidatas
):
    """
    Expansão conservadora de correferência.

    Estratégia:
    1. só tenta expandir pronomes sujeito;
    2. usa entidades encontradas no contexto anterior;
    3. prefere entidades que aparecem mais recentemente;
    4. não substitui se não houver uma candidata plausível.

    Retorna:
        claim_expandida,
        antecedente,
        confianca,
        motivo
    """
    doc_claim = nlp(claim)

    pronome_alvo = None
    for token in doc_claim:
        if (
            token.pos_ == "PRON"
            and token.dep_ in {"nsubj", "nsubjpass", "csubj"}
            and token.text.lower() in PRONOMES_SUJEITO
        ):
            pronome_alvo = token.text
            break

    if pronome_alvo is None:
        return claim, "", 0.0, "no_subject_pronoun"

    if not entidades_candidatas:
        return claim, "", 0.0, "no_candidate_antecedent"

    # Últimas entidades têm maior prioridade, mas não confiamos apenas em uma.
    candidatos_recentes = list(reversed(entidades_candidatas))

    antecedente = candidatos_recentes[0]

    # Confiança heurística:
    # 0.90 se o antecedente aparece na sentença imediatamente anterior;
    # 0.75 se está nas duas anteriores;
    # 0.60 se está no restante da janela.
    confianca = 0.60

    contexto_normalizado = " ".join(contexto_anterior).lower()

    if contexto_anterior:
        ultima = contexto_anterior[-1].lower()
        if antecedente.lower() in ultima:
            confianca = 0.90
        elif antecedente.lower() in contexto_normalizado:
            confianca = 0.75

    claim_expandida = re.sub(
        rf"\b{re.escape(pronome_alvo)}\b",
        antecedente,
        claim,
        count=1,
        flags=re.IGNORECASE
    )

    if claim_expandida == claim:
        return claim, "", 0.0, "replacement_failed"

    return (
        normalizar_texto(claim_expandida),
        antecedente,
        confianca,
        "controlled_subject_pronoun_expansion"
    )

def extrair_relacao_argumental(sentenca_texto, palavra_alvo):
    doc = nlp(sentenca_texto)
    alvo = normalizar_texto(palavra_alvo).lower()

    if not alvo:
        return None, None, None

    token_alvo = next(
        (
            token for token in doc
            if token.text.lower().strip() == alvo
            or token.lemma_.lower().strip() == alvo
        ),
        None
    )

    if not token_alvo:
        return None, None, None

    verbo = token_alvo if token_alvo.pos_ == "VERB" else token_alvo.head

    if verbo.pos_ != "VERB" and verbo.head.pos_ == "VERB":
        verbo = verbo.head

    elif verbo.pos_ != "VERB":
        verbo = next(
            (
                t for t in doc
                if t.dep_ == "ROOT" and t.pos_ == "VERB"
            ),
            verbo
        )

    if verbo.pos_ != "VERB":
        return None, None, None

    sujeito = next(
        (
            w for w in verbo.children
            if w.dep_ in ("nsubj", "nsubjpass", "csubj")
        ),
        None
    )

    objeto = next(
        (
            w for w in verbo.children
            if w.dep_ in (
                "dobj", "pobj", "acomp",
                "xcomp", "ccomp", "attr", "obj"
            )
        ),
        None
    )

    if not objeto:
        for prep in (
            w for w in verbo.children
            if w.dep_ == "prep"
        ):
            pobjs = [
                w for w in prep.children
                if w.dep_ == "pobj"
            ]
            if pobjs:
                objeto = pobjs[0]
                break

    return sujeito, verbo, objeto

def auditar_dependencia_na_fonte(documento_fonte, sujeito, verbo, objeto):
    """
    Auditoria estrutural conservadora.
    Retorna True apenas quando encontra sujeito + verbo + objeto
    na mesma relação sintática básica.
    """
    doc_fonte = nlp(documento_fonte)

    s_lemma = sujeito.lemma_.lower()
    v_lemma = verbo.lemma_.lower()
    o_lemma = objeto.lemma_.lower()

    for sent in doc_fonte.sents:
        for token in sent:
            if (
                token.lemma_.lower() == v_lemma
                and token.pos_ == "VERB"
            ):
                has_s = any(
                    child.lemma_.lower() == s_lemma
                    and child.dep_ in (
                        "nsubj", "nsubjpass", "csubj"
                    )
                    for child in token.children
                )

                has_o = any(
                    child.lemma_.lower() == o_lemma
                    and child.dep_ in (
                        "dobj", "pobj", "acomp",
                        "xcomp", "ccomp", "attr", "obj"
                    )
                    for child in token.children
                )

                if not has_o:
                    for child in token.children:
                        if child.dep_ == "prep":
                            if any(
                                pc.lemma_.lower() == o_lemma
                                and pc.dep_ == "pobj"
                                for pc in child.children
                            ):
                                has_o = True
                                break

                if has_s and has_o:
                    return True

    return False

def classificar_perturbacao(
    documento_fonte,
    claim_injetada,
    palavra_injetada
):
    if not palavra_injetada:
        return "UNVERIFIABLE"

    if "FUTURE_" in palavra_injetada.upper():
        return "UNVERIFIABLE"

    sujeito, verbo, objeto = extrair_relacao_argumental(
        claim_injetada,
        palavra_injetada
    )

    if not (sujeito and verbo and objeto):
        return "UNVERIFIABLE"

    if auditar_dependencia_na_fonte(
        documento_fonte,
        sujeito,
        verbo,
        objeto
    ):
        return "SUPPORTED_BY_SOURCE"

    return "NOT_VERIFIED_IN_SOURCE"

def _execute():

    print(f"Lendo base de dados: {ARQUIVO_ENTRADA_CSV}")
    df_fonte = read_injected_csv(ARQUIVO_ENTRADA_CSV)

    df_lote = df_fonte.iloc[INICIO:FIM]

    print("Iniciando Módulo de Extração (T5) na GPU...")
    extrator = ClaimExtractor(**MODEL_OPTIONS)

    dados_processados = []

    total_candidate = 0
    total_supported = 0
    total_unver = 0

    for row_idx, linha in tqdm(
        df_lote.iterrows(),
        total=len(df_lote),
        desc=f"Pipeline Fase I ({INICIO}-{FIM})"
    ):

        fonte = str(linha["fonte"])

        for tipo_nome, coluna_resumo in TIPOS_AVALIACAO.items():

            valor_resumo = linha[coluna_resumo]

            if pd.isna(valor_resumo):
                continue

            resumo_texto = normalizar_texto(valor_resumo)

            if not resumo_texto:
                continue

            # ---------------------------------------------------------
            # 1. SEGMENTAÇÃO
            # ---------------------------------------------------------
            doc = nlp(resumo_texto)

            sentencas_brutas = [
                sent.text.strip()
                for sent in doc.sents
                if len(sent.text.strip()) > 10
            ]

            if not sentencas_brutas:
                continue

            sentencas_limpas_para_t5 = []
            metadados_das_sentencas = []

            padrao_mascara = r"\[ERROR_(ENTITY|PREDICATE|COREFERENCE):\s*(.*?)\]"

            # ---------------------------------------------------------
            # 2. LIMPEZA + MEMÓRIA DE ENTIDADES
            # ---------------------------------------------------------
            historico_entidades = []

            for idx_sent, sent in enumerate(sentencas_brutas):

                match = re.search(
                    padrao_mascara,
                    sent
                )

                if match:
                    sent_limpa = re.sub(
                        padrao_mascara,
                        "",
                        sent
                    )
                else:
                    sent_limpa = sent

                sent_limpa = normalizar_texto(
                    sent_limpa.replace("  ", " ")
                )

                # Apenas entidades da sentença atual entram na memória.
                doc_sent = nlp(sent_limpa)

                entidades_sentenca = [
                    ent.text.strip()
                    for ent in doc_sent.ents
                    if ent.label_ in COREF_ENTITY_LABELS
                ]

                entidades_sentenca = unique_preservando_ordem(
                    entidades_sentenca
                )

                if entidades_sentenca:
                    historico_entidades.extend(
                        entidades_sentenca
                    )

                historico_entidades = unique_preservando_ordem(
                    historico_entidades
                )[-10:]

                # Contexto anterior usado posteriormente.
                inicio_ctx = max(
                    0,
                    idx_sent - COREF_CONTEXT_SENTENCES
                )

                contexto_anterior = sentencas_brutas[
                    inicio_ctx:idx_sent
                ]

                entidades_candidatas = extrair_entidades_recentes(
                    contexto_anterior
                )

                # Para erros de correferência, o marcador pode conter
                # o alvo alterado; mantemos a âncora separadamente.
                ancora_memoria = (
                    entidades_candidatas[-1]
                    if entidades_candidatas
                    else (
                        historico_entidades[-1]
                        if historico_entidades
                        else ""
                    )
                )

                if match:
                    gabarito = f"{match.group(1).lower()}_error"
                    palavra = match.group(2).strip()
                else:
                    gabarito = "none"
                    palavra = ""

                metadados_das_sentencas.append({
                    "gabarito_original": gabarito,
                    "palavra_injetada": palavra,
                    "ancora_memoria": ancora_memoria,
                    "entidades_candidatas": entidades_candidatas,
                    "contexto_anterior": contexto_anterior
                })

                sentencas_limpas_para_t5.append(
                    sent_limpa
                )

            # ---------------------------------------------------------
            # 3. EXTRAÇÃO T5
            # ---------------------------------------------------------
            claims_extraidos = []

            lista_de_listas = extrator.process_batch(
                sentencas_limpas_para_t5
            )

            for idx_sentenca, sublista_claims in enumerate(
                lista_de_listas
            ):

                metadado = metadados_das_sentencas[
                    idx_sentenca
                ]

                sentenca_origem = sentencas_limpas_para_t5[
                    idx_sentenca
                ]

                inicio_ctx = max(
                    0,
                    idx_sentenca - COREF_CONTEXT_SENTENCES
                )

                fim_ctx = min(
                    len(sentencas_limpas_para_t5),
                    idx_sentenca + 2
                )

                contexto_local = sentencas_limpas_para_t5[
                    inicio_ctx:fim_ctx
                ]

                for claim in sublista_claims:

                    claim_original = normalizar_texto(claim)

                    if not claim_original:
                        continue

                    # -------------------------------------------------
                    # 4. EXPANSÃO CONTROLADA DE CORREFERÊNCIA
                    # -------------------------------------------------
                    (
                        claim_expandida,
                        antecedente,
                        coref_confidence,
                        coref_reason
                    ) = resolver_pronome_controladamente(
                        claim_original,
                        sentenca_origem,
                        metadado["contexto_anterior"],
                        metadado["entidades_candidatas"]
                    )

                    # -------------------------------------------------
                    # 5. SOBREVIVÊNCIA DO ERRO
                    #
                    # Testamos tanto a claim original quanto a
                    # expandida, evitando perder o marcador.
                    # -------------------------------------------------
                    gabarito_final = "none"

                    if metadado["gabarito_original"] != "none":

                        palavra_lower = metadado[
                            "palavra_injetada"
                        ].lower()

                        if (
                            palavra_lower in
                            claim_original.lower()
                        ):
                            gabarito_final = metadado[
                                "gabarito_original"
                            ]

                        elif (
                            palavra_lower in
                            claim_expandida.lower()
                        ):
                            gabarito_final = metadado[
                                "gabarito_original"
                            ]

                    palavra_injetada = (
                        metadado["palavra_injetada"]
                        if gabarito_final != "none"
                        else ""
                    )

                    status_auditoria = "NONE"

                    # -------------------------------------------------
                    # 6. AUDITORIA S-V-O
                    # -------------------------------------------------
                    if gabarito_final != "none":

                        status_auditoria = classificar_perturbacao(
                            fonte,
                            claim_expandida,
                            palavra_injetada
                        )

                        if (
                            status_auditoria ==
                            "SUPPORTED_BY_SOURCE"
                        ):
                            gabarito_final = "supported_by_source"
                            total_supported += 1

                        elif (
                            status_auditoria ==
                            "UNVERIFIABLE"
                        ):
                            gabarito_final = "unverifiable"
                            total_unver += 1

                        else:
                            total_candidate += 1

                    # -------------------------------------------------
                    # 7. CONTEXTO FUTURO
                    # -------------------------------------------------
                    contexto_posterior = sentencas_limpas_para_t5[
                        idx_sentenca + 1:
                        min(
                            len(sentencas_limpas_para_t5),
                            idx_sentenca + 3
                        )
                    ]

                    # -------------------------------------------------
                    # 8. SAÍDA
                    #
                    # "texto" continua sendo a claim expandida para
                    # manter o retriever atual compatível.
                    # A claim original nunca é perdida.
                    # -------------------------------------------------
                    claims_extraidos.append({
                        "texto": claim_expandida,
                        "claim_original": claim_original,
                        "claim_resolvida": claim_expandida,
                        "sentenca_origem": sentenca_origem,
                        "indice_sentenca": idx_sentenca,

                        "contexto_anterior": (
                            metadado["contexto_anterior"]
                        ),
                        "contexto_local": contexto_local,
                        "contexto_posterior": contexto_posterior,

                        "antecedente_coref": antecedente,
                        "coref_confidence": float(
                            coref_confidence
                        ),
                        "coref_reason": coref_reason,

                        "entidades_candidatas": (
                            metadado["entidades_candidatas"]
                        ),

                        "gabarito_injetado": gabarito_final,
                        "palavra_injetada": palavra_injetada,
                        "status_auditoria": status_auditoria
                    })

            dados_processados.append({
                "doc_idx": row_idx,
                "tipo_resumo": tipo_nome,
                "documento_fonte": fonte,
                "resumo_texto": resumo_texto,
                "claims": claims_extraidos
            })

    # =================================================================
    # SALVAMENTO
    # =================================================================
    print("\nSalvando claims enriquecidas...")

    df_saida = pd.DataFrame(
        dados_processados
    )

    df_saida.to_parquet(
        ARQUIVO_SAIDA_PARQUET,
        engine="pyarrow",
        index=False
    )

    print("\n" + "=" * 80)
    print("📊 RESULTADO DO FILTRO E EXPANSÃO DE CORREFERÊNCIA")
    print("=" * 80)
    print(
        f"✅ Candidatos a Erro: {total_candidate}"
    )
    print(
        f"⚠️ Aprovados pela Fonte: {total_supported}"
    )
    print(
        f"❌ Não Verificáveis: {total_unver}"
    )
    print("=" * 80)

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
