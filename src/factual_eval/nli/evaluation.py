"""Adapted from pipeline/03_compute_nli_v7_melhorado_novo.py; scientific rules retained. See docs/migration.md."""
import gc
import os
import re
import torch
import pandas as pd
import spacy
from tqdm import tqdm
import math
import numpy as np
from .model import NLIAligner


SUPPORT_THRESHOLD = 0.64

CONTRADICTION_THRESHOLD = 0.62

CONTRADICTION_MARGIN = 0.10

SUPPORT_MARGIN = 0.10

STRUCTURAL_MIN_LEXICAL = 0.40

SUPPORT_EXACT_WEIGHT = 0.30
SUPPORT_EXPANDED_WEIGHT = 0.70
CONTRADICTION_EXACT_WEIGHT = 0.75
CONTRADICTION_EXPANDED_WEIGHT = 0.25
QUALITY_RETRIEVAL_WEIGHT = 0.60
QUALITY_CROSS_WEIGHT = 0.40
QUALITY_FLOOR = 0.80
QUALITY_SCALE = 0.20


def normalize(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).lower()
    ).strip()

def tokens(text):
    return set(
        t for t in re.findall(
            r"\b[a-z0-9]+\b",
            normalize(text)
        )
        if len(t) > 2
    )

def lexical_overlap(a, b):
    ta = tokens(a)
    tb = tokens(b)

    if not ta:
        return 0.0

    return len(ta & tb) / len(ta)

def content_lexical_overlap(a, b):
    """
    Mede sobreposição lexical removendo sinais estruturais
    que podem mudar em uma perturbação controlada.
    Isso reduz o risco de declarar conflito apenas porque
    uma entidade/número/data é diferente.
    """
    a = str(a)
    b = str(b)

    structural_patterns = [
        r"(?<!\w)\$?\d+(?:[.,]\d+)?%?(?!\w)",
        r"\b(?:19|20)\d{2}\b",
    ]

    for pattern in structural_patterns:
        a = re.sub(pattern, " ", a)
        b = re.sub(pattern, " ", b)

    a_tokens = tokens(a)
    b_tokens = tokens(b)

    if not a_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens)

def extract_numbers(text):
    return re.findall(
        r"(?<!\w)\$?\d+(?:[.,]\d+)?%?(?!\w)",
        str(text),
    )

def normalize_number(value):
    value = str(value)
    value = value.replace(",", "")
    value = value.replace("$", "")
    value = value.replace("%", "")
    return value.strip()

def extract_dates(text):
    """
    Extrai anos e formas de data simples.
    """
    years = re.findall(
        r"\b(?:19|20)\d{2}\b",
        str(text),
    )

    return set(years)

def extract_entities(text):
    doc = nlp(str(text))

    return [
        {
            "text": ent.text,
            "label": ent.label_,
        }
        for ent in doc.ents
        if ent.label_ in {
            "PERSON", "ORG", "GPE", "LOC",
            "FAC", "LAW", "PRODUCT",
        }
    ]

def extract_polarity(text):
    doc = nlp(str(text))

    neg_words = {
        "not", "no", "never",
        "neither", "without",
        "n't",
    }

    return any(
        token.text.lower() in neg_words
        for token in doc
    )

def extract_predicates(text):
    doc = nlp(str(text))

    return [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"VERB", "AUX"}
    ]

def detect_structural_conflicts(claim, evidence):
    """
    Detecta divergências factuais explícitas.

    Importante:
    esses sinais NÃO são tratados como prova isolada.
    Eles servem como evidência adicional contra o viés lexical.
    """

    conflicts = []

    claim_numbers = extract_numbers(claim)
    evidence_numbers = extract_numbers(evidence)

    # --------------------------------------------------------
    # NÚMEROS
    # --------------------------------------------------------

    if claim_numbers and evidence_numbers:

        cnums = {
            normalize_number(x)
            for x in claim_numbers
        }

        enums = {
            normalize_number(x)
            for x in evidence_numbers
        }

        if cnums != enums:

            conflicts.append({
                "type": "NUMBER",
                "claim_values": sorted(cnums),
                "evidence_values": sorted(enums),
            })

    # --------------------------------------------------------
    # ANOS
    # --------------------------------------------------------

    claim_years = extract_dates(claim)
    evidence_years = extract_dates(evidence)

    if claim_years and evidence_years:

        if claim_years != evidence_years:

            conflicts.append({
                "type": "DATE",
                "claim_values": sorted(claim_years),
                "evidence_values": sorted(evidence_years),
            })

    # --------------------------------------------------------
    # NEGACAO
    # --------------------------------------------------------

    if extract_polarity(claim) != extract_polarity(evidence):

        # Só trata como forte quando existe sobreposição
        # lexical suficiente para sugerir que falamos da
        # mesma proposição.
        if content_lexical_overlap(claim, evidence) >= STRUCTURAL_MIN_LEXICAL:

            conflicts.append({
                "type": "NEGATION"
            })

    # --------------------------------------------------------
    # ENTIDADES
    # --------------------------------------------------------

    claim_entities = extract_entities(claim)
    evidence_entities = extract_entities(evidence)

    for ce in claim_entities:

        same_type = [
            ee for ee in evidence_entities
            if ee["label"] == ce["label"]
        ]

        if not same_type:
            continue

        c_text = normalize(ce["text"])

        if not any(
            c_text == normalize(ee["text"])
            for ee in same_type
        ):

            # Para entidade, só considerar conflito quando o restante
            # da proposição também possui boa correspondência lexical.
            if content_lexical_overlap(claim, evidence) >= STRUCTURAL_MIN_LEXICAL:

                conflicts.append({
                    "type": "ENTITY",
                    "claim_entity": ce["text"],
                    "claim_label": ce["label"],
                    "evidence_entities": [
                        ee["text"]
                        for ee in same_type
                    ],
                })

    return conflicts

def make_claim_variants(claim, claim_context):
    """
    Cria variantes de hipótese para NLI.

    1. claim original;
    2. claim contextualizada.

    Não faz substituição cega de pronomes.
    """

    variants = [
        {
            "text": claim,
            "type": "original",
        }
    ]

    context = claim_context or {}

    local = str(
        context.get(
            "local_context",
            ""
        )
    ).strip()

    if local:

        contextualized = (
            f"Context: {local} "
            f"Claim: {claim}"
        )

        variants.append({
            "text": contextualized,
            "type": "contextualized",
        })

    return variants

def evaluate_single_evidence(
    claim,
    evidence,
    probs,
):
    ent, contr, neut = probs

    structural_conflicts = (
        detect_structural_conflicts(
            claim,
            evidence
        )
    )

    # Sinal de conflito estrutural.
    # Não transforma automaticamente em contradição.
    structural_types = {
        c["type"]
        for c in structural_conflicts
    }

    strong_structural = bool(
        structural_types
        & {
            "NUMBER",
            "DATE",
            "NEGATION",
        }
    )

    entity_conflict = (
        "ENTITY" in structural_types
    )

    return {
        "ent": ent,
        "contr": contr,
        "neut": neut,
        "structural_conflicts": structural_conflicts,
        "structural_types": sorted(
            structural_types
        ),
        "strong_structural": strong_structural,
        "entity_conflict": entity_conflict,
    }

def aggregate_nli_scores(results):
    """
    Agrega os escores por evidência.

    Estratégia:
    - suporte recebe maior peso do contexto expandido;
    - contradição recebe maior peso da evidência exata;
    - sinais estruturais não alteram os scores do NLI;
    - a qualidade da recuperação recebe apenas um peso suave;
    - o melhor suporte e a melhor contradição são escolhidos
      entre as evidências disponíveis.
    """

    if not results:
        return {
            "best_support": None,
            "best_contradiction": None,
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "support_structural": False,
            "contradiction_structural": False,
        }

    candidates = []

    for r in results:
        # Exata é mais confiável para detectar conflito factual;
        # expandida é mais útil para confirmar suporte contextual.
        support_base = (
            SUPPORT_EXACT_WEIGHT * r["ent"]
            + SUPPORT_EXPANDED_WEIGHT * r["expanded_ent"]
        )

        contradiction_base = (
            CONTRADICTION_EXACT_WEIGHT * r["contr"]
            + CONTRADICTION_EXPANDED_WEIGHT * r["expanded_contr"]
        )

        # Os scores usados na decisão são exclusivamente neurais.
        # Conflitos estruturais permanecem disponíveis para diagnóstico
        # e taxonomia, evitando aumentar falsos positivos de contradição.
        support_adjusted = support_base
        contradiction_adjusted = contradiction_base

        # Qualidade da evidência: recuperação + reranking.
        retrieval_score = float(r.get("retrieval_score", 0.0))
        cross_score = float(r.get("cross_score", 0.0))

        quality = np.clip(
            QUALITY_RETRIEVAL_WEIGHT * retrieval_score + QUALITY_CROSS_WEIGHT * cross_score,
            0.0,
            1.0,
        )

        # Peso deliberadamente suave: evidência ruim não é descartada,
        # mas também não deve vencer uma evidência claramente melhor.
        quality_weight = QUALITY_FLOOR + QUALITY_SCALE * quality

        support_adjusted *= quality_weight
        contradiction_adjusted *= quality_weight

        item = dict(r)
        item["support_adjusted"] = float(np.clip(support_adjusted, 0.0, 1.0))
        item["contradiction_adjusted"] = float(np.clip(contradiction_adjusted, 0.0, 1.0))
        item["quality_weight"] = float(quality_weight)
        item["has_structural"] = bool(r.get("structural_types", []))

        candidates.append(item)

    best_support = max(
        candidates,
        key=lambda r: (
            r["support_adjusted"],
            r["expanded_ent"],
            r["ent"],
        ),
    )

    best_contradiction = max(
        candidates,
        key=lambda r: (
            r["contradiction_adjusted"],
            r["contr"],
            r["expanded_contr"],
        ),
    )

    return {
        "best_support": best_support,
        "best_contradiction": best_contradiction,
        "support_score": best_support["support_adjusted"],
        "contradiction_score": best_contradiction["contradiction_adjusted"],
        "support_structural": bool(best_support["has_structural"]),
        "contradiction_structural": bool(best_contradiction["has_structural"]),
    }

def decide_nli_status(
    support_score,
    contradiction_score,
    contradiction_structural=False,
):
    """
    Decisão factual baseada prioritariamente no NLI.

    Sinais estruturais não criam uma segunda rota de contradição.
    Eles permanecem disponíveis no registro para diagnóstico e
    posterior taxonomia.
    """

    diff_con = contradiction_score - support_score
    diff_sup = support_score - contradiction_score

    if (
        contradiction_score >= CONTRADICTION_THRESHOLD
        and diff_con >= CONTRADICTION_MARGIN
    ):
        return (
            "CONTRADICTED",
            "nli_contradiction_strong_margin",
        )

    if (
        support_score >= SUPPORT_THRESHOLD
        and diff_sup >= SUPPORT_MARGIN
    ):
        return (
            "SUPPORTED",
            "nli_support_strong_margin",
        )

    return (
        "NOT_SUPPORTED",
        "nli_ambiguous_or_insufficient",
    )

def _execute():

    print(
        f"Lendo evidências: {ARQUIVO_ENTRADA}"
    )

    df = pd.read_parquet(
        ARQUIVO_ENTRADA
    )

    print(
        "Inicializando DeBERTa NLI..."
    )

    aligner = NLIAligner(**MODEL_OPTIONS)

    dados_nli = []

    counters = {
        "SUPPORTED": 0,
        "CONTRADICTED": 0,
        "NOT_SUPPORTED": 0,
    }

    # Diagnóstico específico dos erros controlados.
    diagnostic = {
        "injected_total": 0,
        "injected_contradicted": 0,
        "injected_supported": 0,
        "injected_not_supported": 0,
    }
    
    # --- INSERÇÃO 1: INICIALIZAR MATRIZ ---
    matriz_confusao = {
        "injected_error": {"CONTRADICTED": 0, "NOT_SUPPORTED": 0, "SUPPORTED": 0},
        "original_truth": {"CONTRADICTED": 0, "NOT_SUPPORTED": 0, "SUPPORTED": 0}
    }

    for row_idx, linha in tqdm(
        df.iterrows(),
        total=len(df),
        desc="NLI Multi-Evidence Calibrado"
    ):

        claim = str(
            linha["claim"]
        ).strip()

        claim_context = linha.get(
            "claim_context",
            {}
        )

        if not isinstance(
            claim_context,
            dict
        ):
            claim_context = {}

        # ----------------------------------------------------
        # Desserialização robusta
        # ----------------------------------------------------

        top_sentences = linha.get(
            "top_sentences",
            []
        )

        top_paragraphs = linha.get(
            "top_paragraphs",
            []
        )

        if isinstance(
            top_sentences,
            np.ndarray
        ):
            top_sentences = top_sentences.tolist()

        if isinstance(
            top_paragraphs,
            np.ndarray
        ):
            top_paragraphs = top_paragraphs.tolist()

        if top_sentences is None:
            top_sentences = []

        if top_paragraphs is None:
            top_paragraphs = []

        # ----------------------------------------------------
        # Construção das evidências
        # ----------------------------------------------------

        evidence_items = []

        seen = set()

        for item in (
            list(top_sentences)
            + list(top_paragraphs)
        ):

            if not isinstance(
                item,
                dict
            ):
                continue

            exact = str(
                item.get(
                    "text",
                    ""
                )
            ).strip()

            expanded = str(
                item.get(
                    "expanded_text",
                    exact
                )
            ).strip()

            if not exact and not expanded:
                continue

            key = normalize(
                exact or expanded
            )

            if key in seen:
                continue

            seen.add(key)

            evidence_items.append({
                "exact": exact or expanded,
                "expanded": expanded or exact,
                "sentence_index": item.get(
                    "sentence_index",
                    item.get(
                        "anchor_sentence_index",
                        -1
                    )
                ),
                "retrieval_score": float(
                    item.get(
                        "retrieval_score",
                        0.0
                    )
                ),
                "cross_score": float(
                    item.get(
                        "cross_score",
                        item.get(
                            "anchor_score",
                            0.0
                        )
                    )
                ),
            })

        # ----------------------------------------------------
        # Variantes da claim
        # ----------------------------------------------------

        variants = make_claim_variants(
            claim,
            claim_context
        )

        nli_results = []

        for ev_idx, evidence in enumerate(
            evidence_items
        ):

            exact = evidence["exact"]
            expanded = evidence["expanded"]

            # -----------------------------------------------
            # NLI W0 e W2 para a claim original.
            # -----------------------------------------------

            pairs = [
                (exact, claim),
                (expanded, claim),
            ]

            probs = aligner.process_batch(
                pairs,
                disable_prog_bar=True
            )

            ent_exact = float(
                probs[0][0].item()
            )

            con_exact = float(
                probs[1][0].item()
            )

            neu_exact = float(
                probs[2][0].item()
            )

            ent_expanded = float(
                probs[0][1].item()
            )

            con_expanded = float(
                probs[1][1].item()
            )

            neu_expanded = float(
                probs[2][1].item()
            )

            # -----------------------------------------------
            # Verifica também uma hipótese contextual,
            # mas usa-a apenas como apoio, não como decisão.
            # -----------------------------------------------

            contextual_ent = 0.0
            contextual_con = 0.0

            if (
                len(variants) > 1
                and variants[1]["type"]
                == "contextualized"
            ):

                context_text = variants[1]["text"]

                probs_context = aligner.process_batch(
                    [
                        (
                            expanded,
                            context_text
                        )
                    ],
                    disable_prog_bar=True
                )

                contextual_ent = float(
                    probs_context[0][0].item()
                )

                contextual_con = float(
                    probs_context[1][0].item()
                )

            structural = evaluate_single_evidence(
                claim,
                exact,
                [
                    ent_exact,
                    con_exact,
                    neu_exact,
                ]
            )

            # O NLI contextual é mantido somente para diagnóstico.
            # Não é usado diretamente na decisão porque a hipótese
            # contém a própria claim ("Claim: ..."), o que pode
            # inflar artificialmente entailment/contradiction.
            effective_expanded_ent = ent_expanded
            effective_expanded_con = con_expanded

            nli_results.append({

                "evidence_index": ev_idx,

                "passage_exact": exact,

                "passage_expanded": expanded,

                "sentence_index": evidence[
                    "sentence_index"
                ],

                "retrieval_score": evidence[
                    "retrieval_score"
                ],

                "cross_score": evidence[
                    "cross_score"
                ],

                "ent": ent_exact,

                "contr": con_exact,

                "neut": neu_exact,

                "expanded_ent": effective_expanded_ent,

                "expanded_contr": effective_expanded_con,

                "expanded_neut": neu_expanded,

                "contextual_ent": contextual_ent,

                "contextual_contr": contextual_con,

                "structural_conflicts": structural[
                    "structural_conflicts"
                ],

                "structural_types": structural[
                    "structural_types"
                ],

                "strong_structural": structural[
                    "strong_structural"
                ],

                "entity_conflict": structural[
                    "entity_conflict"
                ],
            })




        # ----------------------------------------------------
        # GABARITO / GRUPO
        # ----------------------------------------------------

        gabarito = str(
            linha.get(
                "gabarito_injetado",
                "none"
            )
        )

        if gabarito.endswith("_error"):

            grupo_gabarito = "injected_error"

            diagnostic[
                "injected_total"
            ] += 1

        else:

            grupo_gabarito = "original_truth"


        # ----------------------------------------------------
        # DECISÃO NLI
        # ----------------------------------------------------

        score_data = aggregate_nli_scores(
            nli_results
        )

        sup_score = score_data[
            "support_score"
        ]

        con_score = score_data[
            "contradiction_score"
        ]

        status, decision_reason = decide_nli_status(
            sup_score,
            con_score,
        )


        # ----------------------------------------------------
        # MATRIZ DE CONFUSÃO
        # ----------------------------------------------------

        matriz_confusao[
            grupo_gabarito
        ][status] += 1

        counters[status] += 1


        # ----------------------------------------------------
        # DIAGNÓSTICO DOS ERROS INJETADOS
        # ----------------------------------------------------

        if grupo_gabarito == "injected_error":

            if status == "CONTRADICTED":

                diagnostic[
                    "injected_contradicted"
                ] += 1

            elif status == "SUPPORTED":

                diagnostic[
                    "injected_supported"
                ] += 1

            else:

                diagnostic[
                    "injected_not_supported"
                ] += 1






        # --- INSERÇÃO 2: CALCULAR NLI PURO E ALIMENTAR MATRIZ ---
        # Aqui pegamos os scores matemáticos que o DeBERTa calculou
        # sup_score = decision.get("support_score", 0.0)
        # con_score = decision.get("contradiction_score", 0.0)
        
        # # Simulamos a decisão ignorando totalmente o auditor de conflitos
        # if con_score >= CONTRADICTION_THRESHOLD and (con_score - sup_score) >= CONTRADICTION_MARGIN:
        #     status_nli_puro = "CONTRADICTED"
        # elif sup_score >= SUPPORT_THRESHOLD and (sup_score - con_score) >= SUPPORT_MARGIN:
        #     status_nli_puro = "SUPPORTED"
        # else:
        #     status_nli_puro = "NOT_SUPPORTED"
            
        # # Alimentamos a contagem da Tabela 1
        # matriz_confusao[grupo_gabarito][status_nli_puro] += 1

        if status == "CONTRADICTED":
            best = score_data["best_contradiction"]
        else:
            best = score_data["best_support"]

        if best is None:

            source_passage = ""

            selected_probs = [
                0.0, 0.0, 0.0
            ]

            selected_expanded_probs = [
                0.0, 0.0, 0.0
            ]

            selected_index = -1

        else:

            source_passage = best[
                "passage_expanded"
            ]

            selected_probs = [
                best["ent"],
                best["contr"],
                best["neut"],
            ]

            selected_expanded_probs = [
                best["expanded_ent"],
                best["expanded_contr"],
                best["expanded_neut"],
            ]

            selected_index = best[
                "sentence_index"
            ]

        dados_nli.append({

            "doc_idx": linha[
                "doc_idx"
            ],

            "tipo_resumo": linha[
                "tipo_resumo"
            ],

            "claim": claim,

            "gabarito_injetado": gabarito,

            "claim_context": claim_context,

            "source_passage": source_passage,

            "source_sentence_index": selected_index,

            "probs": selected_probs,

            "expanded_probs": selected_expanded_probs,

            "status_factual": status,

            "status_factual_nli": status,

            "decision_reason": decision_reason,

            "support_score": sup_score,

            "contradiction_score": con_score,

            "support_structural": score_data["support_structural"],
            "contradiction_structural": score_data["contradiction_structural"],

            "structural_conflicts": [
                c
                for r in nli_results
                for c in r["structural_conflicts"]
            ],

            "nli_details": nli_results,

            "documento_fonte": linha[
                "documento_fonte"
            ],

        })

    # ========================================================
    # SALVAMENTO
    # ========================================================

    df_saida = pd.DataFrame(
        dados_nli
    )

    os.makedirs(
        os.path.dirname(ARQUIVO_SAIDA),
        exist_ok=True,
    )

    df_saida.to_parquet(
        ARQUIVO_SAIDA,
        engine="pyarrow",
        index=False,
    )

    # ========================================================
    # MÉTRICAS
    # ========================================================

    print("\n" + "=" * 80)
    print("RESULTADO DA NOVA DECISÃO NLI")
    print("=" * 80)

    total = sum(
        counters.values()
    )

    for label in (
        "CONTRADICTED",
        "NOT_SUPPORTED",
        "SUPPORTED",
    ):

        count = counters[
            label
        ]

        pct = (
            100.0 * count / total
            if total
            else 0.0
        )

        print(
            f"{label:18s}: "
            f"{count:5d} "
            f"({pct:6.2f}%)"
        )

    print("\n" + "=" * 80)
    print("DIAGNÓSTICO DOS ERROS INJETADOS")
    print("=" * 80)

    injected_total = diagnostic[
        "injected_total"
    ]

    if injected_total:

        print(
            f"Total injected_error: "
            f"{injected_total}"
        )

        print(
            "Detectados como CONTRADICTED: "
            f"{diagnostic['injected_contradicted']} "
            f"("
            f"{100.0 * diagnostic['injected_contradicted'] / injected_total:.2f}%"
            ")"
        )

        print(
            "Falsos negativos NOT_SUPPORTED: "
            f"{diagnostic['injected_not_supported']}"
        )

        print(
            "Falsos negativos SUPPORTED: "
            f"{diagnostic['injected_supported']}"
        )
        
    # --- INSERÇÃO 3: IMPRIMIR MATRIZ ---
    print("\n" + "=" * 80)
    print("🧠 TABELA 1: MATRIZ DE CONFUSÃO — DECISÃO NLI CALIBRADA")
    print("=" * 80)
    #print("Esta tabela mostra como o modelo neural classificou os fatos ANTES da taxonomia.")
    #print("O ideal é que 'injected_error' seja CONTRADICTED e 'original_truth' seja SUPPORTED.\n")
    print(f"{'PREDIÇÃO NLI (DeBERTa)':<30} {'CONTRADICTED':>14} {'NOT_SUPPORTED':>14} {'SUPPORTED':>10} {'TOTAL':>6}")
    print("GABARITO (Verdade vs Mentira)                                               ")
    
    for group in ["injected_error", "original_truth"]:
        c = matriz_confusao[group]["CONTRADICTED"]
        ns = matriz_confusao[group]["NOT_SUPPORTED"]
        s = matriz_confusao[group]["SUPPORTED"]
        t = c + ns + s
        print(f"{group:<30} {c:14d} {ns:14d} {s:10d} {t:6d}")
        
    tot_c = matriz_confusao["injected_error"]["CONTRADICTED"] + matriz_confusao["original_truth"]["CONTRADICTED"]
    tot_ns = matriz_confusao["injected_error"]["NOT_SUPPORTED"] + matriz_confusao["original_truth"]["NOT_SUPPORTED"]
    tot_s = matriz_confusao["injected_error"]["SUPPORTED"] + matriz_confusao["original_truth"]["SUPPORTED"]
    tot_t = tot_c + tot_ns + tot_s
    print(f"{'TOTAL':<30} {tot_c:14d} {tot_ns:14d} {tot_s:10d} {tot_t:6d}")

    # ========================================================
    # MÉTRICAS DE CALIBRAÇÃO
    # ========================================================
    inj = matriz_confusao["injected_error"]
    orig = matriz_confusao["original_truth"]

    injected_recall = (
        100.0 * inj["CONTRADICTED"] / sum(inj.values())
        if sum(inj.values()) else 0.0
    )
    original_false_contradiction = (
        100.0 * orig["CONTRADICTED"] / sum(orig.values())
        if sum(orig.values()) else 0.0
    )
    original_non_contradicted = (
        100.0 * (orig["NOT_SUPPORTED"] + orig["SUPPORTED"]) / sum(orig.values())
        if sum(orig.values()) else 0.0
    )

    print("\n" + "=" * 80)
    print("MÉTRICAS DE CALIBRAÇÃO")
    print("=" * 80)
    print(f"Recall de injected_error (CONTRADICTED): {injected_recall:.2f}%")
    print(f"Falso CONTRADICTED em original_truth    : {original_false_contradiction:.2f}%")
    print(f"Original_truth não contradita             : {original_non_contradicted:.2f}%")

    print(
        f"\n✅ Arquivo salvo: {ARQUIVO_SAIDA}"
    )

    del aligner
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(input_path, output_path, config):
    global ARQUIVO_ENTRADA, ARQUIVO_SAIDA, MODEL_OPTIONS, nlp
    from factual_eval.config import device_for
    from factual_eval.schemas import validate_retrieval, carry_claim_ids
    from factual_eval.schemas import stage_paths
    input_path,output_path=stage_paths(input_path,output_path)
    validate_retrieval(input_path)
    options = config['nli']
    for key in ('support_threshold','contradiction_threshold','support_margin','contradiction_margin',
                'structural_min_lexical','support_exact_weight','support_expanded_weight',
                'contradiction_exact_weight','contradiction_expanded_weight',
                'quality_retrieval_weight','quality_cross_weight','quality_floor','quality_scale'):
        globals()[key.upper()] = options[key]
    MODEL_OPTIONS = {key: options[key] for key in ('model_name','revision','batch_size','max_length')}
    MODEL_OPTIONS['device'] = device_for(config, 'nli')
    nlp = spacy.load(options['spacy_model'])
    ARQUIVO_ENTRADA, ARQUIVO_SAIDA = str(input_path), str(output_path)
    _execute()
    carry_claim_ids(input_path, output_path)
    return output_path
