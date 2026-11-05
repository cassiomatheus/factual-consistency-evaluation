"""Adapted from pipeline/04_classify_factual_status_melhorado_novo.py; scientific rules retained. See docs/migration.md."""
import gc
import os
import re
import torch
import pandas as pd
import spacy
from tqdm import tqdm
import ast
import json
from collections import Counter
from sentence_transformers import SentenceTransformer, util


SBERT_MODEL = "all-MiniLM-L6-v2"

ENTITY_SEMANTIC_THRESHOLD = 0.78

PREDICATE_SEMANTIC_THRESHOLD = 0.72

ARGUMENT_MATCH_THRESHOLD = 0.70

MIN_LEXICAL_OVERLAP = 0.25

VALID_TAXONOMIES = {
    "coreference_error",
    "entity_error",
    "predicate_error",
    "other",
    "none",
}

def normalizar(texto):
    return re.sub(r"\s+", " ", str(texto or "").lower()).strip()

def tokens(texto):
    return {
        t
        for t in re.findall(r"\b[a-z0-9]+\b", normalizar(texto))
        if len(t) > 2
    }

def lexical_overlap(a, b):
    ta = tokens(a)
    tb = tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def parse_context(value):
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    return {}

def text_from_context(context):
    context = parse_context(context)
    pieces = []

    for key in (
        "local_context",
        "previous_sentence",
        "next_sentence",
        "sentence_context",
        "context",
    ):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            pieces.append(value.strip())

    # Algumas versões armazenam lista de sentenças.
    for key in ("sentences", "context_sentences"):
        value = context.get(key)
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text", "")
                else:
                    text = str(item)
                if text.strip():
                    pieces.append(text.strip())

    return " ".join(dict.fromkeys(pieces))

def get_nli_status(row):
    return str(row.get("status_factual_nli", "")).strip().upper()

def normalize_ground_truth(value):
    if pd.isna(value):
        return "none"

    value = str(value).strip().lower()

    if value in VALID_TAXONOMIES:
        return value

    return "none"

def extract_numbers(text):
    return re.findall(
        r"(?<!\w)\$?\d+(?:[.,]\d+)?%?(?!\w)",
        str(text),
    )

def normalize_number(value):
    return str(value).replace(",", "").replace("$", "").replace("%", "").strip()

def extract_years(text):
    return set(re.findall(r"\b(?:19|20)\d{2}\b", str(text)))

def has_negation(text):
    negations = {
        "not", "no", "never", "neither", "without", "n't"
    }
    doc = NLP(str(text))
    return any(tok.text.lower() in negations for tok in doc)

ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "LAW", "PRODUCT"
}

def extract_entities(text):
    doc = NLP(str(text))
    return [
        {
            "text": ent.text.strip(),
            "label": ent.label_,
        }
        for ent in doc.ents
        if ent.label_ in ENTITY_LABELS
    ]

def normalized_entity_text(text):
    return normalizar(text).replace(".", "")

def entity_match_score(a, b):
    a = normalizar(a)
    b = normalizar(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    ta = tokens(a)
    tb = tokens(b)
    if ta and tb:
        lexical = len(ta & tb) / max(len(ta), len(tb))
    else:
        lexical = 0.0

    emb = SBERT.encode(
        [a, b],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    semantic = float(util.cos_sim(emb[0], emb[1]).item())

    return max(lexical, semantic)

def extract_predicate_structures(text):
    doc = NLP(str(text))
    structures = []

    for token in doc:
        if token.pos_ not in {"VERB", "AUX"}:
            continue

        subjects = [
            child for child in token.children
            if child.dep_ in {"nsubj", "nsubjpass", "csubj"}
        ]

        objects = [
            child for child in token.children
            if child.dep_ in {
                "dobj", "obj", "attr", "oprd", "pobj", "dative", "iobj"
            }
        ]

        # Quando o verbo está em passiva, o argumento pode estar como nsubjpass.
        subject = subjects[0].subtree if subjects else []
        obj = objects[0].subtree if objects else []

        subject_text = " ".join(tok.text for tok in subject).strip()
        object_text = " ".join(tok.text for tok in obj).strip()

        structures.append({
            "verb": token.lemma_.lower(),
            "verb_text": token.text,
            "subject": subject_text,
            "object": object_text,
            "token_index": token.i,
        })

    return structures

def best_argument_match(a, b):
    a = normalizar(a)
    b = normalizar(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    lexical = lexical_overlap(a, b)

    emb = SBERT.encode(
        [a, b],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    semantic = float(util.cos_sim(emb[0], emb[1]).item())

    return max(lexical, semantic)

def predicates_are_comparable(claim_struct, evidence_struct):
    subject_score = best_argument_match(
        claim_struct.get("subject", ""),
        evidence_struct.get("subject", ""),
    )

    object_score = best_argument_match(
        claim_struct.get("object", ""),
        evidence_struct.get("object", ""),
    )

    # Exigência conservadora: pelo menos um argumento forte e o outro
    # não pode estar claramente em conflito.
    comparable = (
        subject_score >= ARGUMENT_MATCH_THRESHOLD
        and object_score >= MIN_LEXICAL_OVERLAP
    ) or (
        object_score >= ARGUMENT_MATCH_THRESHOLD
        and subject_score >= MIN_LEXICAL_OVERLAP
    )

    return comparable, subject_score, object_score

def verb_similarity(a, b):
    a = normalizar(a)
    b = normalizar(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    emb = SBERT.encode(
        [a, b],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return float(util.cos_sim(emb[0], emb[1]).item())

PRONOUNS = {
    "he", "she", "it", "they", "them", "him", "her", "his", "hers",
    "its", "their", "theirs", "this", "that", "these", "those",
    "who", "which", "whom"
}

def relevant_pronouns(claim):
    doc = NLP(str(claim))
    return [tok.text.lower() for tok in doc if tok.text.lower() in PRONOUNS]

def candidate_antecedents(claim_context):
    context = parse_context(claim_context)
    context_text = text_from_context(context)
    if not context_text:
        return []

    entities = extract_entities(context_text)

    # Mantém a ordem, removendo duplicatas.
    seen = set()
    result = []
    for ent in entities:
        key = (normalizar(ent["text"]), ent["label"])
        if key not in seen:
            seen.add(key)
            result.append(ent)

    return result

def detect_coreference_issue(claim, evidence, claim_context):
    pronouns = relevant_pronouns(claim)

    if not pronouns:
        return {
            "is_conflict": False,
            "reason": "no_relevant_pronoun",
            "pronouns": [],
            "antecedents": [],
            "evidence_entities": [],
        }

    antecedents = candidate_antecedents(claim_context)
    evidence_entities = extract_entities(evidence)

    if not antecedents:
        return {
            "is_conflict": False,
            "reason": "no_contextual_antecedent",
            "pronouns": pronouns,
            "antecedents": [],
            "evidence_entities": evidence_entities,
        }

    # Só consideramos conflito quando existe entidade antecedente no contexto
    # e há entidade nomeada incompatível/ausente na evidência relacionada.
    comparisons = []

    for ant in antecedents:
        matches = []
        for ev in evidence_entities:
            if ev["label"] == ant["label"]:
                score = entity_match_score(ant["text"], ev["text"])
                matches.append({
                    "antecedent": ant["text"],
                    "evidence_entity": ev["text"],
                    "label": ant["label"],
                    "score": score,
                })

        if not matches:
            comparisons.append({
                "antecedent": ant["text"],
                "label": ant["label"],
                "status": "missing_same_type",
            })
            continue

        best = max(matches, key=lambda x: x["score"])
        comparisons.append({
            **best,
            "status": "match" if best["score"] >= ENTITY_SEMANTIC_THRESHOLD else "mismatch",
        })

    mismatches = [
        x for x in comparisons
        if x.get("status") == "mismatch"
    ]

    # "missing_same_type" sozinho é fraco demais para afirmar erro de coreferência.
    is_conflict = bool(mismatches)

    return {
        "is_conflict": is_conflict,
        "reason": "antecedent_mismatch" if is_conflict else "no_antecedent_mismatch",
        "pronouns": pronouns,
        "antecedents": antecedents,
        "evidence_entities": evidence_entities,
        "comparisons": comparisons,
    }

def detect_entity_issue(claim, evidence):
    claim_entities = extract_entities(claim)
    evidence_entities = extract_entities(evidence)

    if not claim_entities or not evidence_entities:
        return {
            "is_conflict": False,
            "reason": "insufficient_entities",
            "claim_entities": claim_entities,
            "evidence_entities": evidence_entities,
            "matches": [],
        }

    overlap = lexical_overlap(claim, evidence)
    matches = []
    mismatches = []

    for claim_ent in claim_entities:
        same_type = [
            e for e in evidence_entities
            if e["label"] == claim_ent["label"]
        ]

        if not same_type:
            continue

        best = max(
            same_type,
            key=lambda e: entity_match_score(
                claim_ent["text"], e["text"]
            )
        )

        score = entity_match_score(
            claim_ent["text"], best["text"]
        )

        item = {
            "claim_entity": claim_ent["text"],
            "claim_label": claim_ent["label"],
            "best_evidence_entity": best["text"],
            "similarity": score,
        }

        if score >= ENTITY_SEMANTIC_THRESHOLD:
            item["status"] = "match"
            matches.append(item)
        else:
            item["status"] = "mismatch"
            mismatches.append(item)

    # Uma troca de entidade só é forte quando as sentenças compartilham
    # material factual suficiente para indicar que é o mesmo evento.
    is_conflict = bool(mismatches) and overlap >= MIN_LEXICAL_OVERLAP

    return {
        "is_conflict": is_conflict,
        "reason": "entity_mismatch" if is_conflict else "no_reliable_entity_mismatch",
        "claim_entities": claim_entities,
        "evidence_entities": evidence_entities,
        "matches": matches,
        "mismatches": mismatches,
        "lexical_overlap": overlap,
    }

def detect_predicate_issue(claim, evidence):
    claim_structs = extract_predicate_structures(claim)
    evidence_structs = extract_predicate_structures(evidence)

    if not claim_structs or not evidence_structs:
        return {
            "is_conflict": False,
            "reason": "insufficient_predicate_structure",
            "matches": [],
        }

    matches = []

    for cs in claim_structs:
        best_candidate = None

        for es in evidence_structs:
            comparable, subject_score, object_score = predicates_are_comparable(
                cs, es
            )

            if not comparable:
                continue

            v_sim = verb_similarity(
                cs["verb"],
                es["verb"],
            )

            candidate = {
                "claim_verb": cs["verb"],
                "evidence_verb": es["verb"],
                "claim_subject": cs["subject"],
                "evidence_subject": es["subject"],
                "claim_object": cs["object"],
                "evidence_object": es["object"],
                "subject_similarity": subject_score,
                "object_similarity": object_score,
                "verb_similarity": v_sim,
            }

            if best_candidate is None:
                best_candidate = candidate
            else:
                current = (
                    candidate["subject_similarity"]
                    + candidate["object_similarity"]
                )
                previous = (
                    best_candidate["subject_similarity"]
                    + best_candidate["object_similarity"]
                )
                if current > previous:
                    best_candidate = candidate

        if best_candidate:
            matches.append(best_candidate)

    predicate_conflict = []

    for match in matches:
        # Caso 1: verbos diferentes e semanticamente incompatíveis.
        verb_conflict = (
            normalizar(match["claim_verb"]) != normalizar(match["evidence_verb"])
            and match["verb_similarity"] < PREDICATE_SEMANTIC_THRESHOLD
        )

        if verb_conflict:
            predicate_conflict.append({
                **match,
                "type": "VERB_INCOMPATIBILITY",
            })

    # Caso 2: negação explícita com argumentos comparáveis.
    negation_conflict = has_negation(claim) != has_negation(evidence)
    if negation_conflict and lexical_overlap(claim, evidence) >= MIN_LEXICAL_OVERLAP:
        predicate_conflict.append({
            "type": "NEGATION",
            "claim_negation": has_negation(claim),
            "evidence_negation": has_negation(evidence),
        })

    # Caso 3: mudança temporal/numérica com material lexical compartilhado.
    claim_years = extract_years(claim)
    evidence_years = extract_years(evidence)
    if (
        claim_years
        and evidence_years
        and claim_years != evidence_years
        and lexical_overlap(claim, evidence) >= MIN_LEXICAL_OVERLAP
    ):
        predicate_conflict.append({
            "type": "DATE_SHIFT",
            "claim_years": sorted(claim_years),
            "evidence_years": sorted(evidence_years),
        })

    claim_nums = {
        normalize_number(x) for x in extract_numbers(claim)
    }
    evidence_nums = {
        normalize_number(x) for x in extract_numbers(evidence)
    }
    if (
        claim_nums
        and evidence_nums
        and claim_nums != evidence_nums
        and lexical_overlap(claim, evidence) >= MIN_LEXICAL_OVERLAP
    ):
        predicate_conflict.append({
            "type": "NUMBER_SHIFT",
            "claim_numbers": sorted(claim_nums),
            "evidence_numbers": sorted(evidence_nums),
        })

    return {
        "is_conflict": bool(predicate_conflict),
        "reason": "predicate_or_attribute_conflict" if predicate_conflict else "no_reliable_predicate_conflict",
        "matches": matches,
        "conflicts": predicate_conflict,
    }

def classify_taxonomy(claim, evidence, claim_context):
    """
    Classifica SOMENTE uma claim que o script 03 já marcou como CONTRADICTED.

    Prioridade:
        1. coreference_error
        2. entity_error
        3. predicate_error
        4. other

    O rótulo none não é produzido para CONTRADICTED.
    """

    coref = detect_coreference_issue(
        claim,
        evidence,
        claim_context,
    )

    if coref["is_conflict"]:
        return "coreference_error", "coreference_structural_conflict", coref, None, None

    entity = detect_entity_issue(
        claim,
        evidence,
    )

    if entity["is_conflict"]:
        return "entity_error", "entity_structural_conflict", coref, entity, None

    predicate = detect_predicate_issue(
        claim,
        evidence,
    )

    if predicate["is_conflict"]:
        return "predicate_error", "predicate_or_attribute_conflict", coref, entity, predicate

    return "other", "contradiction_without_reliable_structural_type", coref, entity, predicate

def imprimir_matriz(df_taxonomia):
    categorias = [
        "coreference_error",
        "entity_error",
        "predicate_error",
        "other",
        "none",
    ]

    df_eval = df_taxonomia.copy()
    df_eval["ground_truth_taxonomy"] = (
        df_eval["gabarito_injetado"]
        .apply(normalize_ground_truth)
    )

    matriz = pd.crosstab(
        df_eval["ground_truth_taxonomy"],
        df_eval["taxonomy_label"],
    )

    matriz = matriz.reindex(
        index=[
            "coreference_error",
            "entity_error",
            "none",
            "predicate_error",
        ],
        columns=categorias,
        fill_value=0,
    )

    matriz["TOTAL"] = matriz.sum(axis=1)

    total_row = matriz[categorias].sum(axis=0)
    total_row["TOTAL"] = total_row.sum()
    matriz.loc["TOTAL"] = total_row

    print("\n" + "=" * 90)
    print("🏆 MATRIZ DE CONFUSÃO DA TAXONOMIA")
    print("=" * 90)
    print("Somente as claims classificadas como CONTRADICTED pelo script 03")
    print(matriz.to_string())
    print("=" * 90)

    return matriz

def imprimir_metricas_condicionais(df_taxonomia):
    print("\n" + "=" * 90)
    print("📊 RECALL TAXONÔMICO CONDICIONAL À DETECÇÃO NLI")
    print("=" * 90)

    df_eval = df_taxonomia.copy()
    df_eval["ground_truth_taxonomy"] = (
        df_eval["gabarito_injetado"]
        .apply(normalize_ground_truth)
    )

    for categoria in [
        "entity_error",
        "predicate_error",
        "coreference_error",
    ]:
        subset = df_eval[
            df_eval["ground_truth_taxonomy"] == categoria
        ]

        total = len(subset)
        corretos = int(
            (
                subset["taxonomy_label"] == categoria
            ).sum()
        )

        recall = corretos / total if total else 0.0

        print(
            f"{categoria:<22}: "
            f"{corretos:>3}/{total:<3} "
            f"({100.0 * recall:6.2f}%)"
        )

    print("=" * 90)

def imprimir_taxonomia_injetada(df_taxonomia):
    print("\n" + "=" * 90)
    print("📌 TAXONOMIA DOS ERROS INJETADOS DETECTADOS PELO NLI")
    print("=" * 90)

    df_eval = df_taxonomia.copy()
    df_eval["ground_truth_taxonomy"] = (
        df_eval["gabarito_injetado"]
        .apply(normalize_ground_truth)
    )

    injected = df_eval[
        df_eval["ground_truth_taxonomy"].isin({
            "entity_error",
            "predicate_error",
            "coreference_error",
        })
    ]

    print(f"Erros injetados detectados pelo NLI: {len(injected)}")

    if injected.empty:
        return

    corretos = int(
        (
            injected["taxonomy_label"]
            == injected["ground_truth_taxonomy"]
        ).sum()
    )

    print(
        f"Classificados na categoria correta: "
        f"{corretos}/{len(injected)} "
        f"({100.0 * corretos / len(injected):.2f}%)"
    )

    print("\nDistribuição prevista:")
    print(
        injected["taxonomy_label"]
        .value_counts()
        .to_string()
    )

def _execute():
    print(f"Lendo saída NLI: {ARQUIVO_ENTRADA}")

    if not os.path.exists(ARQUIVO_ENTRADA):
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado:\n{ARQUIVO_ENTRADA}"
        )

    df = pd.read_parquet(ARQUIVO_ENTRADA)

    print(f"Total de claims: {len(df)}")

    # ========================================================
    # FILTRO CRÍTICO
    # ========================================================
    df_contradicted = df[
        df["status_factual_nli"].astype(str).str.upper().eq("CONTRADICTED")
    ].copy()

    print(
        f"Claims CONTRADICTED pelo script 03: "
        f"{len(df_contradicted)}"
    )

    print(
        "\n⚠️ A taxonomia será aplicada SOMENTE às claims "
        "CONTRADICTED pelo script 03."
    )

    dados = []

    for _, linha in tqdm(
        df_contradicted.iterrows(),
        total=len(df_contradicted),
        desc="Classificação taxonômica",
    ):
        claim = str(linha.get("claim", "")).strip()
        evidence = str(linha.get("source_passage", "")).strip()
        context = parse_context(linha.get("claim_context", {}))

        taxonomy_label, taxonomy_reason, coref, entity, predicate = classify_taxonomy(
            claim,
            evidence,
            context,
        )

        linha_dict = linha.to_dict()

        # O status factual continua sendo o produzido pelo 03.
        linha_dict["status_factual"] = "CONTRADICTED"
        linha_dict["status_factual_nli"] = "CONTRADICTED"

        linha_dict["taxonomy_label"] = taxonomy_label
        linha_dict["taxonomy_reason"] = taxonomy_reason

        linha_dict["coreference_diagnostic"] = coref
        linha_dict["entity_diagnostic"] = entity
        linha_dict["predicate_diagnostic"] = predicate

        dados.append(linha_dict)

    df_saida = pd.DataFrame(dados)

    if df_saida.empty:
        raise RuntimeError(
            "Nenhuma claim CONTRADICTED foi encontrada no resultado do script 03."
        )

    print("\nSalvando resultado taxonômico...")

    os.makedirs(
        os.path.dirname(ARQUIVO_SAIDA),
        exist_ok=True,
    )

    df_saida.to_parquet(
        ARQUIVO_SAIDA,
        engine="pyarrow",
        index=False,
    )

    print(f"✅ Arquivo salvo:\n{ARQUIVO_SAIDA}")

    matriz = imprimir_matriz(df_saida)
    imprimir_metricas_condicionais(df_saida)
    imprimir_taxonomia_injetada(df_saida)

    print("\n" + "=" * 90)
    print("📊 DISTRIBUIÇÃO DA TAXONOMIA")
    print("=" * 90)
    print(
        df_saida["taxonomy_label"]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 90)
    print("📌 STATUS NLI CONSUMIDO PELO 04")
    print("=" * 90)
    print(
        df["status_factual_nli"]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 90)
    print("📌 FLUXO INJETADO → NLI → TAXONOMIA")
    print("=" * 90)

    total_injected = int(
        df["gabarito_injetado"]
        .astype(str)
        .str.lower()
        .str.endswith("_error")
        .sum()
    )

    detected_injected = int(
        df_contradicted["gabarito_injetado"]
        .astype(str)
        .str.lower()
        .str.endswith("_error")
        .sum()
    )

    print(f"Injected_error total: {total_injected}")
    print(f"Injected_error detectados pelo 03: {detected_injected}")
    print(
        f"Injected_error não detectados pelo 03: "
        f"{total_injected - detected_injected}"
    )

    matriz.to_csv(ARQUIVO_MATRIZ)

    print(f"\n✅ Matriz salva em:\n{ARQUIVO_MATRIZ}")

    # SBERT foi inicializado no escopo global; não usar "del SBERT" aqui,
    # pois isso faz o Python tratá-lo como variável local dentro de main().
    globals().pop("SBERT", None)
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n✅ Finalização concluída.")


def run(input_path, output_path, config, matrix_path=None):
    global ARQUIVO_ENTRADA, ARQUIVO_SAIDA, ARQUIVO_MATRIZ, NLP, SBERT
    from factual_eval.config import device_for
    from factual_eval.schemas import validate_nli, validate_taxonomy
    from factual_eval.provenance import record_model
    from factual_eval.schemas import stage_paths
    input_path,output_path=stage_paths(input_path,output_path)
    if matrix_path is not None:
        from pathlib import Path
        matrix_path=Path(matrix_path).resolve()
        if matrix_path.exists():
            raise FileExistsError(matrix_path)
        matrix_path.parent.mkdir(parents=True,exist_ok=True)
    elif output_path.with_suffix('.confusion.csv').exists():
        raise FileExistsError(output_path.with_suffix('.confusion.csv'))
    validate_nli(input_path)
    options = config['taxonomy']
    for key, name in {'entity_threshold':'ENTITY_SEMANTIC_THRESHOLD',
        'predicate_threshold':'PREDICATE_SEMANTIC_THRESHOLD', 'argument_threshold':'ARGUMENT_MATCH_THRESHOLD',
        'min_lexical_overlap':'MIN_LEXICAL_OVERLAP'}.items():
        globals()[name] = options[key]
    ARQUIVO_ENTRADA, ARQUIVO_SAIDA = str(input_path), str(output_path)
    ARQUIVO_MATRIZ = str(matrix_path or output_path.with_suffix('.confusion.csv'))
    NLP = spacy.load(options['spacy_model'])
    SBERT = SentenceTransformer(options['embedding_model'], device=device_for(config, 'taxonomy'), revision=options['embedding_revision'])
    record_model('taxonomy.embedding', options['embedding_model'], SBERT)
    _execute()
    validate_taxonomy(input_path, output_path)
    return output_path
