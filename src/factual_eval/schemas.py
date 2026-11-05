"""Add stable occurrence IDs and enforce contracts without changing predictions."""
from collections import Counter
import hashlib
import json
import re
from pathlib import Path


def document_id(index, source):
    external = re.sub(r'[^a-zA-Z0-9_-]', '_', str(index))
    return f'doc_{external}_{hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]}'


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def stage_paths(input_path, output_path):
    """Protect callers of the reusable stage API as well as CLI users."""
    source,target=Path(input_path).resolve(),Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite stage output: {target}')
    target.parent.mkdir(parents=True,exist_ok=True)
    return source,target


def read_injected_csv(path, reader=None):
    import pandas as pd
    frame = (reader or pd.read_csv)(path)
    required = {'fonte','resumo_original','resumo_entity','resumo_predicado','resumo_corref'}
    if not required <= set(frame):
        raise ValueError(f'Missing injected CSV columns: {sorted(required-set(frame))}')
    if 'source_index' in frame:
        indices = frame['source_index']
        if indices.isna().any() or indices.duplicated().any() or any(int(x) != x for x in indices):
            raise ValueError('source_index must contain unique integer source positions')
        frame.index = indices.astype(int)
    if frame.empty:
        raise ValueError('Empty input dataset')
    if any(not isinstance(value,str) or not value.strip() for value in frame['fonte']):
        raise ValueError('Source documents must be nonempty strings')
    return frame


def _read(path, required):
    import pandas as pd
    frame = pd.read_parquet(path)
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if frame.empty:
        raise ValueError(f'{path}: empty stage input')
    return frame


def validate_extraction(path):
    frame = _read(path, ('doc_idx','tipo_resumo','documento_fonte','claims'))
    identities = []
    for row in frame.to_dict('records'):
        if not isinstance(row['documento_fonte'], str) or not row['documento_fonte'].strip():
            raise ValueError('Extraction contains an empty source document')
        claims = row['claims']
        if claims is None or not hasattr(claims, '__iter__'):
            raise ValueError('claims must be a list')
        for claim in claims:
            if not isinstance(claim, dict) or not (claim.get('claim_original') or claim.get('texto') or '').strip():
                raise ValueError('Malformed or empty claim; refusing silent exclusion')
            if claim.get('claim_id'):
                identities.append(claim['claim_id'])
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate claim_id in extraction')
    return frame


def add_extraction_ids(input_path, output_path, start=0, num_docs=None):
    import pandas as pd
    frame = validate_extraction(output_path)
    source = read_injected_csv(input_path).iloc[start:None if num_docs is None else start+num_docs]
    columns = {'original':'resumo_original','entity':'resumo_entity','predicate':'resumo_predicado','coreference':'resumo_corref'}
    expected = {(idx, kind) for idx, row in source.iterrows() for kind, col in columns.items()
                if isinstance(row[col],str) and row[col].strip()}
    actual = [(r['doc_idx'],r['tipo_resumo']) for r in frame.to_dict('records')]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Extraction omitted or duplicated document/variant records')
    records=[]
    for row in frame.to_dict('records'):
        did = document_id(row['doc_idx'], row['documento_fonte'])
        row['doc_id'] = did
        annotated=[]
        for index, claim in enumerate(row['claims']):
            claim = dict(claim)
            claim['claim_id'] = f'{did}:{row["tipo_resumo"]}:C{index+1:05d}'
            sent_index = claim.get('sentence_index', claim.get('indice_sentenca'))
            claim['summary_sentence_id'] = f'{did}:{row["tipo_resumo"]}:SUM{sent_index+1:05d}'
            annotated.append(claim)
        row['claims'] = annotated
        records.append(row)
    pd.DataFrame(records).to_parquet(output_path,index=False,engine='pyarrow')


def extraction_claims(path):
    rows=[]
    for record in validate_extraction(path).to_dict('records'):
        did = record.get('doc_id') or document_id(record['doc_idx'],record['documento_fonte'])
        for index, claim in enumerate(record['claims']):
            rows.append({'doc_idx':record['doc_idx'], 'tipo_resumo':record['tipo_resumo'],
                         'doc_id':did, 'claim':(claim.get('claim_original') or claim.get('texto')).strip(),
                         'claim_id':claim.get('claim_id') or f'{did}:{record["tipo_resumo"]}:C{index+1:05d}',
                         'summary_sentence_id':claim.get('summary_sentence_id'),
                         'generator_label':claim.get('gabarito_injetado','none'),
                         'is_gold':False})
    return rows


def _key(row):
    return row['doc_idx'],row['tipo_resumo'],row['claim']


def add_retrieval_ids(input_path, output_path):
    import pandas as pd
    expected = extraction_claims(input_path)
    frame = validate_retrieval(output_path)
    records=frame.to_dict('records')
    if [_key(r) for r in records] != [_key(r) for r in expected]:
        raise ValueError('Retrieval changed claim order, text or count')
    for row, original in zip(records,expected):
        for key in ('claim_id','doc_id','summary_sentence_id'):
            row[key] = original[key]
        for category in ('top_sentences','top_paragraphs'):
            passages=[]
            for item in row[category]:
                item=dict(item)
                if category == 'top_sentences':
                    item['sentence_id']=f'{row["doc_id"]}:S{item["sentence_index"]+1:05d}'
                elif 'start_sentence_index' in item:
                    item['window_id']=f'{row["doc_id"]}:W{item["start_sentence_index"]+1:05d}-{item["end_sentence_index"]:05d}'
                passages.append(item)
            row[category]=passages
    pd.DataFrame(records).to_parquet(output_path,index=False,engine='pyarrow')


def validate_retrieval(path):
    frame = _read(path, ('doc_idx','tipo_resumo','claim','documento_fonte','top_sentences','top_paragraphs'))
    for row in frame.to_dict('records'):
        if not isinstance(row['claim'],str) or not row['claim'].strip():
            raise ValueError('Empty retrieval claim')
        for name in ('top_sentences','top_paragraphs'):
            if row[name] is None:
                raise ValueError(f'{name} must be a list')
            for item in row[name]:
                if not isinstance(item,dict) or not isinstance(item.get('text'),str):
                    raise ValueError(f'Malformed {name} passage')
    return frame


def carry_claim_ids(input_path, output_path):
    import pandas as pd
    original = _read(input_path, ('doc_idx','tipo_resumo','claim')).to_dict('records')
    frame = validate_nli(output_path)
    rows=frame.to_dict('records')
    if [_key(r) for r in original] != [_key(r) for r in rows]:
        raise ValueError('NLI changed claim order, text or count')
    # Historic inputs have no claim_id: create occurrence IDs, never a text-only key.
    seen=Counter()
    for source,row in zip(original,rows):
        did=source.get('doc_id') or document_id(source['doc_idx'],source['documento_fonte'])
        seen[(did,row['tipo_resumo'])]+=1
        row['doc_id']=did
        row['claim_id']=source.get('claim_id') or f'{did}:{row["tipo_resumo"]}:C{seen[(did,row["tipo_resumo"])]:05d}'
        row['summary_sentence_id']=source.get('summary_sentence_id')
        index=row.get('source_sentence_index',-1)
        row['source_sentence_id']=f'{did}:S{index+1:05d}' if index >= 0 else None
    if len({r['claim_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate claim IDs after NLI')
    pd.DataFrame(rows).to_parquet(output_path,index=False,engine='pyarrow')


def validate_nli(path):
    frame = _read(path, ('doc_idx','tipo_resumo','claim','status_factual_nli','source_passage','support_score','contradiction_score'))
    if not set(frame['status_factual_nli']) <= {'SUPPORTED','CONTRADICTED','NOT_SUPPORTED'}:
        raise ValueError('Unknown NLI status')
    return frame


def validate_taxonomy(input_path, output_path):
    source=validate_nli(input_path)
    result=_read(output_path, ('doc_idx','tipo_resumo','claim','status_factual_nli','taxonomy_label'))
    want=source[source['status_factual_nli'].eq('CONTRADICTED')]
    if Counter(map(_key,want.to_dict('records'))) != Counter(map(_key,result.to_dict('records'))):
        raise ValueError('Taxonomy must preserve exactly the contradicted subset')
    if not result['status_factual_nli'].eq('CONTRADICTED').all():
        raise ValueError('Taxonomy altered factual status')
    if not set(result['taxonomy_label']) <= {'coreference_error','entity_error','predicate_error','other'}:
        raise ValueError('Invalid taxonomy label')
    if 'claim_id' in want and list(want['claim_id']) != list(result['claim_id']):
        raise ValueError('Taxonomy changed claim IDs')
    return result
