"""Technical reversibility audit. This never establishes factual contradiction."""
import ast
from collections import Counter
import re

KINDS = {
    'predicate':('resumo_predicado','tokens_predicado','qtd_erros_predicado'),
    'coreference':('resumo_corref','tokens_corref','qtd_erros_corref'),
    'entity':('resumo_entity','tokens_entity','qtd_erros_entity'),
}


def changes_list(value):
    if isinstance(value,str):
        value=ast.literal_eval(value)
    if not isinstance(value,list) or any(not isinstance(x,dict) for x in value):
        raise ValueError('Change log must be a list of mappings')
    return value


def inverse_edits(text, changes, kind):
    """Pair markers with ordered logs, preserving repeated replacement strings."""
    markers=list(re.finditer(r' \[ERROR_'+kind.upper()+r':\s*([^\]]*)\]',text))
    if len(markers)!=len(changes):
        raise ValueError('Marker count differs from change log (legacy formats require a separate adapter)')
    edits=[]
    displacement=0
    for marker,change in zip(markers,changes):
        new=change.get('token_injetado',change.get('expressao_injetada'))
        old=change.get('token_original',change.get('expressao_original'))
        if not isinstance(new,str) or not isinstance(old,str):
            raise ValueError('Missing original/injected expression')
        start=marker.start()-len(new)
        if marker.group(1)!=new or text[start:marker.start()]!=new:
            raise ValueError('Marker order/content does not match the log')
        edits.append({'mutated_start':start,'mutated_end':marker.end(),
                      'original_start':start-displacement,'original_end':start-displacement+len(old),
                      'original':old,'injected':new})
        displacement += marker.end()-start-len(old)
    return edits


def reconstruct_original(text, changes, kind):
    for edit in reversed(inverse_edits(text,changes,kind)):
        text=text[:edit['mutated_start']]+edit['original']+text[edit['mutated_end']:]
    return text


def audit_dataframe(frame):
    records=[]
    counts=Counter()
    for index,row in frame.iterrows():
        for kind,(textcol,logcol,countcol) in KINDS.items():
            errors=[]
            try:
                changes=changes_list(row[logcol])
                if len(changes)!=int(row[countcol]):
                    errors.append('change_count_mismatch')
                sentences=[c['sentenca_index'] for c in changes]
                if len(set(sentences))!=len(sentences):
                    errors.append('multiple_changes_in_sentence')
                restored=reconstruct_original(row[textcol],changes,kind)
                if restored!=row['resumo_original']:
                    errors.append('original_not_exactly_restored')
            except (ValueError,TypeError,KeyError,SyntaxError) as exc:
                errors.append(str(exc))
            counts['checked_variants']+=1
            counts['technical_pass' if not errors else 'technical_fail']+=1
            records.append({'source_index':int(row.get('source_index',index)),
                            'injection_type':kind,'technical_valid':not errors,
                            'errors':errors,'factual_validation':'PENDING','is_gold':False})
    return {'scope':'TECHNICAL_REVERSIBILITY_ONLY','counts':dict(counts),'records':records,
            'warning':'Technical success does not establish contradiction, reference validity or human gold.'}
