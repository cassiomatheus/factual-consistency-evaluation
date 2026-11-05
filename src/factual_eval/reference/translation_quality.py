"""Conservative translation warnings, never factual decisions or corrections."""
import re
import unicodedata
from collections import Counter

QUALITY_VERSION = 'govreport-qa-v1'


def plain(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                   if unicodedata.category(c) != 'Mn')


_WORDS = {
    'zero':0, 'one':1, 'um':1, 'uma':1, 'two':2, 'dois':2, 'duas':2,
    'three':3, 'tres':3, 'four':4, 'quatro':4, 'five':5, 'cinco':5,
    'six':6, 'seis':6, 'seven':7, 'sete':7, 'eight':8, 'oito':8,
    'nine':9, 'nove':9, 'ten':10, 'dez':10, 'eleven':11, 'onze':11,
    'twelve':12, 'doze':12, 'thirteen':13, 'treze':13,
    'fourteen':14, 'catorze':14, 'quatorze':14, 'fifteen':15, 'quinze':15,
    'sixteen':16, 'dezesseis':16, 'seventeen':17, 'dezessete':17,
    'eighteen':18, 'dezoito':18, 'nineteen':19, 'dezenove':19,
    'twenty':20, 'vinte':20, 'thirty':30, 'trinta':30, 'forty':40,
    'quarenta':40, 'fifty':50, 'cinquenta':50, 'sixty':60, 'sessenta':60,
    'seventy':70, 'setenta':70, 'eighty':80, 'oitenta':80,
    'ninety':90, 'noventa':90, 'hundred':100, 'cem':100, 'cento':100,
}
_SCALES = {'thousand':1000, 'mil':1000, 'million':1000000,
           'milhao':1000000, 'milhoes':1000000, 'billion':1000000000,
           'bilhao':1000000000, 'bilhoes':1000000000}


def numbers(text, portuguese=False):
    """Common cardinal words, locale separators and month names; heuristic only."""
    text = plain(text)
    text = re.sub(r'\bpor cento\b', 'percentual', text)
    months = [('january','janeiro'),('february','fevereiro'),('march','marco'),
              ('april','abril'),('may','maio'),('june','junho'),('july','julho'),
              ('august','agosto'),('september','setembro'),('october','outubro'),
              ('november','novembro'),('december','dezembro')]
    result = Counter()
    for i, names in enumerate(months, 1):
        # English "may" is also a modal: require an adjacent day/year.
        for name in names:
            pattern = rf'\b{name}\b'
            if name == 'may':
                pattern = r'\bmay\b(?=\s+\d)'
            hits = len(re.findall(pattern, text))
            if hits:
                result[f'month:{i}'] += hits
    tokens = re.findall(r'\d+(?:[.,]\d+)*|[a-z]+', text)
    total = 0
    active = False
    for token in tokens + ['END']:
        if token in _WORDS:
            value = _WORDS[token]
            total = total * value if value == 100 and total else total + value
            active = True
            continue
        if token in _SCALES and active:
            total *= _SCALES[token]
            continue
        if token in ('and', 'e') and active:
            continue
        if active:
            result[str(total)] += 1
            total, active = 0, False
        if token[0].isdigit():
            # Locale-aware decimal and thousands separators.
            thousands, decimal = ('.', ',') if portuguese else (',', '.')
            value = token.replace(thousands, '').replace(decimal, '.')
            result[value.rstrip('0').rstrip('.') if '.' in value else str(int(value))] += 1
    return result


def quality_warnings(source, translated, done_reason='stop'):
    warnings = []
    if not translated.strip():
        return ['Saída vazia.']
    if source.strip() == translated.strip():
        warnings.append('Saída idêntica ao inglês: tradução possivelmente ausente.')
    if done_reason != 'stop':
        warnings.append('Tradução incompleta: geração interrompida ou limite de tokens.')
    src_numbers, dst_numbers = numbers(source), numbers(translated, portuguese=True)
    # "um/uma" are also indefinite articles. Avoid spurious number warnings.
    if '1' not in src_numbers:
        dst_numbers.pop('1', None)
    if src_numbers != dst_numbers:
        warnings.append('Possível diferença numérica ou de data.')
    en, pt = plain(source), plain(translated)
    neg_en = bool(re.search(r"\b(not|no|never|neither|without|lack|lacks|failed)\b|n['’]t\b", en))
    neg_pt = bool(re.search(r'\b(nao|nunca|jamais|nenhum|nenhuma|nem|sem|deixou de|falta)\b', pt))
    if neg_en != neg_pt:
        warnings.append('Possível mudança de negação.')
    modalities = [
        (r'\b(could|might|possibly|likely|appear|appears)\b|\bmay\b(?!\s+\d)',
         r'\b(pode|podem|poderia|poderiam|podera|poderao|talvez|possivel|provavel|provavelmente|parece|parecem)\b'),
        (r'\b(should|ought)\b', r'\b(deveria|deveriam|deve|devem|recomenda)\b'),
        (r'\b(must|required|shall)\b', r'\b(deve|devem|obrigatori|necessari|exigid|exige)\w*'),
    ]
    for source_pattern, target_pattern in modalities:
        if bool(re.search(source_pattern, en)) != bool(re.search(target_pattern, pt)):
            warnings.append('Possível mudança de modalidade ou incerteza.')
            break
    for symbol, pattern_en, pattern_pt in [
        ('percentual', r'%|\bpercent(?:age)?\b', r'%|\bpor cento\b|\bpercentual\b'),
        ('quantificador', r'\b(only|solely)\b', r'\b(apenas|somente|so|unico|unica)\b'),
    ]:
        if bool(re.search(pattern_en, en)) != bool(re.search(pattern_pt, pt)):
            warnings.append(f'Possível mudança de {symbol}.')
    acronyms = set(re.findall(r'\b[A-Z][A-Z0-9-]{1,}\b', source))
    if any(not re.search(r'\b' + re.escape(a) + r'\b', translated) for a in acronyms):
        warnings.append('Sigla divergente ou ausente.')
    if set(re.findall(r'\b[A-Z][A-Z0-9-]{1,}\b', translated)) - acronyms:
        warnings.append('Possível acréscimo de sigla ou entidade.')
    # Conservative proper-name check: hints only; official translations may trigger it.
    names = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', source))
    common = {'The','A','An','It','They','These','Those','Some','Such','Only','We','In','As',
              'However','For','Each','This','That','There','Most','According','All','No',
              'January','February','March','April','May','June','July','August','September',
              'October','November','December'}
    names.update(m.group() for m in re.finditer(r'\b[A-Z][a-z]+\b', source)
                 if m.group() not in common and plain(m.group()) not in _WORDS
                 and source[:m.start()].rstrip() and source[:m.start()].rstrip()[-1] not in '.!?')
    if any(plain(name) not in pt for name in names):
        warnings.append('Entidade identificável possivelmente divergente.')
    src_len, dst_len = len(source.split()), len(translated.split())
    if src_len >= 6 and dst_len < src_len * .45:
        warnings.append('Possível omissão de conteúdo ou sentença.')
    if dst_len > max(12, src_len * 2.3):
        warnings.append('Possível acréscimo de conteúdo.')
    if re.search(r'<think>|</think>|```|(?:^|\n)\s*(?:tradu[cç][aã]o|nota|explica[cç][aã]o)\s*:', translated, re.I):
        warnings.append('Saída contém possíveis comentários ou formatação adicional.')
    if re.search(r'[.!?]["”\']?\s*$', source) and not re.search(r'[.!?]["”\']?\s*$', translated):
        warnings.append('Saída possivelmente incompleta: pontuação final ausente.')
    return list(dict.fromkeys(warnings))
