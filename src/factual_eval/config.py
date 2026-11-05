"""Configuration and paths, independent of the execution directory."""
from copy import deepcopy
from pathlib import Path
import math
import os

_source_root = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('FACTUAL_EVAL_ROOT', _source_root if (_source_root/'pyproject.toml').exists() else Path.cwd())).resolve()


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _merge(base, override, trail=''):
    for key, value in override.items():
        if key not in base:
            raise ValueError(f'Unknown configuration key: {trail}{key}')
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f'Expected a mapping: {trail}{key}')
            _merge(base[key], value, f'{trail}{key}.')
        else:
            base[key] = value
    return base


def load_config(path=None, reference=False):
    import yaml
    default = ROOT / 'configs' / ('reference.yaml' if reference else 'baseline.yaml')
    if not default.is_file():
        default = Path(__file__).resolve().parent / 'resources' / default.name
    config = yaml.safe_load(default.read_text())
    if path:
        override = yaml.safe_load(project_path(path).read_text())
        if not isinstance(override, dict):
            raise ValueError('Configuration must be a YAML mapping')
        config = _merge(deepcopy(config), override)
    if config['schema_version'] != 1:
        raise ValueError('Unsupported configuration schema_version')
    if reference:
        p = config['preparation']
        if p['top_k'] not in (3, 5) or min(p['dense_pool'], p['bm25_pool'], p['rerank_pool']) < p['top_k']:
            raise ValueError('Reference pools must cover top_k (3 or 5)')
        if p['backend'] not in ('hf', 'tfidf'):
            raise ValueError('Reference backend must be hf or tfidf')
        s = config['sampling']
        fractions = [s['fraction_a'], s['fraction_b'], s['fraction_c']]
        if s['sample_size'] < 1 or any(x < 0 for x in fractions) or abs(sum(fractions) - 1) > 1e-6:
            raise ValueError('Invalid reference sample or fractions')
    else:
        if config['claims']['variant'] not in ('baseline', 'enriched'):
            raise ValueError('Unknown claims variant')
        for section in ('injection', 'claims', 'retrieval', 'nli', 'taxonomy'):
            for name, value in config[section].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                    raise ValueError(f'Non-finite parameter: {section}.{name}')
        r = config['retrieval']
        for key in ('sentence_top_k','window_top_k','semantic_pool','lexical_pool','max_coref_variants'):
            if not isinstance(r[key], int) or isinstance(r[key], bool) or r[key] < 1:
                raise ValueError(f'retrieval.{key} must be a positive integer')
        if not isinstance(r['context_radius'], int) or r['context_radius'] < 0:
            raise ValueError('context_radius must be a nonnegative integer')
        for section, keys in {'claims':('batch_size','max_input_length','max_output_length','num_beams'), 'nli':('batch_size','max_length')}.items():
            if any(not isinstance(config[section][key], int) or isinstance(config[section][key], bool) or config[section][key] < 1 for key in keys):
                raise ValueError(f'{section}: lengths and batches must be positive integers')
    return config


def execution_environment(config):
    import os
    threads = config['runtime']['threads']
    if not isinstance(threads, int) or threads < 1:
        raise ValueError('runtime.threads must be a positive integer')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    import torch
    torch.set_num_threads(threads)


def device_for(config, stage):
    """Keep explicit CUDA for extraction/NLI and original auto choice elsewhere."""
    import torch
    value = config['runtime']['device']
    if stage in ('retrieval', 'taxonomy') and value == 'cuda:0':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if value == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if str(value).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Use --device cpu explicitly for a CPU experiment.')
    return value
