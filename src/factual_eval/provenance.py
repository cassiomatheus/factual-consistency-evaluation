"""Record actual model metadata, without downloading models for inspection."""
from importlib import metadata

MODELS = []


def record_model(role, identifier, model, requested_revision=None):
    config = getattr(model, 'config', None)
    if config is None:
        config = getattr(getattr(model, 'model', None), 'config', None)
    if config is None and hasattr(model, '_first_module'):
        config = getattr(getattr(model._first_module(), 'auto_model', None), 'config', None)
    row = {'role': role, 'identifier': identifier,
           'requested_revision': requested_revision,
           'resolved_revision': getattr(config, '_commit_hash', None),
           'id2label': getattr(config, 'id2label', None),
           'package_model_version':getattr(model,'meta',{}).get('version')}
    MODELS.append(row)


def environment_versions():
    import platform
    import os
    names = ('torch','transformers','sentence-transformers','spacy','en-core-web-sm',
             'en-core-web-md','datasets','lemminflect','sentencepiece','numpy','pandas',
             'pyarrow','tqdm','scikit-learn','PyYAML','ollama','pydantic')
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    result={'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions,
            'pythonhashseed':os.environ.get('PYTHONHASHSEED')}
    try:
        import torch
        available=torch.cuda.is_available()
        result['torch_runtime']={'cuda_compiled':torch.version.cuda,'cuda_available':available,
            'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if available else [],
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
    except ImportError:
        result['torch_runtime']=None
    return result
