"""Immutable run folders, stage fingerprints, logs and checked intermediate reuse."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback
import uuid

from .provenance import MODELS, environment_versions


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(path):
    path=Path(path)
    files=[path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    if not files:
        raise ValueError(f'No input files: {path}')
    return {str(p.name if path.is_file() else p.relative_to(path)):sha256(p) for p in files}


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal,self.logfile=terminal,logfile

    def write(self,text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return False


class Run:
    def __init__(self, output_root, config, *, run_id=None, reuse_from=None, command=None):
        self.root=Path(output_root).resolve()
        name=run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
        if Path(name).name!=name or name in ('.','..'):
            raise ValueError('run_id must be a single directory name')
        self.path=self.root/name
        self.path.mkdir(parents=True,exist_ok=False)
        self.config=config
        self.prior_path=Path(reuse_from).resolve() if reuse_from else None
        self.prior=None
        MODELS.clear()
        package=Path(__file__).resolve().parent
        sources={str(p.relative_to(package)):sha256(p) for p in sorted(package.rglob('*'))
                 if p.is_file() and '__pycache__' not in p.parts}
        self.meta={'schema_version':1,'run_id':name,'created_utc':datetime.now(timezone.utc).isoformat(),
                   'status':'running','command':command or sys.argv,'configuration':config,
                   'environment':environment_versions(),'source_sha256':sources,'stages':{},
                   'models':[], 'reference_policy':'Generator labels and model predictions are not independent human reference.'}
        import yaml
        (self.path/'config.resolved.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
        if self.prior_path:
            try:
                self.prior=json.loads((self.prior_path/'manifest.json').read_text())
            except Exception:
                self.meta['status']='failed'
                self.meta['error']='Cannot read reuse manifest'
                self.save()
                raise
        self.save()

    def save(self):
        self.meta['models']=list(MODELS)
        pending=self.path/'manifest.json.tmp'
        pending.write_text(json.dumps(self.meta,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        pending.replace(self.path/'manifest.json')

    def __enter__(self):
        self.log=(self.path/'execution.log').open('w',encoding='utf-8')
        self.stdout=redirect_stdout(Tee(sys.stdout,self.log))
        self.stderr=redirect_stderr(Tee(sys.stderr,self.log))
        self.stdout.__enter__()
        self.stderr.__enter__()
        return self

    def stage(self, name, input_path, operation, *, parameters=None, allow_reuse=True):
        inputs=file_manifest(input_path) if input_path else {}
        signature={'stage':name,'inputs':inputs,'config':self.config,
                   'parameters':parameters or {},'sources':self.meta['source_sha256'],
                   'environment':self.meta['environment']}
        fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        folder=self.path/name
        folder.mkdir()
        record={'status':'running','input_path':str(input_path) if input_path else None,
                'input_sha256':inputs,'parameters':parameters or {},'fingerprint':fingerprint}
        self.meta['stages'][name]=record
        self.save()
        previous=(self.prior or {}).get('stages',{}).get(name)
        if allow_reuse and previous and previous.get('status')=='complete' and previous.get('fingerprint')==fingerprint:
            for relative,expected_hash in previous['outputs'].items():
                source=(self.prior_path/relative).resolve()
                target=(self.path/relative).resolve()
                if not source.is_relative_to(self.prior_path) or not target.is_relative_to(folder):
                    raise ValueError('Unsafe artifact path in reuse manifest')
                if sha256(source)!=expected_hash:
                    raise ValueError(f'Reused artifact hash mismatch: {source}')
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
            result=self.path/previous['primary_output']
            for model in previous.get('models',[]):
                MODELS.append(dict(model))
            record['reused_from']=str(self.prior_path)
            print(f'{name}: checked artifacts reused from {self.prior_path.name}')
        else:
            if previous:
                print(f'{name}: reuse signature differs; executing stage')
            before=len(MODELS)
            result=Path(operation(folder))
            record['models']=list(MODELS[before:])
        result=Path(result).resolve()
        if not result.is_relative_to(folder) or not result.is_file():
            raise ValueError(f'{name}: operation did not return an artifact inside its run folder')
        record.update(status='complete',primary_output=str(result.relative_to(self.path)),
                      outputs={str(p.relative_to(self.path)):sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()})
        if 'reused_from' in record:
            record['models']=previous.get('models',[])
        self.save()
        return result

    def __exit__(self,kind,error,tb):
        if error:
            traceback.print_exception(kind,error,tb)
            self.meta.update(status='failed',error=f'{kind.__name__}: {error}')
            for record in self.meta['stages'].values():
                if record['status']=='running':
                    record['status']='failed'
        else:
            self.meta['status']='complete'
        self.meta['finished_utc']=datetime.now(timezone.utc).isoformat()
        self.save()
        self.stderr.__exit__(kind,error,tb)
        self.stdout.__exit__(kind,error,tb)
        self.log.close()
        return False
