"""Loopback-only audit UI with atomic JSON autosave and versioned local backups."""
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
from urllib.parse import urlsplit
import uuid


class ConflictError(ValueError):
    pass


def encoded(data):
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def read_page(path):
    source = Path(path).read_text(encoding='utf-8')
    match = re.search(r'<script\b[^>]*\bid=["\']dataset["\'][^>]*>(.*?)</script\s*>', source, re.S)
    if not match:
        raise ValueError('HTML sem dataset de auditoria incorporado')
    return match.group(1), json.loads(match.group(1))


class ReviewStore:
    def __init__(self, path, dataset):
        self.path = Path(path).resolve()
        if self.path.suffix != '.json':
            raise ValueError('O arquivo de revisão deve ter extensão .json')
        self.dataset_id = dataset['manifest']['dataset_id']
        self.selection_id = (dataset.get('sample_meta') or {}).get('selection_id')
        self.claim_docs = {c['claim_id']: c['doc_id'] for c in dataset['claims']}
        self.sentence_docs = {s['sentence_id']: d['doc_id'] for d in dataset['documents'] for s in d['sentences']}
        self.mutex = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # OS releases this lock on process exit, including an interrupted terminal.
        import fcntl
        self.lock_file = self.path.with_suffix('.lock').open('a')
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.read()
        except BaseException:
            self.lock_file.close()
            raise

    def close(self):
        self.lock_file.close()

    def validate(self, review):
        if not isinstance(review, dict) or review.get('schema_version') != 1 or review.get('dataset_id') != self.dataset_id:
            raise ValueError('Revisão de outro dataset ou esquema incompatível')
        if review.get('selection_id') not in (None, self.selection_id):
            raise ValueError('Revisão de outro plano amostral')
        claims = review.get('claims')
        if not isinstance(claims, dict) or set(claims) - self.claim_docs.keys():
            raise ValueError('Claims desconhecidas ou estrutura inválida')
        if not isinstance(review.get('reviewer', ''), str):
            raise ValueError('Identificação do revisor inválida')
        def sentence_ids(cid, values):
            if not isinstance(values, list) or any(not isinstance(s, str) or self.sentence_docs.get(s) != self.claim_docs[cid] for s in values):
                raise ValueError('Evidência inexistente ou associada a outro documento')
        for cid, annotation in claims.items():
            if not isinstance(annotation, dict):
                raise ValueError('Anotação inválida')
            sentence_ids(cid, annotation.get('custom_sentence_ids', []))
            sets = annotation.get('gold_evidence_sets', [])
            if not isinstance(sets, list):
                raise ValueError('Conjuntos de evidências inválidos')
            for subset in sets:
                sentence_ids(cid, subset)
            candidates = annotation.get('candidate_annotations', {})
            if not isinstance(candidates, dict) or any(not isinstance(v, dict) for v in candidates.values()):
                raise ValueError('Julgamentos de evidências inválidos')
            sentence_ids(cid, list(candidates))
            if annotation.get('triage_sentence_id'):
                sentence_ids(cid, [annotation['triage_sentence_id']])
        drafts = review.get('draft_sets', {})
        if not isinstance(drafts, dict) or set(drafts) - self.claim_docs.keys():
            raise ValueError('Rascunhos de evidências inválidos')
        for cid, values in drafts.items():
            sentence_ids(cid, values)

    def read(self):
        if not self.path.exists():
            return {'review': None, 'version': None, 'path': str(self.path)}
        data = self.path.read_bytes()
        review = json.loads(data)
        self.validate(review)
        return {'review': review, 'version': sha256(data).hexdigest(), 'path': str(self.path)}

    @staticmethod
    def atomic_write(path, data):
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.'+path.name, delete=False) as handle:
                temp_name = handle.name
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    def save(self, review, expected_version):
        self.validate(review)
        data = encoded(review)
        with self.mutex:
            previous = self.read()
            if previous['review'] is not None and encoded(previous['review']) == data:
                return previous  # Safe retry after a lost HTTP acknowledgement.
            if previous['version'] != expected_version:
                raise ConflictError('O arquivo mudou em outra aba. Exporte seu JSON antes de recarregar.')
            if previous['review'] is not None:
                history = self.path.parent/(self.path.stem+'.historico')
                history.mkdir(exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                self.atomic_write(history/f'{stamp}_{uuid.uuid4().hex[:8]}.json', encoded(previous['review']))
            self.atomic_write(self.path, data)
            return self.read()

    def backup(self, review):
        self.validate(review)
        with self.mutex:
            history = self.path.parent/(self.path.stem+'.historico')
            history.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            path = history/f'{stamp}_navegador_{uuid.uuid4().hex[:8]}.json'
            self.atomic_write(path, encoded(review))
            return {'path': str(path)}


def make_server(html_path, annotations_path=None, port=8765):
    raw, dataset = read_page(html_path)
    if annotations_path is None:
        from factual_eval.config import project_path
        identity = [dataset['manifest']['dataset_id'], (dataset.get('sample_meta') or {}).get('selection_id')]
        name = sha256(json.dumps(identity).encode()).hexdigest()[:20]
        annotations_path = project_path('outputs/annotations')/name/'revisao.json'
    store = ReviewStore(annotations_path, dataset)
    token = secrets.token_urlsafe(32)
    config = json.dumps({'token': token, 'path': str(store.path)}, ensure_ascii=False).replace('<', '\\u003c')
    here = Path(__file__).resolve().parent
    template = (here/'modelo.html').read_text(encoding='utf-8')
    script = (here/'disk_autosave.js').read_text(encoding='utf-8')
    injected = f'<script>window.AUDIT_DISK_CONFIG={config};\n{script}</script>\n'
    page = template.replace('__DATASET_JSON__', raw).replace("<script>\n'use strict';", injected+"<script>\n'use strict';", 1).encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def respond(self, status, data, kind='application/json; charset=utf-8'):
            body = encoded(data) if isinstance(data, dict) else data
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy', "frame-ancestors 'none'; connect-src 'self'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Closing a tab must not obscure a completed disk write.

        def allowed(self, api=False):
            expected = f'127.0.0.1:{self.server.server_port}'
            origin = self.headers.get('Origin')
            valid = self.headers.get('Host') == expected and origin in (None, 'http://'+expected)
            if api:
                valid = valid and secrets.compare_digest(self.headers.get('X-Audit-Token', ''), token)
            if not valid:
                self.respond(403, {'error': 'Acesso permitido somente pela auditoria local'})
            return valid

        def do_GET(self):
            route = urlsplit(self.path).path
            if not self.allowed(api=route == '/api/review'):
                return
            if route in ('/', '/avaliacao.html'):
                self.respond(200, page, 'text/html; charset=utf-8')
            elif route == '/api/review':
                try:
                    with store.mutex:
                        self.respond(200, store.read())
                except (OSError, ValueError) as exc:
                    self.respond(500, {'error': str(exc)})
            else:
                self.respond(404, {'error': 'Recurso não encontrado'})

        def do_PUT(self):
            if not self.allowed(api=True):
                return
            if self.path != '/api/review':
                self.respond(404, {'error': 'Recurso não encontrado'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if self.headers.get('Content-Type') != 'application/json' or not 0 < length <= 8*1024*1024:
                    raise ValueError('JSON inválido ou revisão maior que 8 MiB')
                def reject_constant(value):
                    raise ValueError('Número JSON inválido: '+value)
                request = json.loads(self.rfile.read(length), parse_constant=reject_constant)
                result = (store.backup(request['review']) if request.get('backup_only') is True
                          else store.save(request['review'], request['version']))
                print('Revisão salva em arquivo:', store.path, flush=True)
                self.respond(200, result)
            except ConflictError as exc:
                self.respond(409, {'error': str(exc)})
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {'error': str(exc)})
            except OSError as exc:
                self.respond(500, {'error': 'Não foi possível gravar em disco: '+str(exc)})

        def log_message(self, format, *args):
            # Do not log annotation contents or API tokens.
            if args and str(args[0]).startswith('PUT'):
                return

    try:
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    except BaseException:
        store.close()
        raise
    server.store = store
    server.audit_page = page
    server.daemon_threads = False  # Finish accepted writes before releasing the file lock.
    return server
