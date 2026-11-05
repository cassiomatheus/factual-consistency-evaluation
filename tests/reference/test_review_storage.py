"""Disk persistence and compatibility; no sockets, browser, models or GPU needed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from _support import ROOT
from factual_eval.reference.review_server import ConflictError, ReviewStore, make_server
from factual_eval.reference.html import run as html_run
from factual_eval.reference.export import validate
from reference.test_translation import fixture


def dataset():
    return {'manifest':{'dataset_id':'test'},'sample_meta':{'selection_id':'selection'},
            'claims':[{'claim_id':'c1','doc_id':'d1'}],
            'documents':[{'doc_id':'d1','sentences':[{'sentence_id':'s1'}]}]}


class ReviewStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root/'revisao.json'
        self.store = ReviewStore(self.path,dataset())
        self.addCleanup(lambda: self.store.close() if not self.store.lock_file.closed else None)
        self.review = {'schema_version':1,'dataset_id':'test','selection_id':'selection',
                       'reviewer':'Pesquisador','claims':{'c1':{'notes':'Primeira análise'}},
                       'draft_sets':{'c1':['s1']}}

    def test_persist_restore_history_and_safe_retry(self):
        first = self.store.save(self.review,None)
        self.assertEqual(json.loads(self.path.read_text()),self.review)
        self.assertEqual(self.store.save(self.review,None)['version'],first['version'])
        second = copy.deepcopy(self.review)
        second['claims']['c1']['notes'] = 'Segunda análise'
        self.store.save(second,first['version'])
        history = list((self.root/'revisao.historico').glob('*.json'))
        self.assertEqual(len(history),1)
        self.assertEqual(json.loads(history[0].read_text()),self.review)
        self.store.close()
        reopened = ReviewStore(self.path,dataset())
        try:
            self.assertEqual(reopened.read()['review'],second)
        finally:
            reopened.close()

    def test_conflict_never_overwrites_and_second_process_is_locked_out(self):
        first = self.store.save(self.review,None)
        other = copy.deepcopy(self.review)
        other['reviewer'] = 'Outra aba'
        with self.assertRaises(ConflictError):
            self.store.save(other,None)
        self.assertEqual(self.store.read()['version'],first['version'])
        with self.assertRaises(OSError):
            ReviewStore(self.path,dataset())

    def test_failed_atomic_replace_preserves_last_valid_revision(self):
        first = self.store.save(self.review,None)
        updated = copy.deepcopy(self.review)
        updated['reviewer'] = 'Modificação'
        with patch('factual_eval.reference.review_server.os.replace',side_effect=OSError('Disco indisponível')):
            with self.assertRaises(OSError):
                self.store.save(updated,first['version'])
        self.assertEqual(self.store.read()['review'],self.review)
        self.assertEqual(list(self.root.glob('.revisao.json*')),[])

    def test_rejects_wrong_dataset_selection_ids_and_corrupted_existing_file(self):
        for field,value in [('dataset_id','outro'),('selection_id','outra'),('claims',{'unknown':{}}),
                            ('draft_sets',{'c1':['other-document-sentence']})]:
            with self.subTest(field=field):
                review = {**self.review,field:value}
                with self.assertRaises(ValueError):
                    self.store.save(review,None)
        self.store.close()
        self.path.write_text('arquivo corrompido')
        with self.assertRaises(ValueError):
            ReviewStore(self.path,dataset())
        self.assertEqual(self.path.read_text(),'arquivo corrompido')

    def test_backup_local_revision_without_changing_current_file(self):
        first = self.store.save(self.review,None)
        local = {**self.review,'reviewer':'Anotação local diferente'}
        result = self.store.backup(local)
        self.assertEqual(json.loads(Path(result['path']).read_text()),local)
        self.assertEqual(self.store.read()['version'],first['version'])

    def test_existing_html_stays_unchanged_and_disk_json_exports_as_gold(self):
        raw = fixture(self.root/'raw')
        page = html_run(raw,self.root/'review.html')
        before = page.read_bytes()
        with patch('factual_eval.reference.review_server.ThreadingHTTPServer'):
            server = make_server(page,self.root/'real-review.json')
        try:
            content = server.audit_page.decode()
            self.assertIn('AUDIT_DISK_CONFIG',content)
            self.assertIn('window.auditDisk',content)
            self.assertEqual(page.read_bytes(),before)
            cid = 'doc1:original:C00001'
            review = {'schema_version':1,'dataset_id':'fixture','reviewer':'Teste',
                'claims':{cid:{'claim_valid':'SIM','factual_label':'SUPPORTED',
                'full_document_checked':True,'search_complete':True,'review_status':'FINALIZADO',
                'gold_evidence_sets':[['doc1:S00001']]}},'draft_sets':{cid:['doc1:S00002']}}
            server.store.save(review,None)
            report = validate(raw,server.store.path,self.root/'gold')
            self.assertEqual(report['claims_gold_validated'],1)
        finally:
            server.store.close()


if __name__=='__main__':
    unittest.main()
