"""Verifica desenho amostral, grupo de controle e impossibilidade de gold automático."""
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from _support import ROOT

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

prep=load(ROOT/'src/factual_eval/reference/preparation.py','prep_sel')
select=load(ROOT/'src/factual_eval/reference/sampling.py','select_mod')
html=load(ROOT/'src/factual_eval/reference/html.py','html_sel')
validate=load(ROOT/'src/factual_eval/reference/export.py','gold_sel')

class SelectiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.raw=self.root/'raw'
        args=Namespace(entrada=str(ROOT/'tests/fixtures/claims_sinteticas.json'),
            saida=str(self.raw),variantes='',campo_claim='claim_original',top_k=3,
            dense_pool=20,bm25_pool=20,rerank_pool=35,max_registros=0,
            max_claims=0,backend='tfidf',embedding_model='unused',reranker_model='unused',
            device='cpu',embedding_batch=16,reranker_batch=16)
        prep.run(args)
        self.meta,self.rows=select.select(self.raw,3,(.34,.33,.33),42)
        self.claims=validate.read_jsonl(self.raw/'claims.jsonl')
        self.sentences=validate.read_jsonl(self.raw/'sentencas.jsonl')
        self.manifest=json.loads((self.raw/'manifest.json').read_text())
    def tearDown(self):
        self.temp.cleanup()

    def test_selection_groups_disjoint_and_deterministic(self):
        self.assertEqual(len(self.rows),4)
        self.assertEqual(sum(r['selected'] for r in self.rows),3)
        self.assertEqual(sum(r['selection_group']=='A_ALEATORIA' for r in self.rows),1)
        self.assertEqual(sum(r['selection_group']=='B_INCERTA' for r in self.rows),1)
        self.assertEqual(sum(r['selection_group']=='C_ESTRUTURAL' for r in self.rows),1)
        self.assertAlmostEqual(self.meta['group_a_inclusion_probability'],.25)
        again,_=select.select(self.raw,3,(.34,.33,.33),42)
        self.assertEqual(again['selection_id'],self.meta['selection_id'])
        record=self.rows[0]
        self.assertNotIn('gabarito_injetado',record)
        self.assertNotIn('palavra_injetada',record)
        self.assertIsInstance(record['priority_score'],float)

    def test_factual_disagreement_optional(self):
        cid=self.claims[0]['claim_id']
        pred=self.root/'modelos.jsonl'
        pred.write_text('\n'.join([
            json.dumps({'claim_id':cid,'model':'deberta','label':'SUPPORTED','probabilities':{'SUPPORTED':.6,'CONTRADICTED':.2,'NEUTRAL':.2}}),
            json.dumps({'claim_id':cid,'model':'minicheck','label':'UNSUPPORTED'})])+'\n')
        _,rows=select.select(self.raw,3,(.34,.33,.33),42,str(pred))
        item=next(r for r in rows if r['claim_id']==cid)
        self.assertEqual(item['discordancia_modelos'],1.0)
        self.assertTrue(item['priority_score']>=0)

    def test_html_shows_new_modes_and_selection(self):
        path=html.run(self.raw,self.root/'review.html')
        content=path.read_text()
        self.assertIn('Triagem rápida',content)
        self.assertIn('groupFilter',content)
        self.assertIn('sample_meta',content)
        self.assertNotIn('gabarito_injetado',content)
        self.assertNotIn('ERROR_ENTITY:',content)

    def test_triage_never_creates_gold_and_outside_excluded(self):
        chosen=next(r for r in self.rows if r['selected'])
        outside=next(r for r in self.rows if not r['selected'])
        sid=next(s['sentence_id'] for s in self.sentences if s['doc_id']==chosen['doc_id'])
        out_sid=next(s['sentence_id'] for s in self.sentences if s['doc_id']==outside['doc_id'])
        ann={'schema_version':1,'dataset_id':self.manifest['dataset_id'],
             'selection_id':self.meta['selection_id'],'reviewer':'TEST',
             'claims':{chosen['claim_id']:{'triage_status':'SUPPORT_CANDIDATE',
                                          'triage_sentence_id':sid,
                                          'candidate_annotations':{sid:{'relevance':'SIM','relation':'SUPPORTS','sufficient':'SIM'}},
                                          'gold_evidence_sets':[]}}}
        path=self.root/'triage.json'
        path.write_text(json.dumps(ann))
        report=validate.validate(self.raw,path,self.root/'output0')
        self.assertEqual(report['claims_gold_validated'],0)
        self.assertEqual(report['triage_only_or_preliminary'],1)
        self.assertEqual(len(validate.read_jsonl(self.root/'output0'/'gold_claims.jsonl')),0)
        finished=lambda sid:{'review_status':'FINALIZADO','claim_valid':'SIM',
                 'full_document_checked':True,'search_complete':True,
                 'factual_label':'SUPPORTED','gold_evidence_sets':[[sid]]}
        ann['claims'][chosen['claim_id']].update(finished(sid))
        ann['claims'][outside['claim_id']]=finished(out_sid)
        path.write_text(json.dumps(ann))
        report=validate.validate(self.raw,path,self.root/'output1')
        gold=validate.read_jsonl(self.root/'output1'/'gold_claims.jsonl')
        self.assertEqual(len(gold),1)
        self.assertEqual(gold[0]['claim_id'],chosen['claim_id'])
        self.assertEqual(gold[0]['selection_group'],chosen['selection_group'])
        self.assertTrue(report['triage_only_or_preliminary']==1)
        report_out=validate.validate(self.raw,path,self.root/'output2',include_outside_sample=True)
        self.assertEqual(report_out['claims_gold_validated'],2)

    def test_selection_id_must_match_when_supplied(self):
        path=self.root/'bad.json'
        path.write_text(json.dumps({'schema_version':1,'dataset_id':self.manifest['dataset_id'],
                                    'selection_id':'invalid','claims':{}}))
        with self.assertRaises(ValueError):
            validate.validate(self.raw,path,self.root/'bad_gold')

if __name__=='__main__':unittest.main()
