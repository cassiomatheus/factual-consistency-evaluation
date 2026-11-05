"""Testes de contrato, sem baixar modelos de linguagem."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _support import ROOT


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep=load(ROOT/'src/factual_eval/reference/preparation.py','prep')
html=load(ROOT/'src/factual_eval/reference/html.py','html')
validator=load(ROOT/'src/factual_eval/reference/export.py','validator')


class ReferenceDeviceTests(unittest.TestCase):
    def test_models_receive_resolved_or_explicit_device(self):
        for requested, cuda_available, expected in (
            ('auto', False, 'cpu'),
            ('auto', True, 'cuda'),
            ('cpu', True, 'cpu'),
            ('cuda:0', True, 'cuda:0'),
        ):
            with self.subTest(device=requested, cuda_available=cuda_available), \
                 patch('torch.cuda.is_available', return_value=cuda_available), \
                 patch('sentence_transformers.SentenceTransformer') as encoder, \
                 patch('sentence_transformers.CrossEncoder') as reranker, \
                 patch('factual_eval.provenance.record_model'):
                prep.get_dense_and_reranker(
                    'hf', requested, 'embedding', 'reranker', 'embedding-rev', 'reranker-rev'
                )
                encoder.assert_called_once_with(
                    'embedding', device=expected, revision='embedding-rev'
                )
                reranker.assert_called_once_with(
                    'reranker', device=expected, max_length=384, revision='reranker-rev'
                )


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        folder=Path(cls.tmp.name)
        from argparse import Namespace
        cls.raw=folder/'raw'
        args=Namespace(entrada=str(ROOT/'tests/fixtures/claims_sinteticas.json'),
           saida=str(cls.raw),variantes='',campo_claim='claim_original',
           top_k=3,dense_pool=20,bm25_pool=20,rerank_pool=35,max_registros=0,
           max_claims=0,backend='tfidf',embedding_model='unused',reranker_model='unused',
           device='cpu',embedding_batch=16,reranker_batch=16)
        prep.run(args)
        cls.html=html.run(cls.raw,folder/'review.html')
        cls.manifest=json.loads((cls.raw/'manifest.json').read_text())
        cls.claims=validator.read_jsonl(cls.raw/'claims.jsonl')
        cls.sentences=validator.read_jsonl(cls.raw/'sentencas.jsonl')
        cls.folder=folder

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_contract_and_stability(self):
        self.assertEqual(self.manifest['number_claims'],4)
        self.assertEqual(self.manifest['number_documents'],2)
        self.assertEqual(len(self.claims),4)
        self.assertEqual(len(set(c['claim_id'] for c in self.claims)),4)
        docs=validator.read_jsonl(self.raw/'documentos.jsonl')
        indexed={d['doc_id']:d for d in docs}
        for s in self.sentences:
            d=indexed[s['doc_id']]
            self.assertEqual(s['text'],d['source_text'][s['char_start']:s['char_end']])
        self.assertNotIn('gabarito_injetado',self.html.read_text())
        self.assertIn('CrossEncoder',self.html.read_text())

    def test_rank_and_no_factual_inference(self):
        rows=validator.read_jsonl(self.raw/'candidatos.jsonl')
        self.assertTrue(rows)
        for r in rows:
            self.assertIsNone(r['score_cross_encoder_logit'])
            self.assertTrue(-1<=r['score_cosine']<=1)
            self.assertIn(r['ranking_method'],('rrf_tfidf_demo',))
            self.assertNotIn('gold_label',r)

    def test_gold_only_valid_human(self):
        cs=self.claims
        ss={docid:[s for s in self.sentences if s['doc_id']==docid] for docid in set(c['doc_id'] for c in cs)}
        marks={}
        for i,c in enumerate(cs):
            marks[c['claim_id']]={
                'review_status':'FINALIZADO','claim_valid':'SIM',
                'full_document_checked':True,'search_complete':True,
                'factual_label':'SUPPORTED','gold_evidence_sets':[[ss[c['doc_id']][0]['sentence_id']]],
                'candidate_annotations':{}
            }
        # Primeiro registro não possui evidência: deve ser excluído.
        marks[cs[0]['claim_id']]['gold_evidence_sets']=[]
        # Segundo registro não revisou documento completo: excluído.
        marks[cs[1]['claim_id']]['full_document_checked']=False
        ann={'schema_version':1,'dataset_id':self.manifest['dataset_id'],
             'claims':marks,'reviewer':'TEST'}
        target=self.folder/'annotations.json'
        target.write_text(json.dumps(ann))
        report=validator.validate(self.raw,target,self.folder/'validated')
        self.assertEqual(report['claims_gold_validated'],2)
        self.assertEqual(report['claims_pending'],2)

    def test_refuses_other_dataset(self):
        target=self.folder/'incorrect.json'
        target.write_text(json.dumps({'schema_version':1,'dataset_id':'wrong','claims':{}}))
        with self.assertRaises(ValueError):
            validator.validate(self.raw,target,self.folder/'invalid')


if __name__=='__main__': unittest.main()
