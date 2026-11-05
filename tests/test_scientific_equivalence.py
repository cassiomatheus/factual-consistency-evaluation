from _support import ROOT,frozen
import json
import math
import re
from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from factual_eval.nli import evaluation as nli
from factual_eval.retrieval import pipeline as retrieval
from factual_eval.taxonomy import classification as taxonomy
from factual_eval.claims.model import ClaimExtractor
from factual_eval.nli.model import NLIAligner


class ScientificEquivalenceTests(unittest.TestCase):
    def test_historical_scores_and_decisions(self):
        original=frozen('nli',np=np)
        fixtures=json.loads((ROOT/'tests/fixtures/historical_probabilities.json').read_text())
        compatible=0
        for case in fixtures:
            actual=nli.aggregate_nli_scores(case['nli_details'])
            expected=original['aggregate_nli_scores'](case['nli_details'])
            self.assertEqual(actual,expected)
            decision=nli.decide_nli_status(actual['support_score'],actual['contradiction_score'])
            self.assertEqual(decision,original['decide_nli_status'](actual['support_score'],actual['contradiction_score']))
            if case['current_baseline_compatible']:
                compatible+=1
                self.assertEqual(actual['support_score'],case['support_score'])
                self.assertEqual(actual['contradiction_score'],case['contradiction_score'])
                self.assertEqual(decision[0],case['status'])
        self.assertEqual(compatible,18)

    def test_decision_boundaries_and_structural_flags(self):
        original=frozen('nli',np=np)
        for support in (0,0.519999999,0.52,0.54,0.639999999,0.64,0.74,1):
            for contradiction in (0,0.52,0.619999999,0.62,0.72,1):
                for structural in (False,True):
                    self.assertEqual(nli.decide_nli_status(support,contradiction,structural),
                                     original['decide_nli_status'](support,contradiction,structural))
        self.assertEqual(nli.aggregate_nli_scores([]),original['aggregate_nli_scores']([]))

    def test_bm25_queries_windows_and_mmr_match_original(self):
        original=frozen('retrieval',math=math,re=re,Counter=Counter,np=np,torch=torch,util=retrieval.util)
        passages=['The agency approved 12 reports in 2020.','The agency denied three reports.', 'US UK not approved 12.','']
        a=retrieval.BM25Index(passages);b=original['BM25Index'](passages)
        for query in ('agency 12','not US UK','absent',''):
            self.assertEqual(a.candidates(query,30),b.candidates(query,30))
            for text in passages:
                self.assertEqual(retrieval.lexical_overlap(query,text),original['lexical_overlap'](query,text))
        for index in range(len(passages)):
            self.assertEqual(retrieval.build_window(passages,index),original['build_window'](passages,index))
        ranked=[{'idx':i,'cross_logit':float(4-i)} for i in range(4)]
        embeddings=np.array([[1.,0],[1.,0],[0.,1],[0.2,0.5]])
        self.assertEqual(retrieval.mmr_select(ranked,embeddings,3),original['mmr_select'](ranked,embeddings,3))
        bounds=[(0,3),(0,3),(1,4),(4,5)]
        self.assertEqual(retrieval.select_windows(ranked,bounds,4),original['select_windows'](ranked,bounds,4))

    def test_reranker_context_combination_and_ties(self):
        original=frozen('retrieval',math=math,re=re,Counter=Counter,np=np,torch=torch,util=retrieval.util)
        class Cross:
            def predict(self,pairs,**kwargs):
                return np.array([len(p[0])*.1+len(p[1])*.01 for p in pairs])
        kwargs={'context_passages':['a b','c d'],'variants':['alternate hypothesis']}
        candidates=[{'idx':0,'retrieval_score':.4},{'idx':1,'retrieval_score':.4}]
        actual=retrieval.rerank_cross('claim',candidates,['a','c'],Cross(),**kwargs)
        expected=original['rerank_cross']('claim',candidates,['a','c'],Cross(),**kwargs)
        self.assertEqual(actual,expected)
        self.assertEqual(actual[0]['idx'],0)

    def test_taxonomy_priority_and_short_circuit(self):
        for coref,entity,predicate,want in [(True,True,True,'coreference_error'),(False,True,True,'entity_error'),(False,False,True,'predicate_error'),(False,False,False,'other')]:
            with patch.object(taxonomy,'detect_coreference_issue',return_value={'is_conflict':coref}), \
                 patch.object(taxonomy,'detect_entity_issue',return_value={'is_conflict':entity}) as e, \
                 patch.object(taxonomy,'detect_predicate_issue',return_value={'is_conflict':predicate}) as p:
                self.assertEqual(taxonomy.classify_taxonomy('x','y',{})[0],want)
                if coref:
                    e.assert_not_called();p.assert_not_called()

    def test_claim_decoder_and_generation_match_original(self):
        class Encoded(dict):
            def to(self,device):return self
        class Tokenizer:
            def __call__(self,**kwargs):
                self.arguments=kwargs
                return Encoded(input_ids=torch.tensor([[1],[2]]))
            def decode(self,value,**kwargs):
                return ['First claim<n>Second claim','Original newline\nkept'][int(value[0])]
        class Model:
            def generate(self,**kwargs):
                self.arguments=kwargs
                return torch.tensor([[0],[1]])
        original=frozen('claims',torch=torch)
        a=ClaimExtractor.__new__(ClaimExtractor)
        b=original['ClaimExtractor'].__new__(original['ClaimExtractor'])
        for obj in (a,b):
            obj.device='cpu';obj.tokenizer=Tokenizer();obj.model=Model()
        a.max_input_length=256;a.max_output_length=128;a.num_beams=4;a.early_stopping=True
        self.assertEqual(a.process_batch(['a','b']),b.process_batch(['a','b']))
        self.assertEqual(a.tokenizer.arguments,b.tokenizer.arguments)
        self.assertEqual(a.model.arguments['max_length'],b.model.arguments['max_length'])
        self.assertEqual(a.model.arguments['num_beams'],b.model.arguments['num_beams'])

    def test_nli_probability_order(self):
        aligner=NLIAligner.__new__(NLIAligner)
        aligner.batch_size=2
        aligner.score_sample=lambda batch:torch.tensor([[.7,.2,.1] for _ in batch])
        ent,con,neu=aligner.process_batch([('p','h')]*3,disable_prog_bar=True)
        self.assertEqual(len(ent),3)
        self.assertAlmostEqual(float(ent[0]),.7)
        self.assertAlmostEqual(float(con[0]),.1)
        self.assertAlmostEqual(float(neu[0]),.2)

    def test_predicate_transformation_priority(self):
        from factual_eval.data import injection
        token=SimpleNamespace(lemma_='approve',text='approved')
        with patch.object(injection,'get_antonym_lemma',return_value='deny'), \
             patch.object(injection,'get_inflected_antonym',return_value='denied'), \
             patch.object(injection,'build_negated_predicate') as neg:
            result=injection.generate_predicate_transformation(token,None,{})
            self.assertEqual(result['strategia'],'antonym')
            neg.assert_not_called()


if __name__=='__main__':unittest.main()
