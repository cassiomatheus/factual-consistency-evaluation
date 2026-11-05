"""End-to-end orchestration contract with simulated models, no factual results."""
from _support import ROOT
from pathlib import Path
from types import SimpleNamespace
import json
import re
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from factual_eval.cli import main
from factual_eval.claims import extraction
from factual_eval.retrieval import pipeline as retrieval
from factual_eval.nli import evaluation
from factual_eval.taxonomy import classification


class Span:
    def __init__(self,text,start=0):
        self.text=text;self.start_char=start;self.end_char=start+len(text);self.ents=[];self.noun_chunks=[]
    def __iter__(self):return iter(())


class Doc(Span):
    def __init__(self,text):
        super().__init__(text)
        self.sents=[Span(m.group().strip(),m.start()+len(m.group())-len(m.group().lstrip()))
                    for m in re.finditer(r'[^.]+\.?',text) if m.group().strip()]


class NLP:
    meta={'version':'TEST_DOUBLE'}
    def __call__(self,text):return Doc(text)


class Extractor:
    def __init__(self,**kwargs):pass
    def process_batch(self,batch):return [[text] for text in batch]


class Embedder:
    def __init__(self,*a,**kw):pass
    def encode(self,texts,convert_to_tensor=False,**kw):
        values=np.array([[len(text)+1.,len(text.split())+1.,1.] for text in texts],dtype=np.float32)
        values=values/np.linalg.norm(values,axis=1,keepdims=True)
        return torch.tensor(values) if convert_to_tensor else values


class Cross:
    def __init__(self,*a,**kw):pass
    def predict(self,pairs,**kwargs):return np.array([2. if p[0] in p[1] else 1. for p in pairs])


class Aligner:
    def __init__(self,**kwargs):pass
    def process_batch(self,pairs,**kwargs):
        ent=[];con=[];neu=[]
        for premise,hypothesis in pairs:
            contradiction='denied' in hypothesis
            ent.append(torch.tensor(.01 if contradiction else .98))
            con.append(torch.tensor(.98 if contradiction else .01))
            neu.append(torch.tensor(.01))
        return ent,con,neu


class PipelineCPUTests(unittest.TestCase):
    def test_full_pipeline_preserves_ids_counts_and_intermediates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);input=root/'dataset.csv'
            source='The agency approved the plan. The agency published the report.'
            row={'fonte':source,'resumo_original':source,'resumo_entity':source,'resumo_corref':source,
                 'resumo_predicado':'The agency denied [ERROR_PREDICATE: denied] the plan. The agency published the report.'}
            pd.DataFrame([row]).to_csv(input,index=False)
            with patch('spacy.load',return_value=NLP()), \
                 patch.object(extraction,'ClaimExtractor',Extractor), \
                 patch.object(retrieval,'SentenceTransformer',Embedder), \
                 patch.object(retrieval,'CrossEncoder',Cross), \
                 patch.object(evaluation,'NLIAligner',Aligner), \
                 patch.object(classification,'SentenceTransformer',Embedder):
                code=main('pipeline',['--input',str(input),'--device','cpu','--num-docs','1',
                                      '--output-root',str(root/'runs'),'--run-id','contract'])
            self.assertEqual(code,0)
            run=root/'runs/contract'
            retrieval_rows=pd.read_parquet(run/'retrieval/retrieval.parquet')
            nli=pd.read_parquet(run/'nli/nli.parquet')
            taxonomy=pd.read_parquet(run/'taxonomy/taxonomy.parquet')
            self.assertEqual(len(retrieval_rows),8)
            self.assertEqual(len(nli),8)
            self.assertEqual(len(set(nli['claim_id'])),8)
            self.assertEqual(list(nli['claim_id']),list(retrieval_rows['claim_id']))
            self.assertEqual(len(taxonomy),1)
            self.assertEqual(set(taxonomy['claim_id']),set(nli.loc[nli.status_factual_nli.eq('CONTRADICTED'),'claim_id']))
            manifest=json.loads((run/'manifest.json').read_text())
            self.assertEqual(manifest['status'],'complete')
            self.assertEqual(list(manifest['stages']),['claims','retrieval','nli','taxonomy'])
            self.assertTrue((run/'claims/generator_labels.jsonl').is_file())
            self.assertTrue(all(not json.loads(line)['is_gold'] for line in (run/'claims/generator_labels.jsonl').read_text().splitlines()))


if __name__=='__main__':unittest.main()
