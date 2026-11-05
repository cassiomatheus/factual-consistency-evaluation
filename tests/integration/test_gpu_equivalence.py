"""Opt-in neural comparisons with cached real checkpoints. Never part of quick CPU inference."""
from _support import ROOT,frozen
import gc
import os
import unittest
import torch


@unittest.skipUnless(os.environ.get('FACTUAL_EVAL_GPU_TESTS')=='1','Opt-in GPU integration; real cached models required')
class GPUEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA unavailable')

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_real_t5_generation_against_original_method(self):
        from factual_eval.claims.model import ClaimExtractor
        from factual_eval.config import load_config
        cfg=load_config()['claims']
        options={k:cfg[k] for k in ('model_name','revision','batch_size','max_input_length','max_output_length','num_beams','early_stopping')}
        migrated=ClaimExtractor(device='cuda:0',**options)
        original_class=frozen('claims',torch=torch)['ClaimExtractor']
        original=original_class.__new__(original_class)
        original.model=migrated.model;original.tokenizer=migrated.tokenizer;original.device=migrated.device
        sentences=['The agency approved the plan in 2020.','It published the report in 2021.']
        self.assertEqual(migrated.process_batch(sentences),original.process_batch(sentences))

    def test_real_nli_probabilities_against_original_method(self):
        from tqdm import tqdm
        from factual_eval.nli.model import NLIAligner
        from factual_eval.config import load_config
        from factual_eval.utils.batching import chunks
        cfg=load_config()['nli']
        migrated=NLIAligner(device='cuda:0',**{k:cfg[k] for k in ('model_name','revision','batch_size','max_length')})
        original_class=frozen('nli_model',torch=torch,tqdm=tqdm,chunks=chunks)['NLIAligner']
        original=original_class.__new__(original_class)
        for key in ('model','tokenizer','device','batch_size','max_length'):
            setattr(original,key,getattr(migrated,key))
        pairs=[('The agency approved the plan.','The plan was approved.'),
               ('The agency approved the plan.','The agency rejected the plan.')]
        a=migrated.process_batch(pairs,disable_prog_bar=True)
        b=original.process_batch(pairs,disable_prog_bar=True)
        for actual,expected in zip(a,b):
            torch.testing.assert_close(torch.stack(actual),torch.stack(expected),rtol=1e-6,atol=1e-7)
