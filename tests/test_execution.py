from _support import ROOT
from pathlib import Path
import importlib
import json
import tempfile
import unittest
from unittest.mock import patch

from factual_eval.config import load_config,project_path
from factual_eval.runs import Run
from factual_eval.cli import main


class ExecutionTests(unittest.TestCase):
    def test_imports_do_not_load_models(self):
        with patch('transformers.AutoTokenizer.from_pretrained',side_effect=AssertionError('download')), \
             patch('transformers.T5Tokenizer.from_pretrained',side_effect=AssertionError('download')), \
             patch('spacy.load',side_effect=AssertionError('model loaded at import')), \
             patch('sentence_transformers.SentenceTransformer.__init__',side_effect=AssertionError('model loaded at import')):
            for name in ('claims.model','claims.extraction','claims.enriched','retrieval.pipeline','nli.model','nli.evaluation',
                         'taxonomy.classification','data.injection','reference.preparation','reference.sampling','reference.html','reference.export'):
                importlib.reload(importlib.import_module('factual_eval.'+name))

    def test_config_unknown_key_refused_and_baseline_values(self):
        baseline=load_config()
        self.assertEqual(baseline['nli']['contradiction_threshold'],.62)
        self.assertEqual(baseline['selection']['num_docs'],5)
        self.assertEqual(baseline['retrieval']['use_mmr'],False)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.yaml';path.write_text('nli:\n  contrdiction_threshold: 0.7\n')
            with self.assertRaisesRegex(ValueError,'Unknown'):load_config(path)
        self.assertEqual(project_path('configs/baseline.yaml'),ROOT/'configs/baseline.yaml')

    def test_exclusive_runs_reuse_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);input=root/'input.txt';input.write_text('fixed input')
            called=[]
            def operation(folder):
                called.append(1);p=folder/'result.txt';p.write_text('result');return p
            config=load_config()
            with Run(root/'runs',config,run_id='first') as run:
                result=run.stage('test',input,operation)
            with self.assertRaises(FileExistsError):Run(root/'runs',config,run_id='first')
            with Run(root/'runs',config,run_id='second',reuse_from=root/'runs/first') as run:
                new=run.stage('test',input,operation)
                self.assertEqual(new.read_text(),'result')
            self.assertEqual(len(called),1)
            result.write_text('tampered')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                with Run(root/'runs',config,run_id='third',reuse_from=root/'runs/first') as run:
                    run.stage('test',input,operation)
            manifest=json.loads((root/'runs/third/manifest.json').read_text())
            self.assertEqual(manifest['status'],'failed')
            self.assertEqual(manifest['stages']['test']['status'],'failed')

    def test_changed_config_never_reuses_old_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);input=root/'input';input.write_text('data');called=[]
            def op(folder):
                called.append(1);path=folder/'out';path.write_text(str(len(called)));return path
            cfg=load_config()
            with Run(root,cfg,run_id='old') as run:run.stage('nli',input,op)
            cfg=load_config();cfg['nli']['contradiction_threshold']=.7
            with Run(root,cfg,run_id='new',reuse_from=root/'old') as run:run.stage('nli',input,op)
            self.assertEqual(len(called),2)

    def test_reusable_stage_refuses_existing_output_before_model_loading(self):
        from factual_eval.claims.extraction import run
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.csv';output=root/'claims.parquet'
            source.write_text('source');output.write_bytes(b'previous result')
            with self.assertRaises(FileExistsError):
                run(source,output,load_config())
            self.assertEqual(output.read_bytes(),b'previous result')

    def test_packaged_default_configs_match_source_defaults(self):
        for name in ('baseline.yaml','reference.yaml'):
            self.assertEqual((ROOT/'configs'/name).read_bytes(),(ROOT/'src/factual_eval/resources'/name).read_bytes())

    def test_module_dispatcher_shows_subcommand_help(self):
        from contextlib import redirect_stdout
        from io import StringIO
        output=StringIO()
        with redirect_stdout(output),self.assertRaises(SystemExit) as stopped:
            main(argv=['pipeline','--help'])
        self.assertEqual(stopped.exception.code,0)
        self.assertIn('--from-stage',output.getvalue())


if __name__=='__main__':unittest.main()
