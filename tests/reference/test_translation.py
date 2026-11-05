"""Offline translation contracts: no GPU, server, paid APIs or factual labels."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from _support import ROOT
from factual_eval.reference.html import jsonl, run as html_run
from factual_eval.reference.export import validate
from factual_eval.reference.translation import (Cache, OllamaTranslator, collect_units,
    digest, load_translations, segments, translate_reference)
from factual_eval.reference.translation_quality import quality_warnings


EN1 = 'EPA may not appoint 8 members in 2018.'
PT1 = 'EPA pode não nomear oito membros em 2018.'
EN2 = 'GAO visited Idaho in 2035.'
PT2 = 'GAO visitou Idaho em 2035.'


def fixture(root):
    root.mkdir()
    def write(name, rows):
        (root/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root/'manifest.json').write_text(json.dumps({'dataset_id':'fixture', 'top_k':3}))
    write('documentos.jsonl',[{'doc_id':'doc1','source_text':EN1+' '+EN2}])
    write('sentencas.jsonl',[
        {'doc_id':'doc1','sentence_id':'doc1:S00001','sentence_index':0,'text':EN1},
        {'doc_id':'doc1','sentence_id':'doc1:S00002','sentence_index':1,'text':EN2}])
    write('claims.jsonl',[{'doc_id':'doc1','claim_id':f'doc1:original:C{i:05d}',
        'variant':'original','claim_text':text,'claim_original':text,'claim_resolved':text,
        'origin_sentence':text} for i,text in enumerate([EN1,EN2,EN1+' '+EN2],1)])
    write('candidatos.jsonl',[{'doc_id':'doc1','claim_id':f'doc1:original:C{i:05d}',
        'sentence_id':'doc1:S00001','sentence_text':EN1,'rank_final':1,'is_top_k':True,
        'rank_dense':1,'rank_bm25':1,'score_cosine':.8,'score_bm25':2,
        'score_cross_encoder_logit':3} for i in range(1,4)])
    return root


class FakeTranslator:
    metadata = {'model':'fake','model_version':'v1','ollama_version':'test'}
    def __init__(self, fail=False):
        self.calls=[]
        self.fail=fail
    def translate(self, source, context, glossary):
        self.calls.append((source,context))
        if self.fail and source == EN1:
            raise RuntimeError('interrupção simulada')
        return {EN1:PT1,EN2:PT2}[source], 'stop'


class QualityTests(unittest.TestCase):
    def test_numbers_dates_negation_and_modalities_preserved(self):
        self.assertEqual(quality_warnings(EN1,PT1),[])
        for target, word in [(PT1.replace('oito','nove'),'numérica'),
                             (PT1.replace('2018','2019'),'numérica'),
                             (PT1.replace('não ',''),'negação'),
                             (PT1.replace('pode ',''),'modalidade')]:
            with self.subTest(target=target):
                self.assertTrue(any(word in w for w in quality_warnings(EN1,target)))
        self.assertFalse(any('numérica' in w for w in quality_warnings(
            'Only two of 23 sites had 8 plans.', 'Apenas dois de 23 locais tinham oito planos.')))
        self.assertFalse(any('numérica' in w for w in quality_warnings(
            'EPA spent 1,200.50 dollars in January 2018.',
            'EPA gastou 1.200,50 dólares em janeiro de 2018.')))
        self.assertFalse(any('numérica' in w for w in quality_warnings(
            'EPA inspected 8 percent of sites.', 'EPA inspecionou oito por cento dos locais.')))

    def test_entities_acronyms_and_added_content(self):
        self.assertEqual(quality_warnings(EN2,PT2),[])
        for target in [PT2.replace('GAO','EPA'),PT2.replace('Idaho','Missouri')]:
            self.assertTrue(any('Sigla' in w or 'Entidade' in w for w in quality_warnings(EN2,target)))
        self.assertTrue(any('acréscimo' in w for w in quality_warnings('EPA acted.', 'EPA agiu. '+'Texto adicional. '*20)))

    def test_missing_incomplete_and_omitted(self):
        self.assertIn('Saída vazia.',quality_warnings(EN1,''))
        self.assertTrue(any('interrompida' in w for w in quality_warnings(EN1,PT1,'length')))
        self.assertTrue(any('omissão' in w for w in quality_warnings(EN1+' '+EN2,'EPA.')))
        self.assertTrue(any('pontuação' in w for w in quality_warnings(EN1,PT1[:-1])))


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.raw=fixture(self.root/'raw')
        self.cache=self.root/'cache.sqlite3'
        self.originals={p.name:p.read_bytes() for p in self.raw.iterdir()}

    def translate(self, name, translator=None, **kwargs):
        translator=translator or FakeTranslator()
        output=self.root/name
        translate_reference(self.raw,output,self.cache,translator,**kwargs)
        for p in self.raw.iterdir():
            self.assertEqual(p.read_bytes(),self.originals[p.name])
        return jsonl(output/'translations.jsonl'),translator

    def test_segment_alignment_cache_and_context(self):
        rows,t=self.translate('first')
        compound=next(r for r in rows if r['source_en']==EN1+' '+EN2)
        self.assertEqual(compound['translation_pt'],PT1+' '+PT2)
        self.assertEqual(len(compound['segments']),2)
        for row in rows:
            for s in row['segments']:
                self.assertEqual(row['source_en'][s['start']:s['end']],s['source_en'])
                self.assertEqual(s['source_hash'],digest(s['source_en']))
        self.assertEqual(len(t.calls),len(set(t.calls)))
        self.assertGreater(len([c for c in t.calls if c[0]==EN1]),1) # distinct contexts
        again,t=self.translate('second')
        self.assertEqual(t.calls,[])
        self.assertEqual([r['translation_pt'] for r in rows],[r['translation_pt'] for r in again])

    def test_resume_failed_and_pending_units_without_partial_compound(self):
        rows,_=self.translate('failed',FakeTranslator(fail=True))
        compound=next(r for r in rows if r['source_en']==EN1+' '+EN2)
        self.assertEqual(compound['status'],'failed')
        self.assertEqual(compound['translation_pt'],'')
        rows,t=self.translate('retry')
        self.assertTrue(t.calls)
        self.assertTrue(all(r['status']=='translated' for r in rows))
        self.assertTrue(all(source==EN1 for source,_ in t.calls))
        limited= self.root/'limited.sqlite3'
        translate_reference(self.raw,self.root/'limit',limited,FakeTranslator(),max_calls=1)
        partial=jsonl(self.root/'limit/translations.jsonl')
        self.assertTrue(any(r['status']=='pending' for r in partial))
        self.assertEqual(len(partial),len(rows))

    def test_human_verification_survives_model_and_prompt_change(self):
        rows,_=self.translate('first')
        cache=Cache(self.cache)
        for row in rows:
            for part in row['segments']:
                cache.verify(part['cache_key'],'revisor')
        cache.close()
        other=FakeTranslator()
        other.metadata={**other.metadata,'model_version':'v2'}
        rows,t=self.translate('changed',other,prompt_version='v2')
        self.assertEqual(t.calls,[])
        self.assertTrue(all(r['status']=='human_verified' for r in rows))
        self.assertEqual(rows[0]['segments'][0]['model_version'],'v1')

    def test_sampling_prioritizes_only_selected_claims_and_doc_source(self):
        rows=jsonl(self.raw/'claims.jsonl')
        (self.raw/'amostragem.jsonl').write_text(''.join(json.dumps({'claim_id':r['claim_id'],'selected':i==0})+'\n' for i,r in enumerate(rows)))
        units=collect_units(self.raw)
        self.assertEqual({u['unit_id'] for u in units if u['kind']=='sentence'}, {'doc1:S00001','doc1:S00002'})
        self.assertFalse(any(rows[1]['claim_id'] in u['unit_id'] for u in units))

    def test_html_validates_translation_identity_and_gold_stays_human(self):
        self.translate('translated')
        html=html_run(self.raw,self.root/'review.html',self.root/'translated')
        content=html.read_text()
        self.assertIn('languageView',content)
        self.assertIn(PT1,content)
        self.assertIn(EN1,content)
        self.assertNotIn('fetch(',content)
        self.assertNotIn('gabarito_injetado',content)
        ann=self.root/'empty.json'
        ann.write_text(json.dumps({'schema_version':1,'dataset_id':'fixture','reviewer':'test','claims':{}}))
        report=validate(self.raw,ann,self.root/'gold')
        self.assertEqual(report['claims_gold_validated'],0)
        with self.assertRaises(ValueError):
            load_translations(self.root/'translated','other-dataset')
        path=self.root/'translated/translations.jsonl'
        row=json.loads(path.read_text().splitlines()[0]);row['source_hash']='bad'
        path.write_text(json.dumps(row)+'\n')
        with self.assertRaises(ValueError):
            html_run(self.raw,self.root/'invalid.html',self.root/'translated')

    def test_ollama_request_disables_reasoning_and_preserves_raw_output(self):
        with patch.object(OllamaTranslator,'request') as request:
            request.side_effect=[{'version':'0.35.0'},{'models':[{'name':'qwen3.5:9b','digest':'hash'}]},
                                 {'done':True,'done_reason':'length','message':{'content':PT1}}]
            translator=OllamaTranslator()
            pt,reason=translator.translate(EN1,'context',{})
            self.assertEqual((pt,reason),(PT1,'length'))
            payload=request.call_args.args[1]
            self.assertIs(payload['think'],False)
            self.assertEqual(payload['options']['temperature'],0)
        with self.assertRaises(ValueError):
            OllamaTranslator(host='https://example.com')

    def test_cli_does_not_rerun_science_or_modify_parquets(self):
        import pandas as pd
        from factual_eval.cli import main
        files=[]
        for stage,field,value in [('claims','claim',EN1),('nli','status_factual_nli','SUPPORTED'),
                                  ('taxonomy','taxonomy_label','predicate_error')]:
            path=self.root/f'{stage}.parquet'
            pd.DataFrame([{'doc_id':'doc1','claim_id':'doc1:original:C00001',field:value}]).to_parquet(path)
            files.append(path)
        hashes={p:digest(p.read_bytes().hex()) for p in files}
        with patch('factual_eval.reference.translation.OllamaTranslator',return_value=FakeTranslator()), \
             patch('factual_eval.cli.execute_stage',side_effect=AssertionError('Scientific stage called')):
            main('translate-reference',['--input',str(self.raw),'--output-root',str(self.root/'runs'),
                 '--cache',str(self.cache),'--run-id','translation-only'])
        self.assertEqual(hashes,{p:digest(p.read_bytes().hex()) for p in files})


if __name__=='__main__':
    unittest.main()
