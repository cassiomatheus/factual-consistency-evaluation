from _support import ROOT
from pathlib import Path
import tempfile
import unittest
import pandas as pd

from factual_eval.data.validation import reconstruct_original,audit_dataframe,inverse_edits
from factual_eval.schemas import add_extraction_ids,add_retrieval_ids,carry_claim_ids,validate_taxonomy,write_jsonl,document_id


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_reconstruction_with_repeated_replacement_and_offsets(self):
        original='Agency approved the plan. Agency rejected the amendment.'
        mutated='Agency denied [ERROR_PREDICATE: denied] the plan. Agency denied [ERROR_PREDICATE: denied] the amendment.'
        changes=[{'token_original':'approved','token_injetado':'denied','sentenca_index':0},
                 {'token_original':'rejected','token_injetado':'denied','sentenca_index':1}]
        self.assertEqual(reconstruct_original(mutated,changes,'predicate'),original)
        for edit in inverse_edits(mutated,changes,'predicate'):
            self.assertEqual(original[edit['original_start']:edit['original_end']],edit['original'])
        with self.assertRaises(ValueError):reconstruct_original(mutated,changes[:1],'predicate')

    def test_technical_success_never_gold(self):
        row={'resumo_original':'The agency approved the plan.'}
        for suffix,summary in [('predicado','resumo_predicado'),('corref','resumo_corref'),('entity','resumo_entity')]:
            row[summary]=row['resumo_original'];row['tokens_'+suffix]=[];row['qtd_erros_'+suffix]=0
        report=audit_dataframe(pd.DataFrame([row]))
        self.assertEqual(report['counts']['technical_pass'],3)
        self.assertTrue(all(not r['is_gold'] and r['factual_validation']=='PENDING' for r in report['records']))

    def test_duplicate_text_ids_and_roundtrip_preservation(self):
        source='The agency approved this plan. The report has evidence.'
        csv=self.root/'input.csv';extraction=self.root/'claims.parquet';retrieval=self.root/'retrieval.parquet';nli=self.root/'nli.parquet'
        pd.DataFrame([{'fonte':source,'resumo_original':'A summary.', 'resumo_entity':None,
                       'resumo_predicado':None,'resumo_corref':None,'source_index':7}]).to_csv(csv,index=False)
        claim={'texto':'Repeated claim.','claim_original':'Repeated claim.','sentence_index':0}
        pd.DataFrame([{'doc_idx':7,'tipo_resumo':'original','documento_fonte':source,'claims':[claim,dict(claim)]}]).to_parquet(extraction,index=False)
        add_extraction_ids(csv,extraction)
        claims=pd.read_parquet(extraction).iloc[0]['claims']
        self.assertNotEqual(claims[0]['claim_id'],claims[1]['claim_id'])
        records=[{'doc_idx':7,'tipo_resumo':'original','documento_fonte':source,'claim':'Repeated claim.',
                  'top_sentences':[{'text':'The agency approved this plan.','sentence_index':0}],
                  'top_paragraphs':[]} for _ in range(2)]
        pd.DataFrame(records).to_parquet(retrieval,index=False)
        add_retrieval_ids(extraction,retrieval)
        predicted=[dict(r,status_factual_nli='SUPPORTED',source_passage=source,source_sentence_index=0,support_score=.9,contradiction_score=.1) for r in records]
        pd.DataFrame(predicted).to_parquet(nli,index=False)
        carry_claim_ids(retrieval,nli)
        self.assertEqual(list(pd.read_parquet(nli)['claim_id']),[c['claim_id'] for c in claims])
        self.assertEqual(set(pd.read_parquet(nli)['doc_idx']),{7})
        pd.DataFrame(records[:1]).to_parquet(retrieval,index=False)
        with self.assertRaisesRegex(ValueError,'count'):
            add_retrieval_ids(extraction,retrieval)

    def test_taxonomy_rejects_lost_contradictions(self):
        source=self.root/'nli.parquet';output=self.root/'taxonomy.parquet'
        rows=[{'doc_idx':0,'tipo_resumo':'original','claim':str(i),'status_factual_nli':'CONTRADICTED',
               'source_passage':'evidence','support_score':.1,'contradiction_score':.8} for i in range(2)]
        pd.DataFrame(rows).to_parquet(source,index=False)
        pd.DataFrame([dict(rows[0],taxonomy_label='other')]).to_parquet(output,index=False)
        with self.assertRaises(ValueError):validate_taxonomy(source,output)

    def test_reference_accepts_baseline_sentence_metadata_without_changing_ids(self):
        from factual_eval.reference.preparation import prepare_inputs
        records=[{'doc_idx':2,'tipo_resumo':'original','documento_fonte':'A source sentence.',
                  'claims':[{'claim_original':'The extracted claim.','sentence_source':'The summary sentence.', 'sentence_index':3}]}]
        _,claims=prepare_inputs(records,None,'claim_original',0,0)
        self.assertEqual(claims[0]['origin_sentence'],'The summary sentence.')
        self.assertEqual(claims[0]['origin_sentence_index'],3)
        self.assertEqual(claims[0]['claim_id'],document_id(2,'A source sentence.')+':original:C00001')


if __name__=='__main__':unittest.main()
