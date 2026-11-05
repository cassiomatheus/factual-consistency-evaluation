"""Guided review behavior in a real browser, with synthetic data and no model."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from _support import ROOT
from reference.test_translation import fixture
from factual_eval.reference.html import run as html_run
from factual_eval.reference.export import validate


@unittest.skipUnless(os.environ.get('FACTUAL_EVAL_BROWSER_TESTS')=='1','Opt-in guided audit UI')
class AuditUsabilityBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from selenium import webdriver
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.raw=fixture(cls.root/'raw')
        cls.page=html_run(cls.raw,cls.root/'review.html')
        options=webdriver.FirefoxOptions()
        options.add_argument('-headless')
        if os.environ.get('FACTUAL_EVAL_FIREFOX_BINARY'):
            options.binary_location=os.environ['FACTUAL_EVAL_FIREFOX_BINARY']
        cls.browser=webdriver.Firefox(options=options)

    @classmethod
    def tearDownClass(cls):
        cls.browser.quit()
        cls.temp.cleanup()

    def setUp(self):
        self.browser.set_window_size(1600,1100)
        self.browser.get(self.page.as_uri())
        self.browser.execute_script('localStorage.clear();sessionStorage.clear()')
        self.browser.refresh()

    def find(self, css):
        return self.browser.find_element('css selector',css)

    def click(self, css):
        self.browser.execute_script('arguments[0].click()',self.find(css))

    def select(self, css, value):
        from selenium.webdriver.support.ui import Select
        self.find(css).location_once_scrolled_into_view
        Select(self.find(css)).select_by_value(value)

    def notes(self, text):
        self.browser.execute_script("arguments[0].value=arguments[1];arguments[0].dispatchEvent(new Event('input',{bubbles:true}))",self.find('#humanNotes'),text)

    def revision(self):
        return self.browser.execute_script("return JSON.parse(localStorage.getItem('revision:fixture:sem_amostragem'))")

    def test_checklist_requires_saved_evidence_and_completion_keeps_original_labels(self):
        self.assertTrue(self.find('#deepSection').is_displayed())
        self.assertFalse(self.find('#completeReview').is_enabled())
        self.assertEqual(len(self.browser.find_elements('css selector','.step-link')),3)
        self.select('#claimValidity','SIM')
        self.select('#factualLabel','SUPPORTED')
        self.click('[data-claim-bool="full_document_checked"]')
        self.click('[data-claim-bool="search_complete"]')
        self.assertFalse(self.find('#completeReview').is_enabled())
        self.click('#sourceList [data-set-toggle="doc1:S00001"]')
        self.assertIn('1 sentença selecionada',self.find('#workingSet').text)
        self.assertFalse(self.find('#completeReview').is_enabled())
        self.click('#saveSet')
        self.assertTrue(self.find('#completeReview').is_enabled())
        self.click('#completeReview')
        review=self.revision()
        first=review['claims']['doc1:original:C00001']
        self.assertEqual(first['claim_valid'],'SIM')
        self.assertEqual(first['factual_label'],'SUPPORTED')
        self.assertEqual(first['review_status'],'FINALIZADO')
        self.assertEqual(first['gold_evidence_sets'],[['doc1:S00001']])
        self.assertIn('C00002',self.find('#claimInfo').text)
        self.assertIn('1 de 3 revisadas',self.find('#progress').text)
        target=self.root/'complete.json'
        target.write_text(json.dumps(review))
        report=validate(self.raw,target,self.root/'gold-complete')
        self.assertEqual(report['claims_gold_validated'],1)

    def test_invalid_claim_can_be_reviewed_without_fabricating_factual_label(self):
        self.select('#claimValidity','NAO')
        self.assertFalse(self.find('#completeReview').is_enabled())
        self.notes('A extração reuniu afirmações que precisam ser revistas.')
        self.assertTrue(self.find('#completeReview').is_enabled())
        self.assertIn('inválida',self.find('#completeReview').text)
        self.click('#completeReview')
        review=self.revision()
        first=review['claims']['doc1:original:C00001']
        self.assertEqual(first['factual_label'],'')
        self.assertEqual(first['review_status'],'FINALIZADO')
        path=self.root/'invalid.json'
        path.write_text(json.dumps(review))
        report=validate(self.raw,path,self.root/'gold-invalid')
        self.assertEqual(report['claims_gold_validated'],0)

    def test_source_search_reset_and_mobile_layout(self):
        field=self.find('#sourceSearch')
        field.send_keys('Idaho')
        self.assertTrue(self.find('#clearSourceSearch').is_displayed())
        self.assertNotIn('S00001',self.find('#sourceList').text)
        self.click('#clearSourceSearch')
        self.assertIn('S00001',self.find('#sourceList').text)
        self.assertEqual(field.get_attribute('value'),'')
        self.browser.set_window_size(390,900)
        width=self.browser.execute_script('return [document.documentElement.scrollWidth,window.innerWidth]')
        self.assertLessEqual(width[0],width[1]+1)
        self.assertTrue(self.find('#stepDecision').is_displayed())
        self.click('#stepDecision')
        self.assertTrue(self.find('#factualLabel').is_displayed())


if __name__=='__main__':
    unittest.main()
