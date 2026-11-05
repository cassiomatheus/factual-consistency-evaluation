"""Opt-in browser regression, synthetic texts and a fake translator, no Ollama/GPU."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from _support import ROOT
from reference.test_translation import EN1, PT1, EN2, PT2, FakeTranslator, fixture
from factual_eval.reference.html import jsonl, run as html_run
from factual_eval.reference.translation import translate_reference
from factual_eval.reference.export import validate


@unittest.skipUnless(os.environ.get('FACTUAL_EVAL_BROWSER_TESTS')=='1','Opt-in Firefox/Selenium UI regression')
class BilingualBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from selenium import webdriver
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.raw=fixture(cls.root/'raw')
        rows=jsonl(cls.raw/'sentencas.jsonl')
        rows.extend({'doc_id':'doc1','sentence_id':f'doc1:S{i:05d}',
                     'sentence_index':i-1,'text':EN2} for i in range(3,83))
        (cls.raw/'sentencas.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        translate_reference(cls.raw,cls.root/'translations',cls.root/'cache.sqlite3',FakeTranslator())
        cls.html=html_run(cls.raw,cls.root/'review.html',cls.root/'translations')
        options=webdriver.FirefoxOptions()
        options.add_argument('-headless')
        if os.environ.get('FACTUAL_EVAL_FIREFOX_BINARY'):
            options.binary_location=os.environ['FACTUAL_EVAL_FIREFOX_BINARY']
        options.set_preference('browser.download.folderList',2)
        options.set_preference('browser.download.dir',str(cls.root))
        options.set_preference('browser.helperApps.neverAsk.saveToDisk','application/json')
        cls.browser=webdriver.Firefox(options=options)
        cls.browser.set_window_size(1650,1100)

    @classmethod
    def tearDownClass(cls):
        cls.browser.quit()
        cls.temp.cleanup()

    def test_complete_audit_survives_language_navigation_and_json_roundtrip(self):
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import Select, WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        b=self.browser
        b.get(self.html.as_uri())
        wait=WebDriverWait(b,15)
        def find(css): return b.find_element(By.CSS_SELECTOR,css)
        def click(css): b.execute_script('arguments[0].click()',find(css))
        def search(css,value):
            element=find(css)
            element.send_keys(Keys.CONTROL,'a')
            element.send_keys(Keys.BACKSPACE)
            if value:
                element.send_keys(value)
        def language(value): Select(find('#languageView')).select_by_value(value)
        self.assertEqual(find('#languageView').get_attribute('value'),'bilingual')
        self.assertTrue(find('#deepSection').is_displayed())
        for css in ('#claimInfo','#claimList','#candidateList','#sourceList'):
            self.assertIn(PT1,find(css).text)
            self.assertIn(EN1,find(css).text)
        click('#candidateList .technical-details summary')
        self.assertIn('CrossEncoder: 3.0000',find('#candidateList').text)
        language('en')
        self.assertNotIn(PT1,find('#claimInfo').text)
        self.assertIn(EN1,find('#claimInfo').text)
        language('pt')
        self.assertIn(PT1,find('#claimInfo').text)
        self.assertNotIn(EN1,find('#claimInfo').text)
        click('#claimInfo .translation-original summary')
        self.assertIn(EN1,find('#claimInfo').text)
        b.refresh()
        self.assertEqual(find('#languageView').get_attribute('value'),'pt')
        for query in ('visitou','visited'):
            search('#claimSearch',query)
            self.assertIn('2 afirmações',find('#queueCount').text)
        search('#claimSearch','')
        # Filter may select a different claim, so explicitly return to the first.
        click('[data-claim-index="0"]')
        click('#nextBtn')
        self.assertIn(PT2,find('#claimInfo').text)
        click('#prevBtn')
        click('#quickModeBtn')
        click('[data-triage="SUPPORT_CANDIDATE"]')
        self.assertIn('SUPPORT_CANDIDATE',find('#triageStatus').text)
        click('#deepModeBtn')
        self.assertTrue(find('#deepSection').is_displayed())
        self.assertEqual(find('[data-claim-field="factual_label"]').get_attribute('value'),'')
        for query in ('visitou','visited'):
            search('#sourceSearch',query)
            self.assertNotIn('S00001',find('#sourceList').text)
            self.assertIn('S00002',find('#sourceList').text)
        search('#sourceSearch','doc1:S00002')
        click('[data-manual-add="doc1:S00002"]')
        self.assertIn('S00002',find('#candidateList').text)
        search('#sourceSearch','')
        click('#sourceNext')
        self.assertIn('S00081',find('#sourceList').text)
        click('#sourcePrev')
        click('[data-goto="doc1:S00001"]')
        self.assertEqual(find('#sourceSearch').get_attribute('value'),'doc1:S00001')
        click('#sourceList [data-set-toggle="doc1:S00001"]')
        click('#saveSet')
        self.assertIn('S00001',find('#savedSets').text)
        search('#reviewer','Revisor teste')
        click('.review-status summary')
        for field,value in [('claim_valid','SIM'),('factual_label','SUPPORTED'),('review_status','FINALIZADO')]:
            Select(find(f'[data-claim-field="{field}"]')).select_by_value(value)
        click('[data-claim-bool="full_document_checked"]')
        click('[data-claim-bool="search_complete"]')
        search('[data-claim-field="notes"]','Nota humana preservada')
        language('bilingual')
        self.assertEqual(find('[data-claim-field="notes"]').get_attribute('value'),'Nota humana preservada')
        click('#exportBtn')
        def completed_download(_):
            path=next(self.root.glob('revisao_*.json'),None)
            if path:
                try:
                    json.loads(path.read_text())
                    return path
                except (OSError,json.JSONDecodeError):
                    pass
            return False
        exported=wait.until(completed_download)
        data=json.loads(exported.read_text())
        cid='doc1:original:C00001'
        self.assertEqual(data['claims'][cid]['gold_evidence_sets'],[['doc1:S00001']])
        self.assertEqual(data['claims'][cid]['factual_label'],'SUPPORTED')
        self.assertNotIn('translations',data)
        report=validate(self.raw,exported,self.root/'gold')
        self.assertEqual(report['claims_gold_validated'],1)
        search('[data-claim-field="notes"]','Alteração temporária')
        find('#importFile').send_keys(str(exported))
        wait.until(EC.alert_is_present()).accept()
        wait.until(EC.alert_is_present()).accept()
        self.assertEqual(find('[data-claim-field="notes"]').get_attribute('value'),'Nota humana preservada')
        self.assertIn('S00001',find('#savedSets').text)
        b.save_screenshot(str(self.root/'bilingual.png'))
        print('Browser audit roundtrip passed:',b.capabilities['browserVersion'])

    def test_missing_translation_and_warning_leave_original_accessible(self):
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import Select
        root=self.root/'warnings'
        root.mkdir()
        rows=jsonl(self.root/'translations/translations.jsonl')
        rows[0].update(status='needs_review',warnings=['Possível diferença numérica.'])
        for row in rows:
            if row['unit_id']=='doc1:S00002':
                row.update(status='pending',translation_pt='')
        (root/'translations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (root/'translation_manifest.json').write_bytes((self.root/'translations/translation_manifest.json').read_bytes())
        page=html_run(self.raw,self.root/'warnings.html',root)
        b=self.browser
        b.get(page.as_uri())
        Select(b.find_element(By.ID,'languageView')).select_by_value('pt')
        warning=b.find_element(By.CSS_SELECTOR,'#claimInfo .translation-warning')
        self.assertIn('Tradução requer conferência.',warning.text)
        warning.find_element(By.TAG_NAME,'summary').click()
        self.assertIn('Possível diferença numérica.',warning.text)
        source=b.find_element(By.ID,'sourceList').text
        self.assertIn('Tradução indisponível (pendente).',source)
        self.assertIn(EN2,source)


if __name__=='__main__':
    unittest.main()
