"""Real loopback/browser autosave tests; only synthetic annotations, no models."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, build_opener

from _support import ROOT
from reference.test_translation import fixture
from factual_eval.reference.html import run as html_run
from factual_eval.reference.review_server import make_server


@unittest.skipUnless(os.environ.get('FACTUAL_EVAL_BROWSER_TESTS')=='1','Opt-in Firefox/Selenium disk autosave')
class DiskAutosaveBrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = fixture(self.root/'raw')
        self.page = html_run(self.raw,self.root/'review.html')
        self.original_html = self.page.read_bytes()
        self.path = self.root/'annotations/revisao.json'
        self.start_server()
        self.addCleanup(self.stop_server)
        self.browser = None
        self.addCleanup(lambda: self.browser.quit() if self.browser else None)
        self.open_browser()

    def start_server(self, port=0):
        self.server = make_server(self.page,self.path,port)
        self.worker = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.worker.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/'

    def stop_server(self):
        if self.server:
            self.server.shutdown()
            self.worker.join(timeout=5)
            self.server.server_close()
            self.server.store.close()
            self.server = None

    def open_browser(self):
        from selenium import webdriver
        from selenium.webdriver.support.ui import WebDriverWait
        options = webdriver.FirefoxOptions()
        options.add_argument('-headless')
        if os.environ.get('FACTUAL_EVAL_FIREFOX_BINARY'):
            options.binary_location = os.environ['FACTUAL_EVAL_FIREFOX_BINARY']
        self.browser = webdriver.Firefox(options=options)
        self.browser.set_window_size(1500,1000)
        self.browser.get(self.url)
        self.wait = WebDriverWait(self.browser,15)
        self.wait.until(lambda _: not self.browser.execute_script('return document.querySelector(".grid").inert'))

    def find(self, css):
        return self.browser.find_element('css selector',css)

    def fill(self, css, value):
        self.browser.execute_script("arguments[0].value=arguments[1];arguments[0].dispatchEvent(new Event('input',{bubbles:true}))",self.find(css),value)

    def saved(self):
        self.wait.until(lambda _: self.find('#diskSaveStatus').text=='Salvo em arquivo ✓')

    def test_disk_survives_browser_storage_clear_and_server_restart(self):
        self.fill('#reviewer','Pesquisador local')
        self.fill('[data-claim-field="notes"]','Análise persistente em português')
        self.browser.execute_script('arguments[0].click()',self.find('#sourceList [data-set-toggle="doc1:S00002"]'))
        self.saved()
        cid = 'doc1:original:C00001'
        data = json.loads(self.path.read_text())
        self.assertEqual(data['claims'][cid]['notes'],'Análise persistente em português')
        self.assertEqual(data['draft_sets'][cid],['doc1:S00002'])
        self.assertEqual(data['claims'][cid]['gold_evidence_sets'],[])
        self.assertTrue(list(self.path.parent.glob('*.json')))
        self.browser.execute_script('localStorage.clear();sessionStorage.clear()')
        self.browser.quit()
        self.browser = None
        port = self.server.server_port
        self.stop_server()
        self.start_server(port)
        self.open_browser()
        self.assertEqual(self.find('#reviewer').get_attribute('value'),'Pesquisador local')
        self.browser.execute_script('arguments[0].click()',self.find('#deepModeBtn'))
        self.assertEqual(self.find('[data-claim-field="notes"]').get_attribute('value'),'Análise persistente em português')
        self.assertIn('S00002',self.find('#workingSet').text)
        self.assertEqual(self.find('#savedSets').text,'')
        self.assertEqual(self.page.read_bytes(),self.original_html)
        self.fill('[data-claim-field="notes"]','Segunda análise')
        self.saved()
        history = list((self.path.parent/'revisao.historico').glob('*.json'))
        self.assertTrue(any(json.loads(p.read_text())['claims'][cid]['notes']=='Análise persistente em português' for p in history))

    def test_other_tab_and_external_requests_cannot_overwrite(self):
        self.fill('[data-claim-field="notes"]','Antes das abas')
        self.saved()
        first = self.browser.current_window_handle
        self.browser.switch_to.new_window('tab')
        second = self.browser.current_window_handle
        self.browser.get(self.url)
        self.wait.until(lambda _: not self.browser.execute_script('return document.querySelector(".grid").inert'))
        self.browser.switch_to.window(first)
        self.fill('[data-claim-field="notes"]','Salvo pela primeira aba')
        self.saved()
        self.browser.switch_to.window(second)
        self.fill('[data-claim-field="notes"]','Alteração da segunda aba')
        self.wait.until(lambda _: 'Conflito entre abas' in self.find('#diskSaveStatus').text)
        cid = 'doc1:original:C00001'
        self.assertEqual(json.loads(self.path.read_text())['claims'][cid]['notes'],'Salvo pela primeira aba')
        opener = build_opener(ProxyHandler({}))
        token = self.browser.execute_script('return window.AUDIT_DISK_CONFIG.token')
        for headers in ({},{'X-Audit-Token':token,'Origin':'https://example.com'},
                        {'X-Audit-Token':token,'Host':'example.com'}):
            with self.subTest(headers=list(headers)):
                with self.assertRaises(HTTPError) as error:
                    opener.open(Request(self.url+'api/review',headers=headers),timeout=3)
                self.assertEqual(error.exception.code,403)


if __name__=='__main__':
    unittest.main()
