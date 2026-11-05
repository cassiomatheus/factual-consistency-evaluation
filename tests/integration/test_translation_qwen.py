"""Opt-in, local Qwen smoke test; translation warnings are not semantic proof."""
import os
import unittest
from _support import ROOT
from factual_eval.reference.translation import OllamaTranslator
from factual_eval.reference.translation_quality import quality_warnings


@unittest.skipUnless(os.environ.get('FACTUAL_EVAL_QWEN_TESTS')=='1','Opt-in local Ollama integration')
class QwenTranslationTests(unittest.TestCase):
    def test_local_translation(self):
        translator=OllamaTranslator()
        source='8 installations had not considered extreme weather and climate change effects in their plans.'
        pt,reason=translator.translate(source,'',{})
        self.assertTrue(pt)
        self.assertEqual(reason,'stop')
        self.assertNotEqual(pt,source)
        self.assertFalse(any('numérica' in w or 'negação' in w for w in quality_warnings(source,pt,reason)))
        print('Ollama:',translator.metadata,'\nEN:',source,'\nPT:',pt)
