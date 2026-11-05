from _support import ROOT
from pathlib import Path
import json
import random
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd

from factual_eval.config import load_config
from factual_eval.data import injection


class GenerationTests(unittest.TestCase):
    def test_interval_preserves_original_rng_prefix_and_source_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'raw.json'
            raw.write_text(json.dumps([{'report':f'Source document {i}.','summary':f'Original summary {i}.'} for i in range(3)]))
            calls=[]
            def transform(summary,source,num_errors):
                calls.append((summary,random.random()))
                return summary,0,[]
            with patch('spacy.load',return_value=object()), \
                 patch.object(injection,'inject_realistic_predicate_errors',side_effect=transform), \
                 patch.object(injection,'inject_realistic_coreference_errors',side_effect=transform), \
                 patch.object(injection,'inject_realistic_entity_errors',side_effect=transform):
                full=injection.run(root/'full',load_config(),num_docs=3,input_path=raw)
                expected=list(calls);calls.clear()
                subset=injection.run(root/'subset',load_config(),start=2,num_docs=1,input_path=raw)
            self.assertEqual(calls,expected)
            self.assertEqual(list(pd.read_csv(subset)['source_index']),[2])
            self.assertEqual(pd.read_csv(subset)['document_id'].iloc[0],pd.read_csv(full)['document_id'].iloc[2])
            report=json.loads((root/'subset/technical_audit.json').read_text())
            self.assertEqual(report['counts']['technical_pass'],3)


if __name__=='__main__':unittest.main()
