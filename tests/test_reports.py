import json
import tempfile
import unittest
from pathlib import Path
from observer_selection.reports import export_report


class ReportTests(unittest.TestCase):
    def test_report_preserves_measured_values_and_unseen_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run = root/'run'; run.mkdir()
            (run/'training.json').write_text(json.dumps([{'epoch':1,'expected_mpjpe_m':.1,
                'rollin_mpjpe_mm':120,'validation_mpjpe_mm':130}]))
            policy = {'scene_macro_mpjpe_mm':130, 'mpjpe_mm':140,
                'source_coverage':{'0':{'acquisition_counts':{'C':1,'D':0}}},
                'frames':[{'sequence':0,'timestamp':0,'selected_ids':['C'],
                           'required_refresh_ids':['C'],'status':'warmup'},
                          {'sequence':0,'timestamp':.05,'selected_ids':['C'],
                           'required_refresh_ids':[],'status':'ok','mpjpe_mm':140}]}
            (run/'evaluation.json').write_text(json.dumps({'dataset':'synthetic_demo', 'policies':{'learned':policy}}))
            result = export_report(run, root/'figures')
            self.assertIn('sequence_0_selection.png', result['files'])
            self.assertIn('training_curves.pdf', result['files'])
            self.assertIn('130', (root/'figures/policy_summary.csv').read_text(encoding='utf-8-sig'))
            self.assertFalse(result['smoothing'])
            with self.assertRaises(FileExistsError):export_report(run, root/'figures')

    def test_no_fake_results_when_metrics_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):export_report(temp, Path(temp)/'figures')
