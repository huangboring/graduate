"""In-sample orchestration fixture: fake FK and generated observations only."""
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import test_egohumans as fixtures
from test_local_aux import fake_decode, REPO
from observer_selection.local_data import export_local_egohumans
from observer_selection.local_fusion import LocalFusionConfig
from observer_selection.single_sequence import fit_sequence


class SingleSequenceTests(TestCase):
    def test_two_stages_and_figures_and_viewer_keep_in_sample_label(self):
        fixture=fixtures.EgoHumansExportTests();fixture.setUp();self.addCleanup(fixture.tearDown)
        m=fixture.manifest
        for key in ('target_tracker','floor','world_to_zup'):m.pop(key)
        m.update(schema_version=2,camera_calibration={'provenance':'calibrated','description':'EXPLICIT FIXTURE',
                 'target_pose_used':False,'rectified_pinhole':True})
        fixture.save_manifest()
        data=export_local_egohumans(fixture.mp,fixture.root/'data.npz')['output']
        body=fixture.root/'body.npz';body.write_bytes(b'EXPLICIT FAKE FK')
        config=LocalFusionConfig(window_size=3,embed_dim=16,nhead=2,num_layer_spatial=1,
                                 num_layer_temporal=1,ff_dim=32,dropout=0)
        with patch('observer_selection.single_sequence.LocalFusionConfig',return_value=config), \
             patch('observer_selection.local_training.load_body_decoder',return_value=fake_decode), \
             patch('observer_selection.local_fusion.load_body_decoder',return_value=fake_decode):
            result=fit_sequence(data,fixture.root/'run',REPO,body,'cpu',1,1)
        self.assertFalse(result['heldout_results'])
        self.assertIn('IN-SAMPLE',Path(result['viewer']).read_text(encoding='utf-8'))
        self.assertTrue((fixture.root/'run/training_results/figures/policy_comparison.png').is_file())
        self.assertTrue((fixture.root/'run/reconstructor/last.pt').is_file())
        self.assertFalse((fixture.root/'run/reconstructor/best.pt').exists())
        records=json.loads((fixture.root/'run/selector/training.json').read_text())
        self.assertNotIn('validation_mpjpe_mm',records[0])
