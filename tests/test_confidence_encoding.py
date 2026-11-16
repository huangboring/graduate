import json
from pathlib import Path
import unittest
import numpy as np
import test_egohumans as fixtures
from observer_selection.egohumans import _hash, _selector_observations
from observer_selection.local_data import export_local_egohumans


class ConfidenceEncodingTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.EgoHumansExportTests();self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.pair=Path(self.fixture.pairs['aria02'])

    def change_scores(self,values):
        p=self.pair/'selector_observations.npz'
        with np.load(p) as z: data={k:z[k] for k in z.files}
        data['confidence'][0,:len(values)]=values
        np.savez_compressed(p,**data)
        receipt=self.pair/'selector_preparation.json'
        r=json.loads(receipt.read_text());r['output_sha256']=_hash(p)
        receipt.write_text(json.dumps(r))
        return p

    def test_raw_amplitudes_preserved_and_transform_disclosed(self):
        p=self.change_scores([1.06959784,1.01,-.02,.75]);before=_hash(p)
        with self.assertRaisesRegex(ValueError,'confidence encoding'):
            _selector_observations(self.pair,self.fixture.ts)
        m=self.fixture.manifest
        for key in ('target_tracker','floor','world_to_zup'):m.pop(key)
        m.update(schema_version=2,camera_calibration={'provenance':'calibrated','description':'FIXTURE',
            'target_pose_used':False,'rectified_pinhole':True})
        self.fixture.save_manifest()
        result=export_local_egohumans(self.fixture.mp,self.fixture.root/'export.npz')
        with np.load(result['output']) as z:
            np.testing.assert_allclose(z['local_confidence'][0,:4],[1,1,0,.75])
        self.assertEqual(_hash(p),before)
        r=json.loads(Path(result['provenance']).read_text())['confidence_encoding']['aria02']
        self.assertEqual(r['above_one'],2);self.assertEqual(r['below_zero'],1)
        self.assertFalse(r['calibrated_probability'])

    def test_nonfinite_still_rejected(self):
        self.change_scores([np.nan])
        with self.assertRaises(ValueError):
            _selector_observations(self.pair,self.fixture.ts,'clip_heatmap_v1')
