import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from observer_selection.demo_values import export_values, frame_value, sparse_fields

class ValuesTests(unittest.TestCase):
    def test_sparse_offsets(self):
        fields = sparse_fields(np.arange(54))
        self.assertEqual(fields['head']['position_xyz_m'], [36,37,38])
        self.assertEqual(fields['right_wrist']['relative_rotation_6d'], list(range(30,36)))
        self.assertEqual(sparse_fields(np.arange(18))['head']['position_xyz_m'], [12,13,14])

    def test_flattened_fk_output(self):
        a = np.arange(96*22*3).reshape(96,22,3)
        for i in (0, 37, 95):
            result, layout = frame_value('AvatarNet.output.egonet_pos', a, i, 96)
            np.testing.assert_array_equal(result, a[i])
        with self.assertRaises(ValueError):
            frame_value('unknown', a, 0, 96)
        with self.assertRaises(IndexError):
            frame_value('AvatarNet.output.egonet_pos', a, 96, 96)

    def test_layout_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'inspection').mkdir(); (root/'dataset'/'pair').mkdir(parents=True)
            np.savez(root/'dataset'/'pair'/'vitpose_egonet.npz',vitpose_kp=np.zeros((3,13,2)))
            (root/'demo_protocol.json').write_text(json.dumps({'sequence':tmp}))
            np.savez(root/'predictions.npz',raw_frame_ids=[2,3],predicted=np.ones((2,22,3)),reference=np.zeros((2,22,3)))
            np.savez(root/'inspection'/'module_arrays.npz', **{
                'AvatarNet.input.0.sparse_input':np.arange(108).reshape(1,2,54),
                'AvatarNet.output.egonet_pos':np.arange(132).reshape(2,22,3),
                'fusion.spatial_encoder.output':np.arange(32).reshape(2,2,8)})
            out=export_values(root,3); data=json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(data['demo_index_0based'],1)
            self.assertEqual(data['modules']['AvatarNet.output.egonet_pos']['values'],np.arange(66,132).reshape(22,3).tolist())
            self.assertEqual(data['modules']['fusion.spatial_encoder.output']['values'],np.arange(16,32).reshape(2,8).tolist())
            self.assertEqual(len(data['modules']['AvatarNet.input.0.sparse_input']['values']),54)
            self.assertAlmostEqual(data['frame_mpjpe_mm'],np.sqrt(3)*1000)
            with self.assertRaises(ValueError):export_values(root,4)
        with self.assertRaises(ValueError):frame_value('unknown',np.ones((3,7)),0,2)
