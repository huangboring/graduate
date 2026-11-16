import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from observer_selection.identity_audit import audit_identity, annotation_records


class IdentityAuditTests(unittest.TestCase):
    def test_explicit_dataset_ids_not_detection_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); images=root/'exo/cam01/images';images.mkdir(parents=True)
            poses=root/'processed_data/poses2d/cam01/rgb';poses.mkdir(parents=True)
            Image.new('RGB',(640,480)).save(images/'00001.jpg')
            rows=[{'human_name':'aria03','bbox':[10,20,60,120,1], 'keypoints':np.ones((17,3))},
                  {'human_name':'aria01','bbox':[100,20,160,120,1], 'keypoints':np.zeros((17,3))}]
            np.save(poses/'00001.npy',rows,allow_pickle=True)
            report=audit_identity(root,root/'audit')
            self.assertEqual(report['usable_annotation_ids'],{'cam01':['aria03']})
            self.assertTrue((root/'audit/cam01_00001.jpg').is_file())
            self.assertFalse((root/'replay.npz').exists())
            self.assertEqual(len(json.loads((root/'audit/identity_audit.json').read_text())['input_files']),2)
            with self.assertRaises(FileExistsError): audit_identity(root,root/'audit')

    def test_no_guess_when_identity_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'00001.npy'
            np.save(path,[{'bbox':[0,0,10,10],'keypoints':np.ones((17,3))}],allow_pickle=True)
            with self.assertRaisesRegex(ValueError,'will not guess'): annotation_records(path)

    def test_missing_annotations_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'No matching'): audit_identity(root,root/'audit')
            self.assertFalse((root/'audit').exists())
