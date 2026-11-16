import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from observer_selection.replay import make_synthetic
from observer_selection.protocol import audit_partitions,load_protocol,verify_heldout
from types import SimpleNamespace


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.paths=[]
        for i in range(3):
            p=self.root/f'{i}.npz';make_synthetic(p,frames=3,seed=i);self.paths.append(p)
    def tearDown(self):self.temp.cleanup()
    def test_explicit_protocol(self):
        p=self.root/'split.json';p.write_text(json.dumps({'train':['0.npz'],'validation':['1.npz'],'test':['2.npz']}))
        paths,manifest=load_protocol(p)
        self.assertEqual(len(manifest['test']),1)
        self.assertEqual(paths['train'],[self.paths[0]])
    def test_duplicate_bytes_rejected(self):
        copy=self.root/'copy.npz';copy.write_bytes(self.paths[0].read_bytes())
        with self.assertRaisesRegex(ValueError,'leakage'):
            audit_partitions({'train':[self.paths[0]],'test':[copy]})
    def test_same_scene_role_changes_rejected(self):
        for p in self.paths[:2]:
            with np.load(p) as z:a={k:z[k] for k in z.files}
            a['scene_id']=np.array('same_scene');np.savez(p,**a)
        with self.assertRaisesRegex(ValueError,'scene_id'):
            audit_partitions({'train':[self.paths[0]],'validation':[self.paths[1]]})
    def test_holdout_and_backend_checks(self):
        manifest=audit_partitions({'train':[self.paths[0]],'validation':[self.paths[1]]})
        backend=SimpleNamespace(provenance={'backend':'test','hash':'a'})
        net=SimpleNamespace(training_data_manifest=manifest,backend_provenance=backend.provenance)
        self.assertEqual(len(verify_heldout(net,[self.paths[2]],backend)),1)
        with self.assertRaises(ValueError):verify_heldout(net,[self.paths[1]],backend)
        with self.assertRaisesRegex(ValueError,'backend'):
            verify_heldout(net,[self.paths[2]],SimpleNamespace(provenance={'backend':'different'}))
