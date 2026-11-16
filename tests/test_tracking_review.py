import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from observer_selection.appearance_tracking import AppearanceTracker
from observer_selection.raw_tracking import track_view
from observer_selection.preparation_cache import atomic_npz
from observer_selection.candidate_reuse import reuse_candidate_cache


class TrackingReviewTests(unittest.TestCase):
    def test_gradual_turn_uses_recent_observation(self):
        tracker=AppearanceTracker();tracker.initialize(0,[0,0,10,10],[1,0])
        for t in range(1,10):
            angle=t*.25
            self.assertEqual(tracker.step(t,np.array([[0,0,10,10]]),
                np.array([[np.cos(angle),np.sin(angle)]])),0)
        self.assertLess(tracker.diagnostics[0]['gallery_cosine'],.55)
        self.assertGreater(tracker.diagnostics[0]['recent_cosine'],.9)

    def test_recovery_is_explicit_causal_and_resumable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'frame_cache').mkdir()
            for i in range(1,9):
                feature=np.zeros((1,768),'f4');feature[0,0 if i==1 else 1]=1
                atomic_npz(root/f'frame_cache/{i:05}.npz',boxes=np.array([[1,1,15,15]],'f4'),features=feature)
            frames=SimpleNamespace(get=lambda t:np.zeros((20,20,3),np.uint8))
            args=(root,frames,list(range(1,9)),np.arange(8),None,SimpleNamespace(identity='FAKE'),
                  'cpu','aria01','FAKE')
            with patch('builtins.input',side_effect=['0','q']):
                with self.assertRaises(KeyboardInterrupt):track_view(*args,review_tracking=True)
            with patch('builtins.input',side_effect=['0']):
                result=track_view(*args,review_tracking=True)
            self.assertEqual(result['manual_review_count'],1)
            with np.load(root/'detector_boxes.npz') as z:
                np.testing.assert_array_equal(z['bbox_valid'],[1,0,0,0,0,1,1,1])
            self.assertEqual(set(json.loads((root/'identity_choices.json').read_text())),{'1'})
            self.assertEqual(set(json.loads((root/'recovery_choices.json').read_text())),{'6'})
            with patch('builtins.input',side_effect=AssertionError('Must reuse annotations')):
                repeated=track_view(*args,review_tracking=True)
            self.assertEqual(result['valid_frames'],repeated['valid_frames'])

    def test_cache_import_excludes_identity_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'old';new=root/'new'
            (old/'aria02/frame_cache').mkdir(parents=True)
            contract={'sequence':'test','raw_frame_ids':[1],'input_files':[], 'feature_models':{},
                'detector':'fake','device':'cpu','torch':'test','requester':'aria02','sources':[],
                'tracker':{'version':'causal_dinov2_association_v1'}}
            (old/'preparation_run.json').write_text(json.dumps(contract))
            (old/'aria02/identity_choices.json').write_text('{"wrong":"identity"}')
            atomic_npz(old/'aria02/frame_cache/00001.npz',boxes=np.array([[0,0,10,10]]),features=np.ones((1,768)))
            reuse_candidate_cache(old,new,contract)
            self.assertFalse((new/'aria02/identity_choices.json').exists())
            self.assertTrue((new/'aria02/frame_cache/00001.npz').exists())
            with self.assertRaisesRegex(ValueError,'feature_models'):
                reuse_candidate_cache(old,new,{**contract,'feature_models':{'different':'weights'}})
