import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from observer_selection.appearance_tracking import AppearanceTracker
from observer_selection.preparation_cache import atomic_npz,load_candidates,load_or_create_contract
from observer_selection.preparation_quality import check_preparation


class AppearanceTrackingTests(unittest.TestCase):
    def test_recover_after_long_absence_far_from_original_position(self):
        track=AppearanceTracker();track.initialize(0,[0,0,10,10],[1,0,0])
        for t in range(1,20):
            self.assertIsNone(track.step(t,np.empty((0,4)),np.empty((0,3))))
        boxes=np.array([[100,100,110,110],[0,0,10,10]])
        self.assertEqual(track.step(20,boxes,np.array([[1,0,0],[0,1,0]])),0)
        self.assertEqual(track.reason,'appearance_recovery')
        self.assertEqual(track.last_seen,20)

    def test_ambiguous_reappearance_is_missing_not_guess(self):
        track=AppearanceTracker();track.initialize(0,[0,0,10,10],[1,0,0])
        boxes=np.array([[100,100,110,110],[200,200,210,210]])
        self.assertIsNone(track.step(20,boxes,np.array([[1,0,0],[1,0,0]])))
        self.assertEqual(track.reason,'ambiguous')
        self.assertEqual(track.last_seen,0)
        self.assertEqual(len(track.gallery),1)

    def test_same_position_cannot_override_appearance_mismatch(self):
        track=AppearanceTracker();track.initialize(0,[0,0,10,10],[1,0,0])
        self.assertIsNone(track.step(1,np.array([[0,0,10,10]]),np.array([[0,1,0]])))
        self.assertEqual(track.last_seen,0)

    def test_future_or_repeated_frame_is_rejected(self):
        track=AppearanceTracker();track.initialize(2,[0,0,10,10],[1,0,0])
        for t in (1,2):
            with self.assertRaises(ValueError):track.step(t,np.empty((0,4)),np.empty((0,3)))

    def test_resume_rejects_changed_contract_and_bad_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);contract={'target':'aria01','model':'fixed'}
            load_or_create_contract(root/'run.json',contract,False)
            load_or_create_contract(root/'run.json',contract,True)
            with self.assertRaises(ValueError):load_or_create_contract(root/'run.json',{'target':'aria02'},True)
            atomic_npz(root/'frame.npz',boxes=np.array([[0,0,10,10]]),features=np.ones((1,768)))
            self.assertEqual(load_candidates(root/'frame.npz')[1].shape,(1,768))
            atomic_npz(root/'frame.npz',boxes=np.array([[0,0,-1,10]]),features=np.ones((1,768)))
            with self.assertRaises(ValueError):load_candidates(root/'frame.npz')

    def test_training_rejects_legacy_permanent_loss_before_loading_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'preparation_summary.json').write_text(json.dumps({'tracking':{'aria02':{'terminal_track_loss':True}}}))
            with self.assertRaisesRegex(ValueError,'Legacy tracker'):check_preparation(root/'replay.npz')

    def test_training_rejects_incomplete_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'preparation_run.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'incomplete'):check_preparation(root/'replay.npz')
