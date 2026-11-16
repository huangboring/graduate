import unittest
import tempfile
import json
import pickle
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import numpy as np
from observer_selection.raw_prepare import (fisheye_project, upright_camera, Q_UPRIGHT, match_track,
    similarity_camera_extrinsic,validate_setup_restart,check_existing_setup)


class RawGeometryTests(unittest.TestCase):
    def test_similarity_preserves_camera_centre_and_projection(self):
        angle=.4
        R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        for s in (.4,1.,2.7):
            ext=np.eye(4);ext[:3,:3]=s*R;ext[:3,3]=[.5,-.2,1.3]
            normalized,scale=similarity_camera_extrinsic(ext)
            self.assertAlmostEqual(scale,s)
            np.testing.assert_allclose(normalized[:3,3],ext[:3,3]/s)
            camera=upright_camera(ext)
            np.testing.assert_allclose(camera[:3,3],np.linalg.inv(ext)[:3,3],atol=1e-12)
            points=np.array([[.3,-.1,3,1],[-1,.2,4,1],[2,-.3,6,1]])
            original=(ext@points.T).T[:,:3]@Q_UPRIGHT.T
            recovered=(np.linalg.inv(camera)@points.T).T[:,:3]
            # rigid() shares the model's float32 input contract; tolerate only
            # float32 roundoff (well below 0.001 pixels at focal length 704).
            np.testing.assert_allclose(original[:,:2]/original[:,2:],recovered[:,:2]/recovered[:,2:],atol=1e-7)

    def test_reject_shear_nonuniform_scale_reflection_and_bad_bottom_row(self):
        matrices=[np.diag([1,2,1,1]),np.diag([-1,1,1,1]),np.zeros((4,4))]
        shear=np.eye(4);shear[0,1]=.1;matrices.append(shear)
        perspective=np.eye(4);perspective[3,0]=.1;matrices.append(perspective)
        for matrix in matrices:
            with self.assertRaises(ValueError): similarity_camera_extrinsic(matrix)

    def test_restart_preserves_weights_and_refuses_observation_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'model_cache/checkpoints').mkdir(parents=True)
            weights=out/'model_cache/checkpoints/fake.pth';weights.write_bytes(b'keep')
            pair=out/'aria02';pair.mkdir()
            (pair/'meta.json').write_text('{"target":"aria01"}')
            (pair/'timestamps.csv').write_text('frame_idx,t_ns\n0,0\n')
            validate_setup_restart(out,('aria02','aria03','aria04'))
            check_existing_setup(pair,{'target':'aria01'},'frame_idx,t_ns\n0,0\n')
            with self.assertRaises(ValueError): check_existing_setup(pair,{'target':'aria04'},'frame_idx,t_ns\n0,0\n')
            self.assertEqual(weights.read_bytes(),b'keep')
            (pair/'tracking_preview.mp4').write_bytes(b'keep observations')
            with self.assertRaises(FileExistsError): validate_setup_restart(out,('aria02',))
    def test_optical_axis_and_radial_projection(self):
        intr=np.zeros(15);intr[:3]=[600,700,710]
        points=np.array([[0,0,1],[1,0,1],[0,1,1]])
        actual=fisheye_project(points,intr)
        np.testing.assert_allclose(actual,[[700,710],[700+600*np.pi/4,710],[700,710+600*np.pi/4]])

    def test_clockwise_image_and_camera_rotation_agree(self):
        K=np.array([[704,0,703.5],[0,704,703.5],[0,0,1.]])
        point=np.array([.3,-.2,2.])
        native=K@point;native=native[:2]/native[2]
        expected=np.array([1407-native[1],native[0]])
        camera=upright_camera(np.eye(4))
        projected=K@(camera[:3,:3].T@point)
        np.testing.assert_allclose(projected[:2]/projected[2],expected)
        np.testing.assert_allclose(camera[:3,:3],Q_UPRIGHT.T)

    def test_tracker_rejects_ambiguity_and_nonoverlap(self):
        previous=np.array([0,0,10,10])
        self.assertEqual(match_track(previous,np.array([[1,0,11,10],[30,30,40,40]])),0)
        self.assertIsNone(match_track(previous,np.array([[0,0,10,10],[0,0,10,10]])))
        self.assertIsNone(match_track(previous,np.array([[30,30,40,40]])))
        self.assertIsNone(match_track(None,np.array([[0,0,10,10]])))

    def test_raw_pipeline_with_explicit_fake_models(self):
        # Integration fixture only: model and FK outputs are deliberately fake.
        # Exercise actual calibration, provenance, exporter and replay contracts.
        import torch
        import cv2
        from observer_selection.raw_prepare import prepare_raw
        from observer_selection.replay import load_sequence
        counter={'calls':0}
        class Detector:
            def eval(self): return self
            def to(self,device): return self
            def __call__(self,images):
                counter['calls']+=1
                return [{'labels':torch.tensor([1]),'scores':torch.tensor([.9]),
                         'boxes':torch.tensor([[100.,100.,600.,900.]])}]
        class Dino:
            identity='EXPLICIT_FAKE_DINO'
            def __init__(self,*args,**kwargs): pass
            def batch_cls(self,images): return np.ones((len(images),768),'f4')
        class Body:
            def __init__(self,*args,**kwargs): pass
            def eval(self): return self
            def to(self,*args): return self
            def __call__(self,**kwargs):
                points=torch.zeros(1,52,3);points[0,:,1]=torch.arange(52)*.01
                return SimpleNamespace(Jtr=points)
        transformers=ModuleType('transformers')
        transformers.AutoImageProcessor=SimpleNamespace(from_pretrained=lambda *a,**k:None)
        bodymodule=ModuleType('human_body_prior.body_model.body_model');bodymodule.BodyModel=Body
        old_hub=torch.hub.get_dir()
        self.addCleanup(torch.hub.set_dir,old_hub)
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);root=base/'01_tagging/001_tagging';repo=base/'repo';out=base/'output'
            for aria in ('aria02','aria03','aria04'):
                folder=root/'ego'/aria
                (folder/'calib').mkdir(parents=True);(folder/'images/rgb').mkdir(parents=True)
                intr='600 703.5 703.5 '+' '.join(['0']*12)
                ext=' '.join(map(str,np.eye(4)[:3].T.reshape(-1)))
                for frame in (1,2):
                    (folder/f'calib/{frame:05}.txt').write_text('header\naria\n'+('\n'.join([intr,ext]*3)))
                    cv2.imwrite(str(folder/f'images/rgb/{frame:05}.jpg'),np.zeros((1408,1408,3),np.uint8))
            (root/'colmap/workplace').mkdir(parents=True)
            with (root/'colmap/workplace/colmap_from_aria_transforms.pkl').open('wb') as f:
                transforms={a:np.diag([scale,scale,scale,1.]) for a,scale in
                            zip(('aria01','aria02','aria03','aria04'),(1.,2.,3.,4.))}
                pickle.dump(transforms,f)
            (root/'processed_data/smpl').mkdir(parents=True)
            for frame in (1,2):
                np.save(root/f'processed_data/smpl/{frame:05}.npy',{'aria01':{
                    'global_orient':np.zeros(3),'body_pose':np.zeros(69),'transl':np.zeros(3),'betas':np.zeros(10)}})
            helper=repo/'data_prep/shared/vitpose_predictor.py';helper.parent.mkdir(parents=True)
            helper.write_text('import numpy as np\nfrom types import SimpleNamespace\nclass ViTPosePredictor:\n'
                '    def __init__(self,**kwargs): pass\n'
                '    def batch(self,images,boxes): return [SimpleNamespace(xy=np.full((1,13,2),200,dtype="f4"),scores=np.full((1,13),.8,dtype="f4"))]\n')
            vitdir=repo/'third_party/vitpose_hf/vitpose-base-simple';vitdir.mkdir(parents=True)
            for name in ('config.json','preprocessor_config.json','model.safetensors'):(vitdir/name).write_text('EXPLICIT_FIXTURE')
            body=base/'fake_body.npz';body.write_bytes(b'EXPLICIT_FAKE_FK')
            def make_detector(**kwargs):
                cache=Path(torch.hub.get_dir())/'checkpoints';cache.mkdir(parents=True,exist_ok=True)
                (cache/'fasterrcnn_resnet50_fpn-fixture.pth').write_bytes(b'EXPLICIT_FAKE_DETECTOR')
                return Detector()
            with patch.dict('sys.modules',{'transformers':transformers,'human_body_prior.body_model.body_model':bodymodule}), \
                 patch('torchvision.models.detection.fasterrcnn_resnet50_fpn',side_effect=make_detector), \
                 patch('observer_selection.image_features.DinoV2Encoder',Dino):
                with patch('builtins.input',side_effect=['0','q']):
                    with self.assertRaises(KeyboardInterrupt):prepare_raw(root,out,repo=repo,body_model=body,device='cpu')
                self.assertEqual(counter['calls'],3)
                with patch('builtins.input',side_effect=['0','0']):
                    result=prepare_raw(root,out,repo=repo,body_model=body,device='cpu',resume=True)
                self.assertEqual(counter['calls'],6)  # all 3x2 frames inferred exactly once
                self.assertEqual(result['tracking']['aria03']['cache_hits'],1)
                with patch('builtins.input',side_effect=AssertionError('Completed run must not ask again')):
                    repeated=prepare_raw(root,out,repo=repo,body_model=body,device='cpu',resume=True)
                self.assertEqual(counter['calls'],6)
                self.assertEqual(repeated['output'],result['output'])
                with patch('builtins.input',return_value='0'):
                    prepare_raw(root,out,repo=repo,body_model=body,device='cpu',resume=True,reset_views=['aria02'])
                self.assertEqual(counter['calls'],6)  # identity correction reuses image inference
                with patch('builtins.input',side_effect=['0','0','0']):
                    reviewed=prepare_raw(root,base/'reviewed',repo=repo,body_model=body,device='cpu',
                                         review_tracking=True,reuse_candidates=out)
                self.assertEqual(counter['calls'],6)  # algorithm change must reuse detector inference
                self.assertEqual(reviewed['tracking']['aria02']['identity_protocol'],'manual_initial_and_recovery_review')
                review_receipt=json.loads((base/'reviewed/aria02/selector_preparation.json').read_text())
                self.assertEqual(review_receipt['identity_source'],'manual_initial_and_recovery_review')
            sequence=load_sequence(result['output'])
            self.assertEqual(len(sequence),2)
            self.assertEqual(sequence[0].candidates,('aria03','aria04'))
            self.assertEqual(result['tracking']['aria02']['valid_frames'],2)
            self.assertTrue((out/'aria02/tracking_preview.mp4').is_file())
            receipt=json.loads((out/'replay.provenance.json').read_text())
            self.assertFalse(receipt['target_tracker_used'])
            self.assertAlmostEqual(receipt['camera_calibration']['alignment_scales']['aria02'],.5)
            from observer_selection.finalize_raw import finalize_raw
            from observer_selection.preparation_quality import check_preparation
            from observer_selection.egohumans import _hash
            original=_hash(out/'aria02/selector_observations.npz')
            with patch('observer_selection.raw_tracking.extract_candidates',side_effect=AssertionError('No GPU reprocessing')):
                finalized=finalize_raw(out,base/'finalized')
            self.assertEqual(original,_hash(out/'aria02/selector_observations.npz'))
            self.assertEqual(check_preparation(finalized['output'])['frames'],2)
            self.assertEqual(len(load_sequence(finalized['output'])),2)
            # A changed source must fail before any new export is written.
            (root/'ego/aria02/calib/00001.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Original raw input changed'):
                finalize_raw(out,base/'invalid_finalization')
            self.assertFalse((base/'invalid_finalization').exists())
