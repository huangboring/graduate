"""Training/artifact tests use synthetic supervision, not measured pose quality."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
sys.path.insert(0,str(ROOT/'run'))
import numpy as np
import torch
import cv2
from demo_mvp import configure_bundle
from train_mvp import validate_settings
from mvp_demo.selection import SubsetSelector, candidate_subsets, subset_masks, select
from mvp_demo.training import ground_truth, pose_quality, fit_selector, collect_records


def record(sequence, n, seed):
    generator = torch.Generator().manual_seed(seed)
    descriptors = torch.rand(n,32,generator=generator)
    geometry = torch.rand(n,6,generator=generator)
    k = min(3,n-1)
    subsets = candidate_subsets(n,k,16)
    values = descriptors[:,0]
    costs = torch.tensor([float(values.topk(k).values.sum()-values[list(s)].sum())*1000 for s in subsets])
    return dict(sequence=sequence,descriptors=descriptors,geometry=geometry,k=k,subsets=subsets,costs=costs,full_cost_mm=0.)


class TrainingTests(unittest.TestCase):
    def test_set_scores_invariant_to_camera_order(self):
        torch.manual_seed(1)
        model=SubsetSelector().eval()
        descriptors,geometry=torch.randn(7,32),torch.randn(7,6)
        masks=subset_masks([(0,2,5),(1,3,6)],7,'cpu')
        permutation=torch.tensor([6,2,0,4,1,5,3])
        torch.testing.assert_close(model(descriptors,masks,geometry),
                                  model(descriptors[permutation],masks[:,permutation],geometry[permutation]))

    def test_variable_camera_count_and_subset_limit(self):
        for n in (3,4,8,31):
            combinations=candidate_subsets(n,2,8)
            self.assertLessEqual(len(combinations),8)
            self.assertEqual(len(combinations),len(set(combinations)))
            selection=select([torch.randn(n,8,4,4)],[str(i) for i in range(n)],2,'learned',
                             scorer=SubsetSelector(),geometry=torch.randn(n,6),subset_limit=8)
            self.assertEqual(len(selection.indices),2)

    def test_quality_penalizes_missing_people_and_false_positives(self):
        truth=np.zeros((1,15,3)); visibility=np.ones((1,15),bool)
        missed=pose_quality(np.zeros((0,15,3)),truth,visibility)
        self.assertEqual(missed['cost_mm'],500)
        self.assertEqual(missed['fn'],1)
        duplicate=pose_quality(np.zeros((2,15,3)),truth,visibility)
        self.assertEqual(duplicate['fp'],1)
        self.assertEqual(duplicate['cost_mm'],500)
        self.assertEqual(pose_quality(truth,truth,visibility)['cost_mm'],0)

    def test_annotations_axes_units_visibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gt.json'
            joints=np.tile([1.,2.,3.,1.],(19,1))
            path.write_text(json.dumps({'bodies':[{'joints19':joints.reshape(-1).tolist()}]}))
            poses,vis=ground_truth(path)
            np.testing.assert_array_equal(poses[0,0],[10,30,-20])
            self.assertTrue(vis.all())

    def test_fit_exports_demo_compatible_weights_and_rejects_leakage(self):
        torch.set_num_threads(1)
        train=[record('train',n,seed) for seed,n in enumerate([3,4,6]*4)]
        val=[record('validation',n,50+seed) for seed,n in enumerate([3,4,6])]
        with tempfile.TemporaryDirectory() as directory:
            model,metrics=fit_selector(train,val,directory,epochs=4,patience=4)
            checkpoint=torch.load(Path(directory)/'selector.pt',weights_only=True)
            restored=SubsetSelector(checkpoint['dimension'],checkpoint['hidden'])
            restored.load_state_dict(checkpoint['state_dict'],strict=True)
            self.assertEqual(checkpoint['architecture'],'subset_set_v1')
            self.assertEqual(metrics['count'],3)
            self.assertTrue(np.isfinite(metrics['learned']))
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,restored.state_dict()[key])
            with self.assertRaises(ValueError):fit_selector(train,train,directory,epochs=1)

    def test_bundle_integrity_and_runtime_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            for name in ['pose.pt','config.yaml','selector.pt']:
                (directory/name).write_bytes(b'test fixture only')
            manifest=dict(schema_version=1,pose_checkpoint='pose.pt',pose_config='config.yaml',
                selector_checkpoint='selector.pt',policy='learned',k=2,subset_limit=16,threshold=.4,nms_mm=200,
                hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()})
            path=directory/'bundle.json';path.write_text(json.dumps(manifest))
            args=SimpleNamespace(bundle=path,policy=None,k=None,threshold=None,nms_mm=None,subset_limit=None)
            configure_bundle(args)
            self.assertEqual(args.k,2)
            self.assertEqual(args.policy,'learned')
            self.assertEqual(args.subset_limit,16)
            (directory/'selector.pt').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):configure_bundle(args)

    def test_config_prevents_sequence_leakage(self):
        settings=json.loads((ROOT/'configs/train_mvp.json').read_text())
        validate_settings(settings)
        settings['val_sequences']=settings['train_sequences'][:]
        with self.assertRaises(ValueError):validate_settings(settings)

    def test_video_to_labels_then_cache_reuse_with_fake_teacher(self):
        class FakeTeacher:
            calls=0
            def extract_view_features(self,views):
                self.calls+=1
                return [torch.nn.functional.adaptive_avg_pool2d(views[0],(4,4))]
            def __call__(self,views,meta,precomputed_features,threshold):
                offset=float(sum(v.mean() for v in views)) * 10
                return np.tile([10.+offset,30.,-20.],(1,15,1))
        settings=json.loads((ROOT/'configs/train_mvp.json').read_text())
        settings.update(frames_per_sequence=2,frame_stride=1,max_candidate_views=3,k_values=[2])
        cfg=SimpleNamespace(NETWORK=SimpleNamespace(IMAGE_SIZE=[64,48],HEATMAP_SIZE=[16,12]),
                            DATASET=SimpleNamespace(COLOR_RGB=True))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); sequence=root/'train'
            (sequence/'hdVideos').mkdir(parents=True)
            gt_dir=sequence/'hdPose3d_stage1_coco19';gt_dir.mkdir()
            cameras=[]
            for i in range(3):
                name=f'00_{i:02d}'
                cameras.append(dict(name=name,panel=0,node=i,resolution=[64,48],K=[[60,0,32],[0,60,24],[0,0,1]],
                                    R=np.eye(3).tolist(),t=[i*10,0,100],distCoef=[0]*5))
                writer=cv2.VideoWriter(str(sequence/'hdVideos'/f'hd_{name}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(64,48))
                self.assertTrue(writer.isOpened())
                for frame in range(2):writer.write(np.full((48,64,3),30+i*60+frame,dtype=np.uint8))
                writer.release()
            (sequence/'calibration_train.json').write_text(json.dumps({'cameras':cameras}))
            for frame in range(2):
                (gt_dir/f'body3DScene_{frame:08d}.json').write_text(json.dumps({'bodies':[{'joints19':[1,2,3,1]*19}]}))
            teacher=FakeTeacher()
            decode=lambda result,*args:(result,np.ones(len(result)))
            records=collect_records(teacher,cfg,settings,['train'],root,root/'cache','train',decode,device='cpu')
            self.assertEqual(len(records),2)
            self.assertEqual(teacher.calls,6)
            self.assertEqual(len(records[0]['subsets']),3)
            self.assertGreater(float(records[0]['costs'].max()-records[0]['costs'].min()),0)
            cached=collect_records(teacher,cfg,settings,['train'],root,root/'cache','train',decode,device='cpu')
            self.assertEqual(teacher.calls,6)
            torch.testing.assert_close(records[0]['costs'],cached[0]['costs'])


if __name__ == '__main__':
    unittest.main()
