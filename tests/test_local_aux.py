"""Geometry/leakage/training integration, not real model accuracy.

Tiny random official fusion modules, generated observations and explicit fake FK.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from observer_selection.local_fusion import (CONVENTION, OUTPUT_AXES, LocalFusionConfig,
    LocalAuxFusionNet, LocalAuxBackend, build_local_batch, view_features)
from observer_selection.types import Observation, Packet, AcquiredFrame, WarmupError

REPO=Path(__file__).resolve().parents[1]/'external/EgoExoMoCap'


def frame(t, source='C', offset=1.):
    K=np.array([[100,0,50],[0,100,50],[0,0,1]],'f4')
    pose=np.eye(4,dtype='f4'); pose[0,3]=t*.1
    common={'coordinate_convention':CONVENTION,'camera_provenance':'synthetic_demo','output_axes':OUTPUT_AXES}
    xy=np.tile([50,50],(13,1)).astype('f4')
    obs=Observation('B',t,xy,np.ones(13,bool),np.array([0,0,100,100]),
        backend_context={**common,'local_intrinsics':K,'local_camera_to_world':pose,
                         'local_dino_cls':np.ones(768,'f4'),'local_bbox_valid':True})
    remote=pose.copy(); remote[0,3]+=offset
    packet=Packet(source,'B',t,xy.copy(),np.ones(13,bool),np.array([0,0,100,100]),
        payload={**common,'intrinsics':K.copy(),'camera_to_world':remote,
                 'dino_cls':np.ones(768,'f4')*.5,'bbox_valid':True})
    return AcquiredFrame(obs,(packet,))


def fake_decode(output):
    # Differentiable wiring fixture, not SMPL or a scientific baseline.
    points=output['pose_body'].reshape(-1,42,3)[:,:22]
    return points-points[:,:1]


def net():
    torch.manual_seed(12)
    return LocalAuxFusionNet(REPO,LocalFusionConfig(window_size=3,embed_dim=16,nhead=2,
        num_layer_spatial=1,num_layer_temporal=1,ff_dim=32,dropout=0))


class LocalAuxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_camera_geometry_and_shared_window_reference(self):
        history=(frame(0),frame(1,'D')); current=frame(2)
        batch,ids=build_local_batch(current.observation,current.packets,history,3)
        self.assertEqual(ids,['C','D'])
        np.testing.assert_allclose(batch['local_geometry'][0,:,0,0],[-.2,-.1,0],atol=1e-6)
        np.testing.assert_allclose(batch['remote_geometry'][0,0,0,0,:6],[.8,0,0,0,0,1],atol=1e-6)
        self.assertEqual(batch['remote_valid'].tolist(),[[[True,False],[False,True],[True,False]]])
        # Rigid world gauge changes must not affect final requester-frame inputs.
        angle=.8;c,s=np.cos(angle),np.sin(angle)
        G=np.array([[c,-s,0,9],[s,c,0,-3],[0,0,1,2],[0,0,0,1]])
        transformed=[]
        for f in (*history,current):
            ctx=dict(f.observation.backend_context)
            ctx['local_camera_to_world']=G@ctx['local_camera_to_world']
            p=f.packets[0];payload=dict(p.payload);payload['camera_to_world']=G@payload['camera_to_world']
            transformed.append(AcquiredFrame(replace(f.observation,backend_context=ctx),(replace(p,payload=payload),)))
        other,_=build_local_batch(transformed[-1].observation,transformed[-1].packets,transformed[:-1],3)
        for key in batch: torch.testing.assert_close(batch[key],other[key],atol=2e-6,rtol=1e-5)

    def test_no_target_tracker_no_unacquired_history(self):
        f=frame(2)
        batch,ids=build_local_batch(f.observation,f.packets,(frame(0,'D'),frame(1,'D')),3)
        self.assertFalse(batch['remote_valid'][0,0,ids.index('C')])
        f.observation.backend_context['sparse_input']=np.zeros(18)
        with self.assertRaisesRegex(ValueError,'forbidden'):
            build_local_batch(f.observation,f.packets,(frame(0),frame(1)),3)

    def test_warmup_and_bad_history(self):
        f=frame(0)
        with self.assertRaises(WarmupError):build_local_batch(f.observation,f.packets,(),3)
        f.observation.backend_context['local_intrinsics'][0,0]=0
        with self.assertRaisesRegex(ValueError,'intrinsics'):build_local_batch(f.observation,f.packets,(),3)
        with self.assertRaisesRegex(ValueError,'chronological'):
            build_local_batch(frame(1).observation,frame(1).packets,(frame(2),),3)

    def test_masked_views_and_local_only_are_finite(self):
        n=net().eval(); f=frame(2)
        batch,_=build_local_batch(f.observation,f.packets,(frame(0),frame(1,'D')),3)
        a=n(batch)['pose_body']
        mask=~batch['remote_valid']
        batch['remote_geometry'][mask]=10000;batch['remote_dino'][mask]=-999
        torch.testing.assert_close(a,n(batch)['pose_body'],atol=0,rtol=0)
        history=tuple(SimpleNamespace(observation=frame(t).observation,packets=()) for t in (0,1))
        local,_=build_local_batch(f.observation,(),history,3)
        local['local_valid'][:]=False
        result=n(local)
        self.assertEqual(result['pose_body'].shape,(1,1,126))
        self.assertTrue(torch.isfinite(result['pose_body']).all())

    def test_gradients_and_frozen_selector_costs(self):
        from observer_selection.networks import SelectorNetwork
        from observer_selection.runtime import SelectionEngine
        from observer_selection.selection import AcquisitionConfig
        from observer_selection.costs import SubsetCostEvaluator
        from observer_selection.training import expected_mpjpe_loss
        n=net(); f=frame(2)
        batch,_=build_local_batch(f.observation,f.packets,(frame(0),frame(1)),3)
        fake_decode(n(batch)).square().mean().backward()
        self.assertGreater(float(n.local_embed[0].weight.grad.norm()),0)
        n.zero_grad(set_to_none=True)
        backend=LocalAuxBackend(n,fake_decode)
        first=backend.reconstruct(f.observation,f.packet,(frame(0),frame(1)))
        repeated=backend.reconstruct(f.observation,f.packets,(frame(0),frame(1)))
        np.testing.assert_array_equal(first.joints,repeated.joints)
        selector=SelectorNetwork(13,16)
        engine=SelectionEngine(selector,acquisition=AcquisitionConfig(1,None,1,4096))
        for t in (0,1):
            old=frame(t);chosen=engine.score(old.observation,('C','D')).selected_ids
            engine.commit(old.observation,(replace(old.packet,observer_id=chosen[0]),),backend)
        decision=engine.score(f.observation,('C','D'));distribution=engine.subset_distribution(decision)
        packets={'C':f.packet,'D':frame(2,'D',-2).packet}
        costs=SubsetCostEvaluator(backend).evaluate(f.observation,distribution.subsets,packets,
            tuple(engine.history),np.zeros((22,3),'f4'),np.ones(22,bool),'root_relative')
        loss=expected_mpjpe_loss(distribution.logits,torch.tensor(costs.errors_m));loss.backward()
        self.assertTrue(any(p.grad is not None for p in selector.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in n.parameters()))

    def test_export_without_tracker_and_gt_is_labels_only(self):
        from test_egohumans import EgoHumansExportTests
        from observer_selection.local_data import export_local_egohumans
        from observer_selection.replay import load_sequence
        fixture=EgoHumansExportTests();fixture.setUp();self.addCleanup(fixture.tearDown)
        m=fixture.manifest
        for k in ('target_tracker','floor','world_to_zup'):m.pop(k)
        m.update(schema_version=2,camera_calibration={'provenance':'calibrated','description':'TEST FIXTURE',
                 'target_pose_used':False,'rectified_pinhole':True})
        fixture.save_manifest();fixture.trackpath.unlink()
        a=export_local_egohumans(fixture.mp,fixture.root/'local.npz')
        frames=load_sequence(a['output'])
        self.assertEqual(len(frames),4)
        self.assertNotIn('tracker_provenance',frames[0].observation.backend_context)
        f=frames[2]
        history=tuple(AcquiredFrame(x.observation,(next(iter(x.packets.values())),)) for x in frames[:2])
        backend=LocalAuxBackend(net(),fake_decode)
        self.assertEqual(backend.reconstruct(f.observation,tuple(f.packets.values()),history).joints.shape,(22,3))
        for pair in fixture.pairs.values():
            path=Path(pair)/'wearer_smplh_fk.npz'
            with np.load(path) as z: gt=z['joints_pos'].copy()
            gt[:,5,0]+=2;np.savez(path,joints_pos=gt)
        b=export_local_egohumans(fixture.mp,fixture.root/'changed.npz')
        with np.load(a['output']) as x,np.load(b['output']) as y:
            for key in x.files:
                if key!='gt_joints': np.testing.assert_array_equal(x[key],y[key],err_msg=key)
            self.assertFalse(np.array_equal(x['gt_joints'],y['gt_joints']))

    def test_backend_split_leakage(self):
        from observer_selection.local_training import check_backend_splits
        row={'scene_id':'train','sha256':'abc','is_synthetic':False}
        b=SimpleNamespace(provenance={'training_data_manifest':{'train':[row],'validation':[]}})
        with self.assertRaisesRegex(ValueError,'overlaps'):check_backend_splits(b,{'test':[row]})
        check_backend_splits(b,{'train':[row]})

    def test_upstream_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'wrong.pt';torch.save({'params':{}},p)
            with self.assertRaisesRegex(ValueError,'trained local_aux'):
                LocalAuxBackend.from_checkpoint(REPO,p,'missing.npz')

    def test_stage1_training_checkpoint_then_selector_training(self):
        from test_egohumans import EgoHumansExportTests
        from observer_selection.local_data import export_local_egohumans
        from observer_selection.local_training import train_local_reconstructor
        from observer_selection.training import train, TrainConfig, load_network, evaluate
        from observer_selection.selection import AcquisitionConfig
        from observer_selection.protocol import verify_heldout
        from observer_selection.replay import load_sequence
        fixture=EgoHumansExportTests();fixture.setUp();self.addCleanup(fixture.tearDown)
        m=fixture.manifest
        for k in ('target_tracker','floor','world_to_zup'):m.pop(k)
        m.update(schema_version=2,camera_calibration={'provenance':'calibrated','description':'TEST FIXTURE',
                 'target_pose_used':False,'rectified_pinhole':True})
        paths={}
        for split in ('train','validation','test'):
            m['scene_id']=split
            for pair in fixture.pairs.values():
                p=Path(pair)/'meta.json';meta=json.loads(p.read_text());meta['source_scene']=split;p.write_text(json.dumps(meta))
                # Feature receipt hashes metadata, so regenerate its digest after
                # this fixture-only scene rename; production exports never do this.
                from observer_selection.egohumans import _hash
                receipt=Path(pair)/'selector_preparation.json';r=json.loads(receipt.read_text())
                for item in r['input_files']:
                    if Path(item['path'])==p:item['sha256']=_hash(p)
                receipt.write_text(json.dumps(r))
            fixture.save_manifest()
            paths[split]=[export_local_egohumans(fixture.mp,fixture.root/f'{split}.npz')['output']]
        protocol=fixture.root/'protocol.json';protocol.write_text(json.dumps(paths))
        body=fixture.root/'fake_body.npz';body.write_bytes(b'EXPLICIT FAKE FK FIXTURE')
        config=LocalFusionConfig(window_size=3,embed_dim=16,nhead=2,num_layer_spatial=1,num_layer_temporal=1,ff_dim=32,dropout=0)
        with patch('observer_selection.local_training.load_body_decoder',return_value=fake_decode):
            result=train_local_reconstructor(protocol,fixture.root/'reconstructor',REPO,body,
                config,epochs=1,max_sources=1)
        with patch('observer_selection.local_fusion.load_body_decoder',return_value=fake_decode):
            backend=LocalAuxBackend.from_checkpoint(REPO,result['checkpoint'],body)
        before={k:v.clone() for k,v in backend.net.state_dict().items()}
        acq=AcquisitionConfig(1,None,1,4096)
        train(paths['train'],fixture.root/'selector',backend,
            TrainConfig(epochs=1,hidden=16,acquisition=acq),paths['validation'])
        selector=load_network(fixture.root/'selector/best.pt')
        verify_heldout(selector,paths['test'],backend)
        report=evaluate(selector,[load_sequence(paths['test'][0])],backend,policies=['learned'])
        self.assertEqual(report['policies']['learned']['evaluated_frames'],2)
        for k,v in before.items():torch.testing.assert_close(v,backend.net.state_dict()[k],atol=0,rtol=0)
