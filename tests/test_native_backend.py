"""Run actual upstream network implementation; random tiny weights, fake FK.

Checks tensor/interface behavior only, never claims SMPL or pretrained accuracy.
"""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from observer_selection.backends import EgoExoMoCapBackend
from observer_selection.types import WarmupError
from test_backends import frame,fake_fk


class NativeBackendTests(unittest.TestCase):
    def make_backend(self):
        path=Path(__file__).resolve().parents[1]/'external/EgoExoMoCap/networks/egoexomocap.py'
        if not path.is_file():self.skipTest('Upstream source absent')
        spec=importlib.util.spec_from_file_location('native_ego_for_test',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        torch.manual_seed(14)
        net=mod.AvatarNet(egonet_arguments=dict(input_dim=20,num_layer=1,embed_dim=16,nhead=2,
            window_size=3,num_input=1,ff_dim=32),input_dim_ego=18,egonet_out_dim=66,
            num_input=1,embed_dim=16,nhead=2,num_layer_temporal=1,num_layer_spatial=1,
            window_size=3,ff_dim=32,dropout=0.0)
        backend=EgoExoMoCapBackend(SimpleNamespace(net=net,fk=fake_fk,bm=None),3)
        return backend

    def test_native_network_masking_freezing_and_repeatability(self):
        backend=self.make_backend();net=backend.net
        hist=(frame(0,local=True),frame(1,'D',local=True));f=frame(2,'C',local=True)
        before={k:v.clone() for k,v in net.state_dict().items()}
        first=backend.reconstruct(f.observation,f.packet,hist)
        again=backend.reconstruct(f.observation,f.packet,hist)
        self.assertEqual(first.joints.shape,(22,3))
        np.testing.assert_array_equal(first.joints,again.joints)
        for k,v in net.state_dict().items():torch.testing.assert_close(v,before[k])
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in net.parameters()))
        f.packet.payload['bbox_valid']=False
        masked=backend.reconstruct(f.observation,f.packet,hist)
        f.packet.payload['exo_rays']=np.full(39,10000,dtype='f4')
        f.packet.payload['dino_cls']=np.full(768,-999,dtype='f4')
        masked_again=backend.reconstruct(f.observation,f.packet,hist)
        np.testing.assert_array_equal(masked.joints,masked_again.joints)

    def test_export_replay_native_reconstruction_and_selection_loss(self):
        # Tiny generated files + random upstream network + fake FK: interface
        # verification only. No real observations, weights, or body-model claims.
        from test_egohumans import EgoHumansExportTests
        from observer_selection.protocol import describe_sequence
        from observer_selection.replay import load_sequence,ReplayProvider
        from observer_selection.networks import SelectorNetwork
        from observer_selection.runtime import SelectionEngine
        from observer_selection.costs import SubsetCostEvaluator
        from observer_selection.training import expected_mpjpe_loss
        fixture=EgoHumansExportTests();fixture.setUp();self.addCleanup(fixture.tearDown)
        exported=fixture.export()
        self.assertEqual(describe_sequence(exported['output'])['frames'],3)
        frames=load_sequence(exported['output'])
        backend=self.make_backend();network=SelectorNetwork(13,16)
        engine=SelectionEngine(network)
        for frame in frames[:2]:
            decision=engine.score(frame.observation,frame.candidates)
            provider=ReplayProvider(frame)
            prediction=engine.commit(frame.observation,provider.acquire_many(decision.selected_ids,
                frame.observation.target_id,frame.observation.timestamp),backend)
            self.assertIsNone(prediction)
            self.assertEqual(len(provider.requests),2)
            self.assertEqual(len(provider.batch_requests),1)
        frame=frames[2]
        decision=engine.score(frame.observation,frame.candidates)
        history_before=tuple(engine.history)
        counts={s:m.count for s,m in engine.memories.items()}
        distribution=engine.subset_distribution(decision)
        costs=SubsetCostEvaluator(backend).evaluate(frame.observation,distribution.subsets,
            frame.packets,history_before,frame.gt,frame.gt_valid,frame.coordinate_frame)
        self.assertEqual(len(engine.history),2)
        self.assertEqual({s:m.count for s,m in engine.memories.items()},counts)
        loss=expected_mpjpe_loss(distribution.logits,torch.tensor(costs.errors_m))
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(any(p.grad is not None for p in network.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in backend.net.parameters()))
        result=engine.commit(frame.observation,tuple(frame.packets[s] for s in decision.selected_ids),backend)
        np.testing.assert_array_equal(result.joints,
            costs.predictions[distribution.subsets.index(decision.selected_ids)].joints)

    def test_warmup_does_not_hide_bad_coordinates(self):
        from test_backends import FakeNet
        b=EgoExoMoCapBackend(SimpleNamespace(net=FakeNet(),fk=fake_fk,bm=None),3)
        f=frame(0);f.observation.backend_context.pop('coordinate_convention')
        with self.assertRaises(ValueError):b.reconstruct(f.observation,f.packet,())
        f=frame(0)
        with self.assertRaises(WarmupError):b.reconstruct(f.observation,f.packet,())

    def test_native_joint_subset_costs_train_only_selector_scores(self):
        from observer_selection.types import AcquiredFrame
        from observer_selection.costs import SubsetCostEvaluator
        from observer_selection.selection import AdditiveSubsetPolicy
        from observer_selection.training import expected_mpjpe_loss
        backend=self.make_backend()
        c0,d0=frame(0,'C',True),frame(0,'D',True)
        d1,e1=frame(1,'D',True),frame(1,'E',True)
        history=(AcquiredFrame(c0.observation,(c0.packet,d0.packet)),
                 AcquiredFrame(d1.observation,(d1.packet,e1.packet)))
        current={s:frame(2,s,True) for s in ('C','D','E')}
        for i,source in enumerate(current):
            current[source].packet.payload['exo_rays']=np.full(39,i*7+1,dtype='f4')
            current[source].packet.payload['dino_cls']=np.full(768,i*3+1,dtype='f4')
        obs=current['C'].observation;packets={s:f.packet for s,f in current.items()}
        logits=torch.tensor([.2,.0,-.3],requires_grad=True)
        dist=AdditiveSubsetPolicy().distribution(logits,('C','D','E'),2)
        evaluator=SubsetCostEvaluator(backend)
        costs=evaluator.evaluate(obs,dist.subsets,packets,history,np.zeros((22,3)),np.ones(22,bool),'root_relative')
        self.assertGreater(float(costs.errors_m.std()),1e-8)
        expected_mpjpe_loss(dist.logits,torch.tensor(costs.errors_m)).backward()
        self.assertGreater(float(logits.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in backend.net.parameters()))
        old_cd=costs.predictions[0].joints.copy()
        packets['E'].payload['exo_rays']*=100
        cd=evaluator.evaluate(obs,(('C','D'),),packets,history,np.zeros((22,3)),np.ones(22,bool),'root_relative')
        np.testing.assert_array_equal(old_cd,cd.predictions[0].joints)
