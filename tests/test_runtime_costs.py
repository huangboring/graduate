import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from observer_selection.networks import SelectorNetwork
from observer_selection.runtime import SelectionEngine
from observer_selection.selection import AcquisitionConfig
from observer_selection.replay import make_synthetic,load_sequence,ReplayProvider
from observer_selection.backends import WeightedJointBackend
from observer_selection.costs import CandidateCostEvaluator


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        path=Path(self.tmp.name)/'data.npz'
        make_synthetic(path,frames=5,joints=3)
        self.frames=load_sequence(path)
        self.net=SelectorNetwork(3,16)
        self.backend=WeightedJointBackend()

    def tearDown(self): self.tmp.cleanup()

    def test_score_before_acquire_and_only_selected_memory(self):
        engine=SelectionEngine(self.net,acquisition=AcquisitionConfig(1,None));f=self.frames[0];p=ReplayProvider(f)
        d=engine.score(f.observation,f.candidates)
        self.assertEqual(p.requests,[]);self.assertEqual(engine.memories,{})
        packet=p.acquire(d.selected_id,'B',f.observation.timestamp)
        engine.commit(f.observation,packet,self.backend)
        self.assertEqual(p.requests,[d.selected_id])
        self.assertEqual(set(engine.memories),{d.selected_id})

    def test_commit_rejects_replaced_observation_and_future_packet(self):
        engine=SelectionEngine(self.net,acquisition=AcquisitionConfig(1,None));f=self.frames[0]
        d=engine.score(f.observation,f.candidates)
        packet=f.packets[d.selected_id]
        with self.assertRaises(ValueError): engine.commit(replace(f.observation),packet,self.backend)
        with self.assertRaises(ValueError): engine.commit(f.observation,replace(packet,timestamp=1),self.backend)
        self.assertFalse(engine.memories)

    def test_cost_order_invariant_and_no_memory_mutation(self):
        engine=SelectionEngine(self.net);f=self.frames[0]
        d=engine.score(f.observation,f.candidates)
        evaluator=CandidateCostEvaluator(self.backend)
        a=evaluator.evaluate(f.observation,f.candidates,f.packets,tuple(engine.history),f.gt,f.gt_valid,f.coordinate_frame)
        b=evaluator.evaluate(f.observation,f.candidates[::-1],f.packets,tuple(engine.history),f.gt,f.gt_valid,f.coordinate_frame)
        np.testing.assert_allclose(a.errors_m,b.errors_m[::-1])
        self.assertFalse(engine.history);self.assertFalse(engine.memories)
        original=d.logits.detach().clone()
        c=evaluator.evaluate(f.observation,f.candidates,f.packets,(),f.gt+10,f.gt_valid,f.coordinate_frame)
        self.assertFalse(np.allclose(a.errors_m,c.errors_m))
        torch.testing.assert_close(original,d.logits.detach())

    def test_attention_memory_truncation_and_gradient(self):
        net=SelectorNetwork(3,16,'attention',2)
        h=net.initial_state()
        for f in self.frames[:3]: h=net.update(f.packets['C'],h,None)
        self.assertEqual(h.shape,(2,16))
        q=net.demand_sequence([f.observation for f in self.frames[:4]])
        net.score(q,h,torch.zeros(9),.1,True).backward()
        self.assertIsNotNone(net.memory_attention.in_proj_weight.grad)
        self.assertIsNotNone(net.observation_encoder[0].weight.grad)
        self.assertIsNotNone(net.demand_temporal.weight_hh_l0.grad)

    def test_target_reset_required(self):
        engine=SelectionEngine(self.net);f=self.frames[0]
        engine.step(f.observation,f.candidates,ReplayProvider(f),self.backend)
        with self.assertRaises(ValueError):
            engine.score(replace(self.frames[1].observation,target_id='other'),f.candidates)

if __name__=='__main__': unittest.main()
