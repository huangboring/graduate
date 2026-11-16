"""Batch acquisition, fairness guarantees, and true subset-cost gradients."""
import math
import random
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from observer_selection.selection import AcquisitionConfig, AdditiveSubsetPolicy, plan_refresh
from observer_selection.runtime import SelectionEngine, SourceMemory
from observer_selection.networks import SelectorNetwork
from observer_selection.replay import make_synthetic, load_sequence, ReplayProvider
from observer_selection.backends import WeightedJointBackend, EgoExoMoCapBackend
from observer_selection.costs import SubsetCostEvaluator
from observer_selection.training import expected_mpjpe_loss, TrainConfig, train, load_network
from observer_selection.types import Observation, Packet, AcquiredFrame


class RefreshTests(unittest.TestCase):
    def test_adversarial_scores_cannot_starve_any_stable_source(self):
        for n,k,slots in ((5,2,1),(5,1,1),(7,3,2)):
            ids=tuple(f's{i}' for i in range(n));dt=.1;threshold=.4
            config=AcquisitionConfig(k,threshold,slots)
            first={s:0. for s in ids};last={};times={s:[] for s in ids}
            logits=torch.arange(n,dtype=torch.float64)*-1000
            chooser=AdditiveSubsetPolicy()
            for t in range(300):
                now=t*dt
                refresh=plan_refresh(ids,now,first,last,config)
                chosen=chooser.map(logits,ids,k,refresh.required_ids)
                self.assertEqual(len(chosen),k)
                for s in chosen:last[s]=now;times[s].append(now)
            bound=threshold+math.ceil(n/slots)*dt
            for source in ids:
                self.assertGreater(len(times[source]),1)
                gaps=np.diff([0.,*times[source],299*dt])
                self.assertLessEqual(float(gaps.max()),bound+1e-8)

    def test_old_overdue_sources_precede_new_arrivals(self):
        config=AcquisitionConfig(2,1.,1)
        plan=plan_refresh(('new','old'),5.,{'new':5.,'old':0.},{'old':0.},config)
        self.assertEqual(plan.required_ids,('old',))
        self.assertEqual(plan.reasons,('expired',))

    def test_normal_selection_resets_deadline_and_fresh_sources_are_not_forced(self):
        config=AcquisitionConfig(2,1.,1)
        plan=plan_refresh(('C','D'),1.2,{'C':0.,'D':0.},{'C':.5,'D':.7},config)
        self.assertEqual(plan.required_ids,())
        plan=plan_refresh(('C','D'),1.6,{'C':0.,'D':0.},{'C':.5,'D':.7},config)
        self.assertEqual(plan.required_ids,('C',))

    def test_explicit_no_refresh_ablation_has_no_guarantee(self):
        plan=plan_refresh(('C','D'),100.,{'C':0.,'D':0.},{},AcquisitionConfig(2,None))
        self.assertEqual(plan.required_ids,())

    def test_invalid_configuration_rejected(self):
        for kwargs in ({'max_sources':0},{'refresh_after_s':0},{'refresh_after_s':float('nan')},
                       {'refresh_slots':3},{'refresh_slots':0},{'max_exact_subsets':0}):
            with self.assertRaises(ValueError):AcquisitionConfig(**kwargs)


class SubsetPolicyTests(unittest.TestCase):
    def test_map_matches_exact_distribution_and_respects_refresh(self):
        policy=AdditiveSubsetPolicy();ids=('C','D','E','F')
        z=torch.tensor([3.,2.,1.,-100.],requires_grad=True)
        distribution=policy.distribution(z,ids,2,('F',))
        self.assertEqual(distribution.subsets,(('C','F'),('D','F'),('E','F')))
        chosen=policy.map(z,ids,2,('F',))
        self.assertEqual(chosen,distribution.subsets[int(distribution.logits.argmax())])
        loss=expected_mpjpe_loss(distribution.logits,torch.tensor([.04,.08,.01]))
        loss.backward()
        self.assertAlmostEqual(float(z.grad[3]),0.,places=7)
        self.assertGreater(float(z.grad[:3].abs().sum()),0)

    def test_exact_subset_gradient_equals_inclusion_formula(self):
        ids=('C','D','E');z=torch.tensor([.2,-.1,.4],dtype=torch.float64,requires_grad=True)
        dist=AdditiveSubsetPolicy().distribution(z,ids,2)
        costs=torch.tensor([.10,.01,.06],dtype=torch.float64,requires_grad=True)
        loss=expected_mpjpe_loss(dist.logits,costs)
        loss.backward()
        membership=torch.tensor([[1,1,0],[1,0,1],[0,1,1]],dtype=torch.float64)
        analytic=membership.T@(dist.probabilities.detach()*(costs.detach()-loss.detach()))
        torch.testing.assert_close(z.grad,analytic)
        self.assertIsNone(costs.grad)

    def test_enumeration_limit_never_silently_truncates_runtime_map_still_works(self):
        policy=AdditiveSubsetPolicy();ids=tuple(map(str,range(30)));z=torch.arange(30.)
        self.assertEqual(len(policy.map(z,ids,10)),10)
        with self.assertRaisesRegex(ValueError,'No silent truncation'):
            policy.distribution(z,ids,10,max_subsets=100)

    def test_sampling_is_seeded_unique_and_always_contains_refresh(self):
        dist=AdditiveSubsetPolicy().distribution(torch.zeros(5),tuple('CDEFG'),3,('G',))
        a=random.Random(3);b=random.Random(3)
        for _ in range(50):
            subset=dist.sample(a)
            self.assertEqual(subset,dist.sample(b));self.assertIn('G',subset)
            self.assertEqual(len(set(subset)),3)


class BatchRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'fixture.npz'
        make_synthetic(self.path,frames=8,joints=3)
        self.frames=load_sequence(self.path)
        self.net=SelectorNetwork(3,16)
        self.engine=SelectionEngine(self.net,acquisition=AcquisitionConfig(2,.1,1))
        self.backend=WeightedJointBackend()

    def test_complete_immutable_decision_precedes_single_provider_call(self):
        f=self.frames[0];engine=self.engine
        class Provider(ReplayProvider):
            def acquire_many(provider,ids,target,ts):
                self.assertEqual(engine._pending.selected_ids,ids)
                self.assertEqual(engine.memories,{})
                with self.assertRaises(FrozenInstanceError):engine._pending.selected_ids=('E',)
                return super().acquire_many(ids,target,ts)
        provider=Provider(f)
        decision,_=engine.step(f.observation,f.candidates,provider,self.backend)
        self.assertEqual(provider.batch_requests,[decision.selected_ids])
        self.assertEqual(len(provider.requests),2)
        self.assertEqual(set(engine.memories),set(decision.selected_ids))
        self.assertEqual(len(engine.history[0].packets),2)
        self.assertTrue(all(m.hidden.grad_fn is None for m in engine.memories.values()))

    def test_partial_bad_extra_and_duplicate_batches_are_atomic(self):
        f=self.frames[0];engine=self.engine;decision=engine.score(f.observation,f.candidates)
        packets=tuple(f.packets[s] for s in decision.selected_ids)
        bad=[packets[:1],(*packets[:1],replace(packets[1],timestamp=99)),
             tuple(f.packets.values()),(packets[0],packets[0])]
        for batch in bad:
            with self.assertRaises(ValueError):engine.commit(f.observation,batch,self.backend)
            self.assertEqual(engine.memories,{})
            self.assertEqual(engine.history,[])
        engine.commit(f.observation,packets,self.backend)
        self.assertEqual(set(engine.memories),set(decision.selected_ids))

    def test_later_batch_does_not_update_unselected_memory(self):
        for frame in self.frames[:2]:
            self.engine.step(frame.observation,frame.candidates,ReplayProvider(frame),self.backend)
        before={s:(m.hidden.clone(),m.last_timestamp,m.count) for s,m in self.engine.memories.items()}
        f=self.frames[2];d,_=self.engine.step(f.observation,f.candidates,ReplayProvider(f),self.backend)
        for source in set(before)-set(d.selected_ids):
            m=self.engine.memories[source];torch.testing.assert_close(m.hidden,before[source][0])
            self.assertEqual((m.last_timestamp,m.count),before[source][1:])

    def test_joint_subset_cost_is_not_sum_of_singleton_errors(self):
        obs=Observation('B',0.,np.zeros((2,2)),np.ones(2,bool),np.array([0,0,10,10]))
        def pkt(s,x):return Packet(s,'B',0.,obs.xy,obs.valid,obs.bbox,
            payload={'joints3d':np.array([[0,0,0],[x,0,0]],dtype='f4')})
        packets={'C':pkt('C',1.),'D':pkt('D',-1.)}
        costs=SubsetCostEvaluator(self.backend).evaluate(obs,(('C',),('D',),('C','D')),packets,(),
            np.zeros((2,3)),np.ones(2,bool),'root_relative')
        np.testing.assert_allclose(costs.errors_m,[.5,.5,0.])

    def test_backend_preserves_each_sources_actual_history_and_masks(self):
        from test_backends import frame,FakeNet,fake_fk
        net=FakeNet();backend=EgoExoMoCapBackend(SimpleNamespace(net=net,bm=None,fk=fake_fk),3)
        c0,d0=frame(0,'C'),frame(0,'D')
        d1,e1=frame(1,'D'),frame(1,'E')
        c2,e2=frame(2,'C'),frame(2,'E')
        hist=(AcquiredFrame(c0.observation,(c0.packet,d0.packet)),
              AcquiredFrame(d1.observation,(d1.packet,e1.packet)))
        result=backend.reconstruct(c2.observation,(c2.packet,e2.packet),hist)
        self.assertEqual(result.metadata['remote_slot_ids'],['C','D','E'])
        mask=net.seen['exo_pad_mask'][0,:,1:].numpy()
        np.testing.assert_array_equal(mask,[[False,False,True],[True,False,False],[False,True,False]])
        reverse=backend.reconstruct(c2.observation,(e2.packet,c2.packet),hist)
        np.testing.assert_array_equal(result.joints,reverse.joints)

    def test_all_candidates_fit_budget_is_rejected_for_training(self):
        with self.assertRaisesRegex(ValueError,'No subset choice'):
            train(self.path,Path(self.temp.name)/'run',self.backend,
                TrainConfig(epochs=1,acquisition=AcquisitionConfig(3)))

    def test_single_source_v2_checkpoint_cannot_silently_change_objective(self):
        path=Path(self.temp.name)/'v2.pt'
        torch.save({'version':2,'objective':'expected_mpjpe'},path)
        with self.assertRaisesRegex(ValueError,'v2 single-source'):load_network(path)


if __name__=='__main__':unittest.main()
