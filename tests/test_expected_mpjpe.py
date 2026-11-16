"""Research-objective and acquisition-boundary checks, not reconstruction accuracy."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from observer_selection.networks import SelectorNetwork
from observer_selection.runtime import SelectionEngine
from observer_selection.selection import AcquisitionConfig
from observer_selection.costs import CandidateCostEvaluator
from observer_selection.training import expected_mpjpe_loss, load_network
from observer_selection.types import Observation, Packet, Reconstruction, NotReadyError


def observation(timestamp=1.0):
    return Observation('B', timestamp, np.array([[2.,3.],[4.,6.]],np.float32),
                       np.ones(2,bool),np.array([0.,0.,10.,10.],np.float32))


def packet(source='C',timestamp=1.0,shift=0.0):
    o=observation(timestamp)
    return Packet(source,'B',timestamp,o.xy+shift,o.valid,o.bbox)


class NumericBackend:
    def __init__(self):
        self.net=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad(): self.net.weight.fill_(1.)
        self.grad_enabled=[]
        self.histories=[]

    def reconstruct(self,obs,pkt,history):
        self.grad_enabled.append(torch.is_grad_enabled())
        self.histories.append(tuple(f.packet.observer_id for f in history))
        magnitude={'C':.04,'D':.08,'E':.06}[pkt.observer_id]
        value=self.net(torch.tensor([[magnitude]])).item()
        return Reconstruction(np.array([[value,0,0],[value,0,0]],np.float32),'test')


class ExpectedMPJPETests(unittest.TestCase):
    def test_exact_expectation_and_analytic_logit_gradient(self):
        probabilities=torch.tensor([.3,.5,.2],dtype=torch.float64)
        logits=probabilities.log().requires_grad_()
        costs=torch.tensor([.04,.08,.06],dtype=torch.float64,requires_grad=True)
        loss=expected_mpjpe_loss(logits,costs)
        self.assertAlmostEqual(loss.item(),.064,places=12)
        loss.backward()
        torch.testing.assert_close(logits.grad,torch.tensor([-.0072,.008,-.0008],dtype=torch.float64))
        self.assertIsNone(costs.grad)

    def test_equal_costs_give_zero_selection_gradient(self):
        logits=torch.tensor([-.7,2.,.1],requires_grad=True)
        expected_mpjpe_loss(logits,torch.full((3,),.05)).backward()
        torch.testing.assert_close(logits.grad,torch.zeros(3),atol=1e-8,rtol=0)

    def test_costs_are_exact_frozen_backend_values(self):
        backend=NumericBackend()
        evaluator=CandidateCostEvaluator(backend)
        self.assertFalse(backend.net.training)
        self.assertFalse(backend.net.weight.requires_grad)
        o=observation();ids=('C','D','E')
        costs=evaluator.evaluate(o,ids,{i:packet(i) for i in ids},(),np.zeros((2,3)),np.ones(2,bool),'test')
        np.testing.assert_allclose(costs.errors_m,[.04,.08,.06],atol=1e-7)
        self.assertEqual(backend.histories,[(),(),()])
        self.assertEqual(backend.grad_enabled,[False]*3)
        self.assertIsNone(backend.net.weight.grad)

    def test_runtime_selects_highest_logit_not_lowest(self):
        net=SelectorNetwork(2,8)
        engine=SelectionEngine(net,acquisition=AcquisitionConfig(1,None))
        def score(q,h,geometry,age,has_history):
            return q.sum()*0 + next(values)
        values=iter([-.2,1.1,.3])
        net.score=score
        d=engine.score(observation(),('C','D','E'))
        self.assertEqual(d.selected_id,'D')
        torch.testing.assert_close(d.probabilities,torch.softmax(d.logits,0))
        self.assertEqual(engine.history,[])
        self.assertEqual(engine.memories,{})

    def test_expected_cost_updates_encoder_and_gru_but_not_costs(self):
        torch.manual_seed(10)
        net=SelectorNetwork(2,8)
        q=net.demand_sequence([observation(.5),observation(1.)])
        h_c=net.update(packet('C',.5,0),net.initial_state())
        h_d=net.update(packet('D',.5,3),net.initial_state())
        logits=torch.stack([net.score(q,h,torch.zeros(9),.5,True) for h in (h_c,h_d)])
        expected_mpjpe_loss(logits,torch.tensor([.04,.12])).backward()
        for parameter in (net.demand_encoder[0].weight,net.demand_temporal.weight_ih_l0,
                          net.observation_encoder[0].weight,net.memory_cell.weight_ih,
                          net.scorer[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()),0)

    def test_missing_assets_are_not_swallowed_as_warmup(self):
        class BrokenBackend:
            def reconstruct(self,*args): raise NotReadyError('Missing weights')
        engine=SelectionEngine(SelectorNetwork(2,8),acquisition=AcquisitionConfig(1,None));o=observation()
        d=engine.score(o,('C','D'))
        with self.assertRaisesRegex(NotReadyError,'Missing weights'):
            engine.commit(o,packet(d.selected_id),BrokenBackend())
        self.assertEqual(engine.history,[])
        self.assertEqual(engine.memories,{})

    def test_legacy_checkpoint_cannot_silently_change_score_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.pt'
            torch.save({'version':1,'objective':'error_regression'},path)
            with self.assertRaisesRegex(ValueError,'v1 error-regression'):
                load_network(path)

    def test_bad_costs_fail_instead_of_training(self):
        for costs in (torch.tensor([.1,float('nan')]),torch.tensor([.1,-.2])):
            with self.assertRaises(ValueError): expected_mpjpe_loss(torch.zeros(2),costs)


if __name__=='__main__': unittest.main()
