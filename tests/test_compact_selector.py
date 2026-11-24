import math
import unittest
from dataclasses import replace

import numpy as np
import torch

from observer_selection.features import compact_demand_features, compact_packet_features
from observer_selection.selector import ViewSelector, LocalEncoding
from observer_selection.types import Observation, Packet


def observation(timestamp=1.05):
    return Observation('T',timestamp,np.tile([240.,330.],(13,1)),np.ones(13,bool),
                       np.array([100.,50.,300.,450.]),np.full(13,.72))


class CompactSelectorTest(unittest.TestCase):
    def test_numeric_features_and_missing_observations(self):
        now=observation()
        old=replace(now,timestamp=1.,xy=np.tile([232.,326.],(13,1)))
        values=compact_demand_features(now,old,13)
        self.assertEqual(values.shape,(71,))
        np.testing.assert_allclose(values[:5],[.1,.2,.72,.4,.2],atol=1e-6)
        np.testing.assert_allclose(values[-6:],[.5,.625,math.log1p(400),math.log1p(.05),1,1],atol=1e-6)
        first=compact_demand_features(now,None,13)
        np.testing.assert_array_equal(first[:65].reshape(13,5)[:,3:],0)
        self.assertEqual(list(first[-2:]),[1,0])
        absent=replace(now,valid=np.zeros(13,bool),xy=np.full((13,2),np.nan),confidence=np.full(13,np.nan))
        missing=compact_demand_features(absent,old,13)
        np.testing.assert_array_equal(missing[:65],0)
        self.assertEqual(list(missing[-2:]),[0,0])
        unavailable_old=replace(old,valid=np.zeros(13,bool))
        self.assertEqual(compact_demand_features(now,unavailable_old,13)[-1],0)
        packet=Packet('A','T',1.05,now.xy,now.valid,now.bbox,now.confidence,{})
        encoded=compact_packet_features(packet,.85,13)
        self.assertEqual(encoded.shape,(41,))
        np.testing.assert_allclose(encoded[:3],[.1,.2,.72],atol=1e-6)
        np.testing.assert_allclose(encoded[-2:],[1,math.log1p(.2)],atol=1e-6)
        # A real normalized origin is distinct from unavailable padding.
        at_center=replace(now,xy=np.tile([200.,250.],(13,1)))
        self.assertEqual(compact_demand_features(at_center,None,13)[-2],1)
        self.assertEqual(missing[-2],0)

    def test_compact_rejects_inputs_that_need_per_joint_masks(self):
        now=observation()
        mask=now.valid.copy();mask[2]=False
        for item in [replace(now,valid=mask),replace(now,confidence=None),
                     replace(now,confidence=np.full(13,np.nan))]:
            with self.assertRaisesRegex(ValueError,'Compact encoding'):
                compact_demand_features(item,None,13)
        with self.assertRaisesRegex(ValueError,'strictly precede'):
            compact_demand_features(now,now,13)

    def test_models_share_initial_common_weights_and_one_scoring_head(self):
        torch.manual_seed(2026);mlp=ViewSelector('mlp')
        torch.manual_seed(2026);attention=ViewSelector('attention')
        for name,value in mlp.state_dict().items():
            torch.testing.assert_close(value,attention.state_dict()[name])
        self.assertFalse(hasattr(mlp,'local_scorer'))
        self.assertFalse(hasattr(mlp,'scorer'))
        self.assertEqual(mlp.architecture_summary()['local_content'],'LayerNorm(current E)')
        self.assertEqual(mlp.source_projection.in_features,74)
        self.assertEqual(mlp.local_encoder[0].in_features,71)
        self.assertEqual(mlp.source_encoder[0].in_features,41)
        self.assertGreater(attention.architecture_summary()['selector_parameters'],
                           mlp.architecture_summary()['selector_parameters'])

    def test_both_variants_support_permuted_rosters_and_gradients(self):
        for kind in ('mlp','attention'):
            net=ViewSelector(kind)
            q=torch.randn(64,requires_grad=True)
            e=torch.randn(64,requires_grad=True)
            local_encoding=LocalEncoding(e,q)
            memories=[torch.randn(64,requires_grad=True) for _ in range(4)]
            geometry=[np.arange(9,dtype='f4')/9+i*.1 for i in range(4)]
            ages=[0.,.2,.4,1.3]
            score,local=net.score_candidates(local_encoding,memories,geometry,ages)
            order=[2,0,3,1]
            permuted,permuted_local=net.score_candidates(local_encoding,[memories[i] for i in order],
                                         [geometry[i] for i in order],[ages[i] for i in order])
            torch.testing.assert_close(permuted,score[order],atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(local,permuted_local,atol=1e-6,rtol=1e-5)
            (score-local).square().sum().backward()
            self.assertTrue(torch.isfinite(q.grad).all())
            self.assertGreater(float(e.grad.abs().sum()),0)
            self.assertTrue(all(h.grad is not None and torch.isfinite(h.grad).all() for h in memories))
            self.assertGreater(float(net.shared_head[0].weight.grad.abs().sum()),0)
            if kind=='attention':
                self.assertGreater(float(net.attention.in_proj_weight.grad.abs().sum()),0)
            self.assertEqual(net.score_candidates(local_encoding,memories[:1],geometry[:1],ages[:1])[0].shape,(1,))
            with self.assertRaisesRegex(ValueError,'ages'):
                net.score_candidates(local_encoding,memories,geometry,[0.,.2,.4,-1.])


if __name__=='__main__':
    unittest.main()
