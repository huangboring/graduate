import unittest
from dataclasses import replace

import numpy as np
import torch

from observer_selection.features import demand_features, packet_features, geometry_features
from observer_selection.networks import SelectorNetwork
from observer_selection.types import Observation, Packet, Geometry


def observation(time=1.0):
    return Observation("target", time, np.array([[20., 40.], [50., 70.]]),
                       np.array([True, True]), np.array([0., 0., 100., 100.]))


def packet(time=1.0):
    o = observation(time)
    return Packet("C", o.target_id, o.timestamp, o.xy, o.valid, o.bbox)


class FeatureTests(unittest.TestCase):
    def test_cold_start_has_no_fake_motion_or_prior(self):
        x = demand_features(observation())
        self.assertEqual(x.shape, (27,))
        per_joint = x[:-5].reshape(2, 11)
        np.testing.assert_array_equal(per_joint[:, 5:], 0)
        np.testing.assert_array_equal(per_joint[:, 2], 1)
        np.testing.assert_array_equal(per_joint[:, 3:5], 0)

    def test_missing_coordinates_never_produce_nan(self):
        o = replace(observation(), xy=np.array([[np.nan, np.nan], [50., 70.]]),
                    valid=np.array([False, True]), confidence=np.array([0.9, np.nan]))
        x = demand_features(o)[:-5].reshape(2, 11)
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[0], 0)
        self.assertEqual(x[1, 4], 0)

    def test_real_local_history_and_prior(self):
        old = observation(1.)
        current = replace(observation(2.), xy=old.xy + 10,
                          prior_xy=old.xy, prior_timestamp=1.)
        x = demand_features(current, old)[:-5].reshape(2, 11)
        np.testing.assert_allclose(x[:, 5:7], 0.1)
        np.testing.assert_allclose(x[:, 8:10], 0.1)
        np.testing.assert_array_equal(x[:, [7, 10]], 1)

    def test_future_or_wrong_target_history_rejected(self):
        with self.assertRaises(ValueError): demand_features(observation(1), observation(2))
        with self.assertRaises(ValueError): demand_features(observation(), replace(observation(0), target_id="other"))
        with self.assertRaises(ValueError): demand_features(replace(observation(), prior_xy=observation().xy, prior_timestamp=2))
        with self.assertRaises(ValueError): packet_features(packet(1), 1)

    def test_geometry_contract(self):
        np.testing.assert_array_equal(geometry_features(None, 5), np.zeros(9))
        g = Geometry(np.array([.5, .8, 0., 1.]), np.ones(4, dtype=bool), 4., "history")
        self.assertEqual(geometry_features(g, 5).shape, (9,))
        with self.assertRaises(ValueError): geometry_features(replace(g, timestamp=6), 5)
        with self.assertRaises(ValueError): geometry_features(replace(g, provenance="current_external"), 5)
        with self.assertRaises(ValueError): geometry_features(replace(g, timestamp=5), 5)

    def test_available_nan_rejected(self):
        with self.assertRaises(ValueError): demand_features(replace(observation(), xy=np.full((2, 2), np.nan)))

    def test_no_backend_or_packet_payload_leak(self):
        base = observation()
        changed = replace(base, backend_context={"gt_joints": np.full((2, 3), 999.)})
        np.testing.assert_array_equal(demand_features(base), demand_features(changed))
        p = packet()
        np.testing.assert_array_equal(packet_features(p), packet_features(replace(p, payload={"anything": 999})))


class NetworkTests(unittest.TestCase):
    def test_recurrent_memory_receives_future_score_gradient(self):
        torch.manual_seed(10)
        net = SelectorNetwork(2, 16)
        h = net.update(packet(1), net.initial_state(), None)
        q = net.demand(observation(2), observation(1))
        score = net.score(q, h, torch.zeros(9), 1., True)
        self.assertEqual(score.ndim, 0)
        self.assertTrue(torch.isfinite(score))  # v2 score is an unrestricted policy logit
        torch.nn.functional.smooth_l1_loss(score, score.detach() + 1.).backward()
        for module in (net.demand_encoder, net.observation_encoder, net.memory_cell, net.scorer):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)

    def test_no_id_embedding_and_memory_is_not_global(self):
        net = SelectorNetwork(2, 16)
        a, b = net.initial_state(), net.initial_state()
        updated = net.update(packet(), a)
        self.assertFalse(torch.equal(updated, a))
        self.assertTrue(torch.equal(a, b))
        renamed = replace(packet(), observer_id="new_name")
        torch.testing.assert_close(updated, net.update(renamed, a))


if __name__ == "__main__":
    unittest.main()
