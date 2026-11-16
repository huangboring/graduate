import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from observer_selection.backends import EgoExoMoCapBackend, WeightedJointBackend
from observer_selection.types import AcquiredFrame, NotReadyError, Observation, Packet


class FakeNet(nn.Module):
    def __init__(self, window_size=3):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.egonet = SimpleNamespace(window_size=window_size, num_input=1)
        self.fusion = SimpleNamespace(window_size=window_size)
        self.seen = None

    def forward(self, batch):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.seen = {k: v.detach().clone() for k, v in batch.items() if torch.is_tensor(v)}
        signal = batch['sparse_input'][..., :1].clone()
        # Deliberately emulate upstream's in-place input modification.
        batch['sparse_input'].zero_()
        return {'root_orient': signal.repeat(1, 1, 6),
                'pose_body': signal.repeat(1, 1, 126)}


def fake_fk(body_model, global_orientation, joint_rotation, trans=None, betas=None):
    arange = torch.arange(22, device=global_orientation.device).reshape(1, 1, 22, 1)
    return (arange * global_orientation[..., :1, None]).expand(-1, -1, -1, 3)


def frame(t, observer='C', local=False):
    ctx = {'sparse_input': np.full(18, t + 1, np.float32), 'tracker_provenance': 'measured',
           'coordinate_convention':'egoexomocap_headlocal_rays_zup',
           'floor_provenance':'calibrated','floor_height_m':0.0}
    if local:
        ctx.update(local_exo_rays=np.full(39, 9, np.float32),
                   local_dino_cls=np.full(768, 7, np.float32), local_bbox_valid=True)
    obs = Observation('B', float(t), np.ones((13, 2)), np.ones(13, bool),
                      np.array([0, 0, 10, 10]), backend_context=ctx)
    packet = Packet(observer, 'B', float(t), np.ones((13, 2)), np.ones(13, bool),
                    np.array([0, 0, 10, 10]), payload={
                        'exo_rays': np.full(39, t, np.float32),
                        'dino_cls': np.full(768, t, np.float32), 'bbox_valid': True})
    return AcquiredFrame(obs, packet)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.net = FakeNet()
        self.backend = EgoExoMoCapBackend(SimpleNamespace(net=self.net, fk=fake_fk, bm=None), 3)

    def test_warmup_is_explicit(self):
        f = frame(0)
        with self.assertRaises(NotReadyError):
            self.backend.reconstruct(f.observation, f.packet, ())

    def test_last_frame_and_acquired_history_only(self):
        history = (frame(0), frame(1, 'D'), frame(2))
        current = frame(3, 'E', local=True)
        result = self.backend.reconstruct(current.observation, current.packet, history)
        np.testing.assert_allclose(result.joints[1], [4, 4, 4])
        self.assertEqual(result.coordinate_frame, 'root_relative')
        self.assertEqual(result.metadata['timestamps'], [1, 2, 3])
        seen = self.net.seen
        self.assertEqual(tuple(seen['exo_rays'].shape), (1, 3, 4, 39))
        self.assertEqual(result.metadata['remote_slot_ids'], ['C','D','E'])
        np.testing.assert_allclose(seen['exo_rays'][0, :, 1, 0], [0, 2, 0])
        np.testing.assert_allclose(seen['exo_rays'][0, :, 2, 0], [1, 0, 0])
        np.testing.assert_allclose(seen['exo_rays'][0, :, 3, 0], [0, 0, 3])
        self.assertEqual(seen['exo_pad_mask'][0, :, 0].tolist(), [True, True, False])
        self.assertEqual(seen['exo_pad_mask'][0, :, 1].tolist(), [True, False, True])
        # Upstream mutation did not affect caller state or subsequent candidates.
        for f in (*history, current):
            np.testing.assert_allclose(f.observation.backend_context['sparse_input'], f.observation.timestamp + 1)
        again = self.backend.reconstruct(current.observation, current.packet, history)
        np.testing.assert_array_equal(result.joints, again.joints)
        self.assertFalse(self.net.training)

    def test_future_and_cross_target_rejected(self):
        current = frame(3)
        with self.assertRaises(ValueError):
            self.backend.reconstruct(current.observation, current.packet, (frame(4),))
        with self.assertRaises(ValueError):
            self.backend.reconstruct(current.observation, current.packet, (frame(1), frame(1)))

    def test_gt_and_unknown_provenance_rejected(self):
        history = (frame(0), frame(1))
        current = frame(2)
        current.observation.backend_context['poses_gt'] = np.zeros(132)
        with self.assertRaises(ValueError):
            self.backend.reconstruct(current.observation, current.packet, history)
        del current.observation.backend_context['poses_gt']
        current.observation.backend_context['tracker_provenance'] = 'benchmark_gt'
        with self.assertRaises(ValueError):
            self.backend.reconstruct(current.observation, current.packet, history)

    def test_invalid_observer_token_is_masked(self):
        current = frame(2)
        current.packet.payload['bbox_valid'] = False
        self.backend.reconstruct(current.observation, current.packet, (frame(0), frame(1)))
        self.assertTrue(self.net.seen['exo_pad_mask'][0, -1, 1])
        self.assertEqual(float(self.net.seen['exo_rays'][0, -1, 1].sum()), 0)

    def test_missing_assets_fail_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(NotReadyError):
                EgoExoMoCapBackend.from_repository(d, 'config.yaml', 'model.pth')

    def test_weighted_simulation_is_labeled_and_root_relative(self):
        f = frame(0)
        f.packet.payload.update(joints3d=np.array([[0, 0, 0], [4, 2, 0]], np.float32),
                                joint_weights=np.array([1, 3]))
        f.observation.backend_context.update(local_joints3d=np.array([[0, 0, 0], [0, 2, 0]], np.float32))
        result = WeightedJointBackend().reconstruct(f.observation, f.packet, ())
        np.testing.assert_allclose(result.joints, [[0, 0, 0], [3, 2, 0]])
        self.assertTrue(result.metadata['synthetic'])


if __name__ == '__main__':
    unittest.main()
