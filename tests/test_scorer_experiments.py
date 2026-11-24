import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from observer_selection.fit_triangulated import train
from observer_selection.replay import ReplayFrame
from observer_selection.scorer_experiments import restrict_sources
from observer_selection.types import Observation, Packet


def fixture_frames(count=7):
    point = np.array([.2, .1, 4.], dtype='f4')
    k = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype='f4')
    frames = []
    for i in range(count):
        local = np.eye(4)
        xy = np.tile((k@point)[:2]/point[2], (13, 1))
        obs = Observation('T', i*.05, xy, np.ones(13, bool), np.array([1, 1, 640, 480]),
            np.ones(13), backend_context={'local_camera_to_world': local, 'local_intrinsics': k})
        packets = {}
        for n in range(1, 5):
            camera = np.eye(4); camera[0, 3] = n*.5
            p = k@(point-camera[:3, 3])
            packets[f'cam{n:02}'] = Packet(f'cam{n:02}', 'T', i*.05,
                np.tile(p[:2]/p[2], (13, 1)), obs.valid, obs.bbox, obs.confidence,
                {'camera_to_world': camera, 'intrinsics': k})
        frames.append(ReplayFrame(obs, tuple(packets), {}, packets, None, None, 'world'))
    return frames


class TinyReconstructor:
    window, device = 3, 'cpu'

    def __init__(self, *args):
        self.adapter = torch.nn.Linear(39, 66)

    def load_tracking(self, sequence, target, ids):
        return (np.zeros((len(ids), 54), 'f4'), np.tile([.2, .1, 4.], (len(ids), 22, 1)).astype('f4'),
                np.tile(np.eye(3), (len(ids), 1, 1)))

    def reconstruct(self, sparse, head, values, *args):
        return self.adapter(torch.tensor(values.reshape(3, 39))).reshape(3, 22, 3)


class TrainingIntegrationTest(unittest.TestCase):
    def test_train_resume_shared_labels_and_fixed_weight_camera_transfer(self):
        frames = fixture_frames()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); prepared = root/'prepared'; prepared.mkdir()
            sequence = root/'scene'/'sequence'; sequence.mkdir(parents=True)
            (prepared/'manifest.json').write_text(json.dumps({'scene_id': 'scene', 'sequence_id': 'sequence',
                'requester_id': 'local', 'target_id': 'T'}), encoding='utf-8')
            (prepared/'replay.npz').write_bytes(b'replay input fixture')
            for source in ['local', *frames[0].candidates]:
                pair = prepared/source; pair.mkdir()
                (pair/'meta.json').write_text(json.dumps({'raw_frame_ids': list(range(1, 8))}), encoding='utf-8')
                np.savez(pair/'observer.npz', observer_img_size=np.array([640, 480]))
            official = root/'official.pt'; official.write_bytes(b'official fixture hash')
            common = dict(prepared=prepared, sequence=sequence, repo=root, checkpoint=official,
                device='cpu', epochs=1, render=False, sources=['cam01', 'cam02'], save_every=1, label_stride=1)
            with patch('observer_selection.fit_triangulated.SingleExo', TinyReconstructor), \
                 patch('observer_selection.fit_triangulated.load_sequence', return_value=frames), \
                 patch('observer_selection.preparation_quality.check_preparation', return_value={}):
                base = root/'adapter'; train(output=base, stage='adapter', **common)
                mlp = root/'mlp'; train(output=mlp, adapter_from=base, **common)
                attention = root/'attention'
                settings = dict(common, scorer='attention', adapter_from=base, target_cache_from=mlp)
                train(output=attention, **settings)
                train(output=attention, **dict(settings, epochs=2, resume=True))
                state = torch.load(attention/'checkpoint.pt', weights_only=True)
                self.assertEqual(state['epoch'], 2)
                self.assertEqual(len(state['history']), 2)
                self.assertFalse((attention/'subset_cost_cache').exists())
                frozen = torch.load(base/'checkpoint.pt', weights_only=True)['adapter']
                for name, tensor in frozen.items():
                    torch.testing.assert_close(tensor, state['adapter'][name])
                trace = json.loads((attention/'round_trace.json').read_text())
                self.assertEqual(len(trace), 6)
                self.assertIn('source_ages_before_s', trace[0])
                self.assertTrue((attention/'predictions.npz').is_file())
                with self.assertRaisesRegex(ValueError, 'Resume config'):
                    train(output=attention, **dict(settings, resume=True, cost=.004))
                out = attention/'epoch_0002/transfer34'
                train(output=out, **dict(common, scorer='attention', sources=['cam03', 'cam04'],
                    evaluate_from=attention/'epoch_0002'))
                result = json.loads((out/'camera_transfer.json').read_text())
                self.assertEqual(result['sources'], ['cam03', 'cam04'])
                self.assertEqual(len(result['methods']), 4)
                self.assertFalse((out/'checkpoint.pt').exists())
                from observer_selection.scorer_report import report
                report(root)
                self.assertGreater((root/'comparison.png').stat().st_size, 1000)
                # A cache cannot be reused on a different replay even with equal shapes.
                from observer_selection.subset_training import build_targets
                with self.assertRaisesRegex(ValueError, 'Subset costs differ'):
                    data = (restrict_sources(frames, ['cam01', 'cam02']), list(range(1, 8)), {}, set(),
                            'local', *TinyReconstructor().load_tracking(None, None, list(range(7))))
                    build_targets(data, TinyReconstructor(), mlp, 1,
                        json.loads((mlp/'experiment.json').read_text())['frozen_sha256'], 'changed replay')


if __name__ == '__main__':
    unittest.main()
