"""Causality and ID-addressed state, independent of reconstruction accuracy."""
from dataclasses import replace
import random
import unittest

import numpy as np
import torch

from observer_selection.rollout import Rollout
from observer_selection.selection import SelectionConfig, choose_sources, plan_refresh
from observer_selection.selector import LocalEncoding, ViewSelector
from observer_selection.state_bank import SourceStateBank
from test_scorer_experiments import fixture_frames


class StateAndRoundTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6)
        torch.set_num_threads(2)
        self.net = ViewSelector()
        self.frames = fixture_frames(4)

    def test_current_e_is_self_content_and_q_is_common_context(self):
        e = torch.randn(64, requires_grad=True)
        q = torch.randn(64, requires_grad=True)
        h = [torch.randn(64, requires_grad=True)]
        source, local = self.net.score_candidates(LocalEncoding(e, q), h, [np.zeros(9)], [.2])
        changed_source, changed_local = self.net.score_candidates(LocalEncoding(e*2+torch.arange(64), q), h, [np.zeros(9)], [.2])
        torch.testing.assert_close(source, changed_source)
        self.assertNotEqual(float(local.detach()), float(changed_local.detach()))
        (source-local).sum().backward()
        self.assertGreater(float(e.grad.abs().sum()), 0)
        self.assertGreater(float(q.grad.abs().sum()), 0)

    def test_local_gru_uses_bounded_window_and_current_e_stays_separate(self):
        frames = fixture_frames(6)
        observations = [frame.observation for frame in frames]
        full = self.net.encode_local(observations)
        recent = self.net.encode_local(observations[-4:])
        torch.testing.assert_close(full.current, recent.current)
        torch.testing.assert_close(full.context, recent.context)
        three = observations[-3:]
        changed = [replace(three[0], xy=three[0].xy+30), *three[1:]]
        before, after = self.net.encode_local(three), self.net.encode_local(changed)
        torch.testing.assert_close(before.current, after.current)
        self.assertFalse(torch.allclose(before.context, after.context))

    def test_bank_updates_only_matching_id_and_retains_capture_time(self):
        bank = SourceStateBank('T')
        initial = self.frames[0].packets
        bank.update_many(initial.values(), initial, .01, self.net)
        old = bank.records.copy()
        packet = replace(self.frames[1].packets['cam02'], timestamp=.04)
        bank.mark_requested(('cam02',), .05)
        bank.update_many([packet], ['cam02'], .05, self.net)
        self.assertIs(bank.records['cam01'], old['cam01'])
        self.assertEqual(bank.records['cam02'].observed_at, .04)
        self.assertEqual(bank.records['cam02'].requested_at, .05)
        self.assertAlmostEqual(bank.ages(['cam02'], .1)[0], .06)
        self.assertEqual(bank.records['cam02'].update_count, 2)
        bank.detach()
        self.assertIsNone(bank.records['cam02'].hidden.grad_fn)

    def test_bad_batch_cannot_partially_update_hidden_or_camera(self):
        bank = SourceStateBank('T')
        packets = self.frames[0].packets
        bank.update_many(packets.values(), packets, 0., self.net)
        before = bank.records.copy()
        good = self.frames[1].packets['cam01']
        bad = replace(self.frames[1].packets['cam02'], target_id='wrong')
        with self.assertRaisesRegex(ValueError, 'different target'):
            bank.update_many([good, bad], ['cam01', 'cam02'], .05, self.net)
        for source in before:
            self.assertIs(bank.records[source], before[source])
        # Invalid features also fail before any temporary result is committed.
        bad = replace(self.frames[1].packets['cam02'], confidence=None)
        with self.assertRaises(ValueError):
            bank.update_many([good, bad], ['cam01', 'cam02'], .05, self.net)
        self.assertIs(bank.records['cam01'], before['cam01'])

    def test_round_reads_old_memory_then_updates_only_selected_for_next_round(self):
        captured = []
        def score(local, memories, geometry, ages):
            captured.append([h.detach().clone() for h in memories])
            return torch.tensor([1., -1., -1., -1.]), torch.tensor(0.)
        self.net.score_candidates = score
        rollout = Rollout(self.net, self.frames, 'local', np.tile([.2,.1,4.], (4,1)),
            np.tile(np.eye(3),(4,1,1)), dict.fromkeys(['local', *self.frames[0].candidates],640),
            set(self.frames[0].candidates), SelectionConfig())
        old = rollout.bank.records.copy()
        result = rollout.step(self.frames[1], 1)
        self.assertEqual(result.selected, ('cam01',))
        torch.testing.assert_close(captured[0][0], old['cam01'].hidden)
        self.assertIs(rollout.bank.records['cam02'], old['cam02'])
        new = rollout.bank.records['cam01'].hidden.detach().clone()
        result = rollout.step(self.frames[2], 2)
        torch.testing.assert_close(captured[1][0], new)
        self.assertEqual(result.trace['source_update_counts']['cam01'], 3)
        self.assertEqual(result.trace['source_update_counts']['cam02'], 1)

    def test_unselected_current_packets_cannot_change_score_or_memory(self):
        score_function = self.net.score_candidates
        observed_scores = []
        def score_then_keep_local(*args):
            external, local = score_function(*args)
            observed_scores.append(torch.cat((local[None], external)).detach().clone())
            return external.new_full(external.shape, -1.), local.new_zeros(())
        self.net.score_candidates = score_then_keep_local
        def run(frames):
            rollout = Rollout(self.net, frames, 'local', np.tile([.2,.1,4.],(4,1)),
                np.tile(np.eye(3),(4,1,1)), dict.fromkeys(['local', *frames[0].candidates],640),
                set(frames[0].candidates), SelectionConfig())
            result = rollout.step(frames[1], 1, 'learned')
            return result, rollout.bank.records
        first, before = run(self.frames)
        changed = list(self.frames)
        poisoned = {s: replace(p, xy=np.full((13,2), 1e9)) for s,p in changed[1].packets.items()}
        changed[1] = replace(changed[1], packets=poisoned)
        second, after = run(changed)
        torch.testing.assert_close(observed_scores[0], observed_scores[1])
        np.testing.assert_array_equal(first.points, second.points)
        for source in before:
            torch.testing.assert_close(before[source].hidden, after[source].hidden)

    def test_refresh_queue_oldest_first_no_duplicate_and_no_topk_cap(self):
        ids = ('a','b','c','d')
        config = SelectionConfig(1.)
        plan = plan_refresh(ids, 2., dict.fromkeys(ids,0.), {'a':0.,'b':.2,'c':1.8,'d':.5}, config)
        self.assertEqual(plan.required_ids, ('a',))
        next_plan = plan_refresh(ids, 2.1, dict.fromkeys(ids,0.), {'a':2.,'b':.2,'c':1.8,'d':.5}, config)
        self.assertEqual(next_plan.required_ids, ('b',))
        chosen = choose_sources(torch.tensor([4.,3.,2.,-1.]),torch.tensor(0.),ids,('d',))
        self.assertEqual(chosen, ids)


if __name__ == '__main__':
    unittest.main()
