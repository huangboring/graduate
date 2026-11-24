"""One causal round: local -> score -> acquire -> bank update -> geometry.

Source packets are a private offline store owned by ReplayProvider. They are
read only after the complete selected-ID set has been decided. Reconstruction
is a separate consumer of returned geometry, not part of this selector.
"""
from collections import deque
from dataclasses import dataclass
import random

import numpy as np

from .features import geometry_features
from .replay import ReplayProvider
from .selection import choose_sources, matched_random, plan_refresh
from .state_bank import SourceStateBank
from .triangulation import _camera, camera_geometry, triangulate_views


@dataclass(frozen=True)
class StepResult:
    points: np.ndarray
    states: np.ndarray
    selected: tuple[str, ...]
    required: tuple[str, ...]
    trace: dict


class Rollout:
    def __init__(self, selector, frames, requester, head, rotations, widths, fixed, config, seed=2026):
        if not frames:
            raise ValueError('An initialization frame is required')
        self.selector, self.config = selector, config
        self.candidates = frames[0].candidates
        if requester in self.candidates or len(set(self.candidates)) != len(self.candidates):
            raise ValueError('Requester O0 and external source IDs must be distinct')
        self.requester, self.head, self.rotations = requester, head, rotations
        self.widths, self.fixed = widths, set(fixed)
        self.rng = random.Random(seed)
        self.bank = SourceStateBank(frames[0].observation.target_id)
        self.local = deque([frames[0].observation], maxlen=selector.local_window)
        self.first_seen = dict.fromkeys(self.candidates, frames[0].observation.timestamp)
        self._acquire_and_update(frames[0], self.candidates)
        self.latest_scores = self.latest_local_score = None

    def _acquire_and_update(self, frame, selected):
        if not selected:
            return ()
        now = frame.observation.timestamp
        self.bank.mark_requested(selected, now)
        packets = ReplayProvider(frame).acquire_many(selected, frame.observation.target_id, now)
        self.bank.update_many(packets, selected, now, self.selector)
        return packets

    def step(self, frame, index, mode='learned', matched=None):
        obs = frame.observation
        if frame.candidates != self.candidates or obs.target_id != self.bank.target_id:
            raise ValueError('Roster and target must remain fixed within a replay')
        if obs.timestamp <= self.local[-1].timestamp:
            raise ValueError('Local observation times must strictly increase')
        self.local.append(obs)
        now = obs.timestamp
        ages = self.bank.ages(self.candidates, now)
        required = ()
        self.latest_scores = self.latest_local_score = None
        if mode == 'all':
            selected = self.candidates
        elif mode == 'local':
            selected = ()
        elif mode == 'random':
            if matched is None:
                raise ValueError('Matched random requires reference decisions')
            required = tuple(matched['refresh'])
            selected = matched_random(self.candidates, matched, self.rng)
        elif mode in ('learned', 'sample'):
            local = self.selector.encode_local(self.local)
            records = [self.bank.records[source] for source in self.candidates]
            geometries = []
            for source, state in zip(self.candidates, records):
                geometry = camera_geometry(_camera(obs), (state.camera_to_world, state.intrinsics),
                    state.observed_at, source in self.fixed, now,
                    self.widths[self.requester], self.widths[source])
                geometries.append(geometry_features(geometry, now))
            self.latest_scores, self.latest_local_score = self.selector.score_candidates(
                local, [state.hidden for state in records], geometries, ages)
            last = {source: state.observed_at for source, state in self.bank.records.items()}
            required = plan_refresh(self.candidates, now, self.first_seen, last, self.config).required_ids
            selected = choose_sources(self.latest_scores, self.latest_local_score,
                self.candidates, required, self.rng if mode == 'sample' else None)
        else:
            raise ValueError(f'Unknown acquisition policy: {mode}')
        # Selection is locked. Only this point may access current remote packets.
        packets = self._acquire_and_update(frame, selected)
        views = (obs, *packets)
        points, states = triangulate_views(views, self.head[index], self.rotations[index],
                                            [view.confidence for view in views])
        trace = {'timestamp_s': now, 'source_ages_before_s': dict(zip(self.candidates, ages)),
                 'updated_sources': [packet.observer_id for packet in packets],
                 'source_update_counts': {source: state.update_count for source, state in self.bank.records.items()}}
        if self.latest_scores is not None:
            trace['source_scores'] = dict(zip(self.candidates, self.latest_scores.detach().cpu().tolist()))
            trace['local_score'] = float(self.latest_local_score.detach())
        return StepResult(points, states, selected, required, trace)

    def detach(self):
        self.bank.detach()
