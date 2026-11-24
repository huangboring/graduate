"""ID-addressed source state. A GRU updates H; packets supply camera metadata."""
from dataclasses import dataclass, replace
import math

import numpy as np
import torch

from .calibration import rigid


@dataclass(frozen=True)
class SourceState:
    hidden: torch.Tensor
    observed_at: float
    requested_at: float
    camera_to_world: np.ndarray
    intrinsics: np.ndarray
    update_count: int


class SourceStateBank:
    def __init__(self, target_id):
        self.target_id = target_id
        self.records: dict[str, SourceState] = {}
        self.last_requested: dict[str, float] = {}

    def mark_requested(self, sources, now):
        if not math.isfinite(now):
            raise ValueError('Request time must be finite seconds')
        for source in sources:
            self.last_requested[source] = now
            if source in self.records:
                self.records[source] = replace(self.records[source], requested_at=now)

    def update_many(self, packets, selected, now, selector):
        """Validate the complete response before committing any H/camera/time."""
        if not math.isfinite(now):
            raise ValueError('Response time must be finite seconds')
        packets, selected = tuple(packets), tuple(selected)
        names = tuple(packet.observer_id for packet in packets)
        if len(set(selected)) != len(selected) or len(set(names)) != len(names) or set(names) != set(selected):
            raise ValueError('Response must contain exactly one packet per selected source')
        validated = []
        for packet in packets:
            old = self.records.get(packet.observer_id)
            timestamp = float(packet.timestamp)
            if packet.target_id != self.target_id:
                raise ValueError('Source packet belongs to a different target T')
            if not math.isfinite(timestamp) or timestamp > now:
                raise ValueError('Source capture time must be finite and not in the future')
            if old is not None and timestamp <= old.observed_at:
                raise ValueError('Source packet must be newer than its stored observation')
            camera = rigid(packet.payload['camera_to_world'], 'source camera').copy()
            intrinsics = np.asarray(packet.payload['intrinsics'], dtype='f8').copy()
            if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
                    or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
                    or not np.allclose(intrinsics[2], [0, 0, 1])
                    or abs(np.linalg.det(intrinsics)) < 1e-12):
                raise ValueError('Invalid source pinhole intrinsics')
            validated.append((packet, old, camera, intrinsics))
        updates = {}
        for packet, old, camera, intrinsics in validated:
            hidden = selector.update_source(packet, old.hidden if old else selector.initial_state(),
                                            old.observed_at if old else None)
            camera.setflags(write=False)
            intrinsics.setflags(write=False)
            updates[packet.observer_id] = SourceState(
                hidden, float(packet.timestamp), self.last_requested.get(packet.observer_id, now),
                camera, intrinsics, old.update_count + 1 if old else 1)
        self.records.update(updates)

    def ages(self, sources, now):
        ages = [now - self.records[source].observed_at for source in sources]
        if any(not math.isfinite(age) or age < 0 for age in ages):
            raise ValueError('Stored observations cannot be newer than the decision time')
        return ages

    def detach(self):
        self.records = {source: replace(state, hidden=state.hidden.detach())
                        for source, state in self.records.items()}
