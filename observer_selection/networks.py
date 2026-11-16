"""Learnable modules; recurrent memory is per source, parameters are shared."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

from .features import demand_features, packet_features
from .types import Observation, Packet


class SelectorNetwork(nn.Module):
    """Produce dimensionless source-selection logits (higher is preferred).

    Source IDs and memory tables live outside this network. Supervision for the
    demand encoder and GRU comes from expected downstream MPJPE across time, not
    from fake occlusion labels or manually labelled hidden states.
    """
    def __init__(self, joints: int, hidden: int = 64, memory_type: str = 'gru', memory_window: int = 32):
        super().__init__()
        if joints <= 0 or hidden <= 0:
            raise ValueError("joints and hidden must be positive")
        self.joints, self.hidden = int(joints), int(hidden)
        if memory_type not in ('gru', 'attention') or memory_window < 1:
            raise ValueError('memory_type must be gru/attention and memory_window positive')
        self.memory_type, self.memory_window = memory_type, memory_window
        self.demand_encoder = self._encoder(11 * joints + 5, hidden)
        self.demand_temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.observation_encoder = self._encoder(5 * joints + 1, hidden)
        self.memory_cell = nn.GRUCell(hidden, hidden)
        self.memory_attention = nn.MultiheadAttention(hidden, 4 if hidden % 4 == 0 else 1, batch_first=True)
        self.memory_positions = nn.Embedding(memory_window, hidden)
        self.scorer = nn.Sequential(nn.Linear(2 * hidden + 11, hidden), nn.GELU(),
                                    nn.Linear(hidden, max(hidden // 2, 1)), nn.GELU(),
                                    nn.Linear(max(hidden // 2, 1), 1))

    @staticmethod
    def _encoder(inputs: int, hidden: int) -> nn.Module:
        return nn.Sequential(nn.Linear(inputs, 2 * hidden), nn.LayerNorm(2 * hidden),
                             nn.GELU(), nn.Linear(2 * hidden, hidden), nn.GELU())

    def _tensor(self, value: np.ndarray | torch.Tensor, size: int, name: str) -> torch.Tensor:
        parameter = next(self.parameters())
        tensor = torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype)
        if tensor.shape != (size,):
            raise ValueError(f"{name} must have shape [{size}], got {tuple(tensor.shape)}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")
        return tensor

    def initial_state(self) -> torch.Tensor:
        if self.memory_type == 'attention':
            return next(self.parameters()).new_zeros((0, self.hidden))
        return next(self.parameters()).new_zeros(self.hidden)

    def demand_sequence(self, observations: list[Observation]) -> torch.Tensor:
        if not observations:
            raise ValueError('At least one local observation is required')
        encoded = [self.demand(o, observations[i-1] if i else None) for i,o in enumerate(observations)]
        out, _ = self.demand_temporal(torch.stack(encoded).unsqueeze(0))
        return out[0,-1]

    def demand(self, observation: Observation,
               previous: Observation | None = None) -> torch.Tensor:
        features = demand_features(observation, previous, self.joints)
        return self.demand_encoder(self._tensor(features, 11 * self.joints + 5, "demand"))

    def score(self, q: torch.Tensor, h: torch.Tensor, geometry: torch.Tensor,
              age: float, has_history: bool) -> torch.Tensor:
        age = float(age)
        if not math.isfinite(age) or age < 0:
            raise ValueError("source age must be finite and nonnegative")
        q = self._tensor(q, self.hidden, "q")
        if self.memory_type == 'attention':
            if h.ndim != 2 or h.shape[1] != self.hidden or len(h) > self.memory_window:
                raise ValueError('Attention memory must be [L,hidden]')
            if len(h):
                tokens = h + self.memory_positions(torch.arange(len(h),device=q.device))
                h, _ = self.memory_attention(q[None,None],tokens[None],tokens[None],need_weights=False)
                h = h[0,0]
            else:
                h = q.new_zeros(self.hidden)
        else:
            h = self._tensor(h, self.hidden, "h")
        geometry = self._tensor(geometry, 9, "geometry")
        metadata = q.new_tensor([math.log1p(age), float(bool(has_history))])
        return self.scorer(torch.cat((q, h, geometry, metadata))).squeeze(-1)

    def update(self, packet: Packet, old_h: torch.Tensor,
               last_timestamp: float | None = None) -> torch.Tensor:
        features = packet_features(packet, last_timestamp, self.joints)
        encoded = self.observation_encoder(self._tensor(features, 5 * self.joints + 1, "packet"))
        if self.memory_type == 'attention':
            if old_h.ndim != 2 or old_h.shape[1] != self.hidden:
                raise ValueError('Attention history must be [L,hidden]')
            return torch.cat((old_h,encoded[None]),dim=0)[-self.memory_window:]
        old_h = self._tensor(old_h, self.hidden, "old_h")
        return self.memory_cell(encoded.unsqueeze(0), old_h.unsqueeze(0)).squeeze(0)
