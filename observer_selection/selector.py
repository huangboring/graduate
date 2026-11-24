"""Pose-only view scoring: current local E, temporal Q and per-source H.

Symbols in docs/notation.json map to these descriptive Python names. IDs and
camera calibration belong to the runtime; neither is an identity embedding.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from .features import compact_demand_features, compact_packet_features


@dataclass(frozen=True)
class LocalEncoding:
    current: torch.Tensor  # E_k, 64
    context: torch.Tensor  # Q_k, 64


class ViewSelector(nn.Module):
    """Common scoring head; optional, explicitly selected attention ablation."""

    architecture_version = 'current_e_shared_head_v2'
    joints = 13
    hidden = 64
    local_window = 4
    demand_size = 71
    packet_size = 41

    def __init__(self, scorer='mlp'):
        super().__init__()
        if scorer not in ('mlp', 'attention'):
            raise ValueError('scorer must be mlp or attention')
        self.scorer_kind = scorer
        self.local_encoder = self._encoder(self.demand_size)
        self.local_gru = nn.GRU(self.hidden, self.hidden, batch_first=True)
        self.source_encoder = self._encoder(self.packet_size)
        self.source_gru = nn.GRUCell(self.hidden, self.hidden)
        self.source_projection = nn.Linear(74, self.hidden)
        self.content_norm = nn.LayerNorm(self.hidden)
        self.shared_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        # Create common modules first so equal seeds give equal common weights.
        if scorer == 'attention':
            self.attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
            self.attention_norm = nn.LayerNorm(64)

    @staticmethod
    def _encoder(size):
        return nn.Sequential(nn.Linear(size, 128), nn.LayerNorm(128), nn.GELU(),
                             nn.Linear(128, 64), nn.GELU())

    def tensor(self, value, size, name):
        parameter = next(self.parameters())
        result = torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype)
        if result.shape != (size,) or not bool(torch.isfinite(result).all()):
            raise ValueError(f'{name} must be a finite [{size}] vector')
        return result

    def initial_state(self):
        return next(self.parameters()).new_zeros(self.hidden)

    def encode_local(self, observations):
        """Recompute Q from <=4 raw local observations, preserving current E.

        There is no persistent local GRU state between overlapping windows.
        Source GRU states, by contrast, persist independently for each source.
        """
        observations = list(observations)[-self.local_window:]
        if not observations:
            raise ValueError('At least one local observation is required')
        encoded = [self.local_encoder(self.tensor(
            compact_demand_features(obs, observations[i-1] if i else None, self.joints),
            self.demand_size, 'local features')) for i, obs in enumerate(observations)]
        temporal, _ = self.local_gru(torch.stack(encoded).unsqueeze(0))
        return LocalEncoding(encoded[-1], temporal[0, -1])

    def score_candidates(self, local, memories, geometries, ages):
        if not isinstance(local, LocalEncoding):
            raise TypeError('Scoring requires both current E and temporal Q')
        if not len(memories) == len(geometries) == len(ages):
            raise ValueError('Source memories, geometry and ages must align')
        if any(not math.isfinite(float(age)) or age < 0 for age in ages):
            raise ValueError('Source ages must be finite nonnegative seconds')
        current = self.tensor(local.current, 64, 'current E')
        query = self.tensor(local.context, 64, 'temporal Q')
        contents = [current]
        if memories:
            raw = torch.stack([torch.cat((self.tensor(h, 64, 'source H'),
                self.tensor(g, 9, 'geometry G'), query.new_tensor([math.log1p(age)])))
                for h, g, age in zip(memories, geometries, ages)])
            contents.extend(self.source_projection(raw).unbind(0))
        tokens = self.content_norm(torch.stack(contents))
        if self.scorer_kind == 'attention':
            attended, _ = self.attention(tokens[None], tokens[None], tokens[None], need_weights=False)
            tokens = self.attention_norm(tokens + attended[0])
        values = self.shared_head(torch.cat((query.expand(len(tokens), -1), tokens), -1)).squeeze(-1)
        return values[1:], values[0]

    def update_source(self, packet, old_hidden, last_observed_at=None):
        features = compact_packet_features(packet, last_observed_at, self.joints)
        encoded = self.source_encoder(self.tensor(features, 41, 'source features'))
        return self.source_gru(encoded, self.tensor(old_hidden, 64, 'old source H'))

    def architecture_summary(self):
        return {'architecture_version': self.architecture_version, 'scorer': self.scorer_kind,
                'appearance': 'none', 'demand_features': 71, 'packet_features': 41,
                'demand_hidden': 64, 'source_hidden': 64, 'geometry_features': 9,
                'source_token_raw': 74, 'local_content': 'LayerNorm(current E)',
                'head': 'shared [temporal Q64, content64] -> 64 -> 32 -> 1',
                'attention_heads': 4 if self.scorer_kind == 'attention' else 0,
                'local_window': self.local_window,
                'selector_parameters': sum(p.numel() for p in self.parameters()),
                'note': 'Parameter count is not measured runtime or end-to-end FLOPs.'}
