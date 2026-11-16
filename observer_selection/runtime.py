from __future__ import annotations

from dataclasses import dataclass
import math
import random
import numpy as np
import torch

from .features import geometry_features
from .networks import SelectorNetwork
from .types import Observation, Packet, Geometry, AcquiredFrame, Backend, WarmupError, packet_tuple
from .selection import AcquisitionConfig, AdditiveSubsetPolicy, PreAcquisitionState, RefreshPlan, plan_refresh


@dataclass
class SourceMemory:
    hidden: torch.Tensor
    last_timestamp: float | None = None
    count: int = 0


@dataclass(frozen=True)
class Decision:
    target_id: str
    timestamp: float
    candidates: tuple[str, ...]
    logits: torch.Tensor  # Source scores, NOT errors or subset probabilities.
    selected_ids: tuple[str, ...]  # Immutable complete batch, fixed before acquisition.
    demand_vector: torch.Tensor
    budget: int
    refresh: RefreshPlan
    state: PreAcquisitionState
    policy: str

    @property
    def selected_id(self):
        if len(self.selected_ids) != 1:
            raise ValueError('Decision contains multiple sources; use selected_ids')
        return self.selected_ids[0]

    @property
    def probabilities(self):
        if self.budget != 1 or self.refresh.required_ids:
            raise ValueError('Use engine.subset_distribution(decision) for actual batch probabilities')
        return torch.softmax(self.logits, dim=0)


def validate_packet(packet: Packet, obs: Observation, selected_id: str) -> None:
    if packet.observer_id != selected_id or packet.target_id != obs.target_id:
        raise ValueError('Packet identity does not match the requested source/target')
    if not math.isfinite(packet.timestamp) or abs(packet.timestamp - obs.timestamp) > 1e-6:
        raise ValueError('Synchronized current-frame packets required; stale/future packets rejected')


class SelectionEngine:
    """One target/session. Decide whole subset, acquire once, commit atomically.

    Partial/malformed batches never update any source memory. Retry the same
    batch or cancel explicitly; asynchronous failures/delays are not modeled.
    """
    def __init__(self, network: SelectorNetwork, history_limit: int = 256,
                 acquisition: AcquisitionConfig | None = None, subset_policy=None):
        if history_limit < 1:
            raise ValueError('history_limit must be positive')
        self.network = network
        self.history_limit = history_limit
        self.acquisition = acquisition or getattr(network, 'acquisition_config', AcquisitionConfig())
        self.subset_policy = subset_policy or AdditiveSubsetPolicy()
        self.memories: dict[str, SourceMemory] = {}
        self.first_seen: dict[str, float] = {}
        self.history: list[AcquiredFrame] = []
        self.previous: Observation | None = None
        self.previous_selected: tuple[str, ...] = ()
        self.steps = 0
        self.demand_history: list[tuple[float, np.ndarray]] = []
        self.reconstruction_history: list[tuple[float, object]] = []
        self._pending: Decision | None = None
        self._pending_observation = None
        self._target_id = None

    def score(self, observation: Observation, candidates: tuple[str, ...],
              geometry: dict[str, Geometry] | None = None, *,
              sample=False, rng=None, policy='learned') -> Decision:
        if self._pending is not None:
            raise RuntimeError('Commit or cancel the pending request before another decision')
        candidates = tuple(candidates)
        if not candidates or len(set(candidates)) != len(candidates) or not all(isinstance(s,str) and s for s in candidates):
            raise ValueError('Candidate roster must contain unique nonempty string IDs')
        if not math.isfinite(observation.timestamp):
            raise ValueError('Nonfinite timestamp')
        if self._target_id is not None and observation.target_id != self._target_id:
            raise ValueError('Start a separate engine for each target/session')
        if policy not in ('learned','random','round_robin','last','geometry'):
            raise ValueError('Unknown selection policy')
        if sample and policy != 'learned':
            raise ValueError('sample=True applies only to the learned policy')
        if (sample or policy == 'random') and rng is None:
            raise ValueError('Explicit seeded RNG required for stochastic selection')
        if self.previous is not None:
            if observation.target_id != self.previous.target_id:
                raise ValueError('Start a separate engine for each target/session')
            if observation.timestamp <= self.previous.timestamp:
                raise ValueError('Observations must be strictly chronological')
        q = self.network.demand_sequence([f.observation for f in self.history[-3:]] + [observation])
        values, hidden_states, geometry_rows, source_ages, seen_flags = [], [], [], [], []
        for source in candidates:
            memory = self.memories.get(source)
            hidden = memory.hidden if memory else self.network.initial_state()
            seen = memory is not None and memory.last_timestamp is not None
            age = observation.timestamp - memory.last_timestamp if seen else 0.0
            geom = torch.as_tensor(geometry_features((geometry or {}).get(source), observation.timestamp),device=q.device)
            values.append(self.network.score(q,hidden,geom,age,seen))
            hidden_states.append(hidden); geometry_rows.append(geom); source_ages.append(age); seen_flags.append(seen)
        logits = torch.stack(values).reshape(-1)
        if not torch.isfinite(logits).all():
            raise ValueError('Selector produced nonfinite logits')
        # Registration is local roster knowledge; it does not acquire observations.
        first_seen = dict(self.first_seen)
        for source in candidates:
            first_seen.setdefault(source, observation.timestamp)
        refresh = plan_refresh(candidates, observation.timestamp, first_seen,
            {s:m.last_timestamp for s,m in self.memories.items()}, self.acquisition)
        state = PreAcquisitionState(q,tuple(hidden_states),torch.stack(geometry_rows),tuple(source_ages),tuple(seen_flags))
        budget = min(self.acquisition.max_sources,len(candidates))
        required = refresh.required_ids
        if policy == 'learned':
            if sample:
                distribution = self.subset_policy.distribution(logits,candidates,budget,required,
                    self.acquisition.max_exact_subsets,state)
                selected = distribution.sample(rng)
            else:
                selected = self.subset_policy.map(logits,candidates,budget,required,state)
        else:
            free = [s for s in candidates if s not in required]
            offset = (self.steps * budget) % len(candidates)
            rotated = candidates[offset:] + candidates[:offset]
            if policy == 'random':
                ordered = rng.sample(free,len(free))
            elif policy == 'round_robin':
                ordered = [s for s in rotated if s in free]
            elif policy == 'last':
                ordered = [s for s in self.previous_selected if s in free]
                ordered += [s for s in candidates if s in free and s not in ordered]
            else:
                features = {s:g.detach().cpu().numpy() for s,g in zip(candidates,geometry_rows)}
                order_index = {s:i for i,s in enumerate(rotated)}
                ordered = sorted(free,key=lambda s:(-int(features[s][4]),-float(features[s][0]),order_index[s]))
            members = set(required) | set(ordered[:budget-len(required)])
            selected = tuple(s for s in candidates if s in members)
        if len(selected) != budget or len(set(selected)) != budget or not set(required) <= set(selected) or not set(selected) <= set(candidates):
            raise ValueError('Subset policy violated the batch budget/refresh constraint')
        self.first_seen = first_seen
        self._target_id = observation.target_id
        self._pending = Decision(observation.target_id,observation.timestamp,candidates,logits,
                                 tuple(selected),q,budget,refresh,state,policy)
        self._pending_observation = observation
        return self._pending

    def subset_distribution(self, decision):
        if decision is not self._pending:
            raise ValueError('Distribution requires the current uncommitted decision')
        return self.subset_policy.distribution(decision.logits,decision.candidates,decision.budget,
            decision.refresh.required_ids,self.acquisition.max_exact_subsets,decision.state)

    def cancel(self):
        self._pending = None
        self._pending_observation = None

    def commit(self, observation: Observation, packets, backend: Backend):
        decision = self._pending
        if observation is not self._pending_observation:
            raise ValueError('Commit must use the exact local observation that was scored')
        if decision is None or (decision.target_id,decision.timestamp) != (observation.target_id,observation.timestamp):
            raise RuntimeError('No matching pending decision')
        packets = packet_tuple(packets)
        received = {p.observer_id:p for p in packets}
        if set(received) != set(decision.selected_ids):
            raise ValueError('Received batch must exactly match the entire selected subset')
        packets = tuple(received[s] for s in decision.selected_ids)
        updated = {}
        for packet in packets:
            validate_packet(packet,observation,packet.observer_id)
            memory = self.memories.get(packet.observer_id)
            old = memory.hidden if memory else self.network.initial_state()
            last = memory.last_timestamp if memory else None
            new = self.network.update(packet,old,last)
            updated[packet.observer_id] = SourceMemory(new,packet.timestamp,(memory.count if memory else 0)+1)
        # No session/source history mutation until the entire batch validates.
        try:
            reconstruction = backend.reconstruct(observation,packets,tuple(self.history))
        except WarmupError:
            reconstruction = None
        self.memories.update(updated)
        self.history.append(AcquiredFrame(observation,packets))
        self.history = self.history[-self.history_limit:]
        self.demand_history.append((observation.timestamp,decision.demand_vector.detach().cpu().numpy().copy()))
        self.demand_history = self.demand_history[-self.history_limit:]
        if reconstruction is not None:
            self.reconstruction_history.append((observation.timestamp,reconstruction))
            self.reconstruction_history = self.reconstruction_history[-self.history_limit:]
        self.previous = observation
        self.previous_selected = decision.selected_ids
        self.steps += 1
        self._pending = None
        self._pending_observation = None
        return reconstruction

    @torch.no_grad()
    def step(self, observation, candidates, provider, backend, geometry=None):
        decision = self.score(observation,candidates,geometry)
        packets = provider.acquire_many(decision.selected_ids,observation.target_id,observation.timestamp)
        return decision, self.commit(observation,packets,backend)

    def detach_memory(self):
        for memory in self.memories.values():
            memory.hidden = memory.hidden.detach()
