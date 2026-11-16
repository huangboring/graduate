"""Batch subset policy and a separate, deterministic refresh scheduler.

No current external observation is accepted here. The scheduler constrains the
set before any acquisition. The additive policy is replaceable independently
of the acquisition protocol; it does not claim pairwise complementarity.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import math
import random
from typing import Protocol
import torch


@dataclass(frozen=True)
class AcquisitionConfig:
    max_sources: int = 2
    refresh_after_s: float | None = 1.0
    refresh_slots: int = 1
    max_exact_subsets: int = 4096

    def __post_init__(self):
        for name in ('max_sources', 'refresh_slots', 'max_exact_subsets'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.refresh_slots > self.max_sources:
            raise ValueError('refresh_slots cannot exceed max_sources')
        if self.refresh_after_s is not None and (not math.isfinite(self.refresh_after_s) or self.refresh_after_s <= 0):
            raise ValueError('refresh_after_s must be positive seconds or None for an explicit ablation')


@dataclass(frozen=True)
class RefreshPlan:
    required_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    ages_s: tuple[float, ...]  # Since last acquisition, or first appearance if unseen.
    seen: tuple[bool, ...]
    overdue_s: tuple[float, ...]


def plan_refresh(candidates, timestamp, first_seen, last_acquired, config):
    """Earliest deadline first, independent of model scores and packet contents.

    Unseen deadline = first appearance; seen deadline = last success + threshold.
    Thus new arrivals do not continually jump ahead of already overdue sources.
    Normal selections reset the same last-acquisition clock as forced refreshes.
    """
    due, ages, seen_flags, overdue = [], [], [], []
    for source in candidates:
        seen = source in last_acquired
        start = last_acquired[source] if seen else first_seen[source]
        age = timestamp - start
        if not math.isfinite(age) or age < 0:
            raise ValueError('Source timestamps must not be future/nonfinite')
        seen_flags.append(seen); ages.append(age)
        if config.refresh_after_s is None:
            overdue.append(0.0)
            continue
        deadline = start + config.refresh_after_s if seen else start
        overdue.append(max(0.0, timestamp - deadline))
        if timestamp >= deadline:
            due.append((deadline, source, 'expired' if seen else 'unseen'))
    due.sort(key=lambda item: (item[0], item[1]))
    chosen = due[:min(config.refresh_slots, config.max_sources, len(candidates))]
    return RefreshPlan(tuple(x[1] for x in chosen), tuple(x[2] for x in chosen),
                       tuple(ages), tuple(seen_flags), tuple(overdue))


@dataclass(frozen=True)
class SubsetDistribution:
    subsets: tuple[tuple[str, ...], ...]
    logits: torch.Tensor

    @property
    def probabilities(self):
        return torch.softmax(self.logits, dim=0)

    def sample(self, rng: random.Random):
        return self.subsets[rng.choices(range(len(self.subsets)),
            weights=self.probabilities.detach().cpu().tolist(), k=1)[0]]


@dataclass(frozen=True)
class PreAcquisitionState:
    demand: torch.Tensor
    source_memories: tuple[torch.Tensor, ...]
    geometry_features: torch.Tensor
    ages_s: tuple[float, ...]
    seen: tuple[bool, ...]


class SubsetPolicy(Protocol):
    """Extension point for a future conditional/set-aware policy.

    Both methods receive pre-acquisition scores/state only. A replacement must
    keep map() and distribution() consistent and always include required_ids.
    """
    def map(self, logits, candidates, budget, required_ids, state=None): ...
    def distribution(self, logits, candidates, budget, required_ids, max_subsets, state=None): ...


def _validate(logits, candidates, budget, required_ids):
    if (not candidates or len(set(candidates)) != len(candidates)
            or logits.shape != (len(candidates),) or not torch.isfinite(logits).all()):
        raise ValueError('Need unique candidates and matching finite source logits')
    if not 1 <= budget <= len(candidates):
        raise ValueError('Budget must fit the candidate roster')
    if len(set(required_ids)) != len(required_ids) or not set(required_ids) <= set(candidates) or len(required_ids) > budget:
        raise ValueError('Invalid mandatory refresh subset')


class AdditiveSubsetPolicy:
    """P(S) proportional to exp(sum of source logits), fixed cardinality.

    MAP is Top-k with mandatory refresh members. This is an unordered-set
    distribution, not sequential Plackett-Luce or independent Bernoulli draws.
    Runtime MAP is O(N log N); only offline exact cost training enumerates sets.
    """
    def map(self, logits, candidates, budget, required_ids=(), state=None):
        _validate(logits, candidates, budget, required_ids)
        values = logits.detach().cpu().tolist()
        remaining = [i for i, s in enumerate(candidates) if s not in required_ids]
        remaining.sort(key=lambda i: (-values[i], candidates[i]))
        selected = set(required_ids) | {candidates[i] for i in remaining[:budget-len(required_ids)]}
        return tuple(s for s in candidates if s in selected)

    def distribution(self, logits, candidates, budget, required_ids=(), max_subsets=4096, state=None):
        _validate(logits, candidates, budget, required_ids)
        free = tuple(s for s in candidates if s not in required_ids)
        choose = budget - len(required_ids)
        count = math.comb(len(free), choose)
        if count > max_subsets:
            raise ValueError(f'Exact subset objective requires {count} reconstructions/frame (limit {max_subsets}); '
                             'raise the explicit limit or design a documented sampled objective. No silent truncation.')
        subsets = []
        for extra in combinations(free, choose):
            members = set(required_ids) | set(extra)
            subsets.append(tuple(s for s in candidates if s in members))
        index = {s:i for i,s in enumerate(candidates)}
        # Required members add the same constant to every set; remove it before
        # summation to avoid numerical cancellation and ensure zero direct gradient.
        scores = torch.stack([logits[[index[s] for s in subset if s not in required_ids]].sum()
                              for subset in subsets])
        if not torch.isfinite(scores).all():
            raise ValueError('Subset score overflow; cannot form a finite selection distribution')
        return SubsetDistribution(tuple(subsets), scores)
