"""Adaptive source sets and an independent earliest-deadline refresh rule."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SelectionConfig:
    refresh_after_s: float | None = 1.0
    refresh_slots: int = 1

    def __post_init__(self):
        if self.refresh_after_s is not None and (
                not math.isfinite(self.refresh_after_s) or self.refresh_after_s <= 0):
            raise ValueError('Refresh age must be positive seconds, or None for an explicit ablation')
        if isinstance(self.refresh_slots, bool) or not isinstance(self.refresh_slots, int) or self.refresh_slots < 1:
            raise ValueError('Refresh slots must be a positive integer')


@dataclass(frozen=True)
class RefreshPlan:
    required_ids: tuple[str, ...]
    ages_s: tuple[float, ...]


def plan_refresh(candidates, now, first_seen, last_observed, config):
    due, ages = [], []
    for source in candidates:
        seen = source in last_observed
        origin = last_observed[source] if seen else first_seen[source]
        age = now - origin
        if not math.isfinite(age) or age < 0:
            raise ValueError('Refresh clocks must be finite and not in the future')
        ages.append(age)
        if config.refresh_after_s is None:
            continue
        deadline = origin + config.refresh_after_s if seen else origin
        if now >= deadline:
            due.append((deadline, source))
    due.sort()
    return RefreshPlan(tuple(source for _, source in due[:config.refresh_slots]), tuple(ages))


def choose_sources(scores, local_score, candidates, required, rng=None):
    """No Top-k cap. Supply rng only for training's Bernoulli roll-in."""
    if len(scores) != len(candidates) or len(set(candidates)) != len(candidates):
        raise ValueError('Scores must align with unique candidate IDs')
    if not set(required) <= set(candidates):
        raise ValueError('Refresh sources must belong to the candidate roster')
    if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(local_score)):
        raise ValueError('View scores must be finite')
    margins = (scores - local_score).detach()
    selected = set(required)
    for source, margin in zip(candidates, margins):
        if source in selected:
            continue
        take = rng.random() < float(margin.sigmoid()) if rng is not None else float(margin) > 0
        if take:
            selected.add(source)
    return tuple(source for source in candidates if source in selected)


def matched_random(candidates, learned, rng):
    required = set(learned['refresh'])
    available = [source for source in candidates if source not in required]
    count = len(learned['selected']) - len(required)
    if count < 0 or count > len(available) or not required <= set(candidates):
        raise ValueError('Invalid matched-count reference decision')
    selected = required | set(rng.sample(available, count))
    return tuple(source for source in candidates if source in selected)
