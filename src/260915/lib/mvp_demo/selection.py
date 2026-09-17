"""Single-frame, batch-one view selection before fusion.

This is a centralized, simulated transport boundary, not the original When2com
protocol. Only compact descriptors are available to the selection policy.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
import itertools
import math
import random


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def describe(features, dimension=32):
    """Parameter-free compact descriptors; no claim of learned visibility."""
    if dimension < 1 or not features:
        raise ValueError('A positive descriptor dimension and feature levels are required')
    pooled = torch.cat([x.mean((-2, -1)) for x in features], dim=1)
    return F.adaptive_avg_pool1d(pooled.unsqueeze(1), dimension).squeeze(1).float()


class QueryKeySelector(nn.Module):
    """Trainable scene-query/view-key scorer. Requires separately trained weights.

    The scene query is computed from transmitted summaries, not full features.
    This is When2com-inspired, not a reproduction of asymmetric peer handshakes.
    """
    def __init__(self, dimension=32):
        super().__init__()
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)

    def forward(self, descriptors):
        query = self.query(descriptors.mean(0))
        return (self.key(descriptors) * query).sum(-1) / descriptors.shape[-1] ** 0.5


def candidate_subsets(count, k, limit=64, seed=42):
    """Exhaust small sets, sample bounded candidates for large camera arrays."""
    if not 2 <= k <= count or limit < 2:
        raise ValueError('Require 2 <= k <= camera count and subset limit >= 2')
    if math.comb(count, k) <= limit:
        return list(itertools.combinations(range(count), k))
    rng = random.Random(seed)
    # Always include the fixed-first-K baseline for an identical-budget comparison.
    subsets = {tuple(range(k))}
    while len(subsets) < limit:
        subsets.add(tuple(sorted(rng.sample(range(count), k))))
    return sorted(subsets)


def subset_masks(subsets, count, device):
    masks = torch.zeros(len(subsets), count, device=device)
    for row, indices in enumerate(subsets):
        masks[row, list(indices)] = 1.
    return masks


class SubsetSelector(nn.Module):
    """Shared set encoder scores whole combinations, independent of camera IDs.

    Six geometry values are static calibrated camera center / 10 m and forward
    direction in MVGFormer world coordinates. No fixed camera-count parameter.
    """
    architecture = 'subset_set_v1'

    def __init__(self, dimension=32, hidden=64):
        super().__init__()
        self.dimension, self.hidden = dimension, hidden
        self.encoder = nn.Sequential(nn.Linear(dimension + 6, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden * 3 + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, descriptors, masks, geometry):
        encoded = self.encoder(torch.cat((torch.sign(descriptors)*torch.log1p(descriptors.abs()), geometry), -1))
        sizes = masks.sum(-1, keepdim=True)
        selected_mean = masks @ encoded / sizes
        selected_second = masks @ encoded.square() / sizes
        variance = (selected_second - selected_mean.square()).clamp_min(0)
        context = encoded.mean(0).expand(len(masks), -1)
        count_features = torch.cat((sizes / len(encoded), torch.log1p(sizes)), -1)
        return self.head(torch.cat((selected_mean, variance, context, count_features), -1)).squeeze(-1)


@dataclass
class Selection:
    indices: list
    scores: list
    descriptor_bytes: int
    decision_bytes: int
    feature_bytes: int
    all_feature_bytes: int

    @property
    def total_bytes(self):
        return self.descriptor_bytes + self.decision_bytes + self.feature_bytes

    @property
    def saving_fraction(self):
        return 1 - self.total_bytes / self.all_feature_bytes


def select(features, camera_ids, k, policy='all', fixed_ids=None, scorer=None, dimension=32,
           geometry=None, subset_limit=64):
    count = len(camera_ids)
    if count < 2 or len(set(camera_ids)) != count:
        raise ValueError('At least two distinct cameras are required')
    if not features or any(x.ndim != 4 or x.shape[0] != count for x in features):
        raise ValueError('MVP selection supports batch=1 with view-major feature levels')
    if not 2 <= k <= count:
        raise ValueError('k must be between 2 and the number of cameras')
    all_bytes = sum(tensor_bytes(x) for x in features)
    scores = torch.zeros(count)
    descriptor_bytes = decision_bytes = 0
    if policy == 'all':
        indices = list(range(count))
    elif policy == 'fixed':
        ids = camera_ids[:k] if fixed_ids is None else fixed_ids
        if len(ids) != k or len(set(ids)) != k or any(x not in camera_ids for x in ids):
            raise ValueError('fixed IDs must be k unique candidate cameras')
        indices = [camera_ids.index(x) for x in ids]
    elif policy in ('descriptor', 'learned'):
        descriptors = describe(features, dimension)
        if not torch.isfinite(descriptors).all():
            raise ValueError('Non-finite descriptors')
        descriptor_bytes = tensor_bytes(descriptors)
        # A uint8 selected/not-selected response per candidate; headers excluded.
        decision_bytes = count
        if policy == 'learned':
            if scorer is None:
                raise ValueError('learned policy requires a trained selector checkpoint')
            if isinstance(scorer, SubsetSelector):
                if geometry is None or geometry.shape != (count, 6):
                    raise ValueError('Subset selector requires calibrated geometry for all candidates')
                subsets = candidate_subsets(count, k, subset_limit)
                masks = subset_masks(subsets, count, descriptors.device)
                subset_scores = scorer(descriptors, masks, geometry)
                if not torch.isfinite(subset_scores).all():
                    raise ValueError('Non-finite subset scores')
                indices = list(subsets[subset_scores.argmax().item()])
                # Display mean score of sampled combinations containing each camera.
                scores = (masks.T @ subset_scores) / masks.sum(0).clamp_min(1)
            else:
                scores = scorer(descriptors)
        else:
            # Explicit heuristic baseline: activation magnitude, not visibility.
            scores = descriptors.square().mean(-1)
        if not torch.isfinite(scores).all():
            raise ValueError('Non-finite selector scores')
        if not (policy == 'learned' and isinstance(scorer, SubsetSelector)):
            indices = torch.argsort(scores, descending=True, stable=True)[:k].tolist()
    else:
        raise ValueError(f'Unknown selection policy: {policy}')
    # Keep canonical camera order for feature/metadata alignment.
    indices = sorted(indices)
    return Selection(indices, scores.detach().cpu().tolist(), descriptor_bytes,
                     decision_bytes, all_bytes * len(indices) // count, all_bytes)


def select_payload(features, views, metadata, selection):
    """Remove unselected spatial tensors and matching metadata at the boundary."""
    indices = selection.indices
    return ([x[indices] for x in features], [views[i] for i in indices],
            [metadata[i] for i in indices])
