"""Offline reconstruction costs for the selection objective; no teacher network."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .runtime import validate_packet
from .types import Backend, Observation, Packet, AcquiredFrame, Reconstruction


def mpjpe(prediction: Reconstruction, gt: np.ndarray, valid: np.ndarray,
          coordinate_frame: str) -> float:
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(prediction.joints, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if prediction.coordinate_frame != coordinate_frame:
        raise ValueError('Prediction and GT coordinate frames differ; no implicit alignment allowed')
    if gt.shape != pred.shape or gt.ndim != 2 or gt.shape[1] != 3 or mask.shape != gt.shape[:1]:
        raise ValueError('Prediction/GT joint layouts must match [J,3] and valid [J]')
    if not mask.any() or not np.isfinite(gt[mask]).all() or not np.isfinite(pred[mask]).all():
        raise ValueError('No valid finite pose reference')
    return float(np.linalg.norm(pred[mask] - gt[mask], axis=-1).mean())


@dataclass(frozen=True)
class CandidateCosts:
    errors_m: np.ndarray
    predictions: tuple[Reconstruction, ...]


class CandidateCostEvaluator:
    """Offline, fixed-backend candidate MPJPE evaluation; no teacher network.

    All candidates see exactly the same acquired history; no hypothetical branch
    is committed. GT is used after reconstruction solely by the metric.
    """
    def __init__(self, backend: Backend):
        self.backend = backend
        # Freeze the known neural components without ever touching the selector.
        # The backend may be a pure numeric simulator with no neural modules.
        model = getattr(backend, 'model', None)
        components = [getattr(backend, 'net', None), getattr(model, 'net', None),
                      getattr(model, 'bm', None)]
        for component in components:
            if isinstance(component, torch.nn.Module):
                component.eval()
                component.requires_grad_(False)

    @torch.no_grad()
    def evaluate(self, obs: Observation, candidates: tuple[str, ...], packets: dict[str, Packet],
              history: tuple[AcquiredFrame, ...], gt: np.ndarray, gt_valid: np.ndarray,
              coordinate_frame: str) -> CandidateCosts:
        preds, errors = [], []
        for source in candidates:
            packet = packets[source]
            validate_packet(packet, obs, source)
            pred = self.backend.reconstruct(obs, packet, history)
            preds.append(pred)
            errors.append(mpjpe(pred, gt, gt_valid, coordinate_frame))
        return CandidateCosts(np.asarray(errors, dtype=np.float32), tuple(preds))


class SubsetCostEvaluator(CandidateCostEvaluator):
    """Evaluate actual jointly reconstructed subsets under one fixed history.

    Inherits only backend freezing. No sum/average of singleton error labels.
    The caller supplies the legal subsets after applying the refresh constraint.
    """
    @torch.no_grad()
    def evaluate(self, obs, subsets, packets, history, gt, gt_valid, coordinate_frame):
        if not subsets or any(not s or len(set(s)) != len(s) for s in subsets):
            raise ValueError('Expected nonempty unique-source subsets')
        if len({frozenset(s) for s in subsets}) != len(subsets):
            raise ValueError('Duplicate unordered subsets')
        preds, errors = [], []
        for subset in subsets:
            batch = tuple(packets[s] for s in subset)
            for source, packet in zip(subset,batch):
                validate_packet(packet,obs,source)
            pred = self.backend.reconstruct(obs,batch,history)
            preds.append(pred)
            errors.append(mpjpe(pred,gt,gt_valid,coordinate_frame))
        return CandidateCosts(np.asarray(errors,dtype=np.float32),tuple(preds))
