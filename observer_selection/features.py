"""Causal feature preparation, separate from learnable encoders.

Availability masks say a number exists, NOT that a joint is visible or correct.
The encoder never reads backend_context, packet payloads, or pose ground truth.
"""
from __future__ import annotations

import math
import numpy as np

from .types import Geometry, Observation, Packet


def _time(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite seconds")
    return value


def _mask(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain only boolean/0/1 availability values")
    return array.astype(bool)


def _box(value: np.ndarray) -> tuple[np.ndarray, float]:
    box = np.asarray(value, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("bbox must be finite [x0,y0,x1,y1]")
    dimensions = box[2:] - box[:2]
    if np.any(dimensions <= 0):
        raise ValueError("bbox must have positive width and height")
    return (box[:2] + box[2:]) / 2, float(dimensions.max())


def _observed_features(observation: Observation | Packet, joints: int | None):
    xy = np.asarray(observation.xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0:
        raise ValueError("xy must have shape [J,2] with J > 0")
    if joints is not None and xy.shape[0] != joints:
        raise ValueError(f"expected {joints} joints, got {xy.shape[0]}")
    valid = _mask(observation.valid, (len(xy),), "valid")
    if not np.isfinite(xy[valid]).all():
        raise ValueError("available xy coordinates must be finite")
    center, scale = _box(observation.bbox)
    normalized = np.where(valid[:, None], (xy - center) / scale, 0.0)
    confidence = np.zeros(len(xy), dtype=np.float64)
    confidence_valid = np.zeros(len(xy), dtype=bool)
    if observation.confidence is not None:
        provided = np.asarray(observation.confidence, dtype=np.float64)
        if provided.shape != (len(xy),):
            raise ValueError("confidence must have shape [J]")
        confidence_valid = valid & np.isfinite(provided)
        if np.any((provided[confidence_valid] < 0) | (provided[confidence_valid] > 1)):
            raise ValueError("confidence must be normalized to [0,1] before encoding")
        confidence = np.where(confidence_valid, provided, 0.0)
    return normalized, valid, confidence, confidence_valid, center, scale


def demand_features(observation: Observation, previous: Observation | None = None,
                    joints: int | None = None) -> np.ndarray:
    """Return 11J+5 features without inventing unavailable history.

    Per joint: normalized xy (2), coordinate availability (1), confidence (1),
    confidence availability (1), local-view projected-prior residual (2), its
    availability (1), normalized local 2D displacement/second (2), its availability
    (1). The 2D rate is NOT a world-space motion velocity.

    Globals: bbox center/scale (2), log1p(bbox scale in pixels) (1), log1p(previous
    local observation age) (1), log1p(projected prior source age) (1).
    """
    now = _time(observation.timestamp, "timestamp")
    xy, valid, confidence, confidence_valid, center, scale = _observed_features(observation, joints)
    n = len(xy)
    residual = np.zeros((n, 2), dtype=np.float64)
    residual_valid = np.zeros(n, dtype=bool)
    rate = np.zeros((n, 2), dtype=np.float64)
    rate_valid = np.zeros(n, dtype=bool)
    previous_age = prior_age = 0.0

    if previous is not None:
        if previous.target_id != observation.target_id:
            raise ValueError("previous observation belongs to a different target")
        previous_age = now - _time(previous.timestamp, "previous timestamp")
        if previous_age <= 0:
            raise ValueError("previous observation must strictly precede current time")
        old_xy, old_valid, *_ = _observed_features(previous, n)
        rate_valid = valid & old_valid
        rate = np.where(rate_valid[:, None], (xy - old_xy) / previous_age, 0.0)

    if observation.prior_xy is not None:
        if observation.prior_timestamp is None:
            raise ValueError("projected prior requires its causal source timestamp")
        prior_age = now - _time(observation.prior_timestamp, "prior timestamp")
        if prior_age <= 0:
            raise ValueError("prior source timestamp must strictly precede current time")
        prior = np.asarray(observation.prior_xy, dtype=np.float64)
        if prior.shape != (n, 2):
            raise ValueError("prior_xy must have shape [J,2]")
        prior_valid = (np.isfinite(prior).all(axis=-1) if observation.prior_valid is None
                       else _mask(observation.prior_valid, (n,), "prior_valid"))
        if not np.isfinite(prior[prior_valid]).all():
            raise ValueError("available prior coordinates must be finite")
        residual_valid = valid & prior_valid
        prior_normalized = (prior - center) / scale
        residual = np.where(residual_valid[:, None], xy - prior_normalized, 0.0)
    elif observation.prior_valid is not None or observation.prior_timestamp is not None:
        raise ValueError("prior metadata without prior_xy is ambiguous")

    per_joint = np.column_stack((xy, valid, confidence, confidence_valid,
                                 residual, residual_valid, rate, rate_valid))
    globals_ = [*(center / scale), math.log1p(scale), math.log1p(previous_age),
                math.log1p(prior_age)]
    result = np.concatenate((per_joint.reshape(-1), globals_)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("feature magnitudes exceed finite float32 range")
    return result


def packet_features(packet: Packet, last_timestamp: float | None = None,
                    joints: int | None = None) -> np.ndarray:
    """5J+1 acquired-observation features, no ID embedding or old demand vector."""
    now = _time(packet.timestamp, "packet timestamp")
    xy, valid, confidence, confidence_valid, *_ = _observed_features(packet, joints)
    gap = 0.0
    if last_timestamp is not None:
        gap = now - _time(last_timestamp, "last packet timestamp")
        if gap <= 0:
            raise ValueError("new packet must be newer than the stored source observation")
    values = np.column_stack((xy, valid, confidence, confidence_valid)).reshape(-1)
    result = np.concatenate((values, [math.log1p(gap)])).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("packet feature magnitudes exceed finite float32 range")
    return result


def geometry_features(geometry: Geometry | None, now: float) -> np.ndarray:
    """Four geometry values, four individual masks, log1p(source age).

    Provenance labels are a validation contract, not proof of an honest upstream
    estimator. Callers must never label inaccessible current external telemetry as
    a local estimate. Unknown geometry is represented by zeros AND masks.
    """
    now = _time(now, "now")
    if geometry is None:
        return np.zeros(9, dtype=np.float32)
    if geometry.provenance not in {"local_estimate", "history"}:
        raise ValueError("geometry must come from local_estimate or history")
    age = now - _time(geometry.timestamp, "geometry timestamp")
    if age < 0:
        raise ValueError("future geometry is unavailable before selection")
    if geometry.provenance == "history" and age == 0:
        raise ValueError("current external geometry cannot be marked as history")
    values = np.asarray(geometry.values, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("geometry values must have shape [4]")
    valid = _mask(geometry.valid, (4,), "geometry valid")
    if not np.isfinite(values[valid]).all():
        raise ValueError("available geometry values must be finite")
    for i, (low, high) in enumerate(((0., 1.), (0., 1.), (-1., 1.), (-1., 1.))):
        if valid[i] and not low <= values[i] <= high:
            raise ValueError(f"geometry feature {i} must lie in [{low},{high}]")
    return np.concatenate((np.where(valid, values, 0.0), valid,
                           [math.log1p(age)])).astype(np.float32)


def _compact_observed(observation, joints):
    """Compact encoding assumes whole-view availability and finite confidences.

    Official prepared replays repeat bbox_valid for every joint. Reject mixed
    per-joint masks rather than silently discarding their missingness semantics.
    """
    values = _observed_features(observation,joints)
    _, valid, _, confidence_valid, _, _ = values
    if not np.all(valid == valid[0]):
        raise ValueError('Compact encoding requires whole-view availability; use legacy for per-joint masks')
    if valid[0] and not confidence_valid.all():
        raise ValueError('Compact encoding requires finite confidences for available joints')
    return values


def compact_demand_features(observation, previous=None, joints=None):
    """5J+6: xy/confidence/rate per joint, then four globals and two view masks.

    Globals: center_x/scale, center_y/scale, log1p(scale), log1p(previous age),
    current observation available, both frames available for rate estimation.
    No projected-prior fields or repeated per-joint masks are retained.
    """
    now = _time(observation.timestamp,'timestamp')
    xy,valid,confidence,_,center,scale = _compact_observed(observation,joints)
    rate = np.zeros_like(xy)
    age = 0.
    rate_available = False
    if previous is not None:
        if previous.target_id != observation.target_id:
            raise ValueError('previous observation belongs to a different target')
        age = now-_time(previous.timestamp,'previous timestamp')
        if age <= 0:
            raise ValueError('previous observation must strictly precede current time')
        old,old_valid,*_ = _compact_observed(previous,len(xy))
        rate_available = bool(valid[0] and old_valid[0])
        if rate_available:
            rate = (xy-old)/age
    rows = np.column_stack((xy,confidence,rate))
    globals_ = [*(center/scale),math.log1p(scale),math.log1p(age),float(valid[0]),float(rate_available)]
    result = np.concatenate((rows.reshape(-1),globals_)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Compact features exceed finite float32 range')
    return result


def compact_packet_features(packet,last_timestamp=None,joints=None):
    """3J+2: xy/confidence per joint, view availability and acquisition gap."""
    now = _time(packet.timestamp,'packet timestamp')
    xy,valid,confidence,*_ = _compact_observed(packet,joints)
    gap = 0.
    if last_timestamp is not None:
        gap = now-_time(last_timestamp,'last packet timestamp')
        if gap <= 0:
            raise ValueError('new packet must be newer than the stored source observation')
    result = np.concatenate((np.column_stack((xy,confidence)).reshape(-1),
                             [float(valid[0]),math.log1p(gap)])).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Compact packet features exceed finite float32 range')
    return result
