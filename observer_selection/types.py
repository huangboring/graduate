from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np


@dataclass(frozen=True)
class Observation:
    """Local detector output. Coordinates are pixels, timestamps seconds."""
    target_id: str
    timestamp: float
    xy: np.ndarray
    valid: np.ndarray
    bbox: np.ndarray  # x0,y0,x1,y1; measured local target box
    confidence: np.ndarray | None = None
    # Optional causal pose prior ALREADY projected into A's current camera.
    prior_xy: np.ndarray | None = None
    prior_valid: np.ndarray | None = None
    prior_timestamp: float | None = None
    # Known reconstruction inputs; never passed to the selector feature encoders.
    backend_context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Geometry:
    """Estimated geometry available before request; not an external current summary.

    Four features: overlap fraction, target coverage, sin(relative yaw), cos(yaw).
    Individual availability is explicit; provenance must be local_estimate or history.
    """
    values: np.ndarray
    valid: np.ndarray
    timestamp: float
    provenance: str


@dataclass(frozen=True)
class Packet:
    observer_id: str
    target_id: str
    timestamp: float
    xy: np.ndarray
    valid: np.ndarray
    bbox: np.ndarray
    confidence: np.ndarray | None = None
    # Backend-specific data acquired only after selection (e.g. rays/DINO).
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcquiredFrame:
    observation: Observation
    packets: tuple[Packet, ...]

    def __post_init__(self):
        object.__setattr__(self, 'packets', packet_tuple(self.packets))

    @property
    def packet(self):
        """Single-source fixture convenience; never silently discard extra views."""
        if len(self.packets) != 1:
            raise ValueError('Frame contains multiple packets; use .packets')
        return self.packets[0]


def packet_tuple(packets) -> tuple[Packet, ...]:
    values = (packets,) if isinstance(packets, Packet) else tuple(packets)
    if not values or not all(isinstance(p, Packet) for p in values):
        raise ValueError('Acquisition requires a nonempty collection of Packet objects')
    if len({p.observer_id for p in values}) != len(values):
        raise ValueError('Duplicate observer packets in one acquisition')
    return values


@dataclass(frozen=True)
class Reconstruction:
    joints: np.ndarray  # [J,3], metres
    coordinate_frame: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    """Pure counterfactual-safe reconstruction. Never mutate history or inputs."""
    def reconstruct(self, observation: Observation, packets: tuple[Packet, ...],
                    history: tuple[AcquiredFrame, ...]) -> Reconstruction: ...


class PacketProvider(Protocol):
    def acquire_many(self, observer_ids: tuple[str, ...], target_id: str,
                     timestamp: float) -> tuple[Packet, ...]: ...


class NotReadyError(RuntimeError):
    """Required backend assets/dependencies are unavailable; never silently skip."""


class WarmupError(NotReadyError):
    """Valid inputs, but not enough real past frames for a fixed-window backend."""
