from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
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
    # Read-only compatibility with older replay files; current encoders ignore priors.
    prior_xy: np.ndarray | None = None
    prior_valid: np.ndarray | None = None
    prior_timestamp: float | None = None
    # Known reconstruction inputs; never passed to the selector feature encoders.
    backend_context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Geometry:
    """Estimated geometry available before request; not an external current summary.

    Four value slots plus masks. Current camera_geometry fills overlap and optical
    axis dot product in slots 0 and 3; the two unused slots stay zero and masked.
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
    # Camera metadata acquired after selection; optional DINO is baseline-only.
    payload: dict[str, Any] = field(default_factory=dict)
