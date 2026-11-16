"""Causal image-only association, not a trained person ReID model.

Manual initial identity is the sole identity seed. No geometry, GT, source IDs
or future images are used for association. Generic DINOv2 appearance is a
baseline and cannot certify identity; ambiguous candidates remain missing.
"""
from dataclasses import asdict, dataclass
import numpy as np

VERSION = 'causal_dinov2_association_v1'


@dataclass(frozen=True)
class AssociationConfig:
    tracking_similarity: float = .55
    recovery_similarity: float = .75
    recovery_margin: float = .08
    tracking_margin: float = .08
    minimum_iou: float = .05
    gallery_similarity: float = .85
    gallery_interval: int = 10
    gallery_size: int = 8


def normalized(features):
    a = np.asarray(features, dtype=np.float32)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Appearance features must be finite [N,D]')
    lengths = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(lengths < 1e-8): raise ValueError('Zero appearance descriptor')
    return a / lengths


def box_iou(box, boxes):
    boxes = np.asarray(boxes)
    low, high = np.maximum(box[:2], boxes[:, :2]), np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(high-low, 0).prod(-1)
    union = (box[2:]-box[:2]).prod() + (boxes[:, 2:]-boxes[:, :2]).prod(-1) - intersection
    return intersection / np.maximum(union, 1e-8)


class AppearanceTracker:
    def __init__(self, config=None):
        self.config = config or AssociationConfig()
        self.gallery = []
        self.previous = None
        self.last_seen = self.last_step = self.gallery_frame = -1
        self.recoveries = 0
        self.reason = 'uninitialized'

    @property
    def initialized(self): return bool(self.gallery)

    def initialize(self, frame, box, feature):
        if self.initialized: raise ValueError('Identity may only be initialized once')
        self.gallery = [normalized(np.asarray(feature)[None])[0].copy()]
        self.previous = np.asarray(box, dtype=np.float32).copy()
        if self.previous.shape != (4,) or np.any(self.previous[2:] <= self.previous[:2]):
            raise ValueError('Invalid identity box')
        self.last_seen = self.last_step = self.gallery_frame = int(frame)
        self.reason = 'manual_initial_identity'

    def step(self, frame, boxes, features):
        if frame <= self.last_step: raise ValueError('Tracker requires strictly increasing frame indices')
        self.last_step = frame
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.shape != (len(features), 4) or not np.isfinite(boxes).all():
            raise ValueError('Invalid candidate boxes/features')
        if not self.initialized:
            self.reason = 'uninitialized'; return None
        if not len(boxes):
            self.reason = 'no_detection'; return None
        if np.any(boxes[:, 2:] <= boxes[:, :2]): raise ValueError('Invalid candidate extent')
        features = normalized(features)
        # A gallery only contains previously accepted, high-confidence crops.
        similarities = features @ np.stack(self.gallery).T
        appearance = similarities.max(axis=1)
        ious = box_iou(self.previous, boxes)
        gap = frame - self.last_seen
        cfg = self.config
        selected = None
        if gap <= 5:
            score = .7*appearance + .3*ious
            ranked = np.argsort(score)[::-1]
            best = int(ranked[0])
            margin = float(score[best]-score[ranked[1]]) if len(ranked)>1 else float('inf')
            if (appearance[best] >= cfg.tracking_similarity and ious[best] >= cfg.minimum_iou
                    and margin >= cfg.tracking_margin):
                selected = best
        if selected is None:
            # Always try global appearance association, even after a long gap.
            ranked = np.argsort(appearance)[::-1]
            best = int(ranked[0])
            margin = float(appearance[best]-appearance[ranked[1]]) if len(ranked)>1 else float('inf')
            if appearance[best] >= cfg.recovery_similarity and margin >= cfg.recovery_margin:
                selected = best
                self.reason = 'appearance_recovery'
                self.recoveries += 1
            else:
                self.reason = 'ambiguous' if appearance[best]>=cfg.recovery_similarity else 'appearance_mismatch'
                return None
        else:
            self.reason = 'motion_and_appearance'
        self.previous = boxes[selected].copy()
        self.last_seen = frame
        # Keep the immutable initial seed; cautiously extend the pose gallery.
        if (appearance[selected] >= cfg.gallery_similarity and gap <= 2
                and frame-self.gallery_frame >= cfg.gallery_interval):
            self.gallery.append(features[selected].copy())
            if len(self.gallery)>cfg.gallery_size: self.gallery.pop(1)
            self.gallery_frame = frame
        return selected

    def description(self): return {'version': VERSION, 'config': asdict(self.config)}
