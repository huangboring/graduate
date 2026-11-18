"""Causal camera geometry and post-acquisition multi-view joint triangulation."""
import math

import numpy as np

from .types import Geometry


def _camera(view):
    data = view.backend_context if hasattr(view, 'backend_context') else view.payload
    return np.asarray(data['camera_to_world']), np.asarray(data['intrinsics'])


def _sector(pose, half_fov, radius):
    forward = pose[:2, 2]
    if np.linalg.norm(forward) < 1e-6:
        return None
    yaw = math.atan2(forward[1], forward[0])
    angles = np.linspace(yaw - half_fov, yaw + half_fov, 25)
    base = pose[:2, 3]
    return np.vstack((base, base + radius * np.column_stack((np.cos(angles), np.sin(angles))))).astype('f4')


def camera_geometry(local, source, acquired_at, static, now, local_width, source_width, radius=5.0):
    import cv2

    def sector(calibration, width):
        pose, k = calibration
        # Midpoint principal-point approximation; all inputs use rectified pixels.
        return _sector(pose, math.atan(width / (2 * k[0, 0])), radius)

    left, right = sector(local, local_width), sector(source, source_width)
    overlap = 0.0
    has_overlap = left is not None and right is not None
    if has_overlap:
        # Translate both polygons before the float32 polygon-area computation.
        origin = left[0].copy()
        left, right = left-origin, right-origin
        area, _ = cv2.intersectConvexConvex(left, right)
        overlap = float(np.clip(area / cv2.contourArea(left), 0, 1))
    a, b = local[0][:3, 2], source[0][:3, 2]
    dot = float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    return Geometry(np.array([overlap, 0., 0., dot], dtype='f4'),
                    np.array([has_overlap, False, False, True]),
                    now if static else acquired_at, 'local_estimate' if static else 'history')


def triangulate_views(views, head_position, head_rotation, scores, min_confidence=.15,
                     min_parallax_deg=2., max_error_px=24.):
    """Use all retained views jointly. Return [13,3] and states 0/1/2."""
    cameras = [_camera(v) for v in views]
    poses = np.stack([c[0] for c in cameras]).astype('f8')
    k = np.stack([c[1] for c in cameras]).astype('f8')
    rotation, centres = poses[:, :3, :3], poses[:, :3, 3]
    xy = np.stack([v.xy for v in views]).astype('f8')
    valid = np.stack([v.valid for v in views]).astype(bool) & np.isfinite(xy).all(-1)
    weights = np.clip(np.asarray(scores, dtype='f8'), 0., 1.)
    valid &= np.isfinite(weights) & (weights >= min_confidence)
    pixels = np.concatenate((np.where(valid[..., None], xy, 0), np.ones((*xy.shape[:2], 1))), -1)
    rays = np.einsum('vij,vkj->vki', rotation @ np.linalg.inv(k), pixels)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    values = np.zeros((13, 3), dtype='f4')
    state = np.zeros(13, dtype='u1')
    max_cos = math.cos(math.radians(min_parallax_deg))
    for joint in range(13):
        indices = np.flatnonzero(valid[:, joint])
        if not len(indices):
            continue
        directions = rays[indices, joint]
        origins = centres[indices]
        ws = weights[indices, joint]
        best = np.argmax(ws)
        endpoint = origins[best] + np.linalg.norm(head_position-origins[best]) * directions[best]
        state[joint] = 1
        if len(indices) >= 2 and np.min(np.abs(directions @ directions.T)) < max_cos:
            # Minimise the weighted squared perpendicular distance to all rays.
            projectors = np.eye(3) - directions[:, :, None] * directions[:, None, :]
            weighted = ws[:, None, None] * projectors
            system = weighted.sum(0)
            if np.linalg.cond(system) < 1e6:
                point = np.linalg.solve(system, np.einsum('vij,vj->i', weighted, origins))
                camera = np.einsum('vji,vj->vi', rotation[indices], point-origins)
                projected = np.einsum('vij,vj->vi', k[indices], camera)
                if np.all(camera[:, 2] > 1e-6):
                    errors = np.linalg.norm(projected[:, :2] / projected[:, 2:] - xy[indices, joint], axis=1)
                    if np.max(errors) <= max_error_px:
                        endpoint = point
                        state[joint] = 2
        values[joint] = (head_rotation.T @ (endpoint - head_position)) * ws[best]
    return values, state
