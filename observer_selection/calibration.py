"""Shared calibration validation, independent of any reconstruction network."""
import numpy as np

# Retained prepared-replay schema tags; these describe stored coordinate axes.
CONVENTION = 'local_aux_camera_rays_v1'
OUTPUT_AXES = 'last_requester_camera_opencv'


def rigid(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'{name} must be a finite 4x4 transform')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4)
            or not np.isclose(np.linalg.det(rotation), 1, atol=2e-4)):
        raise ValueError(f'{name} must be a rigid camera-to-world transform')
    return matrix
