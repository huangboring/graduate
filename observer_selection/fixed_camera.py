"""Rectified fixed cameras from the EgoHumans COLMAP text model."""
from pathlib import Path

import numpy as np

from .raw_prepare import similarity_camera_extrinsic


def _rotation(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class FixedCamera:
    def __init__(self, workplace, name, world_to_colmap, paths):
        import cv2

        model = Path(workplace)
        rows = (model/'images.txt').read_text().splitlines()
        start = next(i for i, row in enumerate(rows) if row.strip() and not row.startswith('#'))
        matches = [row.split() for row in rows[start::2] if row.split()
                   and row.split()[-1].replace('\\', '/').split('/')[0] == name]
        if not matches:
            raise ValueError(f'No COLMAP image registered for {name}')
        match = matches[0]
        if any(row[8] != match[8] for row in matches):
            raise ValueError(f'{name} has inconsistent COLMAP intrinsics')
        params = next(row.split()[1:] for row in (model/'cameras.txt').read_text().splitlines()
                      if row.split() and row.split()[0] == match[8])
        kind, width, height, *intr = params
        size = int(width), int(height)
        fx, fy, cx, cy, *dist = map(float, intr)
        self.k = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        poses = []
        for row in matches:
            extr = np.eye(4)
            extr[:3, :3] = _rotation([float(x) for x in row[1:5]])
            extr[:3, 3] = [float(x) for x in row[5:8]]
            extr, self.scale = similarity_camera_extrinsic(extr @ world_to_colmap)
            poses.append(np.linalg.inv(extr))
        poses = np.stack(poses)
        self.pose = np.eye(4)
        u, _, vt = np.linalg.svd(poses[:,:3,:3].mean(0))
        self.pose[:3,:3] = u @ np.diag([1.,1.,np.linalg.det(u@vt)]) @ vt
        self.pose[:3,3] = np.median(poses[:,:3,3],axis=0)
        centre_spread = np.linalg.norm(poses[:,:3,3]-self.pose[:3,3],axis=1).max()
        relative = poses[:,:3,:3] @ self.pose[:3,:3].T
        angular_spread = np.degrees(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1))).max()
        if centre_spread > .15 or angular_spread > 5.:
            raise ValueError(f'{name} moves too far for static calibration: {centre_spread:.3f} m, {angular_spread:.2f} deg')
        self.paths, self.size = paths, size
        downscale = min(1., 1408 / max(size))
        self.output_size = tuple(max(1, round(x * downscale)) for x in size)
        new_k = np.diag([self.output_size[0]/size[0], self.output_size[1]/size[1], 1.]) @ self.k
        if kind == 'OPENCV_FISHEYE':
            self.maps = cv2.fisheye.initUndistortRectifyMap(self.k, np.asarray(dist), np.eye(3),
                new_k, self.output_size, cv2.CV_16SC2)
        elif kind == 'OPENCV':
            self.maps = cv2.initUndistortRectifyMap(self.k, np.asarray(dist), None,
                new_k, self.output_size, cv2.CV_16SC2)
        elif kind == 'PINHOLE':
            self.maps = None
        else:
            raise ValueError(f'Unsupported COLMAP camera model for {name}: {kind}')
        self.k = new_k

    def get(self, index):
        import cv2

        image = cv2.imdecode(np.fromfile(self.paths[index],dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.shape[1::-1] != self.size:
            raise ValueError(f'Fixed camera image does not match calibration: {self.paths[index]}')
        if self.maps is not None:
            image = cv2.remap(image, *self.maps, cv2.INTER_LINEAR)
        elif self.output_size != self.size:
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __iter__(self):
        for index in range(len(self.paths)):
            yield self.get(index)
