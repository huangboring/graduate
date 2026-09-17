"""Panoptic HD video reader and calibration adapter, without GT dependencies."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import bisect
from utils.transforms import get_affine_transform, get_scale


def image_index(path):
    """Both legacy 00000000.jpg and official 00_00_00000000.jpg retain HD frame IDs."""
    suffix = Path(path).stem.rsplit('_', 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f'Cannot parse image frame ID: {path}')
    return int(suffix)


def image_files(directory):
    result = {}
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() in ('.jpg','.jpeg','.png'):
            frame = image_index(path)
            if frame in result:
                raise ValueError(f'Duplicate image frame {frame} in {directory}')
            result[frame] = path
    return result


def annotation_files(sequence):
    """Accept an extra nesting level introduced when unpacking older downloads."""
    result = {}
    for path in sorted((Path(sequence)/'hdPose3d_stage1_coco19').rglob('body3DScene_*.json')):
        frame = image_index(path)
        if frame in result:
            raise ValueError(f'Duplicate 3D annotation for frame {frame}: {path}')
        result[frame] = path
    return result


def inspect_sequence(sequence, camera_ids=None, input_format='auto'):
    sequence = Path(sequence).resolve()
    calibration = sequence / f'calibration_{sequence.name}.json'
    with calibration.open(encoding='utf-8') as stream:
        cameras = {c['name']: c for c in json.load(stream)['cameras']
                   if c['panel'] == 0}
    videos = {}
    for directory in ('hdVideos', 'hdvideos'):
        for path in (sequence / directory).glob('hd_00_*.mp4'):
            videos[path.stem.removeprefix('hd_')] = path
    legacy, official = {}, {}
    for name in cameras:
        for directory in ('hdVideos','hdvideos'):
            path = sequence/directory/f'hd_{name}'
            if path.is_dir() and image_files(path):
                legacy[name] = path
        path = sequence/'hdImgs'/name
        if path.is_dir() and image_files(path):
            official[name] = path
    choices = {'videos':videos, 'legacy_images':legacy, 'hdimgs':official}
    if input_format == 'auto':
        usable = [(name,paths) for name,paths in choices.items()
                  if paths and (camera_ids is None or all(x in paths for x in camera_ids))]
        if not usable:
            raise ValueError('No complete source for requested cameras: expected MP4, hdVideos/hd_00_XX/*.jpg, or hdImgs/00_XX/*')
        # Most available cameras; ties prefer MP4. Never mix formats within a bundle.
        _, selected_paths = max(usable, key=lambda item:len(item[1]))
    elif input_format in choices:
        selected_paths = choices[input_format]
    else:
        raise ValueError(f'Unknown input format: {input_format}')
    ids = sorted(selected_paths) if camera_ids is None else camera_ids
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError('Need at least two unique HD camera IDs, e.g. 00_00 00_01 00_02')
    for name in ids:
        if name not in selected_paths or name not in cameras:
            raise ValueError(f'Missing image/video source or calibration for {name}')
        c = cameras[name]
        if np.asarray(c['K']).shape != (3, 3) or np.asarray(c['R']).shape != (3, 3):
            raise ValueError(f'Invalid calibration matrix for {name}')
    return ids, [selected_paths[x] for x in ids], [cameras[x] for x in ids]


class PanopticImages:
    def __init__(self, paths, start=0, fps=29.97002997):
        if start < 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid start frame or source FPS')
        self.files = [image_files(p) for p in paths]
        self.indices = sorted(self.files[0])
        if not self.indices or any(set(x) != set(self.indices) for x in self.files[1:]):
            raise ValueError('Image frame IDs must match across cameras; missing views are not filled with black images')
        self.position = bisect.bisect_left(self.indices,start)
        self.frame = start
        self.fps = fps

    def read(self):
        if self.position >= len(self.indices):
            return None
        index = self.indices[self.position]
        frames = [cv2.imread(str(files[index])) for files in self.files]
        if any(frame is None for frame in frames):
            raise ValueError(f'Unreadable image at frame {index}')
        self.position += 1
        self.frame = index+1
        return index,frames

    def skip(self,count):
        self.frame += count
        self.position = bisect.bisect_left(self.indices,self.frame)
        return self.position < len(self.indices)

    def close(self):
        pass


def open_source(paths,start=0,image_fps=29.97002997):
    if all(Path(p).is_dir() for p in paths):
        return PanopticImages(paths,start,image_fps)
    if any(Path(p).is_dir() for p in paths):
        raise ValueError('Cannot mix video and extracted image sources')
    return PanopticVideos(paths,start)


class PanopticVideos:
    def __init__(self, paths, start=0):
        if start < 0:
            raise ValueError('start must be nonnegative')
        self.captures = []
        self.frame = start
        try:
            for path in paths:
                capture = cv2.VideoCapture(str(path))
                self.captures.append(capture)
                if not capture.isOpened():
                    raise ValueError(f'Cannot open video: {path}')
                if start and not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                    raise ValueError(f'Cannot seek video: {path}')
            rates = [c.get(cv2.CAP_PROP_FPS) for c in self.captures]
            if not rates or min(rates) <= 0 or max(rates) - min(rates) > 0.01:
                raise ValueError(f'Invalid or mismatched HD frame rates: {rates}')
            self.fps = rates[0]
        except Exception:
            self.close()
            raise

    def read(self):
        frames = []
        for capture in self.captures:
            ok, frame = capture.read()
            if not ok:
                return None  # Stop at the shortest view; never reuse stale frames.
            frames.append(frame)
        index = self.frame
        self.frame += 1
        return index, frames

    def skip(self, count):
        """Advance without decoding image tensors; maintain the shared frame index."""
        for _ in range(count):
            statuses = [capture.grab() for capture in self.captures]
            if not all(statuses):
                return False
            self.frame += 1
        return True

    def close(self):
        for capture in self.captures:
            capture.release()


def prepare_frame(frame, calibration, image_size, heatmap_size, device, color_rgb=True):
    height, width = frame.shape[:2]
    if [width, height] != list(calibration['resolution']):
        raise ValueError('Video resolution differs from calibration; recalibrate/scale K explicitly')
    image_size, heatmap_size = np.asarray(image_size), np.asarray(heatmap_size)
    center = np.array([width / 2., height / 2.])
    scale = get_scale((width, height), image_size)
    affine = np.eye(3)
    affine[:2] = get_affine_transform(center, scale, 0, image_size)
    inverse = np.eye(3)
    inverse[:2] = get_affine_transform(center, scale, 0, image_size, inv=1)
    image = cv2.warpAffine(frame, affine[:2], tuple(map(int, image_size)))
    if color_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.
    tensor = (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
    # Same world-axis conversion and cm -> mm conversion as dataset/panoptic.py.
    axis = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    rotation = np.asarray(calibration['R']) @ axis
    translation = np.asarray(calibration['t']).reshape(3, 1) * 10.
    intrinsic = np.asarray(calibration['K'])
    distortion = np.asarray(calibration['distCoef'])
    camera = dict(R=rotation, T=-rotation.T @ translation, standard_T=translation,
                  fx=intrinsic[0, 0], fy=intrinsic[1, 1], cx=intrinsic[0, 2], cy=intrinsic[1, 2],
                  k=distortion[[0, 1, 4]].reshape(3, 1), p=distortion[[2, 3]].reshape(2, 1))
    hm_scale = heatmap_size / image_size
    # Preserve the upstream augmentation convention for checkpoint compatibility.
    aug = np.diag([hm_scale[1], hm_scale[0], 1.]) @ affine
    meta = dict(center=center, scale=scale, rotation=0., camera=camera,
                camera_Intri=intrinsic, camera_R=rotation, camera_T=camera['T'],
                camera_standard_T=translation, camera_focal=np.array([camera['fx'], camera['fy'], 1.]),
                affine_trans=affine, inv_affine_trans=inverse, aug_trans=aug)
    def convert(value):
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
    return tensor.unsqueeze(0).to(device), convert(meta)


def camera_geometry(metadata):
    """Static control-plane geometry, recomputed if calibrated poses change."""
    return torch.stack([torch.cat((m['camera']['T'][0, :, 0] / 10000.,
                                   m['camera']['R'][0, 2, :])) for m in metadata])
