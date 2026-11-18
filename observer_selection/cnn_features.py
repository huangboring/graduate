"""Frozen MobileNet appearance descriptors for the offline selector replay."""
from dataclasses import replace
import pickle

import numpy as np
import torch

from .egohumans import _hash
from .fixed_camera import FixedCamera
from .preparation_cache import atomic_npz
from .raw_prepare import RectifiedFrames
from .selection_video import image_paths


class MobileNetAppearance:
    def __init__(self, device, weights=None):
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        self.device = device
        self.weights = weights or MobileNet_V3_Small_Weights.IMAGENET1K_V1
        net = mobilenet_v3_small(weights=self.weights)
        self.features = net.features.eval().to(device).requires_grad_(False)
        self.transform = self.weights.transforms()

    @torch.inference_mode()
    def __call__(self, crops):
        batch = torch.stack([self.transform(crop) for crop in crops]).to(self.device)
        return self.features(batch).mean(dim=(-2,-1)).cpu().numpy()


def _square_crop(image, bbox):
    from PIL import Image

    center = (bbox[:2]+bbox[2:])/2
    side = int(np.ceil(max(bbox[2:]-bbox[:2])))
    if not 0<side<=4*max(image.shape[:2]):
        raise ValueError('Invalid prepared CNN crop extent')
    x0,y0 = np.floor(center-side/2).astype(int)
    # Pillow pads out-of-image pixels, using the same box as the DINO path.
    return Image.fromarray(image).crop((x0,y0,x0+side,y0+side))


def add_cnn_features(prepared, sequence, raw, candidates, ids, frames, fixed, device):
    sources = (raw,*candidates)
    cache = prepared/'mobilenet_v3_small_imagenet1k_v1'
    cache.mkdir(exist_ok=True)
    signature = _hash(prepared/'replay.npz')
    paths = image_paths(sequence,sources,ids)
    model = None
    transforms = None
    descriptors = {}
    for source in sources:
        vault = cache/f'{source}.npz'
        if vault.exists():
            with np.load(vault,allow_pickle=False) as saved:
                values = saved['features']
                if str(saved['replay_sha256'].item())!=signature or not np.array_equal(saved['ids'],ids):
                    raise ValueError(f'Stale CNN feature cache: {vault}')
                if values.shape!=(len(ids),576) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid CNN feature cache: {vault}')
            descriptors[source] = values
            print(f'{source}: reuse CNN descriptors',flush=True)
            continue
        if model is None:
            model = MobileNetAppearance(device)
        images = [paths[source][i] for i in ids]
        if source in fixed:
            workplace = sequence/'colmap/workplace'
            if transforms is None:
                with (workplace/'colmap_from_aria_transforms.pkl').open('rb') as stream:
                    transforms = pickle.load(stream)
            reader = FixedCamera(workplace,source,transforms['aria01'],images)
        else:
            calibration = sequence/'ego'/source/'calib'
            reader = RectifiedFrames(images,[calibration/f'{i:05}.txt' for i in ids])
        values = np.zeros((len(ids),576),'f4')
        observations = [f.observation if source==raw else f.packets[source] for f in frames]
        visible = [i for i,o in enumerate(observations) if np.asarray(o.valid).any()]
        for start in range(0,len(visible),16):
            batch = visible[start:start+16]
            values[batch] = model([_square_crop(reader.get(i),observations[i].bbox) for i in batch])
            if start%128==0:
                print(f'{source}: CNN crops {start+len(batch)}/{len(visible)}',flush=True)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite CNN features: {source}')
        atomic_npz(vault,ids=np.array(ids),replay_sha256=np.array(signature),features=values)
        descriptors[source] = values
    enriched = []
    for i,frame in enumerate(frames):
        obs = replace(frame.observation,backend_context={
            **frame.observation.backend_context,'cnn_cls':descriptors[raw][i]})
        packets = {s:replace(p,payload={**p.payload,'cnn_cls':descriptors[s][i]})
                   for s,p in frame.packets.items()}
        enriched.append(replace(frame,observation=obs,packets=packets))
    return enriched
