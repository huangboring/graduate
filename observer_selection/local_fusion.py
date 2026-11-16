"""Local-observation adaptation of EgoExoMoCap's fusion, not its released model.

No target tracker, GT crop, guessed depth, or target-centric ray normalization.
All rays in a window are expressed in the LAST requester's camera coordinates.
Only the last pose is decoded/supervised. SMPL-H/FK remains the pose decoder.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

from .backends import _array, _check_frame, _reject_gt_fields, _file_sha256, _state_sha256
from .types import AcquiredFrame, Reconstruction, WarmupError, NotReadyError, Packet

CONVENTION = 'local_aux_camera_rays_v1'
OUTPUT_AXES = 'last_requester_camera_opencv'


def rigid(value, name):
    a = _array(value, (4, 4), name).astype(np.float64)
    r = a[:3, :3]
    if (not np.allclose(a[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(r.T @ r, np.eye(3), atol=2e-4)
            or not np.isclose(np.linalg.det(r), 1, atol=2e-4)):
        raise ValueError(f'{name} must be a camera-to-world rigid transform')
    return a


def view_features(observation, payload, prefix, reference):
    """Per joint: ray origin(3), unit direction(3), valid, confidence, known.

    Camera translations are metres. No inferred target distance is used. Images
    and keypoints must already be rectified to the supplied pinhole intrinsics.
    """
    _reject_gt_fields(payload, 'view')
    if payload.get('coordinate_convention') != CONVENTION:
        raise ValueError('Local-aux input convention required; old target-head rays are incompatible')
    if payload.get('camera_provenance') not in ('measured', 'estimated', 'calibrated', 'synthetic_demo'):
        raise ValueError('Explicit non-target-GT camera_provenance required')
    transform = rigid(payload[prefix+'camera_to_world'], 'camera_to_world')
    K = _array(payload[prefix+'intrinsics'], (3, 3), 'intrinsics').astype(np.float64)
    if not np.allclose(K[2], [0, 0, 1]) or K[0, 0] <= 0 or K[1, 1] <= 0 or abs(np.linalg.det(K)) < 1e-12:
        raise ValueError('Rectified pinhole intrinsics required')
    xy = np.asarray(observation.xy)
    if xy.shape != (13, 2) or np.asarray(observation.valid).shape != (13,):
        raise ValueError('Expected 13 detector keypoints and validity flags')
    if not np.isin(observation.valid, [0, 1]).all():
        raise ValueError('Joint validity must be boolean')
    valid = np.asarray(observation.valid, bool)
    box_valid = float(_array(payload[prefix+'bbox_valid'], (), 'bbox_valid'))
    if box_valid not in (0, 1):
        raise ValueError('bbox_valid must be boolean')
    valid = valid & bool(box_valid)
    if not np.isfinite(xy[valid]).all():
        raise ValueError('Valid keypoints must be finite')
    xy = np.where(valid[:, None], xy, 0)
    rays = np.c_[xy, np.ones(13)] @ np.linalg.inv(K).T
    relative = np.linalg.inv(reference) @ transform
    direction = rays @ relative[:3, :3].T
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    origin = np.broadcast_to(relative[:3, 3], (13, 3))
    confidence, known = np.zeros(13), np.zeros(13)
    if observation.confidence is not None:
        confidence = _array(observation.confidence, (13,), 'confidence')
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError('Detector confidence must be in [0,1], not an occlusion label')
        known = valid.astype(float)
    features = np.c_[origin, direction, valid, confidence, known]
    features[~valid] = 0
    dino = _array(payload[prefix+'dino_cls'], (768,), 'dino_cls').copy()
    if not box_valid:
        dino[:] = 0
    return features.astype('f4'), dino, bool(box_valid)


def build_local_batch(observation, packets, history, window_size, device='cpu'):
    """Pure assembly: no provider calls, GT access or history mutation."""
    packets = (packets,) if isinstance(packets, Packet) else tuple(packets)
    if len({p.observer_id for p in packets}) != len(packets):
        raise ValueError('Duplicate sources')
    frames = (*history, AcquiredFrame(observation, packets)) if packets else (*history, SimpleNamespace(observation=observation, packets=()))
    previous = -float('inf')
    for f in frames:
        if f.observation.target_id != observation.target_id or not previous < f.observation.timestamp:
            raise ValueError('History must be chronological and belong to this target')
        previous = f.observation.timestamp
        for p in f.packets:
            _check_frame(f.observation, p)
    frames = frames[-window_size:]
    if len(frames) > 2:
        gaps = np.diff([f.observation.timestamp for f in frames])
        if not np.allclose(gaps, gaps[0], rtol=1e-5, atol=1e-6):
            raise ValueError('Local-aux window needs a uniform cadence')
    reference = rigid(observation.backend_context['local_camera_to_world'], 'reference camera')
    if observation.backend_context.get('output_axes') != OUTPUT_AXES:
        raise ValueError('Explicit last requester OpenCV camera output axes required')
    ids = sorted({p.observer_id for f in frames for p in f.packets})
    slots = {s: i for i, s in enumerate(ids)}
    T, N = len(frames), len(ids)
    local, ld, lv = [], [], []
    remote = np.zeros((T, N, 13, 9), 'f4')
    rd = np.zeros((T, N, 768), 'f4')
    rv = np.zeros((T, N), bool)
    for t, f in enumerate(frames):
        ctx = f.observation.backend_context
        if ctx.get('output_axes') != OUTPUT_AXES:
            raise ValueError('Output axes changed inside history')
        if any(k in ctx for k in ('sparse_input', 'local_exo_rays', 'tracker_provenance')):
            raise ValueError('Target-tracker / legacy ray inputs are forbidden in the local-aux backend')
        x, d, v = view_features(f.observation, ctx, 'local_', reference)
        local.append(x); ld.append(d); lv.append(v)
        for p in f.packets:
            i = slots[p.observer_id]
            remote[t, i], rd[t, i], rv[t, i] = view_features(p, p.payload, '', reference)
    if T < window_size:
        raise WarmupError(f'Need {window_size} real causal frames; have {T}')
    def tensor(x):
        return torch.as_tensor(np.asarray(x), device=device).unsqueeze(0)
    return {'local_geometry': tensor(local), 'local_dino': tensor(ld), 'local_valid': tensor(lv),
            'remote_geometry': tensor(remote), 'remote_dino': tensor(rd), 'remote_valid': tensor(rv)}, ids


@dataclass
class LocalFusionConfig:
    window_size: int = 16
    embed_dim: int = 128
    nhead: int = 4
    num_layer_spatial: int = 2
    num_layer_temporal: int = 2
    ff_dim: int = 512
    dropout: float = 0.1


def load_module(path, name):
    path = Path(path)
    if not path.is_file():
        raise NotReadyError(f'Missing upstream source: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalAuxFusionNet(nn.Module):
    """Replace EgoNet/ego embeddings; reuse official fusion layers and SMPL heads.

    Fresh training is required. No claim of released-checkpoint equivalence.
    Local slot is always a readout query, including when its observation is absent.
    """
    def __init__(self, repository, config: LocalFusionConfig):
        super().__init__()
        if config.window_size < 1 or config.embed_dim % config.nhead:
            raise ValueError('Invalid local fusion dimensions')
        self.config = config
        upstream = load_module(Path(repository)/'networks/egoexomocap.py', '_local_aux_upstream')
        args = asdict(config)
        self.fusion = upstream.FusionNet(**args, input_dim_ego=18, num_input=1, egonet_out_dim=0)
        del self.fusion.ego_token_embed
        del self.fusion.exo_token_embed
        dim = config.embed_dim
        self.local_embed = nn.Sequential(nn.Linear(13*9+1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.remote_embed = nn.Linear(13*9+1, dim)
        self.local_image_embed = nn.Linear(768, dim)
        self.missing_local = nn.Parameter(torch.zeros(dim))

    def forward(self, batch):
        local, remote = batch['local_geometry'], batch['remote_geometry']
        ld, rd = batch['local_dino'], batch['remote_dino']
        lv, rv = batch['local_valid'], batch['remote_valid']
        B, T = local.shape[:2]
        if T != self.config.window_size:
            raise ValueError('Window size differs from trained architecture')
        def gated(geom, dino, valid):
            geom = torch.where(valid[..., None, None], geom, torch.zeros_like(geom))
            dino = torch.where(valid[..., None], dino, torch.zeros_like(dino))
            scores = self.fusion.scorenet(dino)
            geometry = torch.cat([geom[..., :6]*scores.unsqueeze(-1), geom[..., 6:]], -1)
            return torch.cat([geometry.flatten(-2), valid.unsqueeze(-1).to(geom.dtype)], -1)
        local_token = self.local_embed(gated(local, ld, lv))
        local_token = local_token + self.local_image_embed(torch.where(lv[..., None], ld, torch.zeros_like(ld)))
        local_token = torch.where(lv[..., None], local_token, self.missing_local)
        remote_tokens = self.remote_embed(gated(remote, rd, rv))
        tokens = torch.cat([local_token.unsqueeze(2), remote_tokens], 2)
        mask = torch.cat([torch.zeros((B,T,1), device=lv.device, dtype=torch.bool), ~rv], 2)
        x = self.fusion.spatial_encoder(tokens.flatten(0,1), src_key_padding_mask=mask.flatten(0,1))
        x = x[:,0].reshape(B,T,-1) + self.fusion.temporal_pe[:T]
        # Every input is <= window end; only the final token is output/supervised.
        # Earlier tokens are NOT offered as causal per-frame predictions.
        x = self.fusion.temporal_encoder(x)[:,-1:]
        return {'root_orient': self.fusion.global_orientation_decoder(x),
                'pose_body': self.fusion.joint_rotation_decoder(x)}


def load_body_decoder(repository, body_model_path, device):
    """Same licensed SMPL-H output representation; no target tracker required."""
    if not body_model_path or not Path(body_model_path).is_file():
        raise NotReadyError('Local-aux still requires a licensed SMPL-H model for pose-to-joint decoding')
    try:
        from human_body_prior.body_model.body_model import BodyModel
    except ImportError as exc:
        raise NotReadyError('Install human_body_prior for the SMPL-H decoder') from exc
    transforms = load_module(Path(repository)/'utils/utils_transform.py', '_local_aux_transforms')
    bm = BodyModel(bm_fname=str(body_model_path), num_betas=16).to(device).eval()
    for p in bm.parameters(): p.requires_grad_(False)
    def decode(output):
        root = transforms.sixd2aa(output['root_orient'].reshape(-1,6)).float()
        body = transforms.sixd2aa(output['pose_body'].reshape(-1,6)).reshape(-1,63).float()
        joints = bm(root_orient=root, pose_body=body, betas=torch.zeros((len(root),16), device=root.device)).Jtr[:,:22]
        return joints - joints[:,:1]
    return decode


class LocalAuxBackend:
    def __init__(self, net, decoder, provenance=None):
        self.net, self.decoder = net, decoder
        self.window_size = net.config.window_size
        self.net.eval()
        for p in self.net.parameters(): p.requires_grad_(False)
        self.provenance = {'backend':'local_aux_v1', 'model_config':asdict(net.config),
            'model_state_sha256':_state_sha256(net), 'coordinate_frame':'root_relative',
            'output_axes':OUTPUT_AXES, 'units':'metres', 'joint_layout':'SMPL-H_first22',
            'shape_provenance':'neutral_assumption', 'target_tracker_required':False,
            'causal_mode':'trailing_window_last_frame_only', **(provenance or {})}

    @property
    def device(self): return next(self.net.parameters()).device

    @classmethod
    def from_checkpoint(cls, repository, checkpoint, body_model_path, device='cpu'):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('format') != 'local_aux_v1' or not state.get('training_data_manifest'):
            raise ValueError('Expected trained local_aux_v1 checkpoint, not original AvatarNet weights')
        expected = state['dependencies']
        files = {'network_source_sha256':Path(repository)/'networks/egoexomocap.py',
                 'transform_source_sha256':Path(repository)/'utils/utils_transform.py',
                 'body_model_sha256':Path(body_model_path), 'adapter_source_sha256':Path(__file__)}
        for key, path in files.items():
            if _file_sha256(path) != expected[key]:
                raise ValueError(f'Local-aux checkpoint dependency differs: {key}')
        net = LocalAuxFusionNet(repository, LocalFusionConfig(**state['config'])).to(device)
        net.load_state_dict(state['state_dict'], strict=True)
        decoder = load_body_decoder(repository, body_model_path, device)
        return cls(net, decoder, {'checkpoint_sha256':_file_sha256(Path(checkpoint)),
                   'training_data_manifest':state['training_data_manifest'], 'dependencies':expected})

    def reconstruct(self, observation, packets, history):
        packets = (packets,) if isinstance(packets, Packet) else tuple(packets)
        batch, ids = build_local_batch(observation, packets, history, self.window_size, self.device)
        self.net.eval()
        with torch.inference_mode():
            joints = self.decoder(self.net(batch))[0].detach().cpu().numpy()
        _array(joints, (22,3), 'local-aux decoded joints')
        return Reconstruction(joints.copy(), 'root_relative', {'backend':'local_aux_v1',
            'output_axes':OUTPUT_AXES, 'selected_observers':[p.observer_id for p in packets],
            'remote_slot_ids':ids, 'target_tracker_required':False})
