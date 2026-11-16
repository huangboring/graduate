"""Reconstruction backends with an explicit acquisition boundary.

EgoExoMoCap is an adapter, not a newly trained reconstruction model. All supplied
features must already be in the upstream coordinate convention and metres.
The weighted-joint backend is only a deterministic simulation/test backend.
"""
from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .types import AcquiredFrame, NotReadyError, WarmupError, Observation, Packet, Reconstruction, packet_tuple


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float32)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite with shape {shape}; got {a.shape}")
    return a


def _check_frame(observation: Observation, packet: Packet) -> None:
    if observation.target_id != packet.target_id:
        raise ValueError("Packet and observation must refer to the same target")
    if (not np.isfinite(observation.timestamp) or not np.isfinite(packet.timestamp)
            or abs(packet.timestamp - observation.timestamp) > 1e-6):
        raise ValueError("V1 requires synchronous packets at exactly the requested timestamp")


def _reject_gt_fields(mapping: dict[str, Any], name: str) -> None:
    for key, value in mapping.items():
        k = str(key).lower()
        if k.startswith('gt_') or k.endswith('_gt') or 'ground_truth' in k:
            raise ValueError(f"{name}.{key}: GT belongs to the offline evaluator, not runtime inputs")
        if isinstance(value, dict):
            _reject_gt_fields(value, f"{name}.{key}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(net: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f'{name}:{tuple(tensor.shape)}:{tensor.dtype}'.encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class EgoExoMoCapBackend:
    """Trailing-window last-frame inference using a loaded upstream model.

    `model` exposes `.net`, `.fk`, `.bm`; `.get_bare_model` is optional. Unlike
    upstream test(), this adapter never calls feed_data(), reads GT, or consumes
    a future observation. The fixed-window MLP-Mixer requires W real acquired
    frames: startup raises WarmupError, with no fabricated future/history.

    Required observation.backend_context: sparse_input [18 or 54] and
    tracker_provenance ('measured', 'estimated', 'calibrated'). Optional local
    observation keys: local_exo_rays [39], local_dino_cls [768], local_bbox_valid.
    Required packet.payload: exo_rays [39], dino_cls [768], bbox_valid scalar.
    Target tracker and ray canonicalization must use the same upstream frame.
    Floor height must be calibrated/estimated externally, never GT feet heights.

    Slot zero is local A. Remaining slots are the union of actually acquired
    remote IDs in the causal window; absent/unselected frames are masked.
    `known_shape=None` explicitly uses neutral zero betas, which may differ from
    published results. For supplied shape, declare its provenance explicitly.
    """

    def __init__(self, model: Any, window_size: int,
                 known_shape: np.ndarray | None = None, *,
                 shape_provenance: str | None = None,
                 allow_benchmark_provenance: bool = False):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.model = model
        self.net = model.get_bare_model(model.net) if hasattr(model, 'get_bare_model') else model.net
        self.window_size = int(window_size)
        for component in (getattr(self.net, 'egonet', None), getattr(self.net, 'fusion', None)):
            if component is not None and getattr(component, 'window_size', window_size) != window_size:
                raise ValueError("Adapter window_size must match the loaded checkpoint architecture")
        if not hasattr(model, 'fk') or not hasattr(model, 'bm'):
            raise ValueError("Loaded model must expose fk and bm")
        self.allow_benchmark_provenance = bool(allow_benchmark_provenance)
        if known_shape is None:
            self.known_shape = np.zeros(16, dtype=np.float32)
            self.shape_provenance = 'neutral_assumption'
        else:
            self.known_shape = _array(known_shape, (16,), 'known_shape').copy()
            self._validate_provenance(shape_provenance, 'shape_provenance')
            self.shape_provenance = str(shape_provenance)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        if hasattr(model.bm, 'eval'):
            model.bm.eval()
        if hasattr(model.bm, 'parameters'):
            for p in model.bm.parameters():
                p.requires_grad_(False)
        self.provenance = {
            'backend': 'egoexomocap', 'adapter_version': 3,
            'window_size': self.window_size, 'coordinate_frame': 'root_relative',
            'joint_layout': 'SMPL-H_first22', 'units': 'metres',
            'shape_provenance': self.shape_provenance,
            'shape_sha256': hashlib.sha256(self.known_shape.tobytes()).hexdigest(),
            'allow_benchmark_provenance': self.allow_benchmark_provenance,
            'model_state_sha256': _state_sha256(self.net),
            'causal_mode': 'trailing_window_last_frame_only',
        }

    def _validate_provenance(self, value: str | None, name: str) -> None:
        allowed = {'measured', 'estimated', 'calibrated'}
        if self.allow_benchmark_provenance:
            allowed.add('benchmark_gt')
        if value not in allowed:
            raise ValueError(f"{name} must explicitly be one of {sorted(allowed)}; got {value!r}")

    @property
    def device(self) -> torch.device:
        first = next(self.net.parameters(), None)
        return first.device if first is not None else torch.device('cpu')

    @classmethod
    def from_repository(cls, repository: str | Path, config: str | Path,
                        checkpoint: str | Path, device: str = 'cpu',
                        known_shape: np.ndarray | None = None, *,
                        body_model_path: str | Path | None = None,
                        shape_provenance: str | None = None,
                        allow_benchmark_provenance: bool = False) -> 'EgoExoMoCapBackend':
        """Load exact upstream network weights without its GT-dependent wrapper.

        Config/checkpoint paths may be absolute or relative to repository.
        Requires PyYAML and upstream human_body_prior/SMPL-H dependencies.
        Strict state_dict loading rejects partial/randomly initialized models.
        """
        repo = Path(repository).resolve()
        def resolve(p: str | Path) -> Path:
            q = Path(p)
            return q if q.is_absolute() else repo / q
        cfg_path, ckpt_path = resolve(config), resolve(checkpoint)
        network_path = repo / 'networks' / 'egoexomocap.py'
        transform_path = repo / 'utils' / 'utils_transform.py'
        for p in (network_path, transform_path, cfg_path, ckpt_path):
            if not p.is_file():
                raise NotReadyError(f"Missing EgoExoMoCap asset: {p}. Obtain repository, matching config and released checkpoint first.")
        try:
            import yaml
            from human_body_prior.body_model.body_model import BodyModel
        except ImportError as exc:
            raise NotReadyError("Install PyYAML and EgoExoMoCap's human_body_prior dependencies in this Python environment") from exc
        with cfg_path.open(encoding='utf-8') as f:
            opt = yaml.safe_load(f)
        if opt.get('net', {}).get('network_name') != 'egoexomocap':
            raise ValueError("Configuration must describe the EgoExoMoCap AvatarNet")
        args = opt['net']['network_arguments']
        ws = int(args['window_size'])
        if args['egonet_arguments']['window_size'] != ws:
            raise ValueError("EgoNet and fusion window sizes disagree")
        if body_model_path is None:
            body = opt['body_model']
            bm_path = resolve(body['smpl_path']) / 'body_models' / 'smplh' / body.get('default_gender', 'neutral') / 'model.npz'
        else:
            bm_path = resolve(body_model_path)
        if not bm_path.is_file():
            raise NotReadyError(f"Missing licensed SMPL-H body model: {bm_path}. Download it separately; it is not bundled.")
        def load_module(name: str, p: Path):
            spec = importlib.util.spec_from_file_location(name, p)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot import {p}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        network = load_module('_observer_selection_egoexomocap', network_path)
        transforms = load_module('_observer_selection_ego_transforms', transform_path)
        net = network.AvatarNet(**args)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'params' in state:
            state = state['params']
        if not isinstance(state, dict):
            raise ValueError("Checkpoint must contain a state_dict or {'params': state_dict}")
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        net.load_state_dict(state, strict=True)
        net.to(torch.device(device)).eval()
        bm = BodyModel(bm_fname=str(bm_path), num_betas=16).to(device).eval()
        for p in bm.parameters():
            p.requires_grad_(False)
        def fk(body_model, global_orientation, joint_rotation, trans=None, betas=None):
            leading = global_orientation.shape[:-1]
            root = transforms.sixd2aa(global_orientation.reshape(-1, 6)).reshape(-1, 3).float()
            body_pose = transforms.sixd2aa(joint_rotation.reshape(-1, 6)).reshape(-1, 63).float()
            kwargs = {'root_orient': root, 'pose_body': body_pose}
            if betas is not None:
                kwargs['betas'] = betas.reshape(-1, 16).float()
            if trans is not None:
                kwargs['trans'] = trans.reshape(-1, 3)
            return body_model(**kwargs).Jtr[..., :22, :].reshape(*leading, 22, 3)
        model = SimpleNamespace(net=net, bm=bm, fk=fk)
        backend = cls(model, ws, known_shape,
                   shape_provenance=shape_provenance,
                   allow_benchmark_provenance=allow_benchmark_provenance)
        backend.provenance.update({
            'checkpoint_sha256': _file_sha256(ckpt_path),
            'config_sha256': _file_sha256(cfg_path),
            'network_source_sha256': _file_sha256(network_path),
            'transform_source_sha256': _file_sha256(transform_path),
            'body_model_sha256': _file_sha256(bm_path),
        })
        return backend

    def reconstruct(self, observation: Observation, packets,
                    history: tuple[AcquiredFrame, ...]) -> Reconstruction:
        packets = packet_tuple(packets)
        for packet in packets:
            _check_frame(observation, packet)
        previous = -float('inf')
        for frame in history:
            for packet in frame.packets:
                _check_frame(frame.observation, packet)
            if frame.observation.target_id != observation.target_id:
                raise ValueError("Backend history must contain only this target")
            t = frame.observation.timestamp
            if not previous < t < observation.timestamp:
                raise ValueError("Backend history must be strictly ordered and strictly before current frame")
            previous = t
        frames = (*history, AcquiredFrame(observation, packets))
        frames = frames[-self.window_size:]
        source_ids = sorted({p.observer_id for f in frames for p in f.packets})
        source_slot = {source:i+1 for i,source in enumerate(source_ids)}
        slots = 1 + len(source_ids)
        if len(frames) > 2:
            gaps = np.diff([f.observation.timestamp for f in frames])
            if not np.allclose(gaps, gaps[0], rtol=1e-5, atol=1e-6):
                raise ValueError('EgoExoMoCap fixed-window inputs must have a uniform cadence; gaps need explicit preprocessing')
        sparse_rows, rays_rows, dino_rows, valid_rows, mask_rows = [], [], [], [], []
        expected_dim = None
        tracker_provenances = set()
        for frame in frames:
            ctx = frame.observation.backend_context
            _reject_gt_fields(ctx, 'backend_context')
            provenance = ctx.get('tracker_provenance')
            self._validate_provenance(provenance, 'tracker_provenance')
            tracker_provenances.add(provenance)
            if ctx.get('coordinate_convention') != 'egoexomocap_headlocal_rays_zup':
                raise ValueError('Explicit head-local ray / z-up tracker coordinate convention required')
            if ctx.get('floor_provenance') not in ('measured','estimated','calibrated'):
                raise ValueError('Floor calibration/estimate required; never infer it from GT feet')
            if not np.isfinite(ctx.get('floor_height_m',float('nan'))):
                raise ValueError('Explicit finite floor_height_m required')
            if 'sparse_input' not in ctx:
                raise ValueError("backend_context requires target B's sparse_input; requester A's tracker cannot substitute")
            sparse = np.asarray(ctx['sparse_input'], dtype=np.float32)
            if sparse.shape not in ((18,), (54,)) or not np.isfinite(sparse).all():
                raise ValueError("sparse_input must be a finite 18D or 54D target tracker vector")
            expected_dim = expected_dim or len(sparse)
            if len(sparse) != expected_dim:
                raise ValueError("Tracker dimensionality changed inside the window")
            num_input = getattr(getattr(self.net, 'egonet', None), 'num_input', None)
            if num_input is not None and len(sparse) != 18 * num_input:
                raise ValueError("Tracker dimensions do not match checkpoint num_input")
            sparse_rows.append(sparse.copy())
            rays = np.zeros((slots, 39), np.float32)
            dino = np.zeros((slots, 768), np.float32)
            valid = np.zeros(slots, np.float32)
            mask = np.ones(slots, bool)
            local_keys = ('local_exo_rays', 'local_dino_cls', 'local_bbox_valid')
            local_count = sum(k in ctx for k in local_keys)
            if local_count not in (0, 3):
                raise ValueError("Local exo rays, DINO and bbox validity must be supplied together")
            if local_count:
                rays[0] = _array(ctx['local_exo_rays'], (39,), 'local_exo_rays')
                dino[0] = _array(ctx['local_dino_cls'], (768,), 'local_dino_cls')
                valid[0] = float(_array(ctx['local_bbox_valid'], (), 'local_bbox_valid'))
            for packet in frame.packets:
                payload = packet.payload
                _reject_gt_fields(payload, 'packet.payload')
                for name in ('exo_rays', 'dino_cls', 'bbox_valid'):
                    if name not in payload:
                        raise ValueError(f"Acquired packet missing EgoExoMoCap field: {name}")
                slot = source_slot[packet.observer_id]
                rays[slot] = _array(payload['exo_rays'], (39,), 'exo_rays')
                dino[slot] = _array(payload['dino_cls'], (768,), 'dino_cls')
                valid[slot] = float(_array(payload['bbox_valid'], (), 'bbox_valid'))
            if not np.isin(valid, [0, 1]).all():
                raise ValueError("bbox_valid must be boolean or 0/1")
            mask[:] = valid == 0
            rays[mask], dino[mask] = 0, 0
            rays_rows.append(rays); dino_rows.append(dino)
            valid_rows.append(valid); mask_rows.append(mask)
        # Validate even the first frame before declaring warmup. Otherwise malformed
        # packets or missing tracker metadata would be silently stored for W frames.
        if len(frames) < self.window_size:
            raise WarmupError(f"Need {self.window_size} acquired causal frames; only {len(frames)} available. Continue acquisition during warmup.")
        def tensor(rows, dtype=torch.float32):
            return torch.tensor(np.stack(rows), dtype=dtype, device=self.device).unsqueeze(0)
        betas = torch.tensor(self.known_shape.copy(), device=self.device).reshape(1, 1, 16).repeat(1, self.window_size, 1)
        batch = {'sparse_input': tensor(sparse_rows), 'exo_rays': tensor(rays_rows),
                 'dino_cls': tensor(dino_rows), 'bbox_valid': tensor(valid_rows),
                 'exo_pad_mask': tensor(mask_rows, torch.bool),
                 'fk': self.model.fk, 'body_model': self.model.bm, 'betas': betas}
        self.net.eval()
        with torch.inference_mode():
            out = self.net(batch)
            root, body = out['root_orient'][:, -1:], out['pose_body'][:, -1:]
            joints = self.model.fk(body_model=self.model.bm, global_orientation=root,
                                   joint_rotation=body, betas=betas[:, -1:])
            joints = joints.reshape(-1, 3)
            if joints.shape != (22, 3) or not torch.isfinite(joints).all():
                raise ValueError("EgoExoMoCap FK must return 22 finite joints")
            joints = (joints - joints[0:1]).cpu().numpy().copy()
        return Reconstruction(joints, 'root_relative', {
            'backend': 'egoexomocap', 'window_size': self.window_size,
            'causal_mode': 'trailing_window_last_frame_only',
            'shape_provenance': self.shape_provenance,
            'tracker_provenance': sorted(tracker_provenances),
            'selected_observers': [p.observer_id for p in packets],
            'remote_slot_ids': source_ids,
            'timestamps': [f.observation.timestamp for f in frames],
            'synthetic': False,
            'note': 'Causal adaptation differs from upstream offline evaluation; real checkpoint accuracy requires validation.',
        })


class WeightedJointBackend:
    """Deterministic simulation only; not an EgoExoMoCap replacement.

    Packet: joints3d [J,3] in metres and optional joint_weights [J]. Local context:
    optional local_joints3d [J,3], local_joint_weights [J]. Both observations must
    share root-relative coordinates; output is anchored at joint zero.
    """
    provenance = {'backend': 'weighted_joint_simulation', 'adapter_version': 3,
                  'coordinate_frame': 'root_relative', 'units': 'metres'}
    def reconstruct(self, observation: Observation, packets,
                    history: tuple[AcquiredFrame, ...]) -> Reconstruction:
        packets = packet_tuple(packets)
        for packet in packets:
            _check_frame(observation, packet)
            if 'joints3d' not in packet.payload:
                raise ValueError("Simulation packet requires joints3d")
        remote = np.asarray(packets[0].payload['joints3d'], np.float32)
        if remote.ndim != 2 or remote.shape[1] != 3 or not len(remote):
            raise ValueError("joints3d must have shape [J,3]")
        _array(remote, remote.shape, 'joints3d')
        n = len(remote)
        numerator = np.zeros_like(remote)
        denominator = np.zeros(n, np.float32)
        for packet in packets:
            points = _array(packet.payload['joints3d'], remote.shape, 'joints3d')
            rw = _array(packet.payload.get('joint_weights', np.ones(n)), (n,), 'joint_weights')
            if (rw < 0).any():
                raise ValueError("Joint weights cannot be negative")
            numerator += points * rw[:,None]
            denominator += rw
        ctx = observation.backend_context
        if 'local_joints3d' in ctx:
            local = _array(ctx['local_joints3d'], remote.shape, 'local_joints3d')
            lw = _array(ctx.get('local_joint_weights', np.ones(n)), (n,), 'local_joint_weights')
            if (lw < 0).any():
                raise ValueError("Local joint weights cannot be negative")
            numerator = numerator + local * lw[:, None]
            denominator += lw
        if (denominator <= 0).any():
            raise ValueError("Every simulated joint needs positive total observation weight")
        joints = numerator / denominator[:, None]
        joints = joints - joints[0:1]
        return Reconstruction(joints.astype(np.float32), 'root_relative', {
            'backend': 'weighted_joint_simulation', 'synthetic': True,
            'selected_observers': [p.observer_id for p in packets],
            'note': 'Synthetic integration backend only; not evidence of real reconstruction quality.',
        })
