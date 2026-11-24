from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from .types import Observation, Packet, Geometry


@dataclass(frozen=True)
class ReplayFrame:
    observation: Observation
    candidates: tuple[str, ...]
    geometry: dict[str, Geometry]
    packets: dict[str, Packet]  # PRIVATE offline store: never pass to model scoring
    gt: np.ndarray
    gt_valid: np.ndarray
    coordinate_frame: str
    metadata: dict = field(default_factory=dict)


class ReplayProvider:
    def __init__(self, frame: ReplayFrame):
        self._frame = frame
        self.requests: list[str] = []
        self.batch_requests: list[tuple[str, ...]] = []

    def acquire_many(self, observer_ids, target_id, timestamp):
        observer_ids = tuple(observer_ids)
        if not observer_ids or len(set(observer_ids)) != len(observer_ids):
            raise ValueError('Batch request requires unique sources')
        if target_id != self._frame.observation.target_id or timestamp != self._frame.observation.timestamp:
            raise ValueError('Replay batch outside current target/frame')
        if not set(observer_ids) <= set(self._frame.packets):
            raise ValueError('Unknown source in requested batch')
        self.batch_requests.append(observer_ids)
        self.requests.extend(observer_ids)
        return tuple(self._frame.packets[s] for s in observer_ids)

    def acquire(self, observer_id: str, target_id: str, timestamp: float) -> Packet:
        if target_id != self._frame.observation.target_id or timestamp != self._frame.observation.timestamp:
            raise ValueError('Replay request outside the current target/frame')
        self.requests.append(observer_id)
        return self._frame.packets[observer_id]




def load_sequence(path: str | Path, allow_benchmark_inputs: bool = False) -> list[ReplayFrame]:
    """Safe numeric/string NPZ, no pickle. Required exact schema documented in README.

    One file = one target/session, full candidate observations are offline only.
    Canonical metres; reference and prediction use identical joint ordering.
    """
    with np.load(path, allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    required = ('timestamps', 'source_ids', 'target_id', 'local_xy', 'local_valid', 'local_bbox',
                'packet_xy', 'packet_valid', 'packet_bbox', 'gt_joints', 'gt_valid',
                'coordinate_frame', 'units', 'input_provenance')
    for key in required:
        if key not in a:
            raise ValueError(f'Missing NPZ field: {key}')
    if str(a['units'].item()) != 'metres':
        raise ValueError('3D units must be metres; convert explicitly during preparation')
    provenance = str(a['input_provenance'].item())
    if provenance not in ('measured', 'estimated', 'synthetic_demo', 'benchmark_gt_derived'):
        raise ValueError('Unknown input provenance')
    if provenance == 'benchmark_gt_derived' and not allow_benchmark_inputs:
        raise ValueError('GT-derived backend inputs require --allow-benchmark-inputs; not deployable evidence')
    ts = a['timestamps']
    if ts.ndim != 1 or not len(ts) or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError('timestamps must be a strictly increasing finite vector')
    ids = tuple(str(x) for x in a['source_ids'])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Unique nonempty source_ids required')
    T, N = len(ts), len(ids)
    xy = a['local_xy']
    if xy.ndim != 3 or xy.shape[0] != T or xy.shape[-1] != 2:
        raise ValueError('local_xy must be [T,J,2]')
    J = xy.shape[1]
    if a['gt_joints'].ndim != 3 or a['gt_joints'].shape[0] != T or a['gt_joints'].shape[-1] != 3:
        raise ValueError('gt_joints must be [T,K,3]')
    shapes = {'local_valid':(T,J),'local_bbox':(T,4),'packet_xy':(T,N,J,2),
              'packet_valid':(T,N,J),'packet_bbox':(T,N,4),'gt_valid':(T,a['gt_joints'].shape[1])}
    for key, shape in shapes.items():
        if a[key].shape != shape:
            raise ValueError(f'{key}: expected {shape}, got {a[key].shape}')
    if a['gt_joints'].ndim != 3 or a['gt_joints'].shape[0] != T or a['gt_joints'].shape[-1] != 3:
        raise ValueError('gt_joints must be [T,K,3]')
    for key in ('packet_timestamps','local_confidence','packet_confidence','geometry_values','geometry_valid',
                'geometry_timestamp','sparse_input','local_exo_rays','local_dino_cls','local_bbox_valid',
                'packet_exo_rays','packet_dino_cls','packet_bbox_valid','packet_joints3d','packet_joint_weights',
                'local_joints3d','prior_xy','prior_valid','prior_timestamp'):
        if key in a and (a[key].ndim == 0 or a[key].shape[0] != T):
            raise ValueError(f'{key} must have T entries')
    optional_shapes = {'packet_timestamps':(T,N),'local_confidence':(T,J),
        'packet_confidence':(T,N,J),'packet_exo_rays':(T,N,39),'packet_dino_cls':(T,N,768),
        'packet_bbox_valid':(T,N),'local_exo_rays':(T,39),'local_dino_cls':(T,768),
        'local_bbox_valid':(T,),'prior_xy':(T,J,2),'prior_valid':(T,J),'prior_timestamp':(T,)}
    optional_shapes.update({'local_camera_to_world':(T,4,4),'local_intrinsics':(T,3,3),
        'packet_camera_to_world':(T,N,4,4),'packet_intrinsics':(T,N,3,3)})
    for key, shape in optional_shapes.items():
        if key in a and a[key].shape != shape:
            raise ValueError(f'{key}: expected {shape}, got {a[key].shape}')
    if any(k in a for k in ('geometry_values','geometry_valid','geometry_timestamp')):
        for k,shape in [('geometry_values',(T,N,4)),('geometry_valid',(T,N,4)),('geometry_timestamp',(T,N))]:
            if k not in a or a[k].shape != shape:
                raise ValueError(f'{k} must have shape {shape}')
        if 'geometry_provenance' not in a:
            raise ValueError('geometry_provenance required')
    target = str(a['target_id'].item())
    frame_name = str(a['coordinate_frame'].item())
    frames = []
    for t in range(T):
        context = {k:a[k][t] for k in ('sparse_input','local_exo_rays','local_dino_cls','local_bbox_valid','local_joints3d','local_camera_to_world','local_intrinsics') if k in a}
        context['input_provenance'] = provenance
        is_local_aux = str(a.get('coordinate_convention', np.array('')).item()) == 'local_aux_camera_rays_v1'
        if not is_local_aux:
            context['tracker_provenance'] = str(a['tracker_provenance'].item()) if 'tracker_provenance' in a else ('benchmark_gt' if provenance == 'benchmark_gt_derived' else provenance)
        elif any(k in a for k in ('sparse_input','tracker_provenance','local_exo_rays','packet_exo_rays')):
            raise ValueError('Local-aux replay cannot contain legacy tracker/ray inputs')
        if is_local_aux:
            from .calibration import OUTPUT_AXES
            if str(a.get('output_axes',np.array('')).item()) != OUTPUT_AXES:
                raise ValueError('Local-aux replay requires explicit requester-camera GT axes')
        for key in ('coordinate_convention','floor_provenance','floor_height_m','camera_provenance','output_axes'):
            if key in a: context[key]=a[key].item()
        obs = Observation(target,float(ts[t]),xy[t],a['local_valid'][t],a['local_bbox'][t],
                          a['local_confidence'][t] if 'local_confidence' in a else None,
                          a['prior_xy'][t] if 'prior_xy' in a else None,
                          a['prior_valid'][t] if 'prior_valid' in a else None,
                          float(a['prior_timestamp'][t]) if 'prior_timestamp' in a else None,context)
        packets, geometry = {}, {}
        for i, source in enumerate(ids):
            payload = {key:a['packet_'+key][t,i] for key in ('exo_rays','dino_cls','bbox_valid','joints3d','joint_weights','camera_to_world','intrinsics') if 'packet_'+key in a}
            if is_local_aux:
                payload.update(coordinate_convention=context['coordinate_convention'],camera_provenance=context.get('camera_provenance'))
            pt = float(a['packet_timestamps'][t,i]) if 'packet_timestamps' in a else float(ts[t])
            packets[source] = Packet(source,target,pt,a['packet_xy'][t,i],a['packet_valid'][t,i],a['packet_bbox'][t,i],
                                     a['packet_confidence'][t,i] if 'packet_confidence' in a else None,payload)
            if 'geometry_values' in a:
                geometry[source] = Geometry(a['geometry_values'][t,i],a['geometry_valid'][t,i],
                    float(a['geometry_timestamp'][t,i]),str(a['geometry_provenance'].item()))
        metadata={k:str(a[k].item()) for k in ('scene_id','sequence_id','requester_id','joint_layout','crop_provenance','temporal_provenance') if k in a}
        metadata['input_provenance']=provenance
        frames.append(ReplayFrame(obs,ids,geometry,packets,a['gt_joints'][t],a['gt_valid'][t],frame_name,metadata))
    return frames
