"""Prepared EgoHumans to tracker-free local/aux replay.

Reference FK joints are read only for supervision, never for crops/cameras/rays.
Existing detector-feature receipts and exact alignment checks are retained.
"""
from pathlib import Path
import json
import numpy as np
from .egohumans import _json, _npz, _hash, _timestamps, _selector_observations, _rotations, _array
from .local_fusion import CONVENTION, OUTPUT_AXES, rigid


def export_local_egohumans(manifest_path, output_path):
    manifest_path = Path(manifest_path).resolve()
    m = _json(manifest_path)
    if m.get('schema_version') != 2 or m.get('source_dataset') != 'egohumans':
        raise ValueError('Local-aux export requires schema_version=2, source_dataset=egohumans')
    if any(k in m for k in ('target_tracker','floor','world_to_zup')):
        raise ValueError('Remove legacy target tracker/floor normalization fields for local-aux export')
    # This is an explicit input contract, not proof that an arbitrary pose file
    # was measured. Record the calibration/tracking origin and rectification.
    calibration = m.get('camera_calibration', {})
    if (calibration.get('provenance') not in ('measured','estimated','calibrated')
            or not calibration.get('description') or calibration.get('target_pose_used') is not False
            or calibration.get('rectified_pinhole') is not True):
        raise ValueError('Declare non-target-GT camera provenance, description and rectified pinhole images')
    target, requester = str(m['target_id']), str(m['requester_id'])
    sources = tuple(map(str,m['source_ids']))
    if len(sources) < 2 or len(set(sources)) != len(sources) or target == requester or {target,requester}&set(sources):
        raise ValueError('Need distinct target/requester and at least two unique remote candidates')
    if set(m['pairs']) != {requester,*sources}:
        raise ValueError('pairs must match the full fixed roster')
    def path(s):
        p=Path(s); return p if p.is_absolute() else manifest_path.parent/p
    pairs, dependencies = {}, [manifest_path]
    grid = world = gt = None
    for source in (requester,*sources):
        pair = path(m['pairs'][source])
        meta = _json(pair/'meta.json')
        for key, expected in {'source_dataset':'egohumans','source_scene':m['scene_id'],
                'source_subseq':m['sequence_id'],'wearer_aria':target,'observer_aria':source}.items():
            if meta.get(key) != expected: raise ValueError(f'Pair {key} disagrees with manifest')
        ts = _timestamps(pair,meta)
        if grid is None:
            grid, world = ts, meta.get('world_frame')
        if world is None or meta.get('world_frame') != world or not np.array_equal(ts,grid):
            raise ValueError('Pair timestamps/world frames must match exactly')
        data, deps = _selector_observations(pair,grid)
        cam = _npz(pair/'observer.npz'); T=len(grid)
        R = _rotations(cam['observer_head_R'],T,'camera device rotation')
        p = _array(cam['observer_head_pos'],(T,3),'camera device translation')
        extrinsic = rigid(cam['observer_T_dev_cam'],'device_from_camera')
        transform = np.tile(np.eye(4),(T,1,1))
        transform[:,:3,:3] = R @ extrinsic[:3,:3]
        transform[:,:3,3] = p + np.einsum('tij,j->ti',R,extrinsic[:3,3])
        K = _array(cam['observer_K'],(3,3),'intrinsics')
        if not np.allclose(K[2],[0,0,1]) or K[0,0]<=0 or K[1,1]<=0 or abs(np.linalg.det(K))<1e-12:
            raise ValueError('Invalid rectified pinhole intrinsics')
        data.update(camera_to_world=transform.astype('f4'),intrinsics=np.tile(K,(T,1,1)).astype('f4'))
        pairs[source]=data
        reference = _array(_npz(pair/'wearer_smplh_fk.npz')['joints_pos'],(T,22,3),'GT joints')
        if gt is None: gt=reference
        elif not np.array_equal(gt,reference): raise ValueError('Target references disagree')
        dependencies.extend([pair/'meta.json',pair/'timestamps.csv',pair/'observer.npz',pair/'wearer_smplh_fk.npz',*deps])
    local=pairs[requester]
    # Root-relative GT, rotated into A's camera at EACH target output timestamp.
    # Translation is deliberately not evaluated; no target global pose is fed.
    gt_camera = np.einsum('tji,tkj->tki',local['camera_to_world'][:,:3,:3],gt-gt[:,:1])
    out={'timestamps':grid.astype(np.float64)/1e9,'source_ids':np.array(sources),
         'target_id':np.array(target),'requester_id':np.array(requester),
         'scene_id':np.array(m['scene_id']),'sequence_id':np.array(m['sequence_id']),
         'joint_layout':np.array('smplh22'),'detector_joint_layout':np.array('vitpose13'),
         'coordinate_frame':np.array('root_relative'),'output_axes':np.array(OUTPUT_AXES),
         'units':np.array('metres'),'input_provenance':np.array('estimated'),
         'camera_provenance':np.array(calibration['provenance']),
         'coordinate_convention':np.array(CONVENTION),'crop_provenance':np.array('local_detector'),
         'temporal_provenance':np.array('causal'),'preparation_manifest_sha256':np.array(_hash(manifest_path)),
         'gt_joints':gt_camera.astype('f4'),'gt_valid':np.ones((T,22),bool)}
    for key in ('xy','bbox','confidence','dino_cls','bbox_valid','camera_to_world','intrinsics'):
        out['local_'+key]=local[key]
        out['packet_'+key]=np.stack([pairs[s][key] for s in sources],axis=1)
    out['local_valid']=np.repeat(local['bbox_valid'][:,None],13,axis=-1)
    out['packet_valid']=np.repeat(out['packet_bbox_valid'][...,None],13,axis=-1)
    output=Path(output_path).resolve(); sidecar=output.with_suffix('.provenance.json')
    if output.exists() or sidecar.exists(): raise FileExistsError('Choose a fresh export path')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f: np.savez_compressed(f,**out)
    receipt={'schema_version':2,'generator':'local_aux_export_v1','output_sha256':_hash(output),
        'input_files':[{'path':str(p.resolve()),'sha256':_hash(p)} for p in dict.fromkeys(dependencies)],
        'camera_calibration':calibration,'target_tracker_used':False,'dropped_initial_frames':0,
        'reference_usage':'GT first22 root-relative joints for loss only; not model inputs',
        'output_axes':OUTPUT_AXES,'shape_policy':'neutral decoder, not GT betas'}
    sidecar.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return {'output':str(output),'provenance':str(sidecar),'frames':T,'candidates':list(sources)}
