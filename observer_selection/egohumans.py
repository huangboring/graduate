"""Strict EgoHumans prepared-pair conversion, without the GT-dependent loader.

The upstream *_egonet features can depend on future frames and GT-normalized
tracking. They are deliberately not accepted as causal selector observations.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ValueError(f'Missing prepared data: {path}')
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _array(a, shape, name):
    a = np.asarray(a)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite {shape}; got {a.shape}')
    return a


def _rotations(a, T, name):
    a = _array(a,(T,3,3),name).astype(np.float64)
    if not np.allclose(a.swapaxes(-1,-2) @ a,np.eye(3),atol=2e-4) or not np.allclose(np.linalg.det(a),1,atol=2e-4):
        raise ValueError(f'{name} must be proper rotations')
    return a


def _timestamps(pair_dir: Path, meta: dict) -> np.ndarray:
    with (pair_dir/'timestamps.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    T=int(meta['n_frames'])
    if len(rows)!=T or [int(r['frame_idx']) for r in rows]!=list(range(T)):
        raise ValueError('timestamp frame indices/count do not match meta.json')
    ts=np.asarray([int(r['t_ns']) for r in rows],dtype=np.int64)
    expected=int(meta['t_start_ns'])+np.arange(T,dtype=np.int64)*int(meta['dt_ns'])
    if T<2 or not np.array_equal(ts,expected) or np.any(np.diff(ts)<=0):
        raise ValueError('timestamps must exactly match the positive regular metadata grid')
    return ts


def build_sparse(tracker: dict, floor_height_m: float, num_input: int = 1) -> np.ndarray:
    """Match upstream column-6D rotations and per-frame deltas; remove first row.

    Floor is an explicit calibrated/estimated scalar, never GT feet statistics.
    """
    if num_input not in (1,3):
        raise ValueError('num_input must be 1 or 3')
    T=len(tracker['wearer_head_pos'])
    names=('head',) if num_input==1 else ('head','lwrist','rwrist')
    pos=[];rotation=[];delta=[]
    def six(R): return np.concatenate((R[..., :,0],R[..., :,1]),axis=-1)
    for name in names:
        p=_array(tracker[f'wearer_{name}_pos'],(T,3),name+'_pos').astype(np.float64).copy()
        R=_rotations(tracker[f'wearer_{name}_R'],T,name+'_R')
        p[:,2]-=floor_height_m
        pos.append(p);rotation.append(six(R)[1:]);delta.append(six(R[:-1].swapaxes(-1,-2)@R[1:]))
    allpos=np.concatenate(pos,axis=-1)
    return np.concatenate([*rotation,*delta,allpos[1:],np.diff(allpos,axis=0)],axis=-1).astype('f4')


def compute_rays(xy, valid, camera, tracker):
    """Upstream head-local ray convention, independent of GT body joints."""
    T=len(xy)
    K=_array(camera['observer_K'],(3,3),'observer_K').astype(np.float64)
    if not np.allclose(K[2],[0,0,1]) or abs(np.linalg.det(K))<1e-12:
        raise ValueError('Expected invertible pinhole intrinsics with last row [0,0,1]')
    transform=_array(camera['observer_T_dev_cam'],(4,4),'observer_T_dev_cam')
    if not np.allclose(transform[3],[0,0,0,1]):
        raise ValueError('Invalid camera homogeneous transform')
    _rotations(transform[:3,:3][None],1,'camera_extrinsic_R')
    oR=_rotations(camera['observer_head_R'],T,'observer_head_R')
    op=_array(camera['observer_head_pos'],(T,3),'observer_head_pos')
    wR=_rotations(tracker['wearer_head_R'],T,'wearer_head_R')
    wp=_array(tracker['wearer_head_pos'],(T,3),'wearer_head_pos')
    Rwc=oR@transform[:3,:3]
    twc=np.einsum('tij,j->ti',oR,transform[:3,3])+op
    pixels=np.concatenate((xy,np.ones((T,13,1))),axis=-1)
    ray=np.einsum('ij,tkj->tki',np.linalg.inv(K),pixels)
    ray*=np.linalg.norm(wp-op,axis=-1)[:,None,None]
    endpoints=np.einsum('tij,tkj->tki',Rwc,ray)+twc[:,None]
    local=np.einsum('tji,tkj->tki',wR,endpoints-wp[:,None])
    return np.where(valid[:,None,None],local,0).reshape(T,39).astype('f4')


def _selector_observations(pair: Path, timestamps: np.ndarray, confidence_policy='strict'):
    data_path=pair/'selector_observations.npz'
    receipt_path=pair/'selector_preparation.json'
    if not receipt_path.is_file() or not data_path.is_file():
        raise ValueError(f'{pair}: causal selector observations missing. Run prepare_local_features with image-detector boxes; official bbox/vitpose[_egonet] are not automatically causal.')
    receipt=_json(receipt_path)
    if receipt.get('schema_version')!=1 or receipt.get('generator')!='observer_selection.prepare_local_features.v1':
        raise ValueError('Unsupported selector preparation receipt')
    if receipt.get('reference_pose_used') is not False or receipt.get('crop_source')!='local_image_detector' or receipt.get('temporal_mode')!='framewise_features':
        raise ValueError('Selector observations must use causal image-detector crops without reference-pose inputs')
    if receipt.get('output_sha256')!=_hash(data_path):
        raise ValueError('Selector observation digest mismatch')
    if not receipt.get('input_files') or not receipt.get('feature_models'):
        raise ValueError('Selector preparation must record input files and feature checkpoints')
    for item in receipt['input_files']:
        source_path=Path(item['path'])
        if not source_path.is_file() or _hash(source_path)!=item['sha256']:
            raise ValueError(f'Selector preprocessing input digest mismatch: {source_path}')
    data=_npz(data_path);T=len(timestamps)
    if not np.array_equal(data['timestamps_ns'],timestamps):
        raise ValueError('Selector observation timestamps do not exactly align')
    last=_array(data['last_input_t_ns'],(T,),'last_input_t_ns')
    if np.any(last>timestamps):
        raise ValueError('Future frame used in selector preprocessing')
    xy=_array(data['xy'],(T,13,2),'xy').astype('f4')
    conf=_array(data['confidence'],(T,13),'confidence').astype('f4')
    valid=_array(data['bbox_valid'],(T,),'bbox_valid')
    if not np.isin(valid,[0,1]).all():
        raise ValueError('Invalid availability encoding: bbox_valid must be boolean')
    if confidence_policy not in ('strict','clip_heatmap_v1'):
        raise ValueError('Unknown confidence encoding policy')
    confidence_encoding={'policy':confidence_policy,'raw_min':float(conf.min()),
        'raw_max':float(conf.max()),'below_zero':int((conf<0).sum()),'above_one':int((conf>1).sum()),
        'calibrated_probability':False}
    if confidence_policy=='strict' and np.any((conf<0)|(conf>1)):
        raise ValueError(f'Invalid confidence encoding: range [{conf.min()}, {conf.max()}]')
    if confidence_policy=='clip_heatmap_v1':
        # Heatmap amplitudes are not probabilities. Keep source NPZ untouched;
        # this explicit export feature transform is recorded in provenance.
        conf=np.clip(conf,0,1)
    valid=valid.astype(bool)
    boxes=_array(data['bbox'],(T,4),'bbox').astype('f4').copy()
    if np.any(boxes[valid,2:]<=boxes[valid,:2]):
        raise ValueError('Valid detector boxes must have positive area')
    # Missing boxes are explicit unavailable observations. Positive normalization
    # placeholder prevents divide-by-zero; it does not provide invented joints.
    image_size=_npz(pair/'observer.npz').get('observer_img_size')
    if image_size is None or np.asarray(image_size).shape!=(2,) or np.any(np.asarray(image_size)<=0):
        raise ValueError('Known observer_img_size [width,height] required for missing-box normalization')
    boxes[~valid]=[0,0,float(image_size[0]),float(image_size[1])]
    xy=xy.copy();conf=conf.copy();xy[~valid]=0;conf[~valid]=0
    dino=_array(data['dino_cls'],(T,768),'dino_cls').astype('f4').copy();dino[~valid]=0
    return {'xy':xy,'confidence':conf,'bbox':boxes,'bbox_valid':valid,'dino_cls':dino,
            'confidence_encoding':confidence_encoding},[data_path,receipt_path]


def export_prepared_egohumans(manifest_path, output_path, allow_benchmark_inputs=False):
    """Convert one explicit target/requester/roster. No implicit scene split.

    Manifest paths resolve relative to the manifest. Frozen roster includes
    sources with currently invalid detections; there is no visibility filtering.
    """
    manifest_path=Path(manifest_path).resolve();m=_json(manifest_path)
    if m.get('schema_version')!=1 or m.get('source_dataset')!='egohumans':
        raise ValueError('Expected schema_version=1 and source_dataset=egohumans')
    required=('scene_id','sequence_id','target_id','requester_id','source_ids','pairs','target_tracker','floor','world_to_zup')
    for k in required:
        if k not in m: raise ValueError(f'Manifest missing {k}')
    def path(s):
        p=Path(s);return p.resolve() if p.is_absolute() else (manifest_path.parent/p).resolve()
    target,requester=str(m['target_id']),str(m['requester_id'])
    sources=tuple(map(str,m['source_ids']))
    if len(sources)<2 or len(set(sources))!=len(sources) or requester in sources or target in sources or requester==target:
        raise ValueError('Require distinct target/requester and at least two unique external candidates; roster excludes both')
    if set(m['pairs'])!={requester,*sources}:
        raise ValueError('pairs must contain exactly requester and candidate IDs')
    floor=m['floor']
    if floor.get('provenance') not in ('calibrated','estimated') or not np.isfinite(floor.get('height_m',np.nan)):
        raise ValueError('Explicit calibrated/estimated floor height required; GT feet normalization prohibited')
    trackinfo=m['target_tracker'];tp=trackinfo.get('provenance')
    if tp not in ('measured','estimated','benchmark_gt'):
        raise ValueError('Target tracker provenance must be measured/estimated/benchmark_gt')
    if tp=='benchmark_gt' and not allow_benchmark_inputs:
        raise ValueError('GT-derived target tracking requires explicit benchmark opt-in')
    trackpath=path(trackinfo['path']);tracker=_npz(trackpath)
    if tp!='benchmark_gt' and trackpath.name in ('wearer_ego.npz','wearer_smplh_fk.npz'):
        raise ValueError('Official EgoHumans wearer_ego/wearer_smplh_fk files are GT-derived; do not label them measured')
    if not trackinfo.get('sha256') or _hash(trackpath)!=trackinfo['sha256']:
        raise ValueError('Target tracker digest missing/mismatched')
    calibration=m['world_to_zup']
    if calibration.get('provenance') not in ('calibrated','estimated'):
        raise ValueError('world_to_zup rotation must be calibrated/estimated without reference pose')
    rotation=_rotations(np.asarray(calibration['rotation'])[None],1,'world_to_zup')[0]
    def transformed_tracker(raw):
        out=dict(raw)
        for part in ('head','lwrist','rwrist'):
            pk,rk=f'wearer_{part}_pos',f'wearer_{part}_R'
            if pk in raw:out[pk]=np.asarray(raw[pk])@rotation.T
            if rk in raw:out[rk]=rotation@np.asarray(raw[rk])
        return out
    tracker=transformed_tracker(tracker)
    dependencies=[manifest_path,trackpath];pairs={};grid=None;gt=None;world_frame=None
    for source in (requester,*sources):
        pair=path(m['pairs'][source]);meta=_json(pair/'meta.json')
        expected={'source_dataset':'egohumans','source_scene':m['scene_id'],
                  'source_subseq':m['sequence_id'],'wearer_aria':target,'observer_aria':source}
        for k,v in expected.items():
            if meta.get(k)!=v: raise ValueError(f'{pair}: metadata {k} disagrees with manifest')
        ts=_timestamps(pair,meta)
        if grid is None: grid=ts;world_frame=meta.get('world_frame')
        elif not np.array_equal(ts,grid) or meta.get('world_frame')!=world_frame:
            raise ValueError('Pair timestamps/world frames differ; no truncation or nearest-frame matching permitted')
        if world_frame is None: raise ValueError('world_frame missing')
        data,dep=_selector_observations(pair,grid)
        camera=_npz(pair/'observer.npz')
        camera['observer_head_pos']=camera['observer_head_pos']@rotation.T
        camera['observer_head_R']=rotation@camera['observer_head_R']
        data['rays']=compute_rays(data['xy'],data['bbox_valid'],camera,tracker)
        pairs[source]=data
        reference=_npz(pair/'wearer_smplh_fk.npz')['joints_pos']
        reference=(_array(reference,(len(grid),22,3),'reference joints_pos')@rotation.T).astype('f4')
        if gt is None:gt=reference
        elif not np.array_equal(reference,gt):raise ValueError('Target reference joints disagree across observer pairs')
        dependencies.extend([pair/'meta.json',pair/'timestamps.csv',pair/'observer.npz',pair/'wearer_smplh_fk.npz',*dep])
    T=len(grid)
    if 'timestamps_ns' not in tracker or not np.array_equal(tracker['timestamps_ns'],grid):
        raise ValueError('Target tracker must include exactly aligned timestamps_ns (never assumed from array order)')
    if 'world_frame' not in tracker or str(tracker['world_frame'].item())!=world_frame:
        raise ValueError('Target tracker world_frame must match observer camera calibration')
    sparse=build_sparse(tracker,float(floor['height_m']),int(m.get('num_input',1)))
    local=pairs[requester]
    out={'timestamps':grid[1:].astype(np.float64)/1e9,'source_ids':np.array(sources),
         'target_id':np.array(target),'requester_id':np.array(requester),
         'scene_id':np.array(m['scene_id']),'sequence_id':np.array(m['sequence_id']),
         'joint_layout':np.array('smplh22'),'detector_joint_layout':np.array('vitpose13'),
         'coordinate_frame':np.array('root_relative'),'units':np.array('metres'),
         'input_provenance':np.array('benchmark_gt_derived' if tp=='benchmark_gt' else tp),
         'tracker_provenance':np.array(tp),'floor_provenance':np.array(floor['provenance']),
         'floor_height_m':np.array(floor['height_m']),
         'world_to_zup_rotation':rotation.astype('f4'),
         'coordinate_convention':np.array('egoexomocap_headlocal_rays_zup'),
         'crop_provenance':np.array('local_detector'),'temporal_provenance':np.array('causal'),
         'preparation_manifest_sha256':np.array(_hash(manifest_path)),
         'sparse_input':sparse,'gt_joints':(gt-gt[:,:1])[1:],'gt_valid':np.ones((T-1,22),bool),
         'local_xy':local['xy'][1:],'local_valid':np.repeat(local['bbox_valid'][1:,None],13,axis=1),
         'local_bbox':local['bbox'][1:],'local_confidence':local['confidence'][1:],
         'local_exo_rays':local['rays'][1:],'local_dino_cls':local['dino_cls'][1:],
         'local_bbox_valid':local['bbox_valid'][1:]}
    for outname,key in [('xy','xy'),('bbox','bbox'),('confidence','confidence'),('exo_rays','rays'),('dino_cls','dino_cls'),('bbox_valid','bbox_valid')]:
        out['packet_'+outname]=np.stack([pairs[s][key][1:] for s in sources],axis=1)
    out['packet_valid']=np.repeat(out['packet_bbox_valid'][...,None],13,axis=-1)
    output_path=Path(output_path).resolve();sidecar=output_path.with_suffix('.provenance.json')
    if output_path.exists() or sidecar.exists():raise FileExistsError('Refusing to overwrite an existing export/provenance file')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('xb') as f:np.savez_compressed(f,**out)
    receipt={'schema_version':1,'source_dataset':'egohumans','scene_id':m['scene_id'],'sequence_id':m['sequence_id'],
        'requester_id':requester,'target_id':target,'source_ids':list(sources),'output_sha256':_hash(output_path),
        'input_files':[{'path':str(p),'sha256':_hash(p)} for p in dict.fromkeys(dependencies)],
        'dropped_initial_frames':1,'reason':'Tracker finite difference needs preceding measured frame',
        'floor':floor,'world_to_zup':calibration,'tracker_provenance':tp,'shape_policy':'backend neutral/independently supplied; reference betas not inserted',
        'causality':'framewise observed crops/features; no selector geometry inferred from external current cameras',
        'benchmark_warning':tp=='benchmark_gt'}
    sidecar.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return {'output':str(output_path),'provenance':str(sidecar),'frames':T-1,'candidates':list(sources)}


def prepare_local_features(pair_dir, boxes_path, boxes_receipt_path, vit, dino,
                           feature_model_ids: dict[str,str], frames=None):
    """Re-extract per-frame ViTPose/DINO from independently detected target boxes.

    `vit.batch(images,[boxes])->objects.xy/scores`, `dino.batch_cls(PIL crops)`
    match official helper APIs. No detector is introduced; caller supplies its
    causal boxes + honest receipt. This function NEVER reads GT bbox files.
    """
    from PIL import Image
    pair=Path(pair_dir);boxpath=Path(boxes_path);receiptpath=Path(boxes_receipt_path)
    destination=pair/'selector_observations.npz';outreceipt=pair/'selector_preparation.json'
    if destination.exists() or outreceipt.exists():raise FileExistsError('Causal features already exist; choose a fresh prepared directory')
    if set(feature_model_ids) not in ({'vitpose','dinov3'}, {'vitpose','dinov2'}) or not all(feature_model_ids.values()):
        raise ValueError('Record exact ViTPose and DINO checkpoint IDs/digests')
    receipt=_json(receiptpath);boxes=_npz(boxpath);meta=_json(pair/'meta.json');ts=_timestamps(pair,meta);T=len(ts)
    if receipt.get('crop_source')!='local_image_detector' or receipt.get('reference_pose_used') is not False or receipt.get('output_sha256')!=_hash(boxpath):
        raise ValueError('Need detector-produced boxes and verifiable non-reference-pose receipt; do not relabel GT/egonet boxes')
    if not receipt.get('detector_checkpoint') or receipt.get('identity_source') not in ('local_tracking','manual_initial_identity','manual_initial_and_recovery_review'):
        raise ValueError('Detector checkpoint and causal target identity association must be recorded')
    recovery_log=None
    if receipt.get('identity_source')=='manual_initial_and_recovery_review':
        recovery_log=pair/'recovery_choices.json'
        if not recovery_log.is_file() or _hash(recovery_log)!=receipt.get('manual_recovery_choices_sha256'):
            raise ValueError('Manual recovery annotation digest mismatch')
    if not np.array_equal(boxes.get('timestamps_ns'),ts):raise ValueError('Box timestamps do not align exactly')
    last=_array(boxes['last_input_t_ns'],(T,),'last_input_t_ns')
    if np.any(last>ts):raise ValueError('Box detector used future frames')
    b=_array(boxes['bbox'],(T,4),'bbox');valid=_array(boxes['bbox_valid'],(T,),'bbox_valid')
    if not np.isin(valid,[0,1]).all():raise ValueError('bbox_valid must be boolean')
    valid=valid.astype(bool)
    if np.any(b[valid,2:]<=b[valid,:2]):raise ValueError('Valid bbox must have positive area')
    video=pair/'observer_full.mp4'
    reader=None
    if frames is None:
        import imageio.v2 as imageio
        if not video.is_file():raise ValueError(f'Missing observer video {video}')
        reader=imageio.get_reader(str(video),'ffmpeg');frames=reader
    xy=np.zeros((T,13,2),'f4');confidence=np.zeros((T,13),'f4');cls=np.zeros((T,768),'f4')
    count=0
    try:
        for t,frame in enumerate(frames):
            if t>=T:raise ValueError('Video has more frames than timestamp grid')
            count+=1
            if not valid[t]:continue
            image=np.asarray(frame)
            if image.ndim!=3 or image.shape[2]!=3:raise ValueError('Expected RGB video frames')
            H,W=image.shape[:2]
            center=(b[t,:2]+b[t,2:])/2;side=int(np.ceil(max(b[t,2:]-b[t,:2])))
            if side>4*max(H,W):raise ValueError('Unreasonable crop extent')
            lo=np.floor(center-side/2).astype(int);hi=lo+side
            crop=np.zeros((side,side,3),dtype=image.dtype)
            x0,y0=np.maximum(lo,0);x1,y1=np.minimum(hi,[W,H])
            if x1>x0 and y1>y0:crop[y0-lo[1]:y1-lo[1],x0-lo[0]:x1-lo[0]]=image[y0:y1,x0:x1]
            kp=vit.batch([image],[b[t:t+1]])[0]
            features=dino.batch_cls([Image.fromarray(crop)])
            if hasattr(features,'detach'):features=features.detach().cpu().numpy()
            xy[t]=_array(kp.xy[0],(13,2),'vitpose xy')
            confidence[t]=_array(kp.scores[0],(13,),'vitpose confidence')
            cls[t]=_array(np.asarray(features)[0],(768,),'dino_cls')
    finally:
        if reader is not None:reader.close()
    if count!=T:raise ValueError(f'Video frame count {count} differs from timestamp grid {T}')
    with destination.open('xb') as f:
        np.savez_compressed(f,timestamps_ns=ts,last_input_t_ns=np.maximum(last,ts),xy=xy,confidence=confidence,bbox=b,bbox_valid=valid,dino_cls=cls)
    out={'schema_version':1,'generator':'observer_selection.prepare_local_features.v1',
         'crop_source':'local_image_detector','reference_pose_used':False,'temporal_mode':'framewise_features',
         'output_sha256':_hash(destination),'feature_models':feature_model_ids,
         'input_files':[{'path':str(p.resolve()),'sha256':_hash(p)} for p in (boxpath,receiptpath,pair/'meta.json',pair/'timestamps.csv')],
         'generator_sha256':_hash(Path(__file__))}
    if video.is_file():out['input_files'].append({'path':str(video.resolve()),'sha256':_hash(video)})
    out['identity_source']=receipt['identity_source']
    if recovery_log is not None:out['input_files'].append({'path':str(recovery_log.resolve()),'sha256':_hash(recovery_log)})
    outreceipt.write_text(json.dumps(out,indent=2),encoding='utf-8')
    return str(destination)
