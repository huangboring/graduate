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


def _selector_observations(pair: Path, timestamps: np.ndarray, confidence_policy='strict', allow_annotation=False):
    data_path=pair/'selector_observations.npz'
    receipt_path=pair/'selector_preparation.json'
    if not receipt_path.is_file() or not data_path.is_file():
        raise ValueError(f'{pair}: causal selector observations missing. Run prepare_local_features with image-detector boxes; official bbox/vitpose[_egonet] are not automatically causal.')
    receipt=_json(receipt_path)
    if receipt.get('schema_version')!=1 or receipt.get('generator')!='observer_selection.prepare_local_features.v1':
        raise ValueError('Unsupported selector preparation receipt')
    annotation = allow_annotation and receipt.get('crop_source')=='egohumans_annotation'
    detector = receipt.get('crop_source')=='local_image_detector' and receipt.get('reference_pose_used') is False
    if not (annotation or detector) or receipt.get('temporal_mode')!='framewise_features':
        raise ValueError('Selector observations require image-detector crops or explicitly enabled EgoHumans annotations')
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




def prepare_local_features(pair_dir, boxes_path, boxes_receipt_path, vit, dino,
                           feature_model_ids: dict[str,str], frames=None, *, allow_annotation=False):
    """Re-extract per-frame ViTPose/DINO from independently detected target boxes.

    `vit.batch(images,[boxes])->objects.xy/scores`, `dino.batch_cls(PIL crops)`
    match official helper APIs. The caller supplies boxes and records whether
    the crops use a detector or official dataset annotations.
    """
    from PIL import Image
    pair=Path(pair_dir);boxpath=Path(boxes_path);receiptpath=Path(boxes_receipt_path)
    destination=pair/'selector_observations.npz';outreceipt=pair/'selector_preparation.json'
    if destination.exists() or outreceipt.exists():raise FileExistsError('Causal features already exist; choose a fresh prepared directory')
    if set(feature_model_ids) not in ({'vitpose','dinov3'}, {'vitpose','dinov2'}) or not all(feature_model_ids.values()):
        raise ValueError('Record exact ViTPose and DINO checkpoint IDs/digests')
    receipt=_json(receiptpath);boxes=_npz(boxpath);meta=_json(pair/'meta.json');ts=_timestamps(pair,meta);T=len(ts)
    annotation = (allow_annotation and receipt.get('crop_source')=='egohumans_annotation'
                  and receipt.get('identity_source')=='dataset_human_name'
                  and receipt.get('annotation_target')==meta.get('wearer_aria'))
    detector = (receipt.get('crop_source')=='local_image_detector' and receipt.get('reference_pose_used') is False
                and receipt.get('detector_checkpoint') and receipt.get('identity_source') in
                ('local_tracking','manual_initial_identity','manual_initial_and_recovery_review'))
    if not (annotation or detector) or receipt.get('output_sha256')!=_hash(boxpath):
        raise ValueError('Need verified detector or explicitly enabled annotated crop receipt')
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
         'crop_source':receipt['crop_source'],'reference_pose_used':receipt.get('reference_pose_used'),'temporal_mode':'framewise_features',
         'output_sha256':_hash(destination),'feature_models':feature_model_ids,
         'input_files':[{'path':str(p.resolve()),'sha256':_hash(p)} for p in (boxpath,receiptpath,pair/'meta.json',pair/'timestamps.csv')],
         'generator_sha256':_hash(Path(__file__))}
    if video.is_file():out['input_files'].append({'path':str(video.resolve()),'sha256':_hash(video)})
    out['identity_source']=receipt['identity_source']
    if recovery_log is not None:out['input_files'].append({'path':str(recovery_log.resolve()),'sha256':_hash(recovery_log)})
    outreceipt.write_text(json.dumps(out,indent=2),encoding='utf-8')
    return str(destination)
