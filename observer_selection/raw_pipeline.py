"""Raw EgoHumans orchestration with immutable contracts and stage resumption."""
from pathlib import Path
import json
import pickle
import time
import numpy as np
from .egohumans import _hash, prepare_local_features
from .raw_prepare import read_calibration, upright_camera, similarity_camera_extrinsic, RectifiedFrames, validate_setup_restart, check_existing_setup
from .appearance_tracking import AppearanceTracker, VERSION
from .preparation_cache import (atomic_json, atomic_npz, load_or_create_contract,
    verify_completed_view, mark_completed_view, archive_partial_features, archive_outputs)


def prepare_raw(sequence, output, target='aria01', requester='aria02', sources=('aria03','aria04'),
                repo='external/EgoExoMoCap', body_model=None, device='cuda', restart_setup=False,
                resume=False, model_cache=None, reset_views=(), review_tracking=False, reuse_candidates=None):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from .image_features import DinoV2Encoder, checkpoint_identity
    from .local_data import export_local_egohumans
    from .raw_tracking import track_view
    from human_body_prior.body_model.body_model import BodyModel
    start=time.monotonic()
    root,out,repo=Path(sequence).resolve(),Path(output).resolve(),Path(repo).resolve()
    roster=(requester,*sources)
    if len(set((target,*roster)))!=len(roster)+1 or len(sources)<2:
        raise ValueError('Need distinct target, requester and at least two helpers')
    if restart_setup and resume: raise ValueError('Choose --resume or --restart-setup, not both')
    if reset_views and (not resume or not set(reset_views)<=set(roster)):
        raise ValueError('--reset-view requires --resume and names from the observer roster')
    if restart_setup: validate_setup_restart(out,roster)
    elif out.exists() and any(out.iterdir()):
        if not resume: raise FileExistsError('Use --resume for this run, or choose a new output directory')
        if not (out/'preparation_run.json').is_file():
            raise ValueError('Old IoU-only outputs cannot be resumed as new tracking. Use a new output and --model-cache pointing to the old model_cache.')
    if not root.is_dir(): raise FileNotFoundError(root)
    def numbered(folder,suffix):
        return {int(p.stem):p for p in folder.glob('*'+suffix) if p.stem.isdecimal()}
    smpl=numbered(root/'processed_data/smpl','.npy');ids=sorted(smpl)
    if not ids or ids!=list(range(ids[0],ids[-1]+1)):
        raise ValueError('SMPL frame IDs must be nonempty and consecutive')
    images,calibs={},{}
    for aria in roster:
        im=numbered(root/'ego'/aria/'images/rgb','.jpg');ca=numbered(root/'ego'/aria/'calib','.txt')
        if set(im)!=set(ids) or set(ca)!=set(ids): raise ValueError(f'Exact frame-ID alignment required: {aria}')
        images[aria]=[im[i] for i in ids];calibs[aria]=[ca[i] for i in ids]
    transform_file=root/'colmap/workplace/colmap_from_aria_transforms.pkl'
    with transform_file.open('rb') as stream: transforms=pickle.load(stream)
    if any(a not in transforms for a in ('aria01',*roster)): raise ValueError('Missing world transforms')
    camera_poses,scales={},{}
    for aria in roster:
        relative=np.linalg.inv(transforms[aria])@transforms['aria01']
        _,scales[aria]=similarity_camera_extrinsic(relative)
        camera_poses[aria]=np.stack([upright_camera(read_calibration(p)[1]@relative) for p in calibs[aria]])
        print(f'{aria}: camera calibration OK; alignment scale={scales[aria]:.9g}',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    cache_dir=Path(model_cache).resolve() if model_cache else out/'model_cache'
    torch.hub.set_dir(str(cache_dir))
    weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    detector=fasterrcnn_resnet50_fpn(weights=weights).eval().to(device)
    detector_file=next(cache_dir.glob('checkpoints/fasterrcnn_resnet50_fpn*.pth'))
    detector_id=str(weights)+';sha256='+_hash(detector_file)
    vit_file=repo/'data_prep/shared/vitpose_predictor.py'
    import importlib.util,sys
    spec=importlib.util.spec_from_file_location('_raw_vitpose',vit_file)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    vit_dir=repo/'third_party/vitpose_hf/vitpose-base-simple'
    vit=module.ViTPosePredictor(model_id=vit_dir,device=device)
    from transformers import AutoImageProcessor
    vit.processor=AutoImageProcessor.from_pretrained(str(vit_dir),local_files_only=True,use_fast=False)
    dino=DinoV2Encoder(repo/'third_party/dinov2_hf/dinov2-base',device)
    models={'vitpose':checkpoint_identity(vit_dir)+';helper='+_hash(vit_file)+';use_fast=False',
            'dinov2':dino.identity+';use_fast=False'}
    model_path=(Path(body_model) if body_model else repo/'third_party/body_models/smplh/neutral/model.npz').resolve()
    print('Verifying raw input hashes for cache/resume provenance...',flush=True)
    all_inputs=[transform_file,model_path,*smpl.values(),*[p for a in roster for p in (*calibs[a],*images[a])]]
    input_files=[{'path':str(p),'sha256':_hash(p)} for p in all_inputs]
    from . import raw_prepare,raw_tracking,appearance_tracking,image_features,egohumans
    contract={'schema_version':3,'sequence':str(root),'target':target,'requester':requester,'sources':list(sources),
              'review_tracking':review_tracking,
              'raw_frame_ids':ids,'input_files':input_files,'feature_models':models,'detector':detector_id,
              'tracker':AppearanceTracker().description(),'device':device,'torch':str(torch.__version__),
              'pipeline_sha256':_hash(Path(__file__)),
              'code_hashes':{m.__name__:_hash(Path(m.__file__)) for m in (raw_prepare,raw_tracking,appearance_tracking,image_features,egohumans)}}
    load_or_create_contract(out/'preparation_run.json',contract,resume)
    if reuse_candidates:
        from .candidate_reuse import reuse_candidate_cache
        reuse_candidate_cache(reuse_candidates,out,contract)
    if reset_views:
        archive_outputs(out,['manifest.json','replay.npz','replay.provenance.json','preparation_summary.json','preparation_complete.json'])
        for aria in reset_views:
            pair=out/aria
            if pair.exists():
                names=['identity_choices.json','recovery_choices.json','tracking_diagnostics.json','identity_selected.jpg','identity_reference.jpg','view_complete.json',
                       'detector_boxes.npz','detector_receipt.json','selector_observations.npz','selector_preparation.json','tracking_preview.mp4']
                archive_outputs(pair,names)
    T=len(ids);timestamps=np.arange(T,dtype=np.int64)*50_000_000
    K=np.array([[704.,0,703.5],[0,704.,703.5],[0,0,1.]])
    pairs,stats={},{};reference=None
    for aria in roster:
        pair=out/aria;pair.mkdir(exist_ok=True);pairs[aria]=str(pair)
        completed=verify_completed_view(pair)
        if completed is not None:
            stats[aria]=completed
            if reference is None: reference=pair/'identity_reference.jpg'
            print(f'{aria}: reuse completed view; valid={completed["valid_frames"]}/{T}',flush=True)
            continue
        meta={'source_dataset':'egohumans','source_scene':root.parent.name,'source_subseq':root.name,
              'wearer_aria':target,'observer_aria':aria,'world_frame':'aria01_local',
              'n_frames':T,'t_start_ns':0,'dt_ns':50_000_000,'raw_frame_ids':ids}
        timestamp_text='frame_idx,t_ns\n'+''.join(f'{i},{t}\n' for i,t in enumerate(timestamps))
        check_existing_setup(pair,meta,timestamp_text)
        atomic_json(pair/'meta.json',meta);(pair/'timestamps.csv').write_text(timestamp_text)
        poses=camera_poses[aria]
        atomic_npz(pair/'observer.npz',observer_K=K,observer_T_dev_cam=np.eye(4),observer_head_R=poses[:,:3,:3],
                   observer_head_pos=poses[:,:3,3],observer_img_size=[1408,1408])
        frames=RectifiedFrames(images[aria],calibs[aria])
        stats[aria]=track_view(pair,frames,ids,timestamps,detector,dino,device,target,detector_id,reference,review_tracking)
        archive_partial_features(pair)
        prepare_local_features(pair,pair/'detector_boxes.npz',pair/'detector_receipt.json',vit,dino,models,frames)
        feature_receipt=pair/'selector_preparation.json';r=json.loads(feature_receipt.read_text(encoding='utf-8'))
        paths={str(p) for p in (transform_file,*calibs[aria],*images[aria])}
        r['input_files'].extend(item for item in input_files if item['path'] in paths)
        r['input_files'].append({'path':str(out/'preparation_run.json'),'sha256':_hash(out/'preparation_run.json')})
        r['raw_generator_sha256']=_hash(Path(__file__));atomic_json(feature_receipt,r)
        mark_completed_view(pair,stats[aria])
        if reference is None: reference=pair/'identity_reference.jpg'
        print(f'{aria} completed: {stats[aria]}',flush=True)
    complete=out/'preparation_complete.json'
    if complete.exists():
        marker=json.loads(complete.read_text(encoding='utf-8'))
        for name,digest in marker.items():
            p=out/name
            if p.parent!=out or not p.is_file() or _hash(p)!=digest: raise ValueError(f'Completed export changed: {p}')
        return json.loads((out/'preparation_summary.json').read_text(encoding='utf-8'))
    del detector,vit,dino
    if device.startswith('cuda'): torch.cuda.empty_cache()
    # Read reference labels ONLY after observation extraction is finalized.
    bm=BodyModel(bm_fname=str(model_path),num_betas=10).eval().to(device);truth=[]
    for frame_id in ids:
        item=np.load(smpl[frame_id],allow_pickle=True).item()
        if target not in item: raise ValueError(f'Missing target GT at frame {frame_id}; no fabricated zero labels')
        item=item[target];expected={'global_orient':(3,),'body_pose':(69,),'transl':(3,),'betas':(10,)}
        arrays={k:np.asarray(item[k],dtype='f4') for k in expected}
        if any(arrays[k].shape!=shape or not np.isfinite(arrays[k]).all() for k,shape in expected.items()):
            raise ValueError(f'Invalid target GT at frame {frame_id}')
        with torch.inference_mode():
            tensor=lambda x:torch.as_tensor(x,device=device).unsqueeze(0)
            joints=bm(root_orient=tensor(arrays['global_orient']),pose_body=tensor(arrays['body_pose'][:63]),
                      trans=tensor(arrays['transl']),betas=tensor(arrays['betas'])).Jtr[0,:22]
        truth.append(joints.cpu().numpy())
    truth=np.stack(truth)
    for pair in pairs.values(): atomic_npz(Path(pair)/'wearer_smplh_fk.npz',joints_pos=truth)
    archive_outputs(out,['replay.npz','replay.provenance.json','preparation_summary.json'])
    manifest={'schema_version':2,'source_dataset':'egohumans','scene_id':root.parent.name,'sequence_id':root.name,
              'target_id':target,'requester_id':requester,'source_ids':list(sources),'pairs':pairs,
              'camera_calibration':{'provenance':'calibrated','target_pose_used':False,'rectified_pinhole':True,
                'alignment_scales':scales,'scale_normalization':'divide full world-to-camera top3x4 by positive uniform scale; preserve camera centre and projection',
                'description':'Official per-frame Aria calibration; aria01 anchor; native CCW / rectification / upright CW; rotate camera axes Q, diagonal K'}}
    mp=out/'manifest.json';atomic_json(mp,manifest)
    exported=export_local_egohumans(mp,out/'replay.npz')
    receipt_path=Path(exported['provenance']);receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    paths={str(p) for p in (model_path,*smpl.values())}
    receipt['input_files'].extend(item for item in input_files if item['path'] in paths)
    receipt['raw_generator_sha256']=_hash(Path(__file__));receipt['tracker_version']=VERSION
    receipt['supervision']='SMPL-H first22 FK from supplied fitted SMPL parameters; dataset fit reference, not new mocap measurements'
    atomic_json(receipt_path,receipt)
    summary={**exported,'tracker_version':VERSION,'tracking':stats,'elapsed_seconds_this_invocation':time.monotonic()-start,
             'review_required':'Inspect identity_selected.jpg and tracking_preview.mp4 for every view. Generic DINO appearance is not certified identity.'}
    atomic_json(out/'preparation_summary.json',summary)
    names=['manifest.json','replay.npz','replay.provenance.json','preparation_summary.json']
    atomic_json(complete,{name:_hash(out/name) for name in names})
    return summary
