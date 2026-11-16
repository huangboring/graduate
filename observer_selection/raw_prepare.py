"""Official EgoHumans raw data -> causal, manually initialized image tracks.

Camera convention follows EgoHumans AriaCamera.get_undistorted_image_aria:
upright RGB -> CCW native -> rectification -> CW upright. Camera axes rotate
with the final image; K stays conventional positive-diagonal pinhole.
GT is loaded separately AFTER image inference, exclusively for supervision.
"""
from pathlib import Path
import json
import pickle
import time
import numpy as np
from .egohumans import _hash
from .local_fusion import rigid


Q_UPRIGHT = np.array([[0.,-1,0],[1,0,0],[0,0,1]])


def read_calibration(path):
    lines = Path(path).read_text().splitlines()[1:]
    if len(lines) != 7: raise ValueError(f'Unexpected calibration: {path}')
    intr = np.fromstring(lines[1], sep=' ')
    ext = np.eye(4); ext[:3] = np.fromstring(lines[2], sep=' ').reshape(4,3).T
    if intr.shape != (15,) or not np.isfinite(intr).all() or intr[0] <= 0:
        raise ValueError('Invalid RGB calibration')
    rigid(np.linalg.inv(ext), 'native camera')
    return intr, ext


def fisheye_project(points, intr):
    """Vectorized radtanthinprism convention used by the official dataset."""
    xy = np.asarray(points)[...,:2] / np.asarray(points)[...,2:3]
    r = np.linalg.norm(xy, axis=-1); theta = np.arctan(r)
    polynomial = np.ones_like(r)
    for i, coefficient in enumerate(intr[3:9]): polynomial += coefficient * theta**(2*i+2)
    scale = np.divide(theta, r, out=np.ones_like(r), where=r>1e-9)*polynomial
    radial = xy*scale[...,None]
    r2 = (radial**2).sum(-1)
    distorted = radial + 2*radial**2*intr[9:11] + r2[...,None]*intr[9:11]
    distorted[...,0] += intr[11]*r2 + intr[12]*r2**2
    distorted[...,1] += intr[13]*r2 + intr[14]*r2**2
    return intr[0]*distorted + intr[1:3]


def make_map(intr, size=1408):
    f, c = size/2, (size-1)/2
    y, x = np.mgrid[:size,:size]
    rays = np.stack([(x-c)/f,(y-c)/f,np.ones_like(x)],axis=-1)
    pixels = fisheye_project(rays,intr)
    return pixels[...,0].astype('f4'),pixels[...,1].astype('f4')


def similarity_camera_extrinsic(transform):
    """Normalize a positive uniform similarity without changing projection.

    E=[sR,t] maps anchor-world coordinates to native-camera coordinates.
    E/s=[R,t/s] preserves image rays AND world camera centre -R.T@t/s.
    Dividing R alone would change the camera centre and is incorrect.
    This does not rescale the anchor world, GT, or pinhole intrinsics.
    """
    a=np.asarray(transform,dtype=np.float64)
    if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1],atol=1e-8,rtol=0):
        raise ValueError('Camera alignment must be a finite affine 4x4 transform')
    determinant=float(np.linalg.det(a[:3,:3]))
    if determinant<=0:
        raise ValueError(f'Camera alignment must have positive scale and no reflection; det={determinant}')
    scale=float(np.cbrt(determinant))
    result=a.copy();result[:3,:]/=scale
    rotation=result[:3,:3]
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=2e-4,rtol=0):
        singular=np.linalg.svd(a[:3,:3],compute_uv=False)
        raise ValueError(f'Camera alignment is not uniform scale times rotation; singular_values={singular.tolist()}')
    return result,scale


def upright_camera(native_world_to_camera):
    native_world_to_camera,_=similarity_camera_extrinsic(native_world_to_camera)
    rotation = np.eye(4); rotation[:3,:3] = Q_UPRIGHT
    return rigid(np.linalg.inv(rotation @ native_world_to_camera), 'upright camera')


def validate_setup_restart(output, roster):
    """Allow retry ONLY before observation extraction, preserving model cache."""
    out=Path(output)
    if not out.exists(): return
    for path in out.iterdir():
        if path.is_symlink(): raise FileExistsError(f'Restart refuses symlink: {path}')
        if path.name=='model_cache' and path.is_dir(): continue
        if path.name not in roster or not path.is_dir():
            raise FileExistsError(f'Restart only supports camera setup failures; found {path}')
        for child in path.iterdir():
            if child.is_symlink() or not child.is_file() or child.name not in ('meta.json','timestamps.csv','observer.npz'):
                raise FileExistsError(f'Observation work already exists; refusing to overwrite {child}')


def check_existing_setup(pair, meta, timestamp_text):
    for name,value in (('meta.json',meta),('timestamps.csv',timestamp_text)):
        path=Path(pair)/name
        if not path.exists(): continue
        old=json.loads(path.read_text(encoding='utf-8')) if name.endswith('.json') else path.read_text()
        if old!=value: raise ValueError(f'Restart settings/frame grid differ from {path}')


class RectifiedFrames:
    def __init__(self, paths, calibrations):
        self.paths, self.calibrations = paths, calibrations
        self.intr = self.maps = None

    def get(self, index):
        import cv2
        intr, _ = read_calibration(self.calibrations[index])
        if self.intr is None or not np.array_equal(intr,self.intr):
            self.maps = make_map(intr); self.intr = intr
        image = cv2.imread(str(self.paths[index]))
        if image is None or image.shape != (1408,1408,3):
            raise ValueError(f'Expected 1408x1408 RGB image: {self.paths[index]}')
        native = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        rectified = cv2.remap(native,*self.maps,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        return cv2.cvtColor(cv2.rotate(rectified,cv2.ROTATE_90_CLOCKWISE),cv2.COLOR_BGR2RGB)

    def __iter__(self):
        for index in range(len(self.paths)):
            if index % 50 == 0: print(f'  feature frame {index+1}/{len(self.paths)}',flush=True)
            yield self.get(index)


def match_track(previous, boxes, minimum_iou=.2, ambiguity_margin=.1):
    """Conservative baseline: reject ambiguous/crossing detections, no GT."""
    if previous is None or not len(boxes): return None
    lo=np.maximum(previous[:2],boxes[:,:2]); hi=np.minimum(previous[2:],boxes[:,2:])
    inter=np.maximum(hi-lo,0).prod(-1)
    union=(previous[2:]-previous[:2]).prod()+(boxes[:,2:]-boxes[:,:2]).prod(-1)-inter
    scores=inter/np.maximum(union,1e-8); order=np.argsort(scores)[::-1]
    if scores[order[0]] < minimum_iou: return None
    if len(order)>1 and scores[order[0]]-scores[order[1]]<ambiguity_margin: return None
    return int(order[0])


def prepare_raw(sequence, output, target='aria01', requester='aria02', sources=('aria03','aria04'),
                repo='external/EgoExoMoCap', body_model=None, device='cuda', restart_setup=False):
    import cv2
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.transforms.functional import to_tensor
    from .image_features import DinoV2Encoder, checkpoint_identity
    from .egohumans import prepare_local_features
    from .local_data import export_local_egohumans
    from human_body_prior.body_model.body_model import BodyModel

    start = time.monotonic()
    root, out, repo = Path(sequence).resolve(), Path(output).resolve(), Path(repo).resolve()
    roster = (requester,*sources)
    if len(set((target,*roster))) != len(roster)+1 or len(sources)<2:
        raise ValueError('Need distinct target, requester and at least two helpers')
    if restart_setup: validate_setup_restart(out,roster)
    elif out.exists() and any(out.iterdir()): raise FileExistsError('Use a fresh preparation directory; --restart-setup only retries failures before observations')
    if not root.is_dir(): raise FileNotFoundError(root)
    def numbered(folder,suffix):
        return {int(p.stem):p for p in folder.glob('*'+suffix) if p.stem.isdecimal()}
    smpl = numbered(root/'processed_data/smpl','.npy')
    ids=sorted(smpl)
    if not ids or ids != list(range(ids[0],ids[-1]+1)):
        raise ValueError('SMPL frame IDs must be nonempty and consecutive')
    images,calibs={},{}
    for aria in roster:
        im=numbered(root/'ego'/aria/'images/rgb','.jpg'); ca=numbered(root/'ego'/aria/'calib','.txt')
        if set(im)!=set(ids) or set(ca)!=set(ids): raise ValueError(f'Exact frame-ID alignment required: {aria}')
        images[aria]=[im[i] for i in ids]; calibs[aria]=[ca[i] for i in ids]
    transform_file=root/'colmap/workplace/colmap_from_aria_transforms.pkl'
    with transform_file.open('rb') as stream: transforms=pickle.load(stream)
    if any(a not in transforms for a in ('aria01',*roster)): raise ValueError('Missing world transforms')
    # Validate every camera BEFORE allocating models/downloading weights.
    camera_poses,scales={},{}
    for aria in roster:
        relative=np.linalg.inv(transforms[aria])@transforms['aria01']
        _,scale=similarity_camera_extrinsic(relative)
        scales[aria]=scale
        camera_poses[aria]=np.stack([upright_camera(read_calibration(p)[1]@relative) for p in calibs[aria]])
        print(f'{aria}: camera calibration OK; alignment scale={scale:.9g}',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    torch.hub.set_dir(str(out/'model_cache'))  # never default to C: for detector download
    weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    detector=fasterrcnn_resnet50_fpn(weights=weights).eval().to(device)
    vit_file=repo/'data_prep/shared/vitpose_predictor.py'
    import importlib.util, sys
    spec=importlib.util.spec_from_file_location('_raw_vitpose',vit_file)
    vit_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=vit_module
    spec.loader.exec_module(vit_module)
    vit_dir=repo/'third_party/vitpose_hf/vitpose-base-simple'
    vit=vit_module.ViTPosePredictor(model_id=vit_dir,device=device)
    from transformers import AutoImageProcessor
    vit.processor=AutoImageProcessor.from_pretrained(str(vit_dir),local_files_only=True,use_fast=False)
    dino=DinoV2Encoder(repo/'third_party/dinov2_hf/dinov2-base',device)
    models={'vitpose':checkpoint_identity(vit_dir)+';helper='+_hash(vit_file)+';use_fast=False',
            'dinov2':dino.identity+';use_fast=False'}
    T=len(ids); timestamps=np.arange(T,dtype=np.int64)*50_000_000
    K=np.array([[704.,0,703.5],[0,704.,703.5],[0,0,1.]])
    pairs={}; stats={}
    for aria in roster:
        pair=out/aria; pair.mkdir(exist_ok=restart_setup); pairs[aria]=str(pair)
        meta={'source_dataset':'egohumans','source_scene':root.parent.name,'source_subseq':root.name,
              'wearer_aria':target,'observer_aria':aria,'world_frame':'aria01_local',
              'n_frames':T,'t_start_ns':0,'dt_ns':50_000_000,'raw_frame_ids':ids}
        timestamp_text='frame_idx,t_ns\n'+''.join(f'{i},{t}\n' for i,t in enumerate(timestamps))
        if restart_setup: check_existing_setup(pair,meta,timestamp_text)
        (pair/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        (pair/'timestamps.csv').write_text(timestamp_text)
        poses=camera_poses[aria]
        np.savez(pair/'observer.npz',observer_K=K,observer_T_dev_cam=np.eye(4),
                 observer_head_R=poses[:,:3,:3],observer_head_pos=poses[:,:3,3],observer_img_size=[1408,1408])
        frames=RectifiedFrames(images[aria],calibs[aria])
        boxes=np.zeros((T,4),'f4'); valid=np.zeros(T,bool)
        previous=None; misses=0; initialized=False; init_record=None; next_prompt=0
        preview=cv2.VideoWriter(str(pair/'tracking_preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(704,704))
        if not preview.isOpened(): raise RuntimeError('Cannot open tracking preview video writer')
        try:
            for t in range(T):
                image=frames.get(t)
                with torch.inference_mode(): prediction=detector([to_tensor(image).to(device)])[0]
                keep=(prediction['labels']==1)&(prediction['scores']>=.7)
                candidates=prediction['boxes'][keep].cpu().numpy()
                display=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                for i,box in enumerate(candidates):
                    a,b,c,d=map(int,box);cv2.rectangle(display,(a,b),(c,d),(160,160,160),2)
                    cv2.putText(display,str(i),(a,max(20,b)),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
                chosen=None
                if not initialized and len(candidates) and t>=next_prompt:
                    file=pair/f'identity_frame_{ids[t]:05}.jpg';cv2.imwrite(str(file),display)
                    print(f'\n{aria}, raw frame {ids[t]}: open {file}\nIdentify target {target}. Enter its box number; n=not visible (check again in 20 frames); q=stop.',flush=True)
                    while True:
                        answer=input('Target box: ').strip().lower()
                        if answer=='q': raise KeyboardInterrupt('Stopped before completion; use new output directory')
                        if answer=='n':
                            next_prompt=t+20
                            break
                        if answer.isdecimal() and int(answer)<len(candidates):
                            chosen=int(answer);initialized=True
                            init_record={'raw_frame_id':ids[t],'box_index':chosen,'bbox':candidates[chosen].tolist()}
                            cv2.imwrite(str(pair/'identity_selected.jpg'),display)
                            break
                        print('Enter a displayed box number, n, or q.',flush=True)
                elif initialized and misses<5:
                    chosen=match_track(previous,candidates)
                if chosen is not None:
                    previous=candidates[chosen].copy();misses=0;boxes[t]=previous;valid[t]=True
                    a,b,c,d=map(int,previous);cv2.rectangle(display,(a,b),(c,d),(0,255,0),4)
                elif initialized: misses+=1
                cv2.putText(display,f'frame {ids[t]} | target valid={valid[t]}',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
                preview.write(cv2.resize(display,(704,704)))
                if t%50==0: print(f'{aria}: detection {t+1}/{T}, valid {int(valid[:t+1].sum())}',flush=True)
        finally: preview.release()
        if not valid.any(): raise ValueError(f'No target observation initialized for {aria}; preparation incomplete')
        np.savez(pair/'detector_boxes.npz',timestamps_ns=timestamps,last_input_t_ns=timestamps,bbox=boxes,bbox_valid=valid)
        detector_file=next((out/'model_cache/checkpoints').glob('fasterrcnn_resnet50_fpn*.pth'))
        receipt={'crop_source':'local_image_detector','reference_pose_used':False,
                 'output_sha256':_hash(pair/'detector_boxes.npz'),
                 'detector_checkpoint':str(weights)+';sha256='+_hash(detector_file),
                 'identity_source':'manual_initial_identity','initial_identity':init_record,
                 'tracker':'causal IoU .2; ambiguity margin .1; no reacquisition after 5 misses',
                 'score_threshold':.7,'generator_sha256':_hash(Path(__file__))}
        (pair/'detector_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        prepare_local_features(pair,pair/'detector_boxes.npz',pair/'detector_receipt.json',vit,dino,models,frames)
        feature_receipt=pair/'selector_preparation.json'
        r=json.loads(feature_receipt.read_text(encoding='utf-8'))
        r['input_files'].extend({'path':str(p),'sha256':_hash(p)} for p in [transform_file,*calibs[aria],*images[aria]])
        r['raw_generator_sha256']=_hash(Path(__file__))
        feature_receipt.write_text(json.dumps(r,indent=2),encoding='utf-8')
        stats[aria]={'frames':T,'valid_frames':int(valid.sum()),'valid_fraction':float(valid.mean()),
                     'terminal_track_loss':bool(misses>=5)}
        print(f'{aria} completed: {stats[aria]}',flush=True)
    del detector,vit,dino
    if device.startswith('cuda'): torch.cuda.empty_cache()
    # Labels are read only after every observation is finalized.
    model_path=(Path(body_model) if body_model else repo/'third_party/body_models/smplh/neutral/model.npz').resolve()
    bm=BodyModel(bm_fname=str(model_path),num_betas=10).eval().to(device)
    truth=[]
    for frame_id in ids:
        item=np.load(smpl[frame_id],allow_pickle=True).item()
        if target not in item: raise ValueError(f'Missing target GT at frame {frame_id}; no fabricated zero labels')
        item=item[target]
        expected={'global_orient':(3,),'body_pose':(69,),'transl':(3,),'betas':(10,)}
        arrays={k:np.asarray(item[k],dtype='f4') for k in expected}
        if any(arrays[k].shape!=shape or not np.isfinite(arrays[k]).all() for k,shape in expected.items()):
            raise ValueError(f'Invalid target GT at frame {frame_id}')
        with torch.inference_mode():
            tensor=lambda x:torch.as_tensor(x,device=device).unsqueeze(0)
            joints=bm(root_orient=tensor(arrays['global_orient']),pose_body=tensor(arrays['body_pose'][:63]),
                      trans=tensor(arrays['transl']),betas=tensor(arrays['betas'])).Jtr[0,:22]
        truth.append(joints.cpu().numpy())
    truth=np.stack(truth)
    for pair in pairs.values(): np.savez(Path(pair)/'wearer_smplh_fk.npz',joints_pos=truth)
    manifest={'schema_version':2,'source_dataset':'egohumans','scene_id':root.parent.name,'sequence_id':root.name,
              'target_id':target,'requester_id':requester,'source_ids':list(sources),'pairs':pairs,
              'camera_calibration':{'provenance':'calibrated','target_pose_used':False,'rectified_pinhole':True,
                'alignment_scales':scales,'scale_normalization':'divide full world-to-camera top3x4 by positive uniform scale; preserve camera centre and projection',
                'description':'Official per-frame Aria calibration; aria01 anchor; native CCW / rectification / upright CW; rotate camera axes Q, diagonal K'}}
    mp=out/'manifest.json';mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    exported=export_local_egohumans(mp,out/'replay.npz')
    receipt_path=Path(exported['provenance']);receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    receipt['input_files'].extend({'path':str(p),'sha256':_hash(p)} for p in [model_path,*smpl.values()])
    receipt['raw_generator_sha256']=_hash(Path(__file__))
    receipt['supervision']='SMPL-H first22 FK from supplied fitted SMPL parameters; dataset fit reference, not new mocap measurements'
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    summary={**exported,'tracking':stats,'elapsed_seconds':time.monotonic()-start,
             'review_required':'Inspect identity_selected.jpg and tracking_preview.mp4 for every view before training.'}
    (out/'preparation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary
