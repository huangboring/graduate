"""Single-pair pretrained EgoExoMoCap demo with an explicitly substituted DINOv2.

Uses official synthetic trackers, floor normalization, body shape, EgoNet crop
proposal and fusion network. NOT the observer-selection research protocol and
NOT an exact DINOv3 reproduction. No training and no manual identity tracker.
"""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import torch
from .egohumans import _hash
from .preparation_cache import atomic_json, atomic_npz


def module_from_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def strict_weights(net,path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if isinstance(state,dict) and 'params' in state:state=state['params']
    state={k.removeprefix('module.'):v for k,v in state.items()}
    net.load_state_dict(state,strict=True)
    return len(state)


def run_demo(sequence,output,repo,checkpoint=None,wearer='aria01',observer='aria03',points=3,device='cuda'):
    import yaml
    import cv2
    from PIL import Image
    from .raw_prepare import RectifiedFrames, upright_camera
    from .image_features import DinoV2Encoder, checkpoint_identity
    root,out,repo=Path(sequence).resolve(),Path(output).resolve(),Path(repo).resolve()
    if wearer==observer:raise ValueError('Wearer and observer must differ')
    ckpt=Path(checkpoint).resolve() if checkpoint else repo/f'model_zoo/egohumans_syn_tracker_{points}pt/model.pth'
    body=repo/'third_party/body_models/smplh/neutral/model.npz'
    config=repo/f'options/test_egohumans_syn_tracker_{points}pt.yaml'
    for path in (root,ckpt,body,config):
        if not path.exists():raise FileNotFoundError(path)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in this Python environment')
    if out.exists():raise FileExistsError('Choose a fresh demo output directory')
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'data_prep/shared'))
    from human_body_prior.body_model.body_model import BodyModel
    from data_loaders.dataset_nymeria import Nymeria_Dataset
    from models.model_egoexomocap import ModelEgoExoMoCap
    from utils import utils_transform
    from torch.utils.data._utils.collate import default_collate
    stage=module_from_file('_demo_official_stage1',repo/'data_prep/egohumans/extract_egohumans_pairs_stage1.py')
    proposal=module_from_file('_demo_official_proposal',repo/'data_prep/shared/extract_egonet_bboxes.py')
    opt=yaml.safe_load(config.read_text());opt['is_train']=False
    opt['gpu_ids']=[0] if device=='cuda' else None
    opt['body_model']['smpl_path']=str(repo/'third_party')
    opt['path']['models']=str(out/'unused_models')
    torch.manual_seed(2026);torch.set_num_threads(min(4,torch.get_num_threads()))
    model=ModelEgoExoMoCap(opt)
    net=model.get_bare_model(model.net)
    count=strict_weights(net,ckpt);net.eval();net.requires_grad_(False)
    print(f'Loaded ALL {count} checkpoint tensors strictly; no training.',flush=True)
    out.mkdir(parents=True)
    disclaimer='PRETRAINED DEMO / DINOv2 substitution; NOT exact paper reproduction. GT-derived head/wrist, floor and shape per official EgoHumans setup.'
    record={'scope':disclaimer,'wearer':wearer,'observer':observer,'points':points,
        'checkpoint':str(ckpt),'checkpoint_sha256':_hash(ckpt),'config_sha256':_hash(config),
        'body_model_sha256':_hash(body),'strict_loaded_tensors':count,
        'differences':['DINOv2 CLS replaces DINOv3 CLS without retraining ScoreNet',
          'Single sequence/pair, not full benchmark',
          'Per-frame rectification and rigid normalization of similarity camera calibration',
          'Dataset loader entry loaded directly to avoid upstream GT-filename discovery check'],
        'code_sha256':_hash(Path(__file__)),'sequence':str(root)}
    atomic_json(out/'demo_protocol.json',record)
    print('Preparing official synthetic tracker inputs and SMPL-H reference...',flush=True)
    data=stage.load_subseq(root)
    if data is None or wearer not in data['aria_names'] or observer not in data['aria_names']:
        raise ValueError('Requested pair not available')
    ids=sorted(int(p.stem) for p in (root/'processed_data/smpl').glob('*.npy') if p.stem.isdecimal())
    if ids!=list(range(1,data['T']+1)) or not data['smpl'][wearer]['valid'].all():
        raise ValueError('Demo requires exact consecutive frames and valid wearer fits; no zero-filled references')
    bm10=BodyModel(bm_fname=str(body),num_betas=10).eval().to(device)
    with torch.inference_mode():fk={wearer:stage.fk_aria(data['smpl'][wearer],bm10,torch.device(device),batch=32)}
    pair=out/'dataset'/f'{root.parent.name}_{root.name}__w{wearer}_o{observer}'
    stage.write_pair(pair,root.parent.name,root.name,wearer,observer,data,fk)
    # Equivalent upright pinhole geometry, with scaled alignment normalized.
    poses=np.stack([upright_camera(e) for e in data['extr_w2c_rgb'][observer]])
    K=np.array([[704.,0,703.5],[0,704.,703.5],[0,0,1.]])
    atomic_npz(pair/'observer.npz',observer_K=K,observer_T_dev_cam=np.eye(4),
        observer_head_pos=poses[:,:3,3],observer_head_R=poses[:,:3,:3],observer_img_size=[1408,1408])
    dopt=dict(opt['datasets']['test']);dopt.update(dataroot=[str(out/'dataset')],include_exo=False)
    dataset=Nymeria_Dataset(dopt);sample=dataset[0]
    sparse=sample['sparse'].to(device);ws=int(dopt['window_size']);n=len(sparse)//ws*ws
    if n<ws:raise ValueError('Sequence shorter than one official inference window')
    print(f'Running official EgoNet crop proposals for {n} frames...',flush=True)
    predictions=[]
    with torch.inference_mode():
        for start in range(0,n,ws):
            pred=net.egonet({'sparse_input':sparse[start:start+ws].unsqueeze(0).clone()})
            aa_root=utils_transform.sixd2aa(pred['root_orient'].reshape(-1,6)).reshape(ws,3)
            aa_body=utils_transform.sixd2aa(pred['pose_body'].reshape(-1,6)).reshape(ws,63)
            joints=bm10(root_orient=aa_root,pose_body=aa_body,
                betas=sample['betas_gt'][start:start+ws,:10].to(device)).Jtr[:,:22]
            head=sample['head_trans4x4_global'][start:start+ws,:3,3].to(device)
            predictions.append((joints-joints[:,15:16]+head[:,None]).cpu().numpy())
    ego=np.concatenate(predictions)
    pix,depth=proposal.project_22_to_pixel(ego,K,np.eye(4),poses[1:n+1,:3,3],poses[1:n+1,:3,:3])
    T=data['T'];rect=np.zeros((T,4),'f4');square=rect.copy();valid=np.zeros(T,bool)
    for i in range(n):rect[i+1],square[i+1],valid[i+1]=proposal.compute_bbox_egonet(pix[i],depth[i],1408,1408)
    atomic_npz(pair/'bbox_egonet.npz',bbox_xyxy=rect,bbox_xyxy_square=square,bbox_valid=valid)
    print(f'EgoNet proposed crops: {valid.sum()}/{n} evaluated frames. No detector or manual IDs.',flush=True)
    if not valid.any():raise ValueError('No usable EgoNet crop; inspect geometry before claiming an ego-exo result')
    del bm10
    vp_module=module_from_file('_demo_vit',repo/'data_prep/shared/vitpose_predictor.py')
    vit_dir=repo/'third_party/vitpose_hf/vitpose-base-simple'
    vit=vp_module.ViTPosePredictor(model_id=vit_dir,device=device)
    dino=DinoV2Encoder(repo/'third_party/dinov2_hf/dinov2-base',device)
    record['dino_identity']=dino.identity;record['vitpose_identity']=checkpoint_identity(vit_dir)
    image_dir=root/'ego'/observer/'images/rgb';calib_dir=root/'ego'/observer/'calib'
    images=[image_dir/f'{i:05}.jpg' for i in ids];calibs=[calib_dir/f'{i:05}.txt' for i in ids]
    if {int(p.stem) for p in image_dir.glob('*.jpg') if p.stem.isdecimal()}!=set(ids):
        raise ValueError('Image frame IDs differ from reference')
    frames=RectifiedFrames(images,calibs)
    xy=np.zeros((T,13,2),'f4');scores=np.zeros((T,13),'f4');cls=np.zeros((T,768),'f4')
    video=cv2.VideoWriter(str(pair/'observer_full.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1408,1408))
    if not video.isOpened():raise RuntimeError('Cannot create observer video')
    try:
        for t in range(T):
            im=frames.get(t);video.write(cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
            if valid[t]:
                x0,y0,x1,y1=rect[t].astype(int);crop=im[y0:y1,x0:x1]
                if crop.size==0:raise ValueError('Empty proposed crop')
                kp=vit.batch([crop],[np.array([[0,0,crop.shape[1],crop.shape[0]]],'f4')])[0]
                xy[t]=kp.xy[0]+[x0,y0];scores[t]=kp.scores[0]
                a,b,c,d=square[t].astype(int);side=max(c-a,d-b)
                if side<=0 or side>4*1408:raise ValueError('Unreasonable proposed square crop')
                padded=np.zeros((side,side,3),np.uint8)
                lo=np.maximum([a,b],0);hi=np.minimum([c,d],[1408,1408])
                if np.all(hi>lo):padded[lo[1]-b:hi[1]-b,lo[0]-a:hi[0]-a]=im[lo[1]:hi[1],lo[0]:hi[0]]
                cls[t]=dino.batch_cls([Image.fromarray(padded)])[0].cpu().numpy()
            if t%50==0:print(f'ViTPose/DINOv2: {t+1}/{T}',flush=True)
    finally:video.release()
    atomic_npz(pair/'vitpose_egonet.npz',vitpose_kp=xy,vitpose_scores=scores)
    atomic_npz(pair/'dino_features_egonet.npz',dino_cls=cls)
    del vit,dino
    if device=='cuda':torch.cuda.empty_cache()
    dataset.include_exo=True
    entry=dataset._load_seq(pair)  # use official decoding; bypass unrelated GT-filename discovery guard
    full=dataset._make_sample_test(entry)
    print('Running official pretrained fusion (DINOv2 substitution)...',flush=True)
    with torch.inference_mode():
        model.feed_data(default_collate([full]));model.test()
        predicted=model.current_prediction();reference=model.current_gt()
        pred_np=predicted['position'].cpu().numpy();gt_np=reference['position'].cpu().numpy()
        pose_np={k:predicted[k].cpu().numpy() for k in ('root_orient','pose_body','trans')}
    if pred_np.shape!=gt_np.shape or pred_np.shape!=(n,22,3) or not np.isfinite(pred_np).all():
        raise ValueError('Invalid prediction/reference alignment')
    atomic_npz(out/'predictions.npz',predicted=pred_np,reference=gt_np,egonet=ego,
        raw_frame_ids=np.arange(2,n+2),**pose_np)
    record.update(evaluated_frames=n,raw_frames=T,omitted_tail_frames=T-1-n,
        exo_crop_valid_frames=int(valid[1:n+1].sum()),mpjpe_mm=float(np.linalg.norm(pred_np-gt_np,axis=-1).mean()*1000))
    atomic_json(out/'demo_protocol.json',record)
    from .official_demo_visuals import render_demo
    render_demo(out,pred_np,gt_np,ego,pair/'observer_full.mp4',disclaimer)
    print(json.dumps({'output':str(out),'mpjpe_mm':record['mpjpe_mm'],'scope':disclaimer},indent=2),flush=True)
    return record


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',required=True);p.add_argument('--output',required=True)
    p.add_argument('--repo',default='external/EgoExoMoCap');p.add_argument('--checkpoint')
    p.add_argument('--wearer',default='aria01');p.add_argument('--observer',default='aria03')
    p.add_argument('--points',type=int,choices=[1,3],default=3);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    run_demo(**vars(p.parse_args()))


if __name__=='__main__':main()
