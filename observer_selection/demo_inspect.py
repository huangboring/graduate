"""Inspect an existing official demo without repeating image preprocessing."""
from pathlib import Path
import json,csv,sys
import numpy as np
import torch


class TensorTrace:
    def __init__(self):self.arrays={};self.handles=[]
    def capture(self,name,value):
        if isinstance(value,torch.Tensor):
            self.arrays.setdefault(name,value.detach().cpu().clone().numpy())
        elif isinstance(value,dict):
            for k,v in value.items():self.capture(name+'.'+str(k),v)
        elif isinstance(value,(tuple,list)):
            for i,v in enumerate(value):self.capture(name+'.'+str(i),v)
    def attach(self,net):
        names=['','egonet','egonet.linear_embedding','fusion','fusion.scorenet',
               'fusion.ego_token_embed','fusion.egonet_token_embed','fusion.exo_token_embed',
               'fusion.spatial_encoder','fusion.temporal_encoder',
               'fusion.global_orientation_decoder','fusion.joint_rotation_decoder']
        modules=dict(net.named_modules())
        for name in names:
            if name not in modules:raise ValueError('Missing expected module: '+name)
            label=name or 'AvatarNet'
            self.handles.append(modules[name].register_forward_pre_hook(
                lambda m,args,kwargs,n=label:(self.capture(n+'.input',args),self.capture(n+'.kwargs',kwargs)) and None,with_kwargs=True))
            self.handles.append(modules[name].register_forward_hook(
                lambda m,args,out,n=label:self.capture(n+'.output',out)))
    def close(self):
        for h in self.handles:h.remove()
    def save(self,out):
        np.savez_compressed(out/'module_arrays.npz',**self.arrays)
        rows={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sample':v.reshape(-1)[:12].tolist(),
                 'min':float(v.min()) if v.size else None,'max':float(v.max()) if v.size else None}
              for k,v in self.arrays.items()}
        (out/'module_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        return rows


def inspect_demo(run,repo,device='cuda'):
    import yaml
    from .official_demo import strict_weights
    from .egohumans import _hash
    run,repo=Path(run).resolve(),Path(repo).resolve()
    record=json.loads((run/'demo_protocol.json').read_text())
    with np.load(run/'predictions.npz') as z:raw_ids=z['raw_frame_ids'].copy()
    out=run/'inspection'
    if out.exists():raise FileExistsError('Inspection already exists; preserve it or rename it before rerunning')
    ckpt=Path(record['checkpoint'])
    if _hash(ckpt)!=record['checkpoint_sha256']:raise ValueError('Checkpoint differs from demo')
    sys.path.insert(0,str(repo))
    from models.model_egoexomocap import ModelEgoExoMoCap
    from data_loaders.dataset_nymeria import Nymeria_Dataset
    opt=yaml.safe_load((repo/f"options/test_egohumans_syn_tracker_{record['points']}pt.yaml").read_text())
    opt['is_train']=False;opt['gpu_ids']=[0] if device=='cuda' else None
    opt['body_model']['smpl_path']=str(repo/'third_party')
    model=ModelEgoExoMoCap(opt);net=model.get_bare_model(model.net)
    strict_weights(net,ckpt);net.eval()
    pairs=list((run/'dataset').iterdir())
    if len(pairs)!=1:raise ValueError('Expected exactly one demo pair')
    pair=pairs[0];dopt=dict(opt['datasets']['test'])
    dopt.update(dataroot=[str(run/'dataset')],include_exo=False)
    ds=Nymeria_Dataset(dopt);ds.include_exo=True
    sample=ds._make_sample_test(ds._load_seq(pair));ws=int(dopt['window_size'])
    data={dst:sample[src][:ws].unsqueeze(0).to(device) for dst,src in
          [('sparse_input','sparse'),('exo_rays','exo_rays'),('dino_cls','dino_cls'),('bbox_valid','bbox_valid'),('betas','betas_gt')]}
    data.update(fk=model.fk,body_model=model.bm)
    trace=TensorTrace();trace.attach(net)
    try:
        with torch.inference_mode():net(data)
    finally:trace.close()
    out.mkdir();summary=trace.save(out)
    root=Path(record['sequence']);observer=record['observer'];wearer=record['wearer']
    with (out/'frame_map.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['demo_index_0based','raw_frame_id','prediction_time_s','observer_image','smpl_file','smpl_person_key','supplied_2d_annotations'])
        for i,raw in enumerate(raw_ids):
            w.writerow([i,int(raw),i/20,str(root/'ego'/observer/'images/rgb'/f'{raw:05}.jpg'),
                str(root/'processed_data/smpl'/f'{raw:05}.npy'),wearer,
                str(root/'processed_data/poses2d'/observer/'rgb'/f'{raw:05}.npy')])
    raw=int(raw_ids[0]);truth=np.load(root/'processed_data/smpl'/f'{raw:05}.npy',allow_pickle=True).item()[wearer]
    np.savez_compressed(out/'first_frame_annotation.npz',**{k:np.asarray(v) for k,v in truth.items() if np.asarray(v).dtype!=object})
    (out/'README.txt').write_text(
        'Recorded first official inference window only: raw frames 2..'+str(ws+1)+'.\n'
        'module_arrays.npz: full tensors. module_summary.json: shape/dtype/min/max/first12 values.\n'
        'Hooks copy inputs BEFORE forward, because official EgoNet modifies its sparse input.\n'
        'Spatial encoder kwargs include src_key_padding_mask (True means masked).\n'
        'B=batch, T=window time, N=observer count, K=13 keypoints, D=embedding width.\n'
        '3pt sparse: 0:18 head/left/right rotation6D; 18:36 relative rotation6D; '
        '36:45 positions; 45:54 per-frame position differences (NOT metres/second).\n'
        '1pt sparse: 0:6 rotation6D; 6:12 relative rotation6D; 12:15 position; 15:18 position difference.\n'
        'Floor normalization uses supplied reference feet. Rotation6D encodes two rotation matrix columns.\n'
        'DINO CLS: learned 768-dimensional feature, not a joint vector. ScoreNet output: sigmoid joint weights, not verified visibility.\n'
        'exo_token_embed input: weighted 13x3 rays plus bbox validity.\n'
        'root decoder: 6D rotation; body decoder: 21x6D rotations. Predictions.npz contains decoded axis-angle and FK joints.\n'
        'first_frame_annotation.npz copies the target entry from the ORIGINAL fitted SMPL file. '
        'SMPL annotation joints are not necessarily the same layout as demo SMPL-H first22 FK joints.\n'
        'Raw fisheye images/2D annotations differ from rectified demo pixel coordinates: do not overlay them directly.\n'
        'Aria identity is a dictionary key, not detector box index. Demo first prediction is raw frame 2, not 1.\n'
        'This is offline windowed inference, not strict streaming. DINOv2 substitution and GT-derived tracker/floor/shape remain disclosed.\n',encoding='utf-8')
    for name,row in summary.items():print(name,row['shape'],row['sample'][:3])
    print('Inspection saved:',out)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--repo',default='external/EgoExoMoCap');p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    inspect_demo(**vars(p.parse_args()))
