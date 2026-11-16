"""Stage 1: train the adapted reconstructor. Stage 2 lives in training.py.

Random acquired subsets vary across frames. Missing history stays missing;
this stage is observation-dropout training, not a learned selection policy.
"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import random
import numpy as np
import torch

from .local_fusion import (LocalFusionConfig, LocalAuxFusionNet, build_local_batch,
                          load_body_decoder, OUTPUT_AXES, CONVENTION)
from .backends import _file_sha256
from .protocol import load_protocol
from .replay import load_sequence


def validate_reference(frame):
    if (frame.coordinate_frame != 'root_relative'
            or frame.observation.backend_context.get('output_axes') != OUTPUT_AXES
            or frame.observation.backend_context.get('coordinate_convention') != CONVENTION):
        raise ValueError('Local-aux GT must use last requester camera axes and joint0 root-relative metres')
    gt = np.asarray(frame.gt)
    valid = np.asarray(frame.gt_valid)
    if gt.shape != (22,3) or valid.shape != (22,) or not np.isin(valid,[0,1]).all() or not valid.any():
        raise ValueError('Need 22-joint reference with at least one valid joint')
    if not valid[0] or not np.allclose(gt[0],0,atol=1e-6) or not np.isfinite(gt[valid.astype(bool)]).all():
        raise ValueError('Invalid finite/root-relative reference')


def check_backend_splits(backend, partitions):
    """Prevent downstream held-out scenes being learned by the reconstructor."""
    learned = backend.provenance.get('training_data_manifest', {})
    if not learned: return
    backend_train=learned.get('train',[])
    backend_val=learned.get('validation',[])
    for split,rows in partitions.items():
        forbidden = backend_train if split=='validation' else backend_train+backend_val if split=='test' else []
        for row in rows:
            if any(row['scene_id']==r['scene_id'] or row['sha256']==r['sha256'] for r in forbidden):
                raise ValueError(f'{split} overlaps reconstructor training/model selection data')
    known=backend_train+backend_val
    if known and any(row['is_synthetic'] != known[0]['is_synthetic'] for rows in partitions.values() for row in rows):
        raise ValueError('Cannot mix synthetic and real reconstruction/selection training')


def train_local_reconstructor(protocol_path, output, repository, body_model_path,
                              config=None, epochs=8, lr=1e-4, max_sources=2,
                              device='cpu', seed=7):
    if epochs<1 or lr<=0 or max_sources<1: raise ValueError('Positive training parameters required')
    config=config or LocalFusionConfig()
    parts, manifests=load_protocol(protocol_path)
    torch.manual_seed(seed)
    net=LocalAuxFusionNet(repository,config).to(device)
    decode=load_body_decoder(repository,body_model_path,device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr)
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use a fresh reconstruction run directory')
    out.mkdir(parents=True,exist_ok=True)
    from . import local_fusion
    dependencies={
        'network_source_sha256':_file_sha256(Path(repository)/'networks/egoexomocap.py'),
        'transform_source_sha256':_file_sha256(Path(repository)/'utils/utils_transform.py'),
        'body_model_sha256':_file_sha256(Path(body_model_path)),
        'adapter_source_sha256':_file_sha256(Path(local_fusion.__file__))}
    manifest={'format':'local_aux_v1','config':asdict(config), 'dependencies':dependencies,
        'training_data_manifest':{k:manifests[k] for k in ('train','validation')},
        'seed':seed,'epochs':epochs,'lr':lr,'max_sources':max_sources,
        'initialization':'fresh; no pretrained AvatarNet weights imported',
        'loss':'last-frame joint0-root-relative MPJPE in metres, first22 incl root, no PA',
        'training_acquisition':'each frame uniform count 0..min(max_sources,N), uniform subset; actual acquired history only',
        'output_axes':OUTPUT_AXES,'target_tracker_required':False,'neutral_shape':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

    def run(paths, training, epoch):
        net.train(training)
        rng=random.Random(seed+epoch if training else seed+100000)
        scenes={}; updates=0
        for path in paths:
            seq=load_sequence(path)
            history=[]; errors=[]
            for frame in seq:
                validate_reference(frame)
                ids=tuple(sorted(rng.sample(list(frame.candidates),rng.randint(0,min(max_sources,len(frame.candidates))))))
                packets=tuple(frame.packets[s] for s in ids)
                if len(history)+1>=config.window_size:
                    batch,_=build_local_batch(frame.observation,packets,tuple(history),config.window_size,device)
                    with torch.set_grad_enabled(training):
                        pred=decode(net(batch))[0]
                        valid=torch.as_tensor(frame.gt_valid,dtype=torch.bool,device=device)
                        # Filter invalid GT before arithmetic (NaN * zero is NaN).
                        truth=torch.as_tensor(frame.gt,dtype=pred.dtype,device=device)
                        loss=torch.linalg.vector_norm(pred[valid]-truth[valid],dim=-1).mean()
                        if not torch.isfinite(loss): raise ValueError('Non-finite reconstruction loss')
                        if training:
                            optimizer.zero_grad(set_to_none=True);loss.backward()
                            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0,error_if_nonfinite=True)
                            optimizer.step();updates+=1
                    errors.append(float(loss.detach()))
                else:
                    # Validate warmup inputs too; malformed frames are not cached.
                    from .types import WarmupError
                    try: build_local_batch(frame.observation,packets,tuple(history),config.window_size,device)
                    except WarmupError: pass
                history.append(SimpleNamespace(observation=frame.observation,packets=packets))
                history=history[-max(1,config.window_size-1):]
            if not errors: raise ValueError(f'Sequence shorter than reconstruction window: {path}')
            scene=next(r['scene_id'] for r in manifests['train' if training else 'validation'] if r['path']==str(Path(path).resolve()))
            scenes.setdefault(scene,[]).append(float(np.mean(errors)))
        return float(np.mean([np.mean(v) for v in scenes.values()])),updates

    best=float('inf'); records=[]
    for epoch in range(epochs):
        train_error,updates=run(parts['train'],True,epoch)
        val_error,_=run(parts['validation'],False,0)
        records.append({'epoch':epoch,'train_mpjpe_mm':train_error*1000,
                        'validation_scene_macro_mpjpe_mm':val_error*1000,'updates':updates})
        state={**manifest,'state_dict':net.state_dict(),'epoch':epoch,'validation_mpjpe_m':val_error}
        torch.save(state,out/'last.pt')
        if val_error<best:
            best=val_error;torch.save(state,out/'best.pt')
        (out/'metrics.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    return {'checkpoint':str((out/'best.pt').resolve()),'validation_mpjpe_mm':best*1000,
            'test_evaluated':False,'note':'Adapted reconstruction checkpoint; freeze for selector training'}
