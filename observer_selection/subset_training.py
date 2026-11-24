"""Exact subset costs under a frozen, shared reconstruction model."""
import json
import random

import numpy as np
import torch

from .preparation_cache import atomic_json, atomic_npz
from .triangulation import triangulate_views


def masks_for(count):
    if not 1 <= count <= 8:
        raise ValueError('Exact training supports 1 to 8 candidates')
    return np.array([[(mask>>j)&1 for j in range(count)] for mask in range(1<<count)],dtype=bool)


def expected_cost_loss(margins, errors, masks, required, cost, temperature=.0005):
    legal = masks[:,required].all(-1) if required else torch.ones(len(masks),dtype=torch.bool,device=margins.device)
    entries = masks[legal].to(margins.dtype)
    energies = entries @ margins
    logp = torch.log_softmax(energies,dim=0)
    p = logp.exp()
    measured = errors[legal] + cost*entries.sum(-1)
    # A constant offset improves numerical scale without changing gradients.
    centered = (measured-measured.min())/.005
    return (p*centered).sum() + temperature/.005*(p*logp).sum()


@torch.no_grad()
def build_targets(data, model, directory, stride, adapter_hash, replay_hash):
    frames, ids, _, _, _, sparse, truth, rotations = data
    masks = masks_for(len(frames[0].candidates))
    window = model.window
    if stride < 1 or window % stride:
        raise ValueError('Label stride must divide the reconstruction window')
    count = len(sparse)//window*window
    contract = json.dumps({'adapter_sha256':adapter_hash, 'replay_sha256':replay_hash,
                           'sources':list(frames[0].candidates),
                           'target_id':frames[0].observation.target_id,'stride':stride,
                           'frames':count,'version':2,'background':'seeded_random_two'},sort_keys=True)
    cache = directory/'subset_cost_cache'
    cache.mkdir(exist_ok=True)
    errors = np.full((count,len(masks)),np.nan,dtype='f4')
    for start in range(0,count,window):
        stop = start+window
        file = cache/f'window_{start//window:04}.npz'
        if file.exists():
            with np.load(file,allow_pickle=False) as saved:
                if str(saved['contract'].item())!=contract:
                    raise ValueError('Subset costs differ from the frozen model/config')
                values=saved['errors_m'].copy()
            if values.shape!=(window//stride,len(masks)) or not np.isfinite(values).all():
                raise ValueError(f'Invalid subset cost cache: {file}')
            errors[start:stop:stride]=values
            continue
        geometry, states = [], []
        for index in range(start+1,stop+1):
            obs = frames[index].observation
            parts=[]
            for mask in masks:
                packets=tuple(frames[index].packets[source] for source,take in
                              zip(frames[index].candidates,mask) if take)
                views=(obs,*packets)
                parts.append(triangulate_views(views,truth[index,15],rotations[index],
                                               [v.confidence for v in views]))
            geometry.append(np.stack([part[0] for part in parts]))
            states.append(np.stack([part[1] for part in parts]))
        geometry,states=np.stack(geometry),np.stack(states)
        rng=random.Random(2026+start)
        background=[]
        for _ in range(window):
            selected=rng.sample(range(masks.shape[1]),min(2,masks.shape[1]))
            background.append(sum(1<<j for j in selected))
        points=geometry[np.arange(window),background]
        status=states[np.arange(window),background]
        target=torch.as_tensor(truth[start+1:stop+1],device=model.device)
        floor=truth[start+1:stop+1,[10,11],2].min()
        for t in range(0,window,stride):
            for j in range(len(masks)):
                changed,changed_states=points.copy(),status.copy()
                changed[t],changed_states[t]=geometry[t,j],states[t,j]
                prediction=model.reconstruct(sparse[start:stop],truth[start+1:stop+1,15],
                    changed,changed_states,(changed_states!=0).any(-1),floor)
                errors[start+t,j]=torch.linalg.vector_norm(prediction[t]-target[t],dim=-1).mean().item()
            print(f'Subset costs: raw frame {ids[start+t+1]} | {len(masks)} alternatives | window {start//window+1}/{count//window}',flush=True)
        measured=errors[start:stop:stride]
        if not np.isfinite(measured).all():
            raise FloatingPointError('Nonfinite subset reconstruction costs')
        atomic_npz(file,errors_m=measured,contract=np.array(contract))
    sampled=errors[::stride]
    curve=[]
    sizes=masks.sum(-1)
    for size in range(masks.shape[1]+1):
        costs=sampled[:,sizes==size]
        curve.append({'requests':size,'best_per_sample_mm':float(costs.min(-1).mean()*1000),
                      'uniform_subset_mean_mm':float(costs.mean()*1000)})
    atomic_json(directory/'subset_diagnostic.json',{
        'scope':'Offline sampled counterfactuals; each tested frame changes inside a fixed random-two background',
        'samples':len(sampled),'subsets_per_sample':len(masks),'cost_by_count':curve,
        'warning':'Best per sample uses reference labels. These values describe cached tests, not a deployed policy.'})
    return errors


def fit_epoch(data, model, selector, config, targets, optimiser, epoch, cost):
    from .rollout import Rollout

    frames, _, widths, fixed, raw, _, truth, rotations = data
    rollout=Rollout(selector,frames,raw,truth[:,15],rotations,widths,fixed,
                    config,2026+epoch)
    masks=torch.as_tensor(masks_for(len(frames[0].candidates)),device=model.device)
    losses=[]
    for start in range(0,len(targets),model.window):
        terms=[]
        for index in range(start+1,start+model.window+1):
            required=rollout.step(frames[index],index,'sample').required
            measured=targets[index-1]
            if np.isfinite(measured).all():
                errors=torch.as_tensor(measured,device=model.device)
                forced=[frames[index].candidates.index(source) for source in required]
                terms.append(expected_cost_loss(rollout.latest_scores-rollout.latest_local_score,
                                                errors,masks,forced,cost))
        if terms:
            loss=torch.stack(terms).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite subset expected loss')
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(),1.,error_if_nonfinite=True)
            optimiser.step()
            losses.append(loss.item())
        rollout.detach()
    return {'subset_loss':float(np.mean(losses)),'optimizer_updates':len(losses)}
