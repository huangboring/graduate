"""Offline counterfactual costs and supervised source-score training."""
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from .egohumans import _hash
from .preparation_cache import atomic_npz
from .triangulation import triangulate_views


@torch.no_grad()
def build_targets(data, model, directory, stride, adapter_hash):
    frames, ids, _, _, _, sparse, truth, rotations = data
    window = model.window
    count = len(sparse)//window*window
    cache = directory/'counterfactual_targets.npz'
    contract = json.dumps({'adapter_sha256':adapter_hash,'stride':stride,
                           'frames':count,'target_version':1},sort_keys=True)
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:
            if str(saved['contract'].item())!=contract:
                raise ValueError('Counterfactual target cache differs from the frozen model or sampling')
            gains = saved['gains_m'].copy()
        if gains.shape!=(count,len(frames[0].candidates)) or not np.isfinite(gains[::stride]).all():
            raise ValueError('Invalid counterfactual target cache')
        return gains
    sources = frames[0].candidates
    gains = np.full((count,len(sources)),np.nan,dtype='f4')
    for start in range(0,count,window):
        stop = start+window
        local_points, local_states = [], []
        candidates = []
        for raw in range(start+1,stop+1):
            obs = frames[raw].observation
            args = truth[raw,15],rotations[raw]
            values,state = triangulate_views((obs,),*args,[obs.confidence])
            local_points.append(values); local_states.append(state)
            candidates.append([
                triangulate_views((obs,frames[raw].packets[source]),*args,
                                  [obs.confidence,frames[raw].packets[source].confidence])
                for source in sources])
        points,states = np.stack(local_points),np.stack(local_states)
        floor = truth[start+1:stop+1,[10,11],2].min()
        target = torch.as_tensor(truth[start+1:stop+1],device=model.device)
        def error(values,status):
            predicted = model.reconstruct(sparse[start:stop],truth[start+1:stop+1,15],
                                           values,status,(status!=0).any(axis=-1),floor)
            return torch.linalg.vector_norm(predicted-target,dim=-1).mean(-1).cpu().numpy()
        baseline = error(points,states)
        for t in range(0,window,stride):
            for j in range(len(sources)):
                changed, changed_states = points.copy(),states.copy()
                changed[t],changed_states[t] = candidates[t][j]
                # Only this frame's Exo observation changes; all other tokens are fixed.
                measured = error(changed,changed_states)
                gains[start+t,j] = baseline[t]-measured[t]
            if t%(stride*4)==0:
                print(f'Counterfactual labels: frame {ids[start+t+1]}, window {start//window+1}/{count//window}',flush=True)
    if not np.isfinite(gains[::stride]).all():
        raise FloatingPointError('Nonfinite counterfactual costs')
    atomic_npz(cache,gains_m=gains,contract=np.array(contract),raw_frame_ids=np.array(ids[1:count+1]))
    return gains


def score_loss(margins, gains, cost, scale=.005):
    target = gains/scale
    regression = F.smooth_l1_loss(margins,target)
    gate = F.binary_cross_entropy_with_logits(margins-cost/scale,(gains>cost).to(margins.dtype))
    left,right = torch.triu_indices(len(gains),len(gains),offset=1,device=gains.device)
    differences = gains[left]-gains[right]
    keep = differences.abs()>.0005
    rank = margins.sum()*0
    if keep.any():
        rank = F.softplus(-differences[keep].sign()*(margins[left[keep]]-margins[right[keep]])).mean()
    return regression+gate+rank


def fit_epoch(data, model, selector, config, targets, optimiser, epoch, cost):
    from .fit_triangulated import Rollout

    frames, _, widths, fixed, raw, _, truth, rotations = data
    # Scores estimate gain relative to local-only, measured in units of 5 mm.
    selector.score_cost = cost/.005
    rollout = Rollout(selector,model,frames,raw,truth[:,15],rotations,widths,fixed,
                      config,random.Random(2026+epoch),'vitpose')
    losses = []
    updates = 0
    count = len(targets)
    for start in range(0,count,model.window):
        terms = []
        for index in range(start+1,start+model.window+1):
            rollout.step(frames[index],index,'sample')
            gain = targets[index-1]
            if np.isfinite(gain).all():
                y = torch.as_tensor(gain,device=model.device)
                terms.append(score_loss(rollout.latest_scores-rollout.latest_local_score,y,cost))
        if terms:
            loss = torch.stack(terms).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite source-ranking loss')
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(),1.,error_if_nonfinite=True)
            optimiser.step()
            losses.append(loss.item())
            updates += 1
        rollout.detach()
    return {'score_loss':float(np.mean(losses)),'optimizer_updates':updates}
