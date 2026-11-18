"""Train and report causal view acquisition with geometric Exo fusion."""
import argparse
import json
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .features import demand_features, geometry_features, packet_features
from .networks import SelectorNetwork
from .preparation_cache import atomic_json, atomic_npz
from .replay import ReplayProvider, load_sequence
from .selection import AcquisitionConfig, plan_refresh
from .single_exo import SingleExo
from .triangulation import _camera, camera_geometry, triangulate_views


def choose(scores, candidates, required, budget, mode, rng):
    """One fixed batch before acquisition. Sample individual actions with STOP."""
    free = [i for i, source in enumerate(candidates) if source not in required]
    chosen = list(required)
    if mode == 'all':
        return candidates, None
    if mode == 'local':
        return (), None
    terms = []
    if mode == 'sample':
        while free and len(chosen) < budget:
            logits = torch.cat((scores[free], scores.new_zeros(1)))
            log_probs = torch.log_softmax(logits, dim=0)
            action = rng.choices(range(len(logits)), weights=log_probs.detach().exp().cpu().tolist())[0]
            terms.append(log_probs[action])
            if action == len(free):
                break
            chosen.append(candidates[free.pop(action)])
    else:
        ranked = sorted(free, key=lambda i: (-float(scores[i]), candidates[i]))
        chosen.extend(candidates[i] for i in ranked[:budget-len(chosen)] if float(scores[i]) > 0)
    return tuple(s for s in candidates if s in chosen), torch.stack(terms).sum() if terms else None


class AppearanceSelector(SelectorNetwork):
    def __init__(self):
        super().__init__(13)
        self.appearance = torch.nn.Sequential(torch.nn.Linear(768,32), torch.nn.LayerNorm(32))
        self.demand_encoder = self._encoder(11*self.joints+5+32,self.hidden)
        self.observation_encoder = self._encoder(5*self.joints+1+32,self.hidden)

    def _appearance(self, data, key):
        value = data.get(key)
        if value is None:
            return next(self.parameters()).new_zeros(32)
        return self.appearance(self._tensor(value,768,'DINOv2 appearance'))

    def demand(self, observation, previous=None):
        features = demand_features(observation,previous,self.joints)
        tensors = (self._tensor(features,11*self.joints+5,'demand'),
                   self._appearance(observation.backend_context,'local_dino_cls'))
        return self.demand_encoder(torch.cat(tensors))

    def update(self, packet, old_h, last_timestamp=None):
        features = packet_features(packet,last_timestamp,self.joints)
        tensors = (self._tensor(features,5*self.joints+1,'packet'),
                   self._appearance(packet.payload,'dino_cls'))
        encoded = self.observation_encoder(torch.cat(tensors))
        return self.memory_cell(encoded,old_h)


class Rollout:
    def __init__(self, selector, model, frames, raw, head, rotations, widths, fixed, config, rng):
        self.selector, self.model = selector, model
        self.candidates = frames[0].candidates
        self.config, self.rng = config, rng
        self.head, self.rotations = head, rotations
        self.widths, self.fixed = widths, fixed
        self.memory, self.cameras, self.last = {}, {}, {}
        self.local = deque(maxlen=4)
        self.first = dict.fromkeys(self.candidates, frames[0].observation.timestamp)
        self.raw = raw
        self._accept(frames[0], self.candidates)
        self.local.append(frames[0].observation)

    def _accept(self, frame, selected):
        if not selected:
            return ()
        now = frame.observation.timestamp
        packets = ReplayProvider(frame).acquire_many(selected, frame.observation.target_id, now)
        for packet in packets:
            source = packet.observer_id
            hidden = self.memory.get(source)
            if hidden is None:
                hidden = self.selector.initial_state()
            self.memory[source] = self.selector.update(packet, hidden, self.last.get(source))
            self.cameras[source] = _camera(packet)
            self.last[source] = now
        return packets

    def step(self, frame, index, mode, count=None):
        obs = frame.observation
        now = obs.timestamp
        self.local.append(obs)
        if mode in ('all','local'):
            selected, logp = choose(None, self.candidates, (), 0, mode, self.rng)
        elif mode == 'random':
            selected = tuple(self.rng.sample(self.candidates,count))
            logp = None
        else:
            demand = self.selector.demand_sequence(list(self.local))
            refresh = plan_refresh(self.candidates, now, self.first, self.last, self.config)
            scores = []
            for source in self.candidates:
                geo = camera_geometry(_camera(obs), self.cameras[source], self.last[source],
                    source in self.fixed, now, self.widths[self.raw], self.widths[source])
                scores.append(self.selector.score(demand, self.memory[source], geometry_features(geo, now),
                    now-self.last[source], True))
            selected, logp = choose(torch.stack(scores), self.candidates, refresh.required_ids,
                                    min(self.config.max_sources,len(self.candidates)), mode, self.rng)
        packets = self._accept(frame, selected)
        views = (obs, *packets)
        confidences = [self.model.confidence(view) for view in views]
        points, state = triangulate_views(views, self.head[index], self.rotations[index], confidences)
        return points, state, selected, logp

    def detach(self):
        self.memory = {source: state.detach() for source,state in self.memory.items()}


def _run(data, model, selector, config, mode, seed, windows, optimiser=None, cost=.005, counts=None):
    frames, ids, widths, fixed, raw, sparse, truth, rotations = data
    results = []
    all_predictions, all_states = [], []
    head = truth[:,15]
    rollout = Rollout(selector,model,frames,raw,head,rotations,widths,fixed,config,random.Random(seed))
    running = None
    for start in range(0, windows*model.window, model.window):
        stop = start+model.window
        rows = [rollout.step(frames[i],i,mode,counts[i-1] if counts is not None else None)
                for i in range(start+1,stop+1)]
        points, states, selected, logps = zip(*rows)
        states = np.stack(states)
        valid = (states!=0).any(axis=1)
        floor = truth[start+1:stop+1,[10,11],2].min()
        target = torch.as_tensor(truth[start+1:stop+1],device=model.device)
        predicted = model.reconstruct(sparse[start:stop],head[start+1:stop+1],
                                      np.stack(points),states,valid,floor)
        error = torch.linalg.vector_norm(predicted-target,dim=-1).mean(-1)
        if not torch.isfinite(error).all():
            raise FloatingPointError('Nonfinite reconstruction; check calibration and checkpoint compatibility')
        if optimiser is not None:
            # One window reconstruction gives a shared reward because temporal fusion
            # uses all tokens. STOP/observations are sampled with scalar source scores.
            objective = error.detach().mean().item() + cost*np.mean([len(s) for s in selected])
            previous_baseline = 0. if running is None else running
            running = .9*previous_baseline + .1*objective
            terms = [term for term in logps if term is not None]
            policy_loss = (torch.stack(terms).sum() / model.window) * (objective - previous_baseline) if terms else 0.
            loss = error.mean() + policy_loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*selector.parameters(),*model.adapter.parameters()],1.,error_if_nonfinite=True)
            optimiser.step()
            rollout.detach()
        all_predictions.extend(predicted.detach().cpu().numpy())
        all_states.extend(states)
        for n, (names, e, st) in enumerate(zip(selected,error.detach().cpu().tolist(),states),start+1):
            results.append({'frame':ids[n],'selected':list(names),'mpjpe_mm':e*1000,
                            'triangulated_joints':int((st==2).sum())})
    return results,np.asarray(all_predictions),np.asarray(all_states)


def train(prepared, sequence, output, repo, checkpoint, device, epochs, budget, cost, max_windows, resume, render):
    from .selection_video import export_outputs, image_paths

    torch.manual_seed(2026)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python environment')
    prepared, sequence, output, repo = map(lambda p:Path(p).resolve(),(prepared,sequence,output,repo))
    checkpoint = Path(checkpoint).resolve() if checkpoint else repo/'model_zoo/egohumans_syn_tracker_3pt/model.pth'
    manifest = json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    if (manifest['scene_id'],manifest['sequence_id']) != (sequence.parent.name,sequence.name):
        raise ValueError('Prepared data and raw sequence identify different sessions')
    from .preparation_quality import check_preparation
    quality = check_preparation(prepared/'replay.npz')
    frames = load_sequence(prepared/'replay.npz')
    raw, target = manifest['requester_id'], manifest['target_id']
    candidates = frames[0].candidates
    widths, ids = {}, None
    for source in (raw,*candidates):
        pair = prepared/source
        meta = json.loads((pair/'meta.json').read_text(encoding='utf-8'))
        grid = meta['raw_frame_ids']
        if ids is not None and grid!=ids:
            raise ValueError(f'Raw frame grid differs for {source}')
        ids = grid
        with np.load(pair/'observer.npz') as camera:
            widths[source] = int(camera['observer_img_size'][0])
    if len(frames)!=len(ids) or ids!=list(range(ids[0],ids[0]+len(ids))):
        raise ValueError('Prepared replay and raw frame IDs must align and be consecutive')
    fixed = {source for source in candidates if (sequence/'exo'/source).is_dir()}
    images = image_paths(sequence,(raw,*candidates),ids) if render else {}
    model = SingleExo(repo,checkpoint,device)
    windows = min((len(frames)-1)//model.window,max_windows or len(frames))
    if windows<1:
        raise ValueError(f'Need an initialization frame and at least {model.window} aligned frames')
    sparse, truth, rotations = model.load_tracking(sequence,target,ids[:1+windows*model.window])
    data = frames, ids, widths, fixed, raw, sparse, truth, rotations
    selector = AppearanceSelector().to(device)
    optimiser = torch.optim.AdamW([*selector.parameters(),*model.adapter.parameters()],lr=3e-4)
    from .egohumans import _hash
    config = AcquisitionConfig(max_sources=budget)
    contract={'prepared':str(prepared),'replay_sha256':_hash(prepared/'replay.npz'),
              'checkpoint_sha256':_hash(checkpoint),'sequence':str(sequence),'epochs_schema':1,
              'budget':budget,'cost_m_per_source':cost,'windows':windows,
              'crop_provenance':frames[0].metadata.get('crop_provenance'),
              'scope':'training-sequence only, synthetic target trackers and reference-derived floor'}
    saved = output/'checkpoint.pt'
    start, history = 0,[]
    if resume:
        previous=json.loads((output/'experiment.json').read_text(encoding='utf-8'))
        if 'crop_provenance' not in previous and contract['crop_provenance']=='local_detector':
            previous['crop_provenance']='local_detector'
        if previous!=contract:
            raise ValueError('Resume inputs/config differ; choose another output')
        if saved.exists():
            state=torch.load(saved,map_location=device,weights_only=True)
            selector.load_state_dict(state['selector']); model.adapter.load_state_dict(state['adapter'])
            optimiser.load_state_dict(state['optimiser'])
            start,history=state['epoch'],state['history']
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a fresh output or use --resume')
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'experiment.json',contract)
    atomic_json(output/'preparation_quality.json',quality)
    if start>epochs:
        raise ValueError('Requested final epoch is lower than saved checkpoint epoch')
    for epoch in range(start,epochs):
        rows,_,_=_run(data,model,selector,config,'sample',2026+epoch,windows,optimiser,cost)
        history.append({'epoch':epoch+1,'mpjpe_mm':float(np.mean([r['mpjpe_mm'] for r in rows])),
                        'mean_requests':float(np.mean([len(r['selected']) for r in rows]))})
        temporary=saved.with_suffix('.tmp')
        torch.save({'selector':selector.state_dict(),'adapter':model.adapter.state_dict(),
                    'optimiser':optimiser.state_dict(),'epoch':epoch+1,'history':history},temporary)
        temporary.replace(saved)
        print(history[-1],flush=True)
    with torch.no_grad():
        learned,predictions,states=_run(data,model,selector,config,'learned',2026,windows)
        local,_,_=_run(data,model,selector,config,'local',2026,windows)
        all_views,_,_=_run(data,model,selector,config,'all',2026,windows)
        random_matched,_,_=_run(data,model,selector,config,'random',2026,windows,
                               counts=[len(row['selected']) for row in learned])
    atomic_npz(output/'predictions.npz',raw_frame_ids=np.array(ids[1:len(learned)+1]),
               predicted=predictions,reference=truth[1:],joint_states=states)
    warmup={'frame':ids[0],'selected':list(candidates),'mpjpe_mm':None,'triangulated_joints':None}
    export_outputs(output,history,learned,local,all_views,warmup,candidates,raw,predictions,images,
                   random_matched=random_matched,
                   omitted_frames=len(ids)-len(learned)-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','sequence','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--repo',default='external/EgoExoMoCap')
    parser.add_argument('--checkpoint')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--budget',type=int,default=2)
    parser.add_argument('--cost',type=float,default=.005,help='Cost in MPJPE metres per extra request')
    parser.add_argument('--max-windows',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--no-video',action='store_false',dest='render')
    args=parser.parse_args()
    if args.epochs<1 or args.budget<1 or not math.isfinite(args.cost) or args.cost<0 or (
            args.max_windows is not None and args.max_windows<1):
        parser.error('Positive epochs/budget/windows and finite nonnegative cost required')
    train(**vars(args))


if __name__=='__main__':
    main()
