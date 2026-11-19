"""Train and report causal view acquisition with geometric Exo fusion."""
import argparse
import json
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .features import (demand_features, geometry_features, packet_features,
                       compact_demand_features, compact_packet_features)
from .networks import SelectorNetwork
from .preparation_cache import atomic_json, atomic_npz
from .replay import ReplayProvider, load_sequence
from .selection import AcquisitionConfig, plan_refresh
from .single_exo import SingleExo
from .triangulation import _camera, camera_geometry, triangulate_views


def choose(scores, local_score, candidates, required, mode, rng):
    free = [i for i, source in enumerate(candidates) if source not in required]
    if not free:
        return tuple(required), None
    margins = scores[free] - local_score
    if mode == 'sample':
        # Each source competes with the same local-only option.
        probabilities = margins.detach().sigmoid().cpu().tolist()
        draws = [rng.random() < probability for probability in probabilities]
        included = torch.as_tensor(draws,device=margins.device,dtype=margins.dtype)
        logp = -torch.nn.functional.binary_cross_entropy_with_logits(margins,included,reduction='sum')
    else:
        draws = (margins.detach()>0).cpu().tolist()
        logp = None
    selected = set(required)
    selected.update(candidates[i] for i, chosen in zip(free, draws) if chosen)
    return tuple(source for source in candidates if source in selected), logp


def matched_random(candidates, learned, rng):
    refresh = set(learned['refresh'])
    available = [source for source in candidates if source not in refresh]
    selected = refresh | set(rng.sample(available,len(learned['selected'])-len(refresh)))
    return tuple(source for source in candidates if source in selected)


class AppearanceSelector(SelectorNetwork):
    def __init__(self, appearance='dino', scorer='mlp', encoding='legacy'):
        super().__init__(13)
        if encoding not in ('legacy','compact'):
            raise ValueError('Unknown feature encoding')
        self.encoding = encoding
        self.demand_size = 71 if encoding=='compact' else 148
        self.packet_size = 41 if encoding=='compact' else 66
        self.scorer_kind = scorer
        if scorer not in ('mlp','attention'):
            raise ValueError('Unknown scorer')
        if scorer == 'attention' and encoding=='legacy':
            # No camera identity/position embedding: support a new camera roster.
            self.source_token = torch.nn.Linear(self.hidden+11,self.hidden)
            self.score_attention = torch.nn.MultiheadAttention(self.hidden,4,batch_first=True,dropout=0.)
            self.score_norm = torch.nn.LayerNorm(self.hidden)
            self.attention_output = torch.nn.Linear(self.hidden,1)
            del self.scorer
        self.appearance_kind = appearance
        inputs = {'dino':768,'cnn':576,'none':0}[appearance]
        size = 32 if inputs else 0
        if scorer == 'mlp' and encoding=='legacy':
            self.local_scorer = torch.nn.Sequential(torch.nn.Linear(2*self.hidden,self.hidden),
                                                   torch.nn.GELU(),torch.nn.Linear(self.hidden,1))
        self.appearance = (torch.nn.Sequential(torch.nn.Linear(inputs,32),torch.nn.LayerNorm(32))
                           if inputs else None)
        self.demand_encoder = self._encoder(self.demand_size+size,self.hidden)
        self.observation_encoder = self._encoder(self.packet_size+size,self.hidden)
        if encoding=='compact':
            # Common tokens and common scoring head for BOTH model variants.
            # Build shared modules before attention so their seeded weights match.
            del self.scorer, self.memory_attention, self.memory_positions
            self.source_token = torch.nn.Linear(self.hidden+10,self.hidden)  # h64 + geometry9 + age1
            self.token_norm = torch.nn.LayerNorm(self.hidden)
            self.shared_scorer = torch.nn.Sequential(
                torch.nn.Linear(2*self.hidden,self.hidden),torch.nn.GELU(),
                torch.nn.Linear(self.hidden,32),torch.nn.GELU(),torch.nn.Linear(32,1))
            if scorer=='attention':
                self.score_attention = torch.nn.MultiheadAttention(self.hidden,4,batch_first=True,dropout=0.)
                self.score_norm = torch.nn.LayerNorm(self.hidden)

    def score_candidates(self, demand, memories, geometries, ages):
        if self.encoding=='compact':
            if not (len(memories)==len(geometries)==len(ages)) or not memories:
                raise ValueError('Need matching nonempty source memories, geometry and ages')
            if any(not math.isfinite(float(t)) or t<0 for t in ages):
                raise ValueError('Source ages must be finite nonnegative seconds')
            raw = torch.stack([torch.cat((self._tensor(h,64,'memory'),self._tensor(g,9,'geometry'),
                               demand.new_tensor([math.log1p(t)])))
                               for h,g,t in zip(memories,geometries,ages)])
            tokens = self.token_norm(torch.cat((demand[None],self.source_token(raw)),dim=0))
            context = tokens
            if self.scorer_kind=='attention':
                attended,_ = self.score_attention(tokens[None],tokens[None],tokens[None],need_weights=False)
                context = self.score_norm(tokens+attended[0])
            scores = self.shared_scorer(torch.cat((demand.expand(len(tokens),-1),context),dim=-1)).squeeze(-1)
            return scores[1:],scores[0]
        if self.scorer_kind == 'mlp':
            return torch.stack([self.score(demand,h,g,age,True)
                                for h,g,age in zip(memories,geometries,ages)])
        tokens = torch.stack([torch.cat((h,self._tensor(g,9,'geometry'),
                              demand.new_tensor([math.log1p(age),1.])))
                              for h,g,age in zip(memories,geometries,ages)])
        tokens = self.source_token(tokens)
        # Local demand is a common token; candidates attend to each other as well.
        tokens = torch.cat((demand[None],tokens),dim=0)[None]
        attended,_ = self.score_attention(tokens,tokens,tokens,need_weights=False)
        context = self.score_norm(tokens+attended)[0]
        scores = self.attention_output(context).squeeze(-1)
        return scores[1:], scores[0]

    def _appearance(self, data, key):
        if self.appearance is None:
            return next(self.parameters()).new_zeros(0)
        if self.appearance_kind=='cnn':
            return self.appearance(self._tensor(data['cnn_cls'],576,'MobileNet appearance'))
        value = data.get(key)
        if value is None:
            return next(self.parameters()).new_zeros(32)
        return self.appearance(self._tensor(value,768,'DINOv2 appearance'))

    def demand(self, observation, previous=None):
        encode = compact_demand_features if self.encoding=='compact' else demand_features
        features = encode(observation,previous,self.joints)
        tensors = (self._tensor(features,self.demand_size,'demand'),
                   self._appearance(observation.backend_context,'local_dino_cls'))
        return self.demand_encoder(torch.cat(tensors))

    def local_score(self, demand, observation, previous):
        if self.scorer_kind == 'attention' or self.encoding=='compact':
            return demand.new_zeros(())  # Replaced by the joint token-scoring result.
        current = self.demand(observation,previous)
        return self.local_scorer(torch.cat((demand,current))).squeeze(-1)

    def update(self, packet, old_h, last_timestamp=None):
        encode = compact_packet_features if self.encoding=='compact' else packet_features
        features = encode(packet,last_timestamp,self.joints)
        tensors = (self._tensor(features,self.packet_size,'packet'),
                   self._appearance(packet.payload,'dino_cls'))
        encoded = self.observation_encoder(torch.cat(tensors))
        return self.memory_cell(encoded,old_h)


class Rollout:
    def __init__(self, selector, model, frames, raw, head, rotations, widths, fixed, config, rng, joint_weights):
        self.selector, self.model = selector, model
        self.candidates = frames[0].candidates
        self.config, self.rng = config, rng
        self.joint_weights = joint_weights
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

    def step(self, frame, index, mode, matched=None):
        obs = frame.observation
        now = obs.timestamp
        self.local.append(obs)
        logp = None
        required = ()
        if mode == 'all':
            selected = self.candidates
        elif mode == 'local':
            selected = ()
        elif mode == 'random':
            required = tuple(matched['refresh'])
            selected = matched_random(self.candidates,matched,self.rng)
        else:
            local = list(self.local)
            demand = self.selector.demand_sequence(local)
            local_score = self.selector.local_score(demand,obs,local[-2] if len(local)>1 else None)
            refresh = plan_refresh(self.candidates, now, self.first, self.last, self.config)
            required = refresh.required_ids
            geometries, ages = [], []
            for source in self.candidates:
                geo = camera_geometry(_camera(obs), self.cameras[source], self.last[source],
                    source in self.fixed, now, self.widths[self.raw], self.widths[source])
                geometries.append(geometry_features(geo,now))
                ages.append(now-self.last[source])
            scored = self.selector.score_candidates(demand,[self.memory[s] for s in self.candidates],geometries,ages)
            if isinstance(scored,tuple):
                scores,local_score = scored
            else:
                scores = scored
            self.latest_scores, self.latest_local_score = scores, local_score
            local_score = local_score + getattr(self.selector,'score_cost',0.)
            if mode=='rank_two':
                free = [i for i,source in enumerate(self.candidates) if source not in required]
                ranked = sorted(free,key=lambda i:(-float(scores[i]),self.candidates[i]))
                selected = (*required,*(self.candidates[i] for i in ranked[:2-len(required)]))
            else:
                selected, logp = choose(scores,local_score,self.candidates,required,mode,self.rng)
        packets = self._accept(frame, selected)
        views = (obs, *packets)
        confidences = ([view.confidence for view in views] if self.joint_weights=='vitpose'
                       else [self.model.confidence(view) for view in views])
        points, state = triangulate_views(views, self.head[index], self.rotations[index], confidences)
        return points, state, selected, logp, required

    def detach(self):
        self.memory = {source: state.detach() for source,state in self.memory.items()}


def _run(data, model, selector, config, mode, seed, windows, optimiser=None, cost=.005, match_rows=None, joint_weights="scorenet"):
    frames, ids, widths, fixed, raw, sparse, truth, rotations = data
    results = []
    all_predictions, all_states = [], []
    head = truth[:,15]
    rollout = Rollout(selector,model,frames,raw,head,rotations,widths,fixed,config,random.Random(seed),joint_weights)
    running = None
    for start in range(0, windows*model.window, model.window):
        stop = start+model.window
        rows = [rollout.step(frames[i],i,mode,match_rows[i-1] if match_rows is not None else None)
                for i in range(start+1,stop+1)]
        points, states, selected, logps, refresh = zip(*rows)
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
        for n, (names, e, st, required) in enumerate(zip(selected,error.detach().cpu().tolist(),states,refresh),start+1):
            results.append({'frame':ids[n],'selected':list(names),'mpjpe_mm':e*1000,'refresh':list(required),
                            'triangulated_joints':int((st==2).sum())})
    return results,np.asarray(all_predictions),np.asarray(all_states)



@torch.no_grad()
def compare_original_fusion(data, model, selector, config, windows, output, joint_weights):
    import csv

    frames, ids, _, _, requester, sparse, truth, rotations = data
    _, triangulated, _ = _run(data,model,selector,config,'all',2026,windows,joint_weights=joint_weights)
    head = truth[:,15]
    original = []
    for start in range(0,windows*model.window,model.window):
        stop=start+model.window
        chunk=frames[start+1:stop+1]
        views=[[f.observation for f in chunk]]
        views.extend([f.packets[source] for f in chunk] for source in chunk[0].candidates)
        floor=truth[start+1:stop+1,[10,11],2].min()
        joints=model.reconstruct_original(sparse[start:stop],head[start+1:stop+1],
                                          rotations[start+1:stop+1],views,floor)
        original.append(joints.cpu().numpy())
    original = np.concatenate(original)
    reference = truth[1:]
    errors = np.stack([np.linalg.norm(x-reference,axis=-1).mean(axis=-1)*1000
                       for x in (original,triangulated)],axis=1)
    if not np.isfinite(errors).all():
        raise FloatingPointError('Nonfinite result in fusion comparison')
    frame_ids = ids[1:1+len(reference)]
    result = {
        'scope':'IN-SAMPLE, same observations, DINOv2 substituted for DINOv3',
        'raw_frame_range':[frame_ids[0],frame_ids[-1]],
        'frames':len(frame_ids),
        'view_ids':[requester,*frames[0].candidates],
        'methods':{'EgoExoMoCap original fusion (DINOv2)':float(errors[:,0].mean()),
                   'Triangulation + single Exo token':float(errors[:,1].mean())},
        'unit':'MPJPE mm','triangulation_weights':joint_weights,
    }
    with (output/'fusion_comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['raw_frame','original_mpjpe_mm','triangulation_mpjpe_mm'])
        writer.writerows(zip(frame_ids,*errors.T))
    atomic_json(output/'fusion_comparison.json',result)
    print(json.dumps(result,indent=2),flush=True)

def train(prepared, sequence, output, repo, checkpoint, device, epochs, cost, max_windows, resume, render, compare_original=False, appearance="dino", joint_weights="scorenet", sensitivity=False, ranking_from=None, label_stride=8, subset_training=False, scorer="mlp", sources=None, evaluate_from=None, save_every=25, target_cache_from=None, encoding="legacy"):
    from .selection_video import export_outputs, image_paths, raw_joint_overlay

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
    from .scorer_experiments import restrict_sources, load_transfer, evaluate_policies
    frames = restrict_sources(frames,sources)
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
    images = image_paths(sequence,(raw,*candidates),ids) if render and not (compare_original or sensitivity) else {}
    model = SingleExo(repo,checkpoint,device)
    windows = min((len(frames)-1)//model.window,max_windows or len(frames))
    if windows<1:
        raise ValueError(f'Need an initialization frame and at least {model.window} aligned frames')
    sparse, truth, rotations = model.load_tracking(sequence,target,ids[:1+windows*model.window])
    if appearance=='cnn':
        from .cnn_features import add_cnn_features
        frames = add_cnn_features(prepared,sequence,raw,candidates,ids,frames,fixed,device)
    data = frames, ids, widths, fixed, raw, sparse, truth, rotations
    selector = AppearanceSelector(appearance,scorer,encoding).to(device)
    from .egohumans import _hash
    config = AcquisitionConfig(max_sources=len(candidates))
    contract={'prepared':str(prepared),'replay_sha256':_hash(prepared/'replay.npz'),
              'checkpoint_sha256':_hash(checkpoint),'sequence':str(sequence),'epochs_schema':2,
              'policy':'above_local_plus_refresh','cost_m_per_source':cost,'windows':windows,
              'crop_provenance':frames[0].metadata.get('crop_provenance'),
              'scope':'training-sequence only, synthetic target trackers and reference-derived floor'}
    if (appearance,joint_weights)!=('dino','scorenet'):
        contract.update(epochs_schema=3,appearance=appearance,joint_weights=joint_weights)
    if sources is not None:
        contract.update(sources=list(candidates),target_id=target,requester_id=raw)
    if encoding=='compact':
        contract.update(encoding='compact',encoding_version=1,scoring_head='shared_q_token_v1',
                        demand_dim=71,packet_dim=41,has_history_feature=False)
    if scorer != 'mlp':
        contract['scorer'] = scorer
    if evaluate_from:
        load_transfer(evaluate_from,selector,model,contract,candidates,sequence,output)
        result = evaluate_policies(data,model,selector,config,windows,joint_weights)
        atomic_json(output/'camera_transfer.json',result)
        print(json.dumps(result,indent=2),flush=True)
        return
    optimiser = torch.optim.AdamW([*selector.parameters(),*model.adapter.parameters()],lr=3e-4)
    if subset_training and not ranking_from:
        raise ValueError('--subset-training requires --ranking-from')
    if ranking_from:
        if appearance!='none' or joint_weights!='vitpose':
            raise ValueError('Ranking experiment uses --appearance none --joint-weights vitpose')
        bootstrap = Path(ranking_from).resolve()
        source_contract = json.loads((bootstrap/'experiment.json').read_text(encoding='utf-8'))
        for key in ('replay_sha256','checkpoint_sha256','sequence','windows','joint_weights'):
            if source_contract.get(key)!=contract.get(key):
                raise ValueError(f'Frozen reconstruction checkpoint differs in {key}')
        if sources is not None and source_contract.get('sources')!=list(candidates):
            raise ValueError('Frozen adapter must have the same training roster; refit without held-out cameras')
        initial = torch.load(bootstrap/'checkpoint.pt',map_location=device,weights_only=True)
        model.adapter.load_state_dict(initial['adapter'])
        model.adapter.requires_grad_(False)
        optimiser = torch.optim.AdamW(selector.parameters(),lr=1e-4)
        contract.update(epochs_schema=4,objective='counterfactual_gain_ranking',
                        frozen_from=str(bootstrap),frozen_sha256=_hash(bootstrap/'checkpoint.pt'),
                        label_stride=label_stride,score_scale_m=.005)
        if subset_training:
            contract.update(epochs_schema=5,objective='exact_subset_expected_cost',
                            target_context='seeded_random_two',entropy_temperature_m=.0005)
        selector.score_cost = 0. if subset_training else cost/.005
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
    if compare_original or sensitivity:
        if not resume or not saved.is_file():
            raise ValueError('Use --resume with a trained run containing checkpoint.pt')
        if sensitivity:
            result = {}
            with torch.no_grad():
                for weights in ('scorenet','vitpose'):
                    rows,_,_=_run(data,model,selector,config,'all',2026,windows,joint_weights=weights)
                    result[weights]={'mpjpe_mm':float(np.mean([r['mpjpe_mm'] for r in rows])),
                                     'mean_triangulated_joints':float(np.mean([r['triangulated_joints'] for r in rows]))}
            atomic_json(output/'weight_sensitivity.json',result)
            print(json.dumps(result,indent=2),flush=True)
        else:
            compare_original_fusion(data,model,selector,config,windows,output,joint_weights)
        return
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'experiment.json',contract)
    atomic_json(output/'preparation_quality.json',quality)
    from .scorer_experiments import architecture_summary
    atomic_json(output/'model_architecture.json',architecture_summary(selector))
    if start>epochs:
        raise ValueError('Requested final epoch is lower than saved checkpoint epoch')
    if ranking_from:
        if subset_training:
            from .subset_training import build_targets, fit_epoch
        else:
            from .ranking_training import build_targets, fit_epoch
        cache_directory = output
        if target_cache_from:
            if not subset_training:
                raise ValueError('Shared target cache requires subset training')
            cache_directory = Path(target_cache_from).resolve()
            other = json.loads((cache_directory/'experiment.json').read_text(encoding='utf-8'))
            keys = ('replay_sha256','sources','frozen_sha256','label_stride','windows','objective')
            if any(other.get(k)!=contract.get(k) for k in keys):
                raise ValueError('Shared target cache has different data/roster/frozen model')
        targets = build_targets(data,model,cache_directory,label_stride,contract['frozen_sha256'])
    for epoch in range(start,epochs):
        if ranking_from:
            details = fit_epoch(data,model,selector,config,targets,optimiser,epoch,cost)
            with torch.no_grad():
                rows,_,_=_run(data,model,selector,config,'learned',2026,windows,joint_weights=joint_weights)
        else:
            details = {}
            rows,_,_=_run(data,model,selector,config,'sample',2026+epoch,windows,optimiser,cost,joint_weights=joint_weights)
        history.append({'epoch':epoch+1,'mpjpe_mm':float(np.mean([r['mpjpe_mm'] for r in rows])),
                        'mean_requests':float(np.mean([len(r['selected']) for r in rows])),**details})
        temporary=saved.with_suffix('.tmp')
        torch.save({'selector':selector.state_dict(),'adapter':model.adapter.state_dict(),
                    'optimiser':optimiser.state_dict(),'epoch':epoch+1,'history':history},temporary)
        temporary.replace(saved)
        atomic_json(output/'learning_curve.json',history)
        if (epoch+1)%save_every == 0 or epoch+1==epochs:
            import shutil
            milestone=output/f'epoch_{epoch+1:04}'
            milestone.mkdir(exist_ok=True)
            shutil.copy2(saved,milestone/'checkpoint.pt')
            atomic_json(milestone/'experiment.json',contract)
            atomic_json(milestone/'metrics.json',evaluate_policies(data,model,selector,config,windows,joint_weights))
        print(history[-1],flush=True)
    with torch.no_grad():
        learned,predictions,states=_run(data,model,selector,config,'learned',2026,windows,joint_weights=joint_weights)
        local,_,_=_run(data,model,selector,config,'local',2026,windows,joint_weights=joint_weights)
        all_views,_,_=_run(data,model,selector,config,'all',2026,windows,joint_weights=joint_weights)
        random_matched,_,_=_run(data,model,selector,config,'random',2026,windows,
                               match_rows=learned,joint_weights=joint_weights)
        fixed_rank,_,_=_run(data,model,selector,config,'rank_two',2026,windows,joint_weights=joint_weights)
        fixed_random,_,_=_run(data,model,selector,config,'random',2026,windows,
                              match_rows=fixed_rank,joint_weights=joint_weights)
    atomic_npz(output/'predictions.npz',raw_frame_ids=np.array(ids[1:len(learned)+1]),
               predicted=predictions,reference=truth[1:],joint_states=states)
    warmup={'frame':ids[0],'selected':list(candidates),'mpjpe_mm':None,'triangulated_joints':None}
    export_outputs(output,history,learned,local,all_views,warmup,candidates,raw,predictions,images,
                   random_matched=random_matched,
                   diagnostics={'Rank (fixed 2)':fixed_rank,'Random (fixed 2)':fixed_random},
                   joints_for=raw_joint_overlay(sequence,images,ids,frames,raw,fixed) if images else None,
                   omitted_frames=len(ids)-len(learned)-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','sequence','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--repo',default='external/EgoExoMoCap')
    parser.add_argument('--checkpoint')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--cost',type=float,default=.005,help='Cost in MPJPE metres per extra request')
    parser.add_argument('--max-windows',type=int)
    parser.add_argument('--appearance',choices=('none','cnn','dino'),default='dino')
    parser.add_argument('--joint-weights',choices=('vitpose','scorenet'),default='scorenet')
    parser.add_argument('--encoding',choices=('legacy','compact'),default='legacy')
    parser.add_argument('--scorer',choices=('mlp','attention'),default='mlp')
    parser.add_argument('--sources',nargs='+',help='Explicit candidate roster; exclude held-out cameras from training')
    parser.add_argument('--evaluate-from',help='Checkpoint file or run directory; fixed-weight evaluation only')
    parser.add_argument('--save-every',type=int,default=25)
    parser.add_argument('--target-cache-from',help='Reuse subset labels from a strictly matching experiment')
    parser.add_argument('--subset-training',action='store_true')
    parser.add_argument('--ranking-from',help='Freeze the adapter from a compatible ViTPose run and train source scores')
    parser.add_argument('--label-stride',type=int,default=8)
    parser.add_argument('--sensitivity',action='store_true',help='Compare both joint weights on a fixed all-view checkpoint')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--compare-original',action='store_true',help='Evaluate trained run and original fusion on the same all-view inputs')
    parser.add_argument('--no-video',action='store_false',dest='render')
    args=parser.parse_args()
    if args.save_every<1 or args.epochs<1 or args.label_stride<1 or 96%args.label_stride or not math.isfinite(args.cost) or args.cost<0 or (
            args.max_windows is not None and args.max_windows<1):
        parser.error('Positive epochs/windows and finite nonnegative cost required')
    train(**vars(args))


if __name__=='__main__':
    main()
