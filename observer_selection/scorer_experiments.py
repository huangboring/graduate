"""Camera-roster ablations and fixed-weight scorer evaluation."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from .preparation_cache import atomic_json


def restrict_sources(frames, sources):
    if sources is None:
        return frames
    roster = tuple(sources)
    if not roster or len(set(roster)) != len(roster):
        raise ValueError('Use a nonempty roster of unique sources')
    available = set(frames[0].candidates)
    if not set(roster) <= available:
        raise ValueError(f'Missing prepared sources: {sorted(set(roster)-available)}')
    return [replace(f,candidates=roster,packets={s:f.packets[s] for s in roster},
                    geometry={s:f.geometry[s] for s in roster if s in f.geometry}) for f in frames]


@torch.no_grad()
def evaluate_policies(data, model, selector, config, windows, weights):
    from .fit_triangulated import _run
    result = {}
    matched = {}
    for name,mode,match in [('Learned','learned',None),('Local only','local',None),
                           ('All sources','all',None),('Random (matched count)','random','Learned'),
                           ('Rank (fixed 2)','rank_two',None),('Random (fixed 2)','random','Rank (fixed 2)')]:
        rows,_,_ = _run(data,model,selector,config,mode,2026,windows,
                       match_rows=matched.get(match),joint_weights=weights)
        matched[name] = rows
        result[name] = {'mpjpe_mm':float(np.mean([r['mpjpe_mm'] for r in rows])),
                        'mean_requests':float(np.mean([len(r['selected']) for r in rows])),
                        'mean_triangulated_joints':float(np.mean([r['triangulated_joints'] for r in rows]))}
    return {'scope':'Same-sequence evaluation; held-out cameras do not establish cross-scene generalization',
            'frames':windows*model.window,'sources':list(data[0][0].candidates),
            'random_seed':2026,'methods':result}


def load_transfer(checkpoint, selector, model, current, candidates, sequence, output):
    path=Path(checkpoint).resolve()
    if path.is_dir():
        path=path/'checkpoint.pt'
    trained=json.loads((path.parent/'experiment.json').read_text(encoding='utf-8'))
    for key,default in [('appearance','dino'),('joint_weights','scorenet'),('scorer','mlp'),('encoding','legacy'),
                        ('encoding_version',None),('scoring_head',None)]:
        if trained.get(key,default)!=current.get(key,default):
            raise ValueError(f'Evaluation checkpoint differs in {key}')
    if trained['checkpoint_sha256']!=current['checkpoint_sha256']:
        raise ValueError('Official reconstruction weights differ')
    for key in ('target_id','requester_id'):
        if trained.get(key)!=current.get(key):
            raise ValueError(f'Evaluation target/local identity differs in {key}')
    state=torch.load(path,map_location=model.device,weights_only=True)
    selector.load_state_dict(state['selector'])
    model.adapter.load_state_dict(state['adapter'])
    selector.score_cost=(trained['cost_m_per_source']/.005
                         if trained.get('objective')=='counterfactual_gain_ranking' else 0.)
    selector.eval().requires_grad_(False)
    model.adapter.eval().requires_grad_(False)
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a fresh evaluation output')
    output.mkdir(parents=True,exist_ok=True)
    old=trained.get('sources',[])
    from .egohumans import _hash
    atomic_json(output/'evaluation_protocol.json',{
        'checkpoint':str(path),'checkpoint_sha256':_hash(path),'evaluation_replay_sha256':current['replay_sha256'],'epoch':state['epoch'],'training_sources':old,
        'evaluation_sources':list(candidates),'unseen_sources':sorted(set(candidates)-set(old)),
        'overlapping_sources':sorted(set(candidates)&set(old)),
        'same_sequence':trained['sequence']==str(sequence),'training_performed':False,
        'initialization':'All evaluation candidates acquired once; identical warm-up for each policy',
        'warning':'Camera transfer within the same sequence; no claim of held-out people or scenes'})


def architecture_summary(selector):
    compact = selector.encoding=='compact'
    return {
        'encoding':selector.encoding,'scorer':selector.scorer_kind,
        'appearance':selector.appearance_kind,
        'demand_features':selector.demand_size,'packet_features':selector.packet_size,
        'demand_hidden':64,'source_hidden':64,'geometry_features':9,
        'has_history_input':not compact,
        'source_token_raw':74 if compact else 75,
        'head':('shared [q64, token64] -> 64 -> 32 -> 1' if compact else
                'legacy attention token64 -> 1' if selector.scorer_kind=='attention' else
                'legacy separate local/source heads'),
        'attention_heads':4 if selector.scorer_kind=='attention' else 0,
        'selector_parameters':sum(p.numel() for p in selector.parameters()),
        'note':'Parameter count is not end-to-end FLOPs or measured runtime; recurrent encoders are included.'}
