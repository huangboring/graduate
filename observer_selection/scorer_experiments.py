"""Fixed-weight camera-roster evaluation of the current scoring architecture."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from .egohumans import _hash
from .preparation_cache import atomic_json


def restrict_sources(frames, sources):
    if sources is None:
        return frames
    roster = tuple(sources)
    if not roster or len(set(roster)) != len(roster):
        raise ValueError('Use a nonempty roster of unique sources')
    if not set(roster) <= set(frames[0].candidates):
        raise ValueError('Requested sources are not all present in prepared data')
    return [replace(frame, candidates=roster, packets={s: frame.packets[s] for s in roster},
        geometry={s: frame.geometry[s] for s in roster if s in frame.geometry}) for frame in frames]


@torch.no_grad()
def evaluate_policies(data, model, selector, config, windows):
    from .fit_triangulated import _run
    selector.eval()
    methods, reference = {}, None
    for name, mode in [('Learned', 'learned'), ('Local only', 'local'),
                       ('All sources', 'all'), ('Random (matched count)', 'random')]:
        rows, _, _ = _run(data, model, selector, config, mode, 2026, windows,
                          match_rows=reference if mode == 'random' else None)
        if mode == 'learned':
            reference = rows
        methods[name] = {'mpjpe_mm': float(np.mean([r['mpjpe_mm'] for r in rows])),
            'mean_requests': float(np.mean([len(r['selected']) for r in rows])),
            'mean_triangulated_joints': float(np.mean([r['triangulated_joints'] for r in rows]))}
    return {'scope': 'Same-sequence evaluation; camera transfer does not establish cross-scene generalization',
        'frames': windows*model.window, 'sources': list(data[0][0].candidates),
        'initialization': 'One all-source acquisition, excluded from scored frames',
        'random_seed': 2026, 'methods': methods}


def load_transfer(checkpoint, selector, model, current, candidates, sequence, output):
    path = Path(checkpoint).resolve()
    if path.is_dir():
        path = path/'checkpoint.pt'
    trained = json.loads((path.parent/'experiment.json').read_text(encoding='utf-8'))
    for key in ('architecture_version', 'appearance', 'joint_weights', 'scorer',
                'checkpoint_sha256', 'target_id', 'requester_id', 'refresh_after_s'):
        if trained.get(key) != current.get(key):
            raise ValueError(f'Evaluation checkpoint differs in {key}; old selectors cannot be resumed')
    if trained.get('stage') != 'selector':
        raise ValueError('Transfer requires a trained selector checkpoint')
    state = torch.load(path, map_location=model.device, weights_only=True)
    if state.get('architecture_version') != selector.architecture_version:
        raise ValueError('Checkpoint does not implement current E/Q self scoring')
    selector.load_state_dict(state['selector'], strict=True)
    model.adapter.load_state_dict(state['adapter'], strict=True)
    selector.eval().requires_grad_(False)
    model.adapter.eval().requires_grad_(False)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a fresh evaluation output')
    output.mkdir(parents=True, exist_ok=True)
    old = trained['sources']
    atomic_json(output/'evaluation_protocol.json', {
        'checkpoint': str(path), 'checkpoint_sha256': _hash(path),
        'evaluation_replay_sha256': current['replay_sha256'], 'epoch': state['epoch'],
        'training_sources': old, 'evaluation_sources': list(candidates),
        'unseen_sources': sorted(set(candidates)-set(old)),
        'overlapping_sources': sorted(set(candidates)&set(old)),
        'same_sequence': trained['sequence'] == str(sequence), 'training_performed': False,
        'initialization': 'All evaluation sources acquired once for every policy',
        'warning': 'Same-sequence camera transfer, not held-out people or scenes'})
