"""Current pose-only training and evaluation entry point.

Stage adapter fits only the geometric Exo residual with all acquired views.
Stage selector freezes that adapter and fits the common E/Q source score head.
"""
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from .egohumans import _hash
from .preparation_cache import atomic_json, atomic_npz
from .replay import load_sequence
from .rollout import Rollout
from .selection import SelectionConfig
from .selector import ViewSelector
from .single_exo import SingleExo


def _run(data, model, selector, config, mode, seed, windows, optimiser=None, match_rows=None):
    """Execute aligned windows. Only the adapter may be optimized here."""
    frames, ids, widths, fixed, requester, sparse, truth, rotations = data
    results, predictions, joint_states = [], [], []
    head = truth[:, 15]
    rollout = Rollout(selector, frames, requester, head, rotations, widths, fixed, config, seed)
    for start in range(0, windows * model.window, model.window):
        stop = start + model.window
        steps = [rollout.step(frames[i], i, mode, match_rows[i-1] if match_rows is not None else None)
                 for i in range(start+1, stop+1)]
        points = np.stack([step.points for step in steps])
        states = np.stack([step.states for step in steps])
        floor = truth[start+1:stop+1, [10, 11], 2].min()
        target = torch.as_tensor(truth[start+1:stop+1], device=model.device)
        predicted = model.reconstruct(sparse[start:stop], head[start+1:stop+1], points,
                                      states, (states != 0).any(-1), floor)
        error = torch.linalg.vector_norm(predicted-target, dim=-1).mean(-1)
        if not bool(torch.isfinite(error).all()):
            raise FloatingPointError('Nonfinite reconstruction; check calibration and weights')
        if optimiser is not None:
            optimiser.zero_grad(set_to_none=True)
            error.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1., error_if_nonfinite=True)
            optimiser.step()
        rollout.detach()
        predictions.extend(predicted.detach().cpu().numpy())
        joint_states.extend(states)
        for index, step, e in zip(range(start+1, stop+1), steps, error.detach().cpu().tolist()):
            results.append({'frame': ids[index], 'selected': list(step.selected),
                'refresh': list(step.required), 'mpjpe_mm': e*1000,
                'triangulated_joints': int((step.states == 2).sum()), **step.trace})
    return results, np.asarray(predictions), np.asarray(joint_states)


def _checkpoint_path(path):
    path = Path(path).resolve()
    return path/'checkpoint.pt' if path.is_dir() else path


def load_adapter(path, model, contract):
    """Explicit adapter-only migration; never load an old selector implicitly."""
    path = _checkpoint_path(path)
    previous = json.loads((path.parent/'experiment.json').read_text(encoding='utf-8'))
    for key in ('replay_sha256', 'checkpoint_sha256', 'sequence', 'windows', 'sources',
                'target_id', 'requester_id'):
        if previous.get(key) != contract.get(key):
            raise ValueError(f'Frozen adapter differs in {key}; refit the adapter for these inputs')
    if previous.get('joint_weights') != 'vitpose':
        raise ValueError('Adapter must have been trained with ViTPose joint confidences')
    state = torch.load(path, map_location=model.device, weights_only=True)
    model.adapter.load_state_dict(state['adapter'], strict=True)
    return path


def _save(path, selector, model, optimiser, epoch, history):
    temporary = path.with_suffix('.tmp')
    torch.save({'architecture_version': selector.architecture_version,
        'selector': selector.state_dict(), 'adapter': model.adapter.state_dict(),
        'optimiser': optimiser.state_dict(), 'epoch': epoch, 'history': history}, temporary)
    temporary.replace(path)


@torch.no_grad()
def compare_original_fusion(data, model, selector, config, windows, output):
    """Modified DINOv2 baseline, not the paper's official 50 mm protocol."""
    frames, ids, _, _, requester, sparse, truth, rotations = data
    _, triangulated, _ = _run(data, model, selector, config, 'all', 2026, windows)
    original = []
    for start in range(0, windows*model.window, model.window):
        stop = start+model.window
        chunk = frames[start+1:stop+1]
        views = [[f.observation for f in chunk]]
        views.extend([f.packets[source] for f in chunk] for source in frames[0].candidates)
        floor = truth[start+1:stop+1, [10, 11], 2].min()
        original.extend(model.reconstruct_original(sparse[start:stop], truth[start+1:stop+1, 15],
            rotations[start+1:stop+1], views, floor).cpu().numpy())
    result = {'scope': 'IN-SAMPLE, same observations, DINOv2 substituted for DINOv3',
        'raw_frame_range': [ids[1], ids[len(triangulated)]], 'frames': len(triangulated),
        'view_ids': [requester, *frames[0].candidates], 'unit': 'MPJPE mm',
        'methods': {name: float(np.linalg.norm(np.asarray(prediction)-truth[1:len(triangulated)+1],
                    axis=-1).mean()*1000) for name, prediction in
                    [('EgoExoMoCap original fusion (DINOv2)', original),
                     ('Triangulation + single Exo token', triangulated)]}}
    atomic_json(output/'fusion_comparison.json', result)
    return result


def train(prepared, sequence, output, repo='external/EgoExoMoCap', checkpoint=None, device='cuda',
          epochs=100, cost=.002, max_windows=None, resume=False, render=True, stage='selector',
          adapter_from=None, scorer='mlp', sources=None, evaluate_from=None, save_every=25,
          target_cache_from=None, label_stride=8, refresh_after_s=1., compare_original=False):
    from .preparation_quality import check_preparation
    from .scorer_experiments import restrict_sources, load_transfer, evaluate_policies
    from .selection_video import export_outputs, image_paths, raw_joint_overlay

    if (epochs < 1 or save_every < 1 or label_stride < 1 or 96 % label_stride
            or not math.isfinite(cost) or cost < 0 or (max_windows is not None and max_windows < 1)):
        raise ValueError('Positive epochs/windows and stride dividing 96; finite nonnegative cost required')
    if stage not in ('adapter', 'selector'):
        raise ValueError('Unknown training stage')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python environment')
    torch.manual_seed(2026)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    prepared, sequence, output, repo = [Path(p).resolve() for p in (prepared, sequence, output, repo)]
    checkpoint = Path(checkpoint).resolve() if checkpoint else repo/'model_zoo/egohumans_syn_tracker_3pt/model.pth'
    manifest = json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    if (manifest['scene_id'], manifest['sequence_id']) != (sequence.parent.name, sequence.name):
        raise ValueError('Prepared data and raw sequence identify different sessions')
    quality = check_preparation(prepared/'replay.npz')
    frames = restrict_sources(load_sequence(prepared/'replay.npz'), sources)
    requester, target = manifest['requester_id'], manifest['target_id']
    if any(frame.observation.target_id != target for frame in frames):
        raise ValueError('Manifest target differs from replay target')
    candidates, widths, ids = frames[0].candidates, {}, None
    for source in (requester, *candidates):
        pair = Path(manifest.get('pairs', {}).get(source, source))
        if not pair.is_absolute():
            pair = prepared/pair
        grid = json.loads((pair/'meta.json').read_text(encoding='utf-8'))['raw_frame_ids']
        if ids is not None and grid != ids:
            raise ValueError(f'Raw frame grid differs for {source}')
        ids = grid
        with np.load(pair/'observer.npz') as camera:
            widths[source] = int(camera['observer_img_size'][0])
    if len(frames) != len(ids) or ids != list(range(ids[0], ids[0]+len(ids))):
        raise ValueError('Replay and raw frame IDs must align and be consecutive')
    fixed = {source for source in candidates if (sequence/'exo'/source).is_dir()}
    model = SingleExo(repo, checkpoint, device)
    windows = min((len(frames)-1)//model.window, max_windows or len(frames))
    if windows < 1:
        raise ValueError(f'Need initialization plus {model.window} aligned frames')
    sparse, truth, rotations = model.load_tracking(sequence, target, ids[:1+windows*model.window])
    data = frames, ids, widths, fixed, requester, sparse, truth, rotations
    selector = ViewSelector(scorer).to(device)
    config = SelectionConfig(refresh_after_s)
    contract = {'architecture_version': selector.architecture_version, 'stage': stage,
        'prepared': str(prepared), 'replay_sha256': _hash(prepared/'replay.npz'),
        'checkpoint_sha256': _hash(checkpoint), 'sequence': str(sequence), 'windows': windows,
        'sources': list(candidates), 'target_id': target, 'requester_id': requester,
        'policy': 'above_local_plus_refresh', 'refresh_after_s': refresh_after_s,
        'cost_m_per_source': cost, 'joint_weights': 'vitpose', 'appearance': 'none',
        'encoding': 'compact', 'scorer': scorer, 'label_stride': label_stride,
        'objective': 'adapter_all_views_mpjpe' if stage == 'adapter' else 'exact_subset_expected_cost',
        'target_context': 'seeded_random_two', 'entropy_temperature_m': .0005,
        'scope': 'IN-SAMPLE; synthetic target trackers and reference-derived floor'}
    if evaluate_from:
        load_transfer(evaluate_from, selector, model, contract, candidates, sequence, output)
        result = evaluate_policies(data, model, selector, config, windows)
        atomic_json(output/'camera_transfer.json', result)
        print(json.dumps(result, indent=2), flush=True)
        return result
    if stage == 'selector':
        if not adapter_from:
            raise ValueError('Selector training requires --adapter-from; first run --stage adapter')
        frozen = load_adapter(adapter_from, model, contract)
        model.adapter.requires_grad_(False)
        contract.update(frozen_from=str(frozen), frozen_sha256=_hash(frozen))
        optimiser = torch.optim.AdamW(selector.parameters(), lr=1e-4)
    else:
        if adapter_from or target_cache_from:
            raise ValueError('Adapter stage does not accept an adapter or selector target cache')
        selector.requires_grad_(False)
        optimiser = torch.optim.AdamW(model.adapter.parameters(), lr=3e-4)
    saved, start, history = output/'checkpoint.pt', 0, []
    if resume:
        old = json.loads((output/'experiment.json').read_text(encoding='utf-8'))
        if old != contract:
            raise ValueError('Resume config/input differs, or old architecture: use a fresh output directory')
        state = torch.load(saved, map_location=device, weights_only=True)
        if state.get('architecture_version') != selector.architecture_version:
            raise ValueError('Old selector checkpoint: use a fresh current-E run')
        selector.load_state_dict(state['selector'], strict=True)
        model.adapter.load_state_dict(state['adapter'], strict=True)
        optimiser.load_state_dict(state['optimiser'])
        start, history = state['epoch'], state['history']
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a fresh output or use --resume')
    if start > epochs:
        raise ValueError('Final epoch is lower than saved epoch')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'experiment.json', contract)
    atomic_json(output/'preparation_quality.json', quality)
    atomic_json(output/'model_architecture.json', selector.architecture_summary())
    if compare_original:
        if not resume:
            raise ValueError('--compare-original requires --resume with an existing checkpoint')
        return compare_original_fusion(data, model, selector, config, windows, output)
    if stage == 'selector':
        from .subset_training import build_targets, fit_epoch
        cache_dir = Path(target_cache_from).resolve() if target_cache_from else output
        targets = build_targets(data, model, cache_dir, label_stride, contract['frozen_sha256'],
                                contract['replay_sha256'])
    for epoch in range(start, epochs):
        selector.train(stage == 'selector')
        if stage == 'adapter':
            rows, _, _ = _run(data, model, selector, config, 'all', 2026, windows, optimiser)
            details = {'optimizer_updates': windows}
        else:
            details = fit_epoch(data, model, selector, config, targets, optimiser, epoch, cost)
            selector.eval()
            with torch.no_grad():
                rows, _, _ = _run(data, model, selector, config, 'learned', 2026, windows)
        history.append({'epoch': epoch+1, 'mpjpe_mm': float(np.mean([r['mpjpe_mm'] for r in rows])),
                        'mean_requests': float(np.mean([len(r['selected']) for r in rows])), **details})
        _save(saved, selector, model, optimiser, epoch+1, history)
        atomic_json(output/'learning_curve.json', history)
        if (epoch+1) % save_every == 0 or epoch+1 == epochs:
            milestone = output/f'epoch_{epoch+1:04}'
            milestone.mkdir(exist_ok=True)
            shutil.copy2(saved, milestone/'checkpoint.pt')
            atomic_json(milestone/'experiment.json', contract)
            if stage == 'selector':
                atomic_json(milestone/'metrics.json', evaluate_policies(data, model, selector, config, windows))
        print(history[-1], flush=True)
    selector.eval()
    if stage == 'adapter':
        return history[-1]
    with torch.no_grad():
        learned, predictions, states = _run(data, model, selector, config, 'learned', 2026, windows)
        local, _, _ = _run(data, model, selector, config, 'local', 2026, windows)
        all_views, _, _ = _run(data, model, selector, config, 'all', 2026, windows)
        random_matched, _, _ = _run(data, model, selector, config, 'random', 2026, windows, match_rows=learned)
    atomic_json(output/'round_trace.json', learned)
    atomic_npz(output/'predictions.npz', raw_frame_ids=np.array(ids[1:len(learned)+1]),
        predicted=predictions, reference=truth[1:len(learned)+1], joint_states=states)
    images = image_paths(sequence, (requester, *candidates), ids) if render else {}
    warmup = {'frame': ids[0], 'selected': list(candidates), 'mpjpe_mm': None, 'triangulated_joints': None}
    export_outputs(output, history, learned, local, all_views, warmup, candidates, requester,
        predictions, images, random_matched=random_matched,
        joints_for=raw_joint_overlay(sequence, images, ids, frames, requester, fixed) if images else None,
        omitted_frames=len(ids)-len(learned)-1)
    return json.loads((output/'metrics.json').read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'sequence', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--repo', default='external/EgoExoMoCap')
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--stage', choices=('adapter', 'selector'), default='selector')
    parser.add_argument('--adapter-from')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--cost', type=float, default=.002, help='Metres of MPJPE per request in offline training cost')
    parser.add_argument('--max-windows', type=int)
    parser.add_argument('--scorer', choices=('mlp', 'attention'), default='mlp')
    parser.add_argument('--sources', nargs='+')
    parser.add_argument('--evaluate-from')
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--target-cache-from')
    parser.add_argument('--label-stride', type=int, default=8)
    parser.add_argument('--refresh-after-s', type=float, default=1.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--compare-original', action='store_true')
    parser.add_argument('--no-video', action='store_false', dest='render')
    train(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
