"""Supervised subset ranking with a frozen pose teacher and sequence-held-out validation."""
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from .selection import SubsetSelector, candidate_subsets, subset_masks, describe, select_payload, Selection
from .source import inspect_sequence, open_source, prepare_frame, camera_geometry, annotation_files


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ground_truth(path):
    """Use the same first-15 joints, root visibility and mm axes as Panoptic loader."""
    bodies = json.loads(Path(path).read_text(encoding='utf-8'))['bodies']
    poses, visibility = [], []
    axis = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
    for body in bodies:
        joints = np.asarray(body['joints19'], dtype=float).reshape(19,4)[:15]
        visible = (joints[:,3] > .1) & np.isfinite(joints[:,:3]).all(-1)
        if not visible[2]:
            continue
        poses.append(np.nan_to_num(joints[:,:3]) @ axis * 10.)
        visibility.append(visible)
    return np.asarray(poses).reshape(-1,15,3), np.asarray(visibility, dtype=bool).reshape(-1,15)


def pose_quality(predictions, truth, visible, match_mm=500.):
    """Custom detection-aware cost in mm, NOT official Panoptic AP/MPJPE.

    Hungarian one-to-one matching; pairs over match_mm are unmatched. Penalize
    missing people and false detections, including frames with no labelled people.
    """
    ng, npred = len(truth), len(predictions)
    distances = np.empty((ng, npred))
    for i in range(ng):
        distances[i] = np.linalg.norm(predictions[:,visible[i]]-truth[i,visible[i]], axis=-1).mean(-1)
    pairs = []
    if ng and npred:
        rows, columns = linear_sum_assignment(distances)
        pairs = [(r,c) for r,c in zip(rows,columns) if distances[r,c] <= match_mm]
    errors = [float(distances[r,c]) for r,c in pairs]
    tp = len(pairs)
    cost = (sum(errors) + match_mm * (ng + npred - 2*tp)) / max(ng,1)
    return dict(cost_mm=cost, matched_error_sum_mm=sum(errors), tp=tp, fp=npred-tp, fn=ng-tp)


def scores_for_record(model, record, device):
    descriptors = record['descriptors'].to(device)
    masks = subset_masks(record['subsets'], len(descriptors), device)
    return model(descriptors, masks, record['geometry'].to(device))


@torch.no_grad()
def evaluate_selector(model, records, device='cpu'):
    model.eval()
    sums = dict(learned=0., fixed=0., random=0., oracle=0., full=0., regret=0., agreements=0.)
    per_k = {}
    for record in records:
        scores = scores_for_record(model, record, device)
        choice = scores.argmax().item()
        costs = record['costs']
        selected = float(costs[choice])
        fixed = record['subsets'].index(tuple(range(record['k'])))
        sums['learned'] += selected
        sums['fixed'] += float(costs[fixed])
        sums['random'] += float(costs.mean())
        sums['oracle'] += float(costs.min())
        sums['full'] += record['full_cost_mm']
        sums['regret'] += selected-float(costs.min())
        sums['agreements'] += float(abs(selected-float(costs.min())) < 1e-4)
        group = per_k.setdefault(str(record['k']), {'count':0, 'learned_cost_mm':0., 'fixed_cost_mm':0.})
        group['count'] += 1
        group['learned_cost_mm'] += selected
        group['fixed_cost_mm'] += float(costs[fixed])
    if not records:
        raise ValueError('Validation requires records from separate sequences')
    for group in per_k.values():
        for key in ('learned_cost_mm','fixed_cost_mm'):
            group[key] /= group['count']
    return dict(count=len(records), **{k:v/len(records) for k,v in sums.items()}, per_k=per_k,
                metric='custom detection-aware cost_mm; random=expected cost over sampled subsets; oracle=best sampled subset')


def fit_selector(train_records, val_records, output, epochs=50, lr=.001, seed=42,
                 temperature_mm=100., patience=10, device='cpu'):
    """One set at a time naturally supports variable N and K, without padding IDs."""
    if not train_records or not val_records or epochs < 1 or temperature_mm <= 0:
        raise ValueError('Nonempty train/validation sets, epochs and temperature are required')
    train_sequences = {r['sequence'] for r in train_records}
    val_sequences = {r['sequence'] for r in val_records}
    if train_sequences & val_sequences:
        raise ValueError('Train/validation sequence overlap would leak neighbouring frames')
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = SubsetSelector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best, stalled = float('inf'), 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_records)))
        rng.shuffle(order)
        losses = []
        for index in order:
            record = train_records[index]
            costs = record['costs'].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = scores_for_record(model, record, device)
            targets = torch.softmax(-costs/temperature_mm, dim=0)
            loss = -(targets * torch.log_softmax(logits, dim=0)).sum()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite selector loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(loss.item())
        report = evaluate_selector(model, val_records, device)
        history.append(dict(epoch=epoch+1, loss=sum(losses)/len(losses), validation=report))
        print(f'epoch={epoch+1} loss={history[-1]["loss"]:.4f} val_cost={report["learned"]:.2f} fixed={report["fixed"]:.2f}', flush=True)
        if report['learned'] < best:
            best, stalled = report['learned'], 0
            torch.save(dict(architecture=SubsetSelector.architecture, state_dict=model.cpu().state_dict(),
                            dimension=32, hidden=64, epoch=epoch+1), output/'selector.pt')
            model.to(device)
        else:
            stalled += 1
        (output/'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if stalled >= patience:
            break
    model.load_state_dict(torch.load(output/'selector.pt', map_location=device, weights_only=True)['state_dict'])
    final = evaluate_selector(model, val_records, device)
    (output/'validation.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
    return model.eval(), final


def collect_records(model, model_cfg, settings, sequences, data_root, cache_dir, split,
                    decode_output, device='cuda'):
    """Cache compact supervision, not full spatial features; resume per frame."""
    cache_dir = Path(cache_dir)/split
    cache_dir.mkdir(parents=True, exist_ok=True)
    records, decoded, missing_gt, no_choice = [], 0, 0, 0
    for sequence in sequences:
        folder = Path(data_root)/sequence
        ids, paths, cameras = inspect_sequence(folder, settings.get('cameras'), settings.get('input_format','auto'))
        annotations = annotation_files(folder)
        source = open_source(paths, settings['start_frame'], settings.get('image_fps',29.97002997))
        seen = 0
        try:
            while seen < settings['frames_per_sequence']:
                bundle = source.read()
                if bundle is None:
                    break
                frame_index, frames = bundle
                seen += 1
                decoded += 1
                gt_file = annotations.get(frame_index)
                cache_file = cache_dir/f'{sequence}_{frame_index:08d}.pt'
                if gt_file is None:
                    missing_gt += 1
                elif cache_file.is_file():
                    records.append(torch.load(cache_file, map_location='cpu', weights_only=True))
                else:
                    # Vary candidate-pool size across frames; do not assume adjacent IDs.
                    rng = random.Random(f'{settings["seed"]}:{sequence}:{frame_index}')
                    max_pool = min(len(ids), settings['max_candidate_views'])
                    if max_pool < 3:
                        raise ValueError('Training selection needs >=3 cameras to provide alternative subsets')
                    pool_size = rng.randint(3, max_pool) if split == 'train' else max_pool
                    pool = sorted(rng.sample(range(len(ids)), pool_size)) if split == 'train' else list(range(max_pool))
                    usable_k = [k for k in settings['k_values'] if 2 <= k < pool_size]
                    if not usable_k:
                        no_choice += 1
                    else:
                        k = usable_k[rng.randrange(len(usable_k))]
                        subsets = candidate_subsets(pool_size, k, settings['subset_limit'])
                        with torch.inference_mode():
                            prepared = [prepare_frame(frames[i], cameras[i], model_cfg.NETWORK.IMAGE_SIZE,
                                model_cfg.NETWORK.HEATMAP_SIZE, device, model_cfg.DATASET.COLOR_RGB) for i in pool]
                            views, metadata = map(list, zip(*prepared))
                            levels = [model.extract_view_features([view]) for view in views]
                            features = [torch.cat(x,0) for x in zip(*levels)]
                            truth, visibility = ground_truth(gt_file)
                            stats = []
                            full_cost = None
                            for subset in subsets + [tuple(range(pool_size))]:
                                selection = Selection(list(subset), [],0,0,0,1)
                                payload, selected_views, selected_meta = select_payload(features,views,metadata,selection)
                                result = model(views=selected_views, meta=selected_meta, precomputed_features=payload,
                                               threshold=settings['threshold'])
                                poses, _ = decode_output(result, 15, settings['threshold'], settings['nms_mm'])
                                quality = pose_quality(poses, truth, visibility, settings['match_mm'])
                                if len(subset) == pool_size:
                                    full_cost = quality['cost_mm']
                                else:
                                    stats.append(quality)
                            record = dict(sequence=sequence, frame=frame_index, camera_ids=[ids[i] for i in pool],
                                descriptors=describe(features).cpu(), geometry=camera_geometry(metadata).cpu(),
                                subsets=subsets, k=k, costs=torch.tensor([s['cost_mm'] for s in stats]),
                                quality=stats, full_cost_mm=full_cost)
                            torch.save(record, cache_file)
                            records.append(record)
                            print(f'{split} {sequence}:{frame_index} cameras={pool_size} k={k} subsets={len(subsets)} cached', flush=True)
                if not source.skip(settings['frame_stride']-1):
                    break
        finally:
            source.close()
    if not records:
        raise ValueError(f'No {split} records: decoded={decoded}, missing_gt={missing_gt}, no_valid_k={no_choice}')
    spreads = [float(r['costs'].max()-r['costs'].min()) for r in records]
    audit = dict(records=len(records), decoded=decoded, missing_gt=missing_gt, no_valid_k=no_choice,
                 informative_records=sum(x > 1e-3 for x in spreads))
    (cache_dir/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    if split == 'train' and not audit['informative_records']:
        raise ValueError('All subset costs are identical; inspect teacher poses/annotations before training')
    return records
