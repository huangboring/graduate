"""Atomic cache/provenance helpers for restartable raw preparation."""
import json
from pathlib import Path
import numpy as np
from .egohumans import _hash


def atomic_json(path, value):
    path = Path(path); temp = path.with_name(path.name+'.pending')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path); temp = path.with_name(path.name+'.pending')
    with temp.open('wb') as stream: np.savez_compressed(stream, **arrays)
    temp.replace(path)


def load_or_create_contract(path, contract, resume):
    path = Path(path)
    if path.exists():
        if not resume: raise FileExistsError('Use --resume to continue this preparation')
        if json.loads(path.read_text(encoding='utf-8')) != contract:
            raise ValueError('Preparation inputs/models/settings changed. Use a fresh output directory; model cache can be shared.')
    else:
        atomic_json(path, contract)


def load_candidates(path):
    with np.load(path, allow_pickle=False) as z:
        boxes, features = z['boxes'], z['features']
    if (boxes.ndim != 2 or boxes.shape[1:] != (4,) or features.shape != (len(boxes),768)
            or not np.isfinite(boxes).all() or not np.isfinite(features).all()
            or np.any(boxes[:,2:]<=boxes[:,:2])):
        raise ValueError(f'Invalid detection cache: {path}')
    return boxes, features


def verify_completed_view(pair):
    pair = Path(pair); marker = pair/'view_complete.json'
    if not marker.exists(): return None
    record = json.loads(marker.read_text(encoding='utf-8'))
    for name,digest in record['outputs'].items():
        path = pair/name
        if path.parent != pair or not path.is_file() or _hash(path)!=digest:
            raise ValueError(f'Completed view changed: {path}')
    return record['statistics']


def mark_completed_view(pair, stats):
    pair = Path(pair)
    names = ('meta.json','timestamps.csv','observer.npz','detector_boxes.npz','detector_receipt.json',
             'selector_observations.npz','selector_preparation.json','tracking_preview.mp4')
    if stats.get('identity_protocol') != 'egohumans_annotation':
        names += ('identity_choices.json','identity_selected.jpg','identity_reference.jpg')
    names += tuple(n for n in ('recovery_choices.json','tracking_diagnostics.json') if (pair/n).is_file())
    atomic_json(pair/'view_complete.json', {'statistics':stats,'outputs':{n:_hash(pair/n) for n in names}})


def archive_partial_features(pair):
    """Move only this incomplete stage's generated output, keeping evidence."""
    pair = Path(pair)
    existing = [pair/name for name in ('selector_observations.npz','selector_preparation.json') if (pair/name).exists()]
    if not existing: return
    i=1
    while (pair/f'interrupted_features_{i}').exists(): i+=1
    destination=pair/f'interrupted_features_{i}';destination.mkdir()
    for path in existing: path.rename(destination/path.name)


def archive_outputs(folder, names):
    folder=Path(folder).resolve()
    paths=[folder/name for name in names if (folder/name).exists()]
    if not paths: return
    if any(p.is_symlink() or not p.is_file() or p.parent!=folder for p in paths):
        raise ValueError('Can only archive generated files directly inside this output folder')
    i=1
    while (folder/f'previous_attempt_{i}').exists(): i+=1
    destination=folder/f'previous_attempt_{i}';destination.mkdir()
    for path in paths: path.rename(destination/path.name)
