"""Reuse inference, never old associations, when changing tracking algorithms."""
from pathlib import Path
import json
import shutil
from .preparation_cache import load_candidates, atomic_json
from .egohumans import _hash


def reuse_candidate_cache(source, output, contract):
    source,out=Path(source).resolve(),Path(output).resolve()
    if source==out: raise ValueError('Candidate import requires a different output directory')
    old=json.loads((source/'preparation_run.json').read_text())
    for key in ('sequence','raw_frame_ids','input_files','feature_models','detector','device','torch','requester','sources'):
        if old.get(key)!=contract.get(key): raise ValueError(f'Candidate cache contract mismatch: {key}')
    if old.get('tracker',{}).get('version') not in ('causal_dinov2_association_v1','causal_dinov2_association_v2'):
        raise ValueError('Unsupported candidate extraction version')
    copied=[]
    for aria in (contract['requester'],*contract['sources']):
        destination=out/aria/'frame_cache';destination.mkdir(parents=True,exist_ok=True)
        for frame in contract['raw_frame_ids']:
            p=source/aria/'frame_cache'/f'{frame:05}.npz'
            if not p.is_file(): continue
            load_candidates(p);digest=_hash(p);q=destination/p.name
            if q.exists():
                if _hash(q)!=digest: raise ValueError(f'Existing candidate cache differs: {q}')
            else:
                pending=q.with_suffix('.pending');shutil.copyfile(p,pending);pending.replace(q)
            copied.append({'source':str(p),'sha256':digest})
    atomic_json(out/'candidate_import.json',{'source_run':str(source),'files':copied,
        'scope':'image detections and appearance features only; identity decisions not imported'})
    print(f'Reused {len(copied)} candidate frames; identity decisions will be made again.',flush=True)
