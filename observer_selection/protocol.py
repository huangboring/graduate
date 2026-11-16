"""Explicit scene-disjoint experimental protocols; no automatic frame splitting."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def describe_sequence(path):
    path=Path(path).resolve()
    digest=sha256_file(path)
    with np.load(path,allow_pickle=False) as a:
        def scalar(key,default=None):
            if key not in a:
                if default is None: raise ValueError(f'{path.name}: required metadata {key} missing')
                return default
            if a[key].shape != (): raise ValueError(f'{key} must be scalar')
            return str(a[key].item())
        provenance=scalar('input_provenance')
        synthetic=provenance=='synthetic_demo'
        ids=[str(x) for x in a['source_ids']]
        if len(ids)<2 or len(set(ids))!=len(ids):
            raise ValueError('Selection experiments require at least two distinct external candidates')
        target=scalar('target_id')
        if synthetic:
            scene=scalar('scene_id','synthetic:'+digest)
            sequence=scalar('sequence_id',digest)
            requester=scalar('requester_id','A')
        else:
            scene,sequence,requester=(scalar(k) for k in ('scene_id','sequence_id','requester_id'))
            if scalar('temporal_provenance')!='causal':
                raise ValueError('Offline/bidirectional preprocessing cannot enter a causal selector')
            if scalar('crop_provenance')!='local_detector':
                raise ValueError('Selector crops must come from local image-only causal detection; GT/EgoNet benchmark crops rejected')
            if scalar('joint_layout')!='smplh22':
                raise ValueError('EgoExoMoCap reference must explicitly use SMPL-H first22 joint order')
            if 'preparation_manifest_sha256' not in a:
                raise ValueError('Real replay requires audited exporter provenance receipt')
            sidecar=path.with_suffix('.provenance.json')
            if not sidecar.is_file(): raise ValueError(f'Preparation receipt missing: {sidecar}')
            receipt=json.loads(sidecar.read_text(encoding='utf-8'))
            recorded=receipt.get('output_sha256')
            if recorded!=digest:
                raise ValueError('Replay file changed after preparation receipt')
        if not scene or not sequence or requester in ids or target in ids or requester==target:
            raise ValueError('Invalid requester/target/observer roster or scene identity')
        if scalar('units')!='metres' or scalar('coordinate_frame')!='root_relative':
            raise ValueError('Protocol requires explicit metres and joint0-root-relative coordinates')
        ts=np.asarray(a['timestamps'])
        if ts.ndim!=1 or not len(ts) or not np.isfinite(ts).all() or np.any(np.diff(ts)<=0):
            raise ValueError('Invalid chronological timestamps')
        return {'path':str(path),'sha256':digest,'scene_id':scene,'sequence_id':sequence,
                'requester_id':requester,'target_id':target,'source_ids':ids,
                'input_provenance':provenance,'is_synthetic':synthetic,'frames':len(ts)}


def audit_partitions(partitions):
    manifests={name:[describe_sequence(p) for p in paths] for name,paths in partitions.items()}
    seen_hashes, seen_scenes = {},{}
    flags=set()
    for split,rows in manifests.items():
        own=set()
        for row in rows:
            flags.add(row['is_synthetic'])
            if row['sha256'] in own: raise ValueError(f'Duplicate sequence bytes in {split}')
            own.add(row['sha256'])
            for key,seen in [('sha256',seen_hashes),('scene_id',seen_scenes)]:
                value=row[key]
                if value in seen and seen[value]!=split:
                    raise ValueError(f'Cross-split {key} leakage: {value} in {seen[value]} and {split}')
                seen[value]=split
    if len(flags)>1:
        raise ValueError('Synthetic fixtures and real research data cannot be mixed in one experiment')
    return manifests


def load_protocol(path):
    path=Path(path).resolve()
    doc=json.loads(path.read_text(encoding='utf-8'))
    if set(doc)!= {'train','validation','test'}:
        raise ValueError('Protocol must explicitly define train, validation and test path lists')
    result={}
    for name,items in doc.items():
        if not isinstance(items,list) or not items or not all(isinstance(p,str) for p in items):
            raise ValueError(f'{name} must contain explicit NPZ paths (no random/frame split)')
        result[name]=[(path.parent/entry).resolve() for entry in items]
    return result,audit_partitions(result)


def verify_heldout(network,test_paths,backend):
    records=[describe_sequence(p) for p in test_paths]
    if getattr(backend,'provenance',{}).get('backend')=='local_aux_v1':
        from .local_training import check_backend_splits
        check_backend_splits(backend,{'test':records})
    train_manifest=getattr(network,'training_data_manifest',None)
    if not train_manifest:
        raise ValueError('Checkpoint has no split provenance; cannot certify held-out evaluation')
    known=[r for rows in train_manifest.values() for r in rows]
    if any(r['scene_id']==t['scene_id'] or r['sha256']==t['sha256'] for r in known for t in records):
        raise ValueError('Test data overlaps checkpoint training/validation scenes or bytes')
    if {r['is_synthetic'] for r in known}!={r['is_synthetic'] for r in records}:
        raise ValueError('Cannot evaluate synthetic-trained checkpoint as a real research result')
    current=getattr(backend,'provenance',{'backend':type(backend).__name__})
    saved=getattr(network,'backend_provenance',None)
    if saved!=current:
        raise ValueError('Frozen reconstruction backend differs from the training checkpoint provenance')
    return records
