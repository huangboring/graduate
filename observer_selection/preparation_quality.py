"""Training preflight: reject incomplete exports and the legacy terminal bug."""
import json
from pathlib import Path
import numpy as np
from .egohumans import _hash


def check_preparation(data):
    data=Path(data).resolve();folder=data.parent
    summary_path=folder/'preparation_summary.json'
    summary=json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
    if any(s.get('terminal_track_loss') for s in summary.get('tracking',{}).values()):
        raise ValueError('Legacy tracker stopped permanently. Reprepare using appearance tracking; do not train this export.')
    if (folder/'preparation_run.json').exists():
        marker=folder/'preparation_complete.json'
        if not marker.exists(): raise ValueError('Raw preparation is incomplete; resume it before training')
        for name,digest in json.loads(marker.read_text(encoding='utf-8')).items():
            path=folder/name
            if path.parent!=folder or not path.is_file() or _hash(path)!=digest:
                raise ValueError(f'Preparation completion digest differs: {path}')
    with np.load(data,allow_pickle=False) as z:
        local=np.asarray(z['local_valid'],bool).any(axis=-1)
        remote=np.asarray(z['packet_valid'],bool).any(axis=-1)
        sources=[str(s) for s in z['source_ids']]
    if not (local.any() or remote.any()): raise ValueError('No observed joints to train from')
    quality={'frames':len(local),'local_observed_frames':int(local.sum()),
             'remote_observed_frames':{s:int(remote[:,i].sum()) for i,s in enumerate(sources)},
             'all_views_missing_frames':int((~local & ~remote.any(axis=-1)).sum()),
             'tracking':summary.get('tracking',{}),
             'meaning':'Availability counts only; do not certify identity or physical visibility.'}
    print('Training input availability: '+json.dumps(quality),flush=True)
    return quality
