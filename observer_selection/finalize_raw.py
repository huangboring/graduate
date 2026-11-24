"""Re-export completed, verified observation stages without GPU inference."""
from pathlib import Path
import json
from .egohumans import _hash
from .local_data import export_local_egohumans
from .preparation_cache import verify_completed_view, atomic_json


def finalize_raw(prepared, output):
    root,out=Path(prepared).resolve(),Path(output).resolve()
    if out.exists(): raise FileExistsError('Use a fresh final export directory')
    contract_path=root/'preparation_run.json';manifest=root/'manifest.json'
    contract=json.loads(contract_path.read_text(encoding='utf-8'))
    m=json.loads(manifest.read_text(encoding='utf-8'))
    if (m['target_id']!=contract['target'] or m['requester_id']!=contract['requester']
            or m['source_ids']!=contract['sources']):
        raise ValueError('Manifest roster differs from original preparation contract')
    print('Verifying completed views and original inputs; no GPU inference...',flush=True)
    tracking={}
    for source in (contract['requester'],*contract['sources']):
        pair=Path(m['pairs'][source])
        if not pair.is_absolute(): pair=root/pair
        if pair.resolve()!=(root/source).resolve(): raise ValueError('Unexpected view location')
        stats=verify_completed_view(pair)
        if stats is None: raise ValueError(f'{source}: observation stage is incomplete; cannot finalize')
        if not (pair/'wearer_smplh_fk.npz').is_file():
            raise ValueError('Reference FK stage is missing; this command cannot invent labels')
        tracking[source]=stats
    for item in contract['input_files']:
        path=Path(item['path'])
        if not path.is_file() or _hash(path)!=item['sha256']:
            raise ValueError(f'Original raw input changed: {path}')
    result=export_local_egohumans(manifest,out/'replay.npz')
    # A finalized export remains directly trainable while reusing verified views.
    final_manifest=dict(m)
    final_manifest['pairs']={source:str((root/Path(path)).resolve())
                             for source,path in m['pairs'].items()}
    atomic_json(out/'manifest.json',final_manifest)
    sidecar=Path(result['provenance']);receipt=json.loads(sidecar.read_text(encoding='utf-8'))
    receipt['input_files'].extend(contract['input_files'])
    receipt['input_files'].append({'path':str(contract_path),'sha256':_hash(contract_path)})
    receipt['reexport_of']=str(root)
    receipt['tracking']=tracking
    atomic_json(sidecar,receipt)
    summary={**result,'tracking':tracking,'confidence_encoding':receipt['confidence_encoding'],
             'review_required':'Export success does not certify tracking identity or adequate observation coverage.'}
    atomic_json(out/'preparation_summary.json',summary)
    # Separate finalization record: never rewrite the original immutable run.
    atomic_json(out/'finalization_run.json',{'source':str(root),'generator_sha256':_hash(Path(__file__))})
    names=['replay.npz','replay.provenance.json','preparation_summary.json','finalization_run.json','manifest.json']
    atomic_json(out/'preparation_complete.json',{n:_hash(out/n) for n in names})
    return summary
