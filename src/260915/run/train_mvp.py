"""One command: preflight -> frozen teacher labels -> selector -> validation -> demo bundle."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'lib'))


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def validate_settings(settings):
    if settings.get('input_format','auto') not in ('auto','videos','legacy_images','hdimgs'):
        raise ValueError('Unsupported input_format')
    if settings.get('image_fps',29.97002997) <= 0:
        raise ValueError('image_fps must be positive')
    for split in ('train_sequences','val_sequences'):
        names = settings[split]
        if not names or len(names) != len(set(names)):
            raise ValueError(f'{split} must contain unique sequence names')
        if any('/' in x or '\\' in x or x in ('.','..') for x in names):
            raise ValueError('Sequence names must be folder basenames')
    if set(settings['train_sequences']) & set(settings['val_sequences']):
        raise ValueError('Training and validation sequences must be disjoint')
    for key in ('frame_stride','frames_per_sequence','epochs','patience','smoke_frames'):
        if not isinstance(settings[key],int) or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if settings['start_frame'] < 0 or settings['max_candidate_views'] < 3 or settings['subset_limit'] < 2:
        raise ValueError('Invalid start frame, pool size, or subset limit')
    if not settings['k_values'] or any(not isinstance(k,int) or k < 2 for k in settings['k_values']):
        raise ValueError('k_values must contain integer budgets >=2')
    if settings['demo_k'] not in settings['k_values']:
        raise ValueError('demo_k must be one of the training k_values')
    if not 0 <= settings['threshold'] <= 1:
        raise ValueError('threshold must be in [0,1]')
    if any(settings[x] <= 0 for x in ('lr','temperature_mm','match_mm','nms_mm')):
        raise ValueError('Learning rate, temperature, matching distance and NMS must be positive')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/train_mvp.json')
    parser.add_argument('--check', action='store_true', help='Check data and dependencies without training')
    parser.add_argument('--resume', action='store_true', help='Reuse verified teacher caches; retrain small selector deterministically')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    validate_settings(settings)
    from demo_mvp import check_environment, load_model, poses_from_output
    environment = check_environment()
    errors, inventory = [], []
    from mvp_demo.source import inspect_sequence, annotation_files, image_files
    data_root = resolve(settings['data_root'])
    for sequence in settings['train_sequences'] + settings['val_sequences']:
        try:
            ids, paths, _ = inspect_sequence(data_root/sequence, settings['cameras'], settings.get('input_format','auto'))
            if len(ids) < 3:
                raise ValueError('Selection training needs >=3 HD views')
            if settings['demo_k'] >= min(len(ids),settings['max_candidate_views']):
                raise ValueError('demo_k must leave at least one unselected camera during training')
            gt_dir = data_root/sequence/'hdPose3d_stage1_coco19'
            annotations = list(annotation_files(data_root/sequence).values())
            if not annotations:
                raise ValueError(f'Missing unpacked 3D annotations: {gt_dir}')
            source_files = [file for p in paths for file in (list(image_files(p).values()) if p.is_dir() else [p])]
            files = source_files + [data_root/sequence/f'calibration_{sequence}.json'] + annotations
            inventory.append(dict(sequence=sequence,cameras=ids,
                files=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in files]))
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{sequence}: {exc}')
    for key in ('pose_checkpoint','pose_config'):
        if not resolve(settings[key]).is_file():
            errors.append(f'Missing {key}: {resolve(settings[key])}')
    errors += [f'Unavailable runtime: {k}' for k,v in environment.items() if v is False]
    report = dict(environment=environment,errors=errors,
        sequences=[{k:v for k,v in x.items() if k != 'files'} for x in inventory])
    print(json.dumps(report,indent=2),flush=True)
    if args.check:
        return 1 if errors else 0
    if errors:
        raise RuntimeError('Preflight failed; correct the reported configuration/data/environment before training')
    import torch
    from mvp_demo.training import sha256, collect_records, fit_selector
    torch.manual_seed(settings['seed'])
    output = resolve(settings['output'])
    signature = dict(settings=settings,inventory=inventory,
        pose_sha256=sha256(resolve(settings['pose_checkpoint'])),
        config_sha256=sha256(resolve(settings['pose_config'])),
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'lib').rglob('*.py'))},
        runner_sha256={p.name:sha256(p) for p in (Path(__file__),ROOT/'run/demo_mvp.py')})
    if output.exists():
        if not args.resume:
            raise ValueError(f'{output} already exists; choose a new output or use --resume')
        old = json.loads((output/'provenance.json').read_text(encoding='utf-8'))
        if old != signature:
            raise ValueError('Resume rejected: data, code, checkpoint or settings changed; use a new output')
    else:
        output.mkdir(parents=True)
        (output/'provenance.json').write_text(json.dumps(signature,indent=2),encoding='utf-8')
    (output/'READY.json').unlink(missing_ok=True)
    model_args = SimpleNamespace(cfg=resolve(settings['pose_config']),checkpoint=resolve(settings['pose_checkpoint']))
    model, cfg = load_model(model_args, settings['demo_k'])
    model.requires_grad_(False)
    train = collect_records(model,cfg,settings,settings['train_sequences'],data_root,output/'cache','train',poses_from_output)
    validation = collect_records(model,cfg,settings,settings['val_sequences'],data_root,output/'cache','validation',poses_from_output)
    del model
    torch.cuda.empty_cache()
    _, metrics = fit_selector(train,validation,output,epochs=settings['epochs'],lr=settings['lr'],seed=settings['seed'],
                             temperature_mm=settings['temperature_mm'],patience=settings['patience'])
    bundle_dir = output/'bundle'
    bundle_dir.mkdir(exist_ok=True)
    shutil.copy2(resolve(settings['pose_checkpoint']),bundle_dir/'pose_model.pt')
    shutil.copy2(resolve(settings['pose_config']),bundle_dir/'pose_config.yaml')
    shutil.copy2(output/'selector.pt',bundle_dir/'selector.pt')
    # Ship a baseline fallback when validation does not support using the learned policy.
    demo_metrics = metrics['per_k'].get(str(settings['demo_k']))
    recommended = 'learned' if demo_metrics and demo_metrics['learned_cost_mm'] < demo_metrics['fixed_cost_mm'] else 'fixed'
    manifest = dict(schema_version=1,pose_checkpoint='pose_model.pt',pose_config='pose_config.yaml',
        selector_checkpoint='selector.pt',policy=recommended,k=settings['demo_k'],subset_limit=settings['subset_limit'],
        threshold=settings['threshold'],nms_mm=settings['nms_mm'],validation=metrics,
        image_fps=settings.get('image_fps',29.97002997),
        trained_max_candidate_views=max(len(r['camera_ids']) for r in train),
        training_scope='frozen pretrained MVGFormer + trained subset selector; not joint end-to-end training',
        hashes={p.name:sha256(p) for p in (bundle_dir/'pose_model.pt',bundle_dir/'pose_config.yaml',bundle_dir/'selector.pt')})
    manifest_path = bundle_dir/'bundle.json'
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    smoke_dir = output/'smoke'
    smoke_index = 0
    while smoke_dir.exists():
        smoke_index += 1
        smoke_dir = output/f'smoke_{smoke_index}'
    command = [sys.executable,str(ROOT/'run/demo_mvp.py'),'--bundle',str(manifest_path),
        '--sequence',str(data_root/settings['val_sequences'][0]),'--max-frames',str(settings['smoke_frames']),
        '--start',str(next((r['frame'] for r in validation if r['sequence']==settings['val_sequences'][0]
            and any(q['tp']+q['fn']>0 for q in r['quality'])),settings['start_frame'])),
        '--headless','--output',str(smoke_dir)]
    command += ['--input-format',settings.get('input_format','auto'),'--image-fps',str(settings.get('image_fps',29.97002997))]
    if settings['cameras']:
        command += ['--cameras',*settings['cameras']]
    subprocess.run(command,check=True)
    result = json.loads((smoke_dir/'summary.json').read_text(encoding='utf-8'))
    if result['frames'] < settings['smoke_frames']:
        raise RuntimeError('Demo smoke test ended before the required frame count')
    if result.get('pose_frames',0) == 0:
        raise RuntimeError('Demo executed but found no people; inspect input, checkpoint and threshold before presenting')
    (output/'READY.json').write_text(json.dumps(dict(bundle=str(manifest_path),smoke=result,
        note='Inference executed and returned poses; visual quality and realtime speed still require inspection'),indent=2),encoding='utf-8')
    print(f'Training and demo smoke completed. Bundle: {manifest_path}',flush=True)
    print(f'Validation-recommended policy: {recommended}; learned policy remains available explicitly.',flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
