"""Panoptic streaming demo. Run --help without loading MVGFormer extensions."""
import argparse
from dataclasses import asdict
from datetime import datetime
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lib'))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, help='Panoptic sequence folder, not dataset root')
    parser.add_argument('--cameras', nargs='+', help='Camera IDs such as 00_00 00_01 00_02')
    parser.add_argument('--input-format',choices=['auto','videos','legacy_images','hdimgs'],default='auto')
    parser.add_argument('--image-fps',type=float,help='Original HD FPS for extracted images; frame IDs must be preserved')
    parser.add_argument('--cfg', type=Path, default=ROOT / 'configs/panoptic/knn5-lr4-q1024.yaml')
    parser.add_argument('--checkpoint', type=Path, help='Full MVGFormer state_dict checkpoint')
    parser.add_argument('--bundle', type=Path, help='bundle.json exported by train_mvp.py')
    parser.add_argument('--policy', choices=['all', 'fixed', 'descriptor', 'learned'])
    parser.add_argument('--k', type=int)
    parser.add_argument('--subset-limit', type=int)
    parser.add_argument('--fixed-cameras', nargs='+')
    parser.add_argument('--selector-checkpoint', type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nms-mm', type=float)
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / ('mvp_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--check', action='store_true', help='Check environment/data without model construction')
    return parser.parse_args()


def configure_bundle(args):
    manifest = {}
    if args.bundle:
        manifest = json.loads(args.bundle.read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 1:
            raise ValueError('Unsupported model bundle version')
        directory = args.bundle.resolve().parent
        for field, target in [('pose_checkpoint','checkpoint'),('pose_config','cfg'),('selector_checkpoint','selector_checkpoint')]:
            path = (directory/manifest[field]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError('Model bundle paths must stay inside the bundle directory')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != manifest['hashes'][path.name]:
                raise ValueError(f'Bundle checksum mismatch: {path.name}')
            setattr(args,target,path)
    for field, fallback in [('policy','all'),('k',3),('threshold',.5),('nms_mm',200.),('subset_limit',64),('image_fps',29.97002997)]:
        if getattr(args,field,None) is None:
            setattr(args,field,manifest.get(field,fallback))
    return manifest


def check_environment():
    packages = ('torch', 'torchvision', 'numpy', 'cv2', 'scipy', 'yaml',
                'easydict', 'matplotlib', 'PIL', 'Deformable')
    report = {name: importlib.util.find_spec(name) is not None for name in packages}
    if report['torch']:
        import torch
        report['torch_version'] = torch.__version__
        report['cuda_available'] = torch.cuda.is_available()
    return report


def load_model(args, number_views):
    import torch
    from core.config import config, update_config
    update_config(str(args.cfg.resolve()))
    config.DATASET.CAMERA_NUM = number_views
    config.TEST.BATCH_SIZE = 1
    config.DEBUG.LOG_VAL_LOSS = False
    config.DEBUG.VISUALIZATION_JUMP_NUM = -1
    config.DECODER.gt_match = False
    config.DECODER.gt_match_test = False
    config.DECODER.init_ref_method = 'sample_space'
    # The existing gate has no compatible pretrained weights and is post-fusion.
    # Use the original pretrained decoder while testing pre-fusion selection.
    config.DECODER.feature_update_method = 'MLP'
    config.DECODER.t_pose_dir = str(ROOT / 'tpose.pt')
    from models.dq_transformer import get_mvp
    model = get_mvp(config, is_train=False)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    # Old local decoder gates are dormant under MLP. Everything active must load.
    dormant = ('.view_query_proj.', '.view_key_proj.', '.comm_gate.')
    missing = [k for k in result.missing_keys if not any(p in k for p in dormant)]
    if missing or result.unexpected_keys:
        raise ValueError(f'Incompatible checkpoint: missing={missing}, unexpected={result.unexpected_keys}')
    model = model.cuda().eval()
    return model, config


def poses_from_output(output, num_joints, threshold, nms_mm):
    import numpy as np
    scores = output['pred_logits'][0, :, 1].sigmoid().detach().cpu().numpy()
    poses = output['pred_poses']['outputs_coord'][0].reshape(-1, num_joints, 3).detach().cpu().numpy()
    kept = []
    for i in np.argsort(-scores):
        if scores[i] < threshold or not np.isfinite(poses[i]).all() or not np.any(poses[i]):
            continue
        if any(np.linalg.norm(poses[i] - poses[j], axis=-1).mean() < nms_mm for j in kept):
            continue
        kept.append(i)
    return poses[kept], scores[kept]


def display(frames, ids, selection, poses, fps, latency_ms):
    import cv2
    import numpy as np
    tiles = []
    for i, frame in enumerate(frames):
        tile = cv2.resize(frame, (320, 180))
        color = (0, 220, 0) if i in selection.indices else (110, 110, 110)
        cv2.rectangle(tile, (1, 1), (318, 178), color, 3)
        cv2.putText(tile, f'{ids[i]} score={selection.scores[i]:.3g}', (8, 24), 0, .55, color, 1)
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    mosaic = np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)])
    canvas = np.zeros((440, 960, 3), dtype=np.uint8)
    # Isometric projection of reconstructed XYZ (mm); not a camera overlay.
    limbs = [(0,1),(0,2),(0,3),(3,4),(4,5),(0,9),(9,10),(10,11),
             (2,6),(6,7),(7,8),(2,12),(12,13),(13,14)]
    for pose in poses:
        points = np.column_stack((480 + .065 * pose[:,0] - .03 * pose[:,1],
                                  380 - .12 * pose[:,2] - .02 * pose[:,1])).astype(int)
        for a, b in limbs:
            cv2.line(canvas, tuple(points[a]), tuple(points[b]), (255, 200, 40), 2)
    lines = [f'3D pose, isometric view | people={len(poses)}',
             f'Inference pipeline {fps:.2f} FPS | {latency_ms:.1f} ms (excludes display)',
             f'SIMULATED tensor traffic: {selection.total_bytes / 1e6:.3f} MB/frame | saving {selection.saving_fraction:.1%}']
    for y, text in enumerate(lines):
        cv2.putText(canvas, text, (12, 24 + y*25), 0, .55, (255,255,255), 1)
    cv2.imshow('MVGFormer MVP - Q or Esc to stop', np.vstack((mosaic, canvas)))
    return cv2.waitKey(1) & 255 not in (ord('q'), 27)


def main():
    args = arguments()
    bundle_settings = configure_bundle(args)
    report = check_environment()
    if args.sequence:
        from mvp_demo.source import inspect_sequence
        ids, paths, cameras = inspect_sequence(args.sequence, args.cameras, args.input_format)
        trained_max = bundle_settings.get('trained_max_candidate_views')
        if trained_max and len(ids) > trained_max:
            print(f'Using {len(ids)} candidates; selector was trained with at most {trained_max}. This is an extrapolation experiment.')
        report['cameras'] = ids
        report['videos'] = list(map(str, paths))
    if args.check:
        print(json.dumps(report, indent=2))
        return
    if not args.sequence or not args.checkpoint or not args.checkpoint.is_file():
        raise ValueError('Provide --sequence and an existing full --checkpoint; use --check for diagnostics')
    missing = [k for k, v in report.items() if v is False]
    if missing:
        raise RuntimeError(f'MVGFormer runtime unavailable: {missing}. Use the upstream CUDA environment.')
    if args.max_frames < 1 or args.start < 0 or not 0 <= args.threshold <= 1 or args.nms_mm <= 0:
        raise ValueError('Invalid frame range, confidence threshold, or NMS radius')
    k = len(ids) if args.policy == 'all' else args.k
    if not 2 <= k <= len(ids):
        raise ValueError('Require 2 <= k <= number of candidate cameras')
    import cv2
    import torch
    from mvp_demo.source import open_source, prepare_frame, camera_geometry
    from mvp_demo.selection import QueryKeySelector, SubsetSelector, select, select_payload
    torch.manual_seed(args.seed)
    model, cfg = load_model(args, k)
    scorer = None
    if args.policy == 'learned':
        if not args.selector_checkpoint:
            raise ValueError('A separately trained --selector-checkpoint is required')
        checkpoint = torch.load(args.selector_checkpoint, map_location='cpu', weights_only=True)
        if checkpoint.get('architecture') == SubsetSelector.architecture:
            scorer = SubsetSelector(checkpoint['dimension'], checkpoint['hidden']).cuda().eval()
            scorer.load_state_dict(checkpoint['state_dict'],strict=True)
        else:
            scorer = QueryKeySelector().cuda().eval()
            scorer.load_state_dict(checkpoint, strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    environment=report, decoder_feature_update='MLP',
                    traffic='simulated uncompressed feature tensors + descriptors + decision mask; excludes headers/calibration/RGB display',
                    evaluation='No ground truth loaded; no accuracy metric claimed')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    source = open_source(paths, args.start,args.image_fps)
    processed = pose_frames = 0
    wall_start = time.perf_counter()
    try:
        with (args.output / 'frames.jsonl').open('w', encoding='utf-8') as log, torch.inference_mode():
            for _ in range(args.max_frames):
                torch.cuda.synchronize()
                started = time.perf_counter()
                bundle = source.read()
                if bundle is None:
                    break
                frame_id, frames = bundle
                prepared = [prepare_frame(f, c, cfg.NETWORK.IMAGE_SIZE, cfg.NETWORK.HEATMAP_SIZE, 'cuda', cfg.DATASET.COLOR_RGB)
                            for f,c in zip(frames,cameras)]
                views, metadata = map(list, zip(*prepared))
                # Simulate local feature extraction on each camera node.
                levels_by_view = [model.extract_view_features([view]) for view in views]
                features = [torch.cat(levels, 0) for levels in zip(*levels_by_view)]
                selection = select(features, ids, k, args.policy, args.fixed_cameras, scorer,
                                   geometry=camera_geometry(metadata), subset_limit=args.subset_limit)
                payload, selected_views, selected_meta = select_payload(features, views, metadata, selection)
                output = model(views=selected_views, meta=selected_meta,
                               precomputed_features=payload, threshold=args.threshold)
                poses, confidence = poses_from_output(output, cfg.NETWORK.NUM_JOINTS, args.threshold, args.nms_mm)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                record = dict(frame=frame_id, source_time_s=frame_id/source.fps,
                              camera_ids=ids, selected_ids=[ids[i] for i in selection.indices],
                              communication=asdict(selection), total_bytes=selection.total_bytes,
                              saving_fraction=selection.saving_fraction, pipeline_ms=elapsed*1000,
                              poses_mm=poses.tolist(), confidence=confidence.tolist())
                log.write(json.dumps(record) + '\n')
                log.flush()
                processed += 1
                pose_frames += int(len(poses) > 0)
                print(f'frame={frame_id} selected={record["selected_ids"]} people={len(poses)} pipeline_ms={elapsed*1000:.1f} simulated_bytes={selection.total_bytes}')
                if not args.headless and not display(frames, ids, selection, poses, 1/elapsed, elapsed*1000):
                    break
    finally:
        source.close()
        cv2.destroyAllWindows()
        duration = time.perf_counter() - wall_start
        (args.output / 'summary.json').write_text(json.dumps(dict(frames=processed, pose_frames=pose_frames, wall_seconds=duration,
            wall_fps=processed/duration if duration else 0), indent=2), encoding='utf-8')
    if not processed:
        raise RuntimeError('No complete synchronized frame bundle decoded; check start frame and videos')


if __name__ == '__main__':
    main()
