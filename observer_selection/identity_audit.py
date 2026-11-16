"""Offline annotation-assisted identity review, never imported by training.

Official schema: processed_data/poses2d/<camera>/rgb/<frame>.npy contains
records with human_name, bbox (xyxy), keypoints and optional is_valid.
Use EXO images to avoid mixing native Aria and rectified coordinates.
"""
from pathlib import Path
import html
import json
import numpy as np
from PIL import Image, ImageDraw
from .egohumans import _hash


def annotation_records(path):
    records = np.load(path, allow_pickle=True).tolist()
    if not isinstance(records, list):
        raise ValueError(f'Expected official list of pose records: {path}')
    result = []
    for record in records:
        if not isinstance(record, dict) or 'human_name' not in record or 'bbox' not in record:
            raise ValueError(f'Missing explicit human_name/bbox; will not guess IDs: {path}')
        name = str(record['human_name'])
        if name not in ('aria01', 'aria02', 'aria03', 'aria04'):
            raise ValueError(f'Unexpected identity {name}: {path}')
        box = np.asarray(record['bbox'], dtype=float).reshape(-1)
        if box.size not in (4, 5) or not np.isfinite(box).all():
            raise ValueError(f'Invalid bbox: {path}')
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        keypoints = np.asarray(record.get('keypoints', []), dtype=float)
        usable = bool(record.get('is_valid', True)) and keypoints.ndim == 2 and keypoints.shape[1] >= 3
        usable = usable and bool(np.any(np.isfinite(keypoints[:, 2]) & (keypoints[:, 2] > 0)))
        result.append((name, box[:4], usable))
    return result


def audit_identity(sequence, output, frame=1):
    root, out = Path(sequence).resolve(), Path(output).resolve()
    if frame < 1: raise ValueError('Frame must be positive')
    if not root.is_dir(): raise FileNotFoundError(root)
    if out.exists(): raise FileExistsError('Choose a new identity audit output directory')
    rows, problems, inputs = [], [], []
    for camera in sorted((root/'exo').glob('cam*')):
        if not camera.is_dir(): continue
        annotation = root/'processed_data/poses2d'/camera.name/'rgb'/f'{frame:05}.npy'
        image = camera/'images'/f'{frame:05}.jpg'
        if not image.is_file(): image = camera/'images/rgb'/f'{frame:05}.jpg'
        if not annotation.is_file() or not image.is_file():
            problems.append(f'{camera.name}: missing image or poses2d at frame {frame}')
            continue
        records = annotation_records(annotation)
        with Image.open(image) as source: canvas = source.convert('RGB')
        canvas.thumbnail((1600, 1600))
        with Image.open(image) as source: sx, sy = canvas.width/source.width, canvas.height/source.height
        draw = ImageDraw.Draw(canvas)
        for name, box, usable in records:
            x1,y1,x2,y2 = box*np.array([sx,sy,sx,sy])
            color = '#00ff50' if usable else '#ffaa00'
            draw.rectangle((x1,y1,x2,y2), outline=color, width=4)
            label = name + ('' if usable else ' [unreliable annotation]')
            x, y = max(0,min(x1,canvas.width-200)), max(0,min(y1-24,canvas.height-24))
            bounds = draw.textbbox((x,y),label,font_size=20)
            draw.rectangle(bounds,fill='black');draw.text((x,y),label,fill=color,font_size=20)
        rows.append((camera.name, canvas, [name for name,_,ok in records if ok]))
        inputs.extend({'path':str(p),'sha256':_hash(p)} for p in (annotation,image))
    if not rows:
        raise ValueError('No matching EXO images/official poses2d. Nothing generated. '+ '; '.join(problems))
    out.mkdir(parents=True)
    body = ['<meta charset="utf-8"><title>Dataset identity audit</title>',
            '<h1>Annotation-assisted identity initialization</h1>',
            '<p>Dataset IDs, NOT detector indices. Review visually; annotations may be wrong. '
            'Orange boxes have unreliable pose evidence. This page is not a training input.</p>',
            f'<p>Raw frame {frame}. Use the first frame for causal initialization; '
            'later-frame review must not be presented as first-frame causal initialization.</p>']
    for camera, canvas, names in rows:
        filename=f'{camera}_{frame:05}.jpg';canvas.save(out/filename)
        body.append(f'<h2>{html.escape(camera)}</h2><img style="max-width:100%" src="{filename}">')
    report={'scope':'offline_annotation_assisted_identity_review_only', 'frame':frame,
            'sequence':str(root),'input_files':inputs,'warnings':problems,
            'usable_annotation_ids':{camera:names for camera,_,names in rows},
            'disclosure':'Manual identity initialization assisted by supplied dataset 2D annotations. '
                         'No automatic identity assignment, training crops, or feature extraction.'}
    (out/'identity_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'index.html').write_text('\n'.join(body),encoding='utf-8')
    return {'index':str(out/'index.html'), **report}
