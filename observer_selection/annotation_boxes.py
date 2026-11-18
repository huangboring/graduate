"""Extract target crops using official EgoHumans 2D annotations."""
from pathlib import Path

import numpy as np

from .egohumans import _hash
from .preparation_cache import atomic_json, atomic_npz


def annotated_box(path, target):
    # EgoHumans stores a NumPy object array in each official annotations file.
    rows = np.load(path, allow_pickle=True)
    matches = [r for r in rows if r['human_name'] == target]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f'Duplicate target annotations: {path}')
    box = np.asarray(matches[0]['bbox'][:4], dtype='f4')
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError(f'Invalid target annotation box: {path}')
    return box


def box_from_reverse_map(maps, box, rotate=False):
    import cv2

    x, y = maps
    x0, y0, x1, y1 = box
    if rotate:
        # Rotate the raw upright annotation into the native input image.
        x0, y0, x1, y1 = y0, x.shape[1]-1-x1, y1, x.shape[1]-1-x0
    mask = ((x>=x0) & (x<=x1) & (y>=y0) & (y<=y1)).astype('u1')
    if rotate:
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    a,b,w,h = cv2.boundingRect(mask)
    return [a,b,a+w,b+h] if w>1 and h>1 else None


def export_annotated_boxes(pair, frames, ids, timestamps, paths, target, fixed):
    import cv2

    pair=Path(pair)
    final=pair/'tracking_preview.mp4'
    pending=pair/'tracking_preview.pending.mp4'
    writer=cv2.VideoWriter(str(pending),cv2.VideoWriter_fourcc(*'mp4v'),20,(704,704))
    if not writer.isOpened():
        raise RuntimeError('Cannot open annotated box preview video')
    maps = cv2.convertMaps(*frames.maps,cv2.CV_32FC1) if fixed and frames.maps is not None else None
    valid=np.zeros(len(ids),bool)
    boxes=np.zeros((len(ids),4),'f4')
    try:
        for t,(raw_id,path) in enumerate(zip(ids,paths)):
            box=annotated_box(path,target)
            image=frames.get(t)
            if box is not None:
                if not fixed:
                    box=box_from_reverse_map(frames.maps,box,rotate=True)
                elif maps is not None:
                    box=box_from_reverse_map(maps,box)
                else:
                    ratio=np.asarray(frames.output_size)/frames.size
                    box=np.clip(box*np.tile(ratio,2),0,np.tile(frames.output_size,2))
                if box is not None and np.all(np.asarray(box[2:])>box[:2]):
                    boxes[t],valid[t]=box,True
            preview=cv2.cvtColor(cv2.resize(image,(704,704)),cv2.COLOR_RGB2BGR)
            if valid[t]:
                a,b,c,d=np.rint(boxes[t]*np.tile([704/image.shape[1],704/image.shape[0]],2)).astype(int)
                cv2.rectangle(preview,(a,b),(c,d),(0,225,30),2)
            cv2.putText(preview,f'{target} | frame {raw_id} | dataset annotation',
                        (12,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,240,255),2)
            writer.write(preview)
            if t%100==0:
                print(f'{pair.name}: annotation boxes {t+1}/{len(ids)}, valid={int(valid.sum())}',flush=True)
    finally:
        writer.release()
    pending.replace(final)
    file=pair/'detector_boxes.npz'
    atomic_npz(file,timestamps_ns=timestamps,last_input_t_ns=timestamps,bbox=boxes,bbox_valid=valid)
    atomic_json(pair/'detector_receipt.json',{'crop_source':'egohumans_annotation',
        'identity_source':'dataset_human_name','output_sha256':_hash(file),
        'annotation_target':target,'generator_sha256':_hash(Path(__file__))})
    return {'frames':len(ids),'valid_frames':int(valid.sum()),'valid_fraction':float(valid.mean()),
            'identity_protocol':'egohumans_annotation','terminal_track_loss':False}
