"""Restartable tracking stage. Candidate caches contain image inference only."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from .appearance_tracking import AppearanceTracker, VERSION
from .preparation_cache import atomic_json, atomic_npz, load_candidates
from .egohumans import _hash


def longest_missing(valid):
    longest = current = 0
    for value in valid:
        current = 0 if value else current+1
        longest = max(longest, current)
    return longest


def extract_candidates(image, detector, dino, device):
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    with torch.inference_mode(): result = detector([to_tensor(image).to(device)])[0]
    keep = (result['labels']==1) & (result['scores']>=.7)
    boxes = result['boxes'][keep].detach().cpu().numpy().astype('f4')
    if not len(boxes): return boxes.reshape(0,4),np.empty((0,768),'f4')
    crops=[]
    for box in boxes:
        low=np.maximum(np.floor(box[:2]).astype(int),0)
        high=np.minimum(np.ceil(box[2:]).astype(int),[image.shape[1],image.shape[0]])
        if np.any(high<=low): raise ValueError('Detector produced empty crop')
        crops.append(Image.fromarray(image[low[1]:high[1],low[0]:high[0]]))
    chunks=[]
    for start in range(0,len(crops),8):
        values=dino.batch_cls(crops[start:start+8])
        if hasattr(values,'detach'): values=values.detach().cpu().numpy()
        chunks.append(np.asarray(values,dtype='f4'))
    features=np.concatenate(chunks)
    if features.shape!=(len(boxes),768) or not np.isfinite(features).all():
        raise ValueError('Invalid candidate appearance features')
    return boxes,features


def initial_choice(pair, raw_id, candidates, annotated, image, target, choices, reference_image=None, recovery=False):
    import cv2
    key=str(raw_id)
    if key in choices:
        event=choices[key]
        if event['candidate_boxes']!=candidates.tolist():
            raise ValueError('Cached identity decision no longer matches detector candidates')
        return event['choice']
    display=cv2.resize(annotated,(704,704))
    if reference_image is not None and Path(reference_image).is_file():
        reference=cv2.imread(str(reference_image))
        if reference is not None:
            panel=np.zeros((704,280,3),np.uint8)
            h,w=reference.shape[:2];factor=min(256/w,600/h)
            resized=cv2.resize(reference,(max(1,int(w*factor)),max(1,int(h*factor))))
            panel[55:55+resized.shape[0],12:12+resized.shape[1]]=resized
            cv2.putText(panel,'Previous view target',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            display=np.hstack([panel,display])
    file=pair/f'identity_frame_{raw_id:05}.jpg';cv2.imwrite(str(file),display)
    cv2.imwrite(str(pair/'identity_current.jpg'),display)
    print(f'\nOpen {file}\nIdentify the SAME target {target}. Numbers are detection indices, NOT Aria IDs.\n'
          + ('RECOVERY/REVIEW: this is an additional manual identity annotation.\n' if recovery else '')
          + 'Enter its box number; n=no matching box (retry after 20 frames); q=save cache and stop.',flush=True)
    while True:
        answer=input('Target box: ').strip().lower()
        if answer=='q': raise KeyboardInterrupt('Cache saved. Resume the same command with --resume.')
        if answer=='n': selected=None;break
        if answer.isdecimal() and int(answer)<len(candidates): selected=int(answer);break
        print('Enter a displayed box number, n, or q.',flush=True)
    choices[key]={'choice':selected,'candidate_boxes':candidates.tolist(),
                  'identity_source':'manual_recovery_review' if recovery else 'manual_initial_identity','raw_frame_id':raw_id}
    atomic_json(pair/('recovery_choices.json' if recovery else 'identity_choices.json'),choices)
    return selected


def track_view(pair, frames, ids, timestamps, detector, dino, device, target, checkpoint_id, reference_image=None, review_tracking=False):
    import cv2
    pair=Path(pair);cache=pair/'frame_cache';cache.mkdir(exist_ok=True)
    choice_file=pair/'identity_choices.json'
    choices=json.loads(choice_file.read_text(encoding='utf-8')) if choice_file.exists() else {}
    tracker=AppearanceTracker()
    T=len(ids);boxes=np.zeros((T,4),'f4');valid=np.zeros(T,bool)
    next_prompt=0;init_record=None;reasons=[];cache_hits=0
    recovery_choices_path=pair/'recovery_choices.json'
    recovery_choices=json.loads(recovery_choices_path.read_text()) if recovery_choices_path.exists() else {}
    misses=0;last_review=-1;manual_reviews=0;automatic_recoveries=0;diagnostics=[]
    pending=pair/'tracking_preview.pending.mp4'
    preview=cv2.VideoWriter(str(pending),cv2.VideoWriter_fourcc(*'mp4v'),20,(704,704))
    if not preview.isOpened(): raise RuntimeError('Cannot open tracking preview video writer')
    try:
        for t,raw_id in enumerate(ids):
            image=frames.get(t)
            candidate_file=cache/f'{raw_id:05}.npz'
            if candidate_file.exists():
                candidates,features=load_candidates(candidate_file);cache_hits+=1
            else:
                candidates,features=extract_candidates(image,detector,dino,device)
                atomic_npz(candidate_file,boxes=candidates,features=features)
            display=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            for i,box in enumerate(candidates):
                a,b,c,d=map(int,box);cv2.rectangle(display,(a,b),(c,d),(160,160,160),2)
                cv2.putText(display,str(i),(a,max(20,b)),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
            selected=None
            if not tracker.initialized and init_record is None:
                if len(candidates) and t>=next_prompt:
                    selected=initial_choice(pair,raw_id,candidates,display,image,target,choices,reference_image)
                    if selected is None: next_prompt=t+20
                    else:
                        tracker.initialize(t,candidates[selected],features[selected])
                        init_record={'raw_frame_id':raw_id,'box_index':selected,'bbox':candidates[selected].tolist()}
                        a,b,c,d=map(int,candidates[selected]);cv2.rectangle(display,(a,b),(c,d),(0,255,0),4)
                        cv2.imwrite(str(pair/'identity_selected.jpg'),display)
                        cv2.imwrite(str(pair/'identity_reference.jpg'),cv2.cvtColor(image[b:d,a:c],cv2.COLOR_RGB2BGR))
            else:
                selected=tracker.step(t,candidates,features)
                if tracker.reason=='appearance_recovery': automatic_recoveries+=1
            diagnostics.append({'raw_frame_id':raw_id,'reason':tracker.reason,'selected':selected,
                                'candidates':tracker.diagnostics})
            misses=misses+1 if selected is None else 0
            # Additional supervision is explicit and chronological. No later
            # choice repairs or interpolates earlier frames.
            if review_tracking and init_record is not None and t>=next_prompt and len(candidates) and (
                    misses>=5 or (last_review>=0 and t-last_review>=100) or str(raw_id) in recovery_choices):
                selected=initial_choice(pair,raw_id,candidates,display,image,target,recovery_choices,
                                        pair/'identity_reference.jpg',recovery=True)
                manual_reviews+=1;last_review=t;next_prompt=t+20
                if selected is not None:
                    tracker=AppearanceTracker();tracker.initialize(t,candidates[selected],features[selected])
                    tracker.reason='manual_recovery_review';misses=0
                else:
                    tracker=AppearanceTracker();tracker.reason='manual_no_matching_box'
            if init_record is not None and last_review<0: last_review=t
            reasons.append(tracker.reason)
            if selected is not None:
                boxes[t]=candidates[selected];valid[t]=True
                a,b,c,d=map(int,boxes[t]);cv2.rectangle(display,(a,b),(c,d),(0,255,0),4)
            cv2.putText(display,f'frame {raw_id}: {tracker.reason}',(20,40),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,255,255),2)
            preview.write(cv2.resize(display,(704,704)))
            if t%50==0:
                print(f'{pair.name}: tracking {t+1}/{T}, valid={int(valid[:t+1].sum())}, '
                      f'recoveries={automatic_recoveries}, manual_reviews={manual_reviews}, cached={cache_hits}',flush=True)
    finally: preview.release()
    if not valid.any(): raise ValueError(f'No target identity initialized for {pair.name}. Use --reset-view to correct saved choices.')
    pending.replace(pair/'tracking_preview.mp4')
    atomic_json(pair/'tracking_diagnostics.json',diagnostics)
    atomic_npz(pair/'detector_boxes.npz',timestamps_ns=timestamps,last_input_t_ns=timestamps,bbox=boxes,bbox_valid=valid)
    receipt={'crop_source':'local_image_detector','reference_pose_used':False,
             'output_sha256':_hash(pair/'detector_boxes.npz'),'detector_checkpoint':checkpoint_id,
             'identity_source':'manual_initial_and_recovery_review' if review_tracking else 'manual_initial_identity','initial_identity':init_record,
             'identity_choices_sha256':_hash(choice_file),'tracker':tracker.description(),
             'appearance_model':dino.identity,'score_threshold':.7,'generator_sha256':_hash(Path(__file__))}
    if review_tracking:
        if not recovery_choices_path.exists(): atomic_json(recovery_choices_path,{})
        receipt['manual_recovery_choices_sha256']=_hash(recovery_choices_path)
        receipt['manual_review_count']=manual_reviews
    atomic_json(pair/'detector_receipt.json',receipt)
    return {'tracker_version':VERSION,'frames':T,'valid_frames':int(valid.sum()),'valid_fraction':float(valid.mean()),
            'automatic_recoveries':automatic_recoveries,'manual_review_count':manual_reviews,
            'identity_protocol':'manual_initial_and_recovery_review' if review_tracking else 'manual_initial_only',
            'longest_missing_frames':longest_missing(valid),
            'reason_counts':dict(Counter(reasons)),'ends_unmatched':not bool(valid[-1]),
            'terminal_track_loss':False,'cache_hits':cache_hits}
