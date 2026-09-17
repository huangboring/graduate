"""Read-only dataset inventory; accepts a sequence directory or its parent.

No torch/CUDA required. JSON archives are read directly, never extracted.
This detects structural problems, not semantic frame alignment or pose accuracy.
"""
import argparse
import json
from pathlib import Path
import tarfile


def frame_id(path):
    tail=Path(path).stem.rsplit('_',1)[-1]
    return int(tail) if tail.isdigit() else None


def inspect_folder(folder):
    warnings=[]
    calibration=folder/f'calibration_{folder.name}.json'
    cameras={}
    if calibration.is_file():
        cameras={c['name']:c for c in json.loads(calibration.read_text(encoding='utf-8-sig'))['cameras'] if c.get('type')=='hd' or c.get('panel')==0}
    else:
        warnings.append('Missing sequence calibration JSON')
    try:
        import cv2
    except ImportError:
        cv2=None
        warnings.append('OpenCV unavailable: video/image decoding was not checked')
    video_paths=sorted(set(folder.glob('hdVideos/hd_00_*.mp4')) | set(folder.glob('hdvideos/hd_00_*.mp4')))
    videos=[]
    for path in video_paths:
        name=path.stem.removeprefix('hd_')
        entry=dict(camera=name,path=str(path.resolve()),bytes=path.stat().st_size,calibrated=name in cameras)
        if cv2:
            cap=cv2.VideoCapture(str(path))
            try:
                entry.update(opened=cap.isOpened(),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),fps=cap.get(cv2.CAP_PROP_FPS),
                    resolution=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))])
            finally:cap.release()
        videos.append(entry)
    images=[]
    for layout,pattern in [('legacy_images','hdVideos/hd_00_*'),('legacy_images','hdvideos/hd_00_*'),('hdimgs','hdImgs/00_*')]:
        for directory in sorted(folder.glob(pattern)):
            if not directory.is_dir():continue
            files=sorted(p for p in directory.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
            indices=[frame_id(p) for p in files]
            valid=sorted(x for x in indices if x is not None)
            name=directory.name.removeprefix('hd_')
            item=dict(layout=layout,camera=name,path=str(directory.resolve()),files=len(files),
                first_frame=valid[0] if valid else None,last_frame=valid[-1] if valid else None,
                duplicate_ids=len(valid)-len(set(valid)),unparseable_ids=sum(x is None for x in indices),calibrated=name in cameras)
            if files and cv2:
                image=cv2.imread(str(files[0]))
                item['first_image_resolution']=[image.shape[1],image.shape[0]] if image is not None else None
                if name in cameras and item['first_image_resolution'] != cameras[name]['resolution']:
                    warnings.append(f'{name}: extracted image size differs from calibration; prefer original videos')
            images.append(item)
    annotation_dir=folder/'hdPose3d_stage1_coco19'
    annotation_paths=sorted(annotation_dir.rglob('body3DScene_*.json'))
    archive_path=folder/'hdPose3d_stage1_coco19.tar'
    ids=[]; people_ids=[]; usable_ids=[]; invalid=[]; max_people=0
    def consume(name,stream):
        nonlocal max_people
        try:
            index=frame_id(name)
            if index is None:raise ValueError('invalid annotation filename')
            data=json.load(stream);bodies=data['bodies'];ids.append(index)
            max_people=max(max_people,len(bodies))
            if bodies:people_ids.append(index)
            if any(len(b.get('joints19',[]))==76 and b['joints19'][2*4+3]>.1 for b in bodies):usable_ids.append(index)
        except (ValueError,KeyError,TypeError) as exc:invalid.append(f'{name}: {exc}')
    if annotation_paths:
        annotation_source='unpacked'
        for path in annotation_paths:
            with path.open(encoding='utf-8-sig') as stream:consume(path.name,stream)
    elif archive_path.is_file():
        annotation_source='tar_only'
        warnings.append('3D annotations exist only in TAR; unpack before training')
        try:
            with tarfile.open(archive_path) as archive:
                for member in archive:
                    if member.isfile() and Path(member.name).name.startswith('body3DScene_') and member.name.endswith('.json'):
                        with archive.extractfile(member) as stream:consume(member.name,stream)
        except tarfile.TarError as exc:invalid.append(str(exc))
    else:
        annotation_source='missing';warnings.append('No 3D annotations found')
    if len(ids)!=len(set(ids)):warnings.append('Duplicate annotation frame IDs')
    available=set(v['camera'] for v in videos) | set(v['camera'] for v in images if v['files'])
    if len(available)<3:warnings.append('Fewer than 3 available cameras: cannot train a nontrivial K>=2 selector')
    if invalid:warnings.append('Invalid annotation files; see annotation_errors')
    for v in videos:
        if cv2 and v['frames'] and ids:
            v['annotations_in_video_range']=sum(0<=i<v['frames'] for i in ids)
    return dict(sequence=folder.name,path=str(folder.resolve()),calibration_hd_count=len(cameras),videos=videos,images=images,
        annotation_source=annotation_source,annotation_count=len(ids),first_annotation=min(ids) if ids else None,
        last_annotation=max(ids) if ids else None,frames_with_people=len(people_ids),
        first_people_frame=min(people_ids) if people_ids else None,
        first_usable_root_frame=min(usable_ids) if usable_ids else None,max_people=max_people,
        annotation_errors=invalid[:20],warnings=warnings,
        alignment='Filename/range checks only; extraction start_number, dropped frames and visual reprojection still need verification')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.data_root.resolve()
    if not root.is_dir():raise ValueError(f'Data folder does not exist: {root}')
    folders=[root] if (root/f'calibration_{root.name}.json').is_file() else sorted(p for p in root.iterdir() if p.is_dir() and (p/f'calibration_{p.name}.json').is_file())
    if not folders:raise ValueError('No calibrated Panoptic sequence folders found')
    report=dict(data_root=str(root),sequences=[inspect_folder(p) for p in folders],
                scope='Observed files on this computer only; does not establish contents of another computer')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
