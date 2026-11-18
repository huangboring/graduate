"""Official EgoHumans raw data -> causal, manually initialized image tracks.

Camera convention follows EgoHumans AriaCamera.get_undistorted_image_aria:
upright RGB -> CCW native -> rectification -> CW upright. Camera axes rotate
with the final image; K stays conventional positive-diagonal pinhole.
GT is loaded separately AFTER image inference, exclusively for supervision.
"""
from pathlib import Path
import json
import numpy as np
from .local_fusion import rigid


Q_UPRIGHT = np.array([[0.,-1,0],[1,0,0],[0,0,1]])


def read_calibration(path):
    lines = Path(path).read_text().splitlines()[1:]
    if len(lines) != 7: raise ValueError(f'Unexpected calibration: {path}')
    intr = np.fromstring(lines[1], sep=' ')
    ext = np.eye(4); ext[:3] = np.fromstring(lines[2], sep=' ').reshape(4,3).T
    if intr.shape != (15,) or not np.isfinite(intr).all() or intr[0] <= 0:
        raise ValueError('Invalid RGB calibration')
    rigid(np.linalg.inv(ext), 'native camera')
    return intr, ext


def fisheye_project(points, intr):
    """Vectorized radtanthinprism convention used by the official dataset."""
    xy = np.asarray(points)[...,:2] / np.asarray(points)[...,2:3]
    r = np.linalg.norm(xy, axis=-1); theta = np.arctan(r)
    polynomial = np.ones_like(r)
    for i, coefficient in enumerate(intr[3:9]): polynomial += coefficient * theta**(2*i+2)
    scale = np.divide(theta, r, out=np.ones_like(r), where=r>1e-9)*polynomial
    radial = xy*scale[...,None]
    r2 = (radial**2).sum(-1)
    distorted = radial + 2*radial**2*intr[9:11] + r2[...,None]*intr[9:11]
    distorted[...,0] += intr[11]*r2 + intr[12]*r2**2
    distorted[...,1] += intr[13]*r2 + intr[14]*r2**2
    return intr[0]*distorted + intr[1:3]


def make_map(intr, size=1408):
    f, c = size/2, (size-1)/2
    y, x = np.mgrid[:size,:size]
    rays = np.stack([(x-c)/f,(y-c)/f,np.ones_like(x)],axis=-1)
    pixels = fisheye_project(rays,intr)
    return pixels[...,0].astype('f4'),pixels[...,1].astype('f4')


def similarity_camera_extrinsic(transform):
    """Normalize a positive uniform similarity without changing projection.

    E=[sR,t] maps anchor-world coordinates to native-camera coordinates.
    E/s=[R,t/s] preserves image rays AND world camera centre -R.T@t/s.
    Dividing R alone would change the camera centre and is incorrect.
    This does not rescale the anchor world, GT, or pinhole intrinsics.
    """
    a=np.asarray(transform,dtype=np.float64)
    if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1],atol=1e-8,rtol=0):
        raise ValueError('Camera alignment must be a finite affine 4x4 transform')
    determinant=float(np.linalg.det(a[:3,:3]))
    if determinant<=0:
        raise ValueError(f'Camera alignment must have positive scale and no reflection; det={determinant}')
    scale=float(np.cbrt(determinant))
    result=a.copy();result[:3,:]/=scale
    rotation=result[:3,:3]
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=2e-4,rtol=0):
        singular=np.linalg.svd(a[:3,:3],compute_uv=False)
        raise ValueError(f'Camera alignment is not uniform scale times rotation; singular_values={singular.tolist()}')
    return result,scale


def upright_camera(native_world_to_camera):
    native_world_to_camera,_=similarity_camera_extrinsic(native_world_to_camera)
    rotation = np.eye(4); rotation[:3,:3] = Q_UPRIGHT
    return rigid(np.linalg.inv(rotation @ native_world_to_camera), 'upright camera')


def validate_setup_restart(output, roster):
    """Allow retry ONLY before observation extraction, preserving model cache."""
    out=Path(output)
    if not out.exists(): return
    for path in out.iterdir():
        if path.is_symlink(): raise FileExistsError(f'Restart refuses symlink: {path}')
        if path.name=='model_cache' and path.is_dir(): continue
        if path.name not in roster or not path.is_dir():
            raise FileExistsError(f'Restart only supports camera setup failures; found {path}')
        for child in path.iterdir():
            if child.is_symlink() or not child.is_file() or child.name not in ('meta.json','timestamps.csv','observer.npz'):
                raise FileExistsError(f'Observation work already exists; refusing to overwrite {child}')


def check_existing_setup(pair, meta, timestamp_text):
    for name,value in (('meta.json',meta),('timestamps.csv',timestamp_text)):
        path=Path(pair)/name
        if not path.exists(): continue
        old=json.loads(path.read_text(encoding='utf-8')) if name.endswith('.json') else path.read_text()
        if old!=value: raise ValueError(f'Restart settings/frame grid differ from {path}')


class RectifiedFrames:
    def __init__(self, paths, calibrations):
        self.paths, self.calibrations = paths, calibrations
        self.intr = self.maps = None

    def get(self, index):
        import cv2
        intr, _ = read_calibration(self.calibrations[index])
        if self.intr is None or not np.array_equal(intr,self.intr):
            self.maps = make_map(intr); self.intr = intr
        image = cv2.imdecode(np.fromfile(self.paths[index],dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.shape != (1408,1408,3):
            raise ValueError(f'Expected 1408x1408 RGB image: {self.paths[index]}')
        native = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        rectified = cv2.remap(native,*self.maps,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        return cv2.cvtColor(cv2.rotate(rectified,cv2.ROTATE_90_CLOCKWISE),cv2.COLOR_BGR2RGB)

    def __iter__(self):
        for index in range(len(self.paths)):
            if index % 50 == 0: print(f'  feature frame {index+1}/{len(self.paths)}',flush=True)
            yield self.get(index)


def match_track(previous, boxes, minimum_iou=.2, ambiguity_margin=.1):
    """Conservative baseline: reject ambiguous/crossing detections, no GT."""
    if previous is None or not len(boxes): return None
    lo=np.maximum(previous[:2],boxes[:,:2]); hi=np.minimum(previous[2:],boxes[:,2:])
    inter=np.maximum(hi-lo,0).prod(-1)
    union=(previous[2:]-previous[:2]).prod()+(boxes[:,2:]-boxes[:,:2]).prod(-1)-inter
    scores=inter/np.maximum(union,1e-8); order=np.argsort(scores)[::-1]
    if scores[order[0]] < minimum_iou: return None
    if len(order)>1 and scores[order[0]]-scores[order[1]]<ambiguity_margin: return None
    return int(order[0])


def prepare_raw(sequence, output, target='aria01', requester='aria02', sources=('aria03','aria04'),
                repo='external/EgoExoMoCap', body_model=None, device='cuda', restart_setup=False,
                resume=False, model_cache=None, reset_views=(), review_tracking=False, reuse_candidates=None, annotation_boxes=False):
    from .raw_pipeline import prepare_raw as run
    return run(sequence, output, target, requester, sources, repo, body_model, device,
               restart_setup, resume, model_cache, reset_views, review_tracking, reuse_candidates, annotation_boxes)
