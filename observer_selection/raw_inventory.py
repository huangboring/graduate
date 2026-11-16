"""Read-only inventory of an official EgoHumans sequence; no inferred poses."""
from pathlib import Path
import json
import numpy as np
from PIL import Image


def inventory_raw(sequence, output):
    root = Path(sequence).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError('Choose a new inventory output')

    def numbered(directory, suffix):
        return {int(p.stem): p for p in directory.glob('*'+suffix)
                if p.stem.isdecimal() and p.is_file()}

    report = {'sequence':str(root), 'cameras':{}, 'status':'inventory_only_not_training_ready'}
    grids = {}
    for directory in sorted((root/'ego').glob('aria*')):
        if not directory.is_dir(): continue
        images = numbered(directory/'images/rgb', '.jpg')
        calibration = numbered(directory/'calib', '.txt')
        grids[directory.name+'/images'] = set(images)
        grids[directory.name+'/calib'] = set(calibration)
        row = {'images':len(images), 'calibrations':len(calibration),
               'first_frame':min(images) if images else None,
               'last_frame':max(images) if images else None,
               'image_calibration_ids_equal':set(images)==set(calibration)}
        if images:
            first = images[min(images)]
            with Image.open(first) as im: row['image_size'] = list(im.size)
        if calibration:
            text = calibration[min(calibration)].read_text().splitlines()
            row['first_calibration_text'] = text
        report['cameras'][directory.name] = row
    smpl = numbered(root/'processed_data/smpl', '.npy')
    grids['smpl'] = set(smpl)
    report['smpl_frames'] = len(smpl)
    # Official dataset SMPL files are pickled dictionaries. Only use trusted
    # files downloaded from the official dataset; never load arbitrary uploads.
    if smpl:
        file = smpl[min(smpl)]
        data = np.load(file, allow_pickle=True).item()
        if not isinstance(data, dict): raise ValueError('Expected official SMPL dictionary')
        report['first_smpl_fields'] = {
            str(person):{str(key):{'shape':list(np.asarray(value).shape),
                                  'dtype':str(np.asarray(value).dtype)}
                         for key,value in fields.items()}
            for person,fields in data.items() if isinstance(fields,dict)}
    union = set.union(*grids.values()) if grids else set()
    common = set.intersection(*grids.values()) if grids else set()
    report['all_frame_ids_equal'] = bool(union) and common == union
    report['common_frame_count'] = len(common)
    report['missing_frame_ids_first20'] = {name:sorted(union-ids)[:20] for name,ids in grids.items()}
    transform = root/'colmap/workplace/colmap_from_aria_transforms.pkl'
    report['world_transform_file_exists'] = transform.is_file()
    report['processed_data_directories'] = sorted(p.name for p in (root/'processed_data').iterdir() if p.is_dir())
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report
