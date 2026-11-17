"""Export full, frame-aligned values from an existing demo inspection (NumPy only)."""
import argparse
import json
from pathlib import Path
import numpy as np

JOINTS = 'pelvis left_hip right_hip spine1 left_knee right_knee spine2 left_ankle right_ankle spine3 left_foot right_foot neck left_collar right_collar head left_shoulder right_shoulder left_elbow right_elbow left_wrist right_wrist'.split()
KEYPOINTS = 'nose left_shoulder right_shoulder left_elbow right_elbow left_wrist right_wrist left_hip right_hip left_knee right_knee left_ankle right_ankle'.split()


def frame_value(name, value, index, length):
    """Only accept the layouts actually recorded by demo_inspect, with B=1."""
    if not 0 <= index < length:
        raise IndexError('Frame index outside recorded window')
    if name == 'AvatarNet.output.egonet_pos' and value.shape == (length, 22, 3):
        return value[index], 'B*T, joints=22, xyz=3; B=1 (flattened FK output)'
    if name.startswith('fusion.spatial_encoder.'):
        if value.ndim >= 2 and value.shape[0] == length:
            return value[index], 'B*T, tokens, [features]; B=1'
    elif value.ndim >= 2 and value.shape[:2] == (1, length):
        return value[0, index], 'B, T, ...; B=1'
    raise ValueError(f'Unrecognized trace layout: {name} {value.shape}; refusing ambiguous frame slicing')


def sparse_fields(x):
    x = np.asarray(x)
    if x.shape not in ((18,), (54,)):
        raise ValueError(f'Unexpected sparse layout: {x.shape}')
    n = x.size // 18
    names = ['head', 'left_wrist', 'right_wrist'][:n]
    return {name: {
        'rotation_6d': x[6*i:6*i+6].tolist(),
        'relative_rotation_6d': x[6*n+6*i:6*n+6*i+6].tolist(),
        'position_xyz_m': x[12*n+3*i:12*n+3*i+3].tolist(),
        'position_difference_xyz_m_per_frame': x[15*n+3*i:15*n+3*i+3].tolist(),
    } for i, name in enumerate(names)}


def export_values(run, raw_frame=2):
    run = Path(run).resolve()
    inspection = run / 'inspection'
    protocol = json.loads((run / 'demo_protocol.json').read_text(encoding='utf-8'))
    with np.load(run / 'predictions.npz', allow_pickle=False) as z:
        predictions = {k: z[k] for k in z.files}
    with np.load(inspection / 'module_arrays.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    sparse = arrays['AvatarNet.input.0.sparse_input']
    if sparse.ndim != 3 or sparse.shape[0] != 1:
        raise ValueError('Expected a single-batch inspection')
    length = sparse.shape[1]
    matches = np.flatnonzero(predictions['raw_frame_ids'] == raw_frame)
    if len(matches) != 1 or matches[0] >= length:
        raise ValueError('Requested frame is outside the recorded first inference window; choose one of: '
                         + str(predictions['raw_frame_ids'][:length].tolist()))
    index = int(matches[0])
    notes = {
        'scope': '指定幀的完整數值；時間模型仍使用整個視窗，不是只以這一幀推論。未重新執行模型。',
        'sparse': '3pt依序為旋轉18、相對旋轉18、位置9、位置差9；1pt為6、6、3、3。位置單位m，z已減去資料集參考地板高度。',
        'rotation_6d': '旋轉矩陣前兩個欄向量的6D表示，不是六個角度。相對旋轉由前幀與當幀旋轉計算；位置差不是m/s。',
        'learned_features': 'DINO、embedding、Transformer每一維是學習出的特徵，沒有獨立的關節或物理單位。保留全部數值以供檢查。',
        'exo_rays': '13個點各3維：2D反投影後按兩人頭部距離縮放，轉到目標頭部局部座標。不是量測到的3D關節。順序見keypoint_order。',
        'scorenet': '13個sigmoid權重，用來乘上對應ray三個分量；不能直接宣稱是遮蔽機率或校準後信心。',
        'exo_token': '輸入為13*3個加權ray值及最後1個bbox_valid；空間token順序為ego、exo1、exo2…。',
        'spatial_mask': 'src_key_padding_mask若存在，True代表不參與注意力的token。bbox_valid是另一個輸入，不能視為同一遮罩。',
        'temporal': '空間融合讀取ego token，加上時間位置編碼，再送進temporal_encoder。',
        'decoder': 'root_orient為6D旋轉；pose_body為21組6D旋轉。predictions中的root_orient/pose_body則是轉換後axis-angle，單位rad。trans為m。',
        'normalization': 'AvatarNet輸入在forward前複製；EgoNet會修改sparse內容，後續模組數值可不同。linear_embedding的額外維度是相對視窗首幀的XY差。',
        'reference': 'predicted/reference/egonet為SMPL-H前22個FK關節，單位m。原始SMPL標記的joints排列不保證相同，不能直接按索引比較。',
        'pixels': 'vitpose_egonet座標是整張校正影像的像素座標，不是原始魚眼影像座標。ViTPose分數不是可靠的遮蔽標記。',
        'protocol': '此demo使用GT衍生tracker、地板與體型；DINOv2替代DINOv3。不是論文數值的等價重現。',
    }
    report = dict(raw_frame_id=raw_frame, demo_index_0based=index,
                  trace_window_raw_ids=predictions['raw_frame_ids'][:length].tolist(),
                  protocol=protocol, explanations=notes, joint_order=JOINTS,
                  keypoint_order=KEYPOINTS, sparse_decoded=sparse_fields(sparse[0,index]),
                  modules={}, prediction={}, prepared_data={}, original_annotation={})
    for name, value in arrays.items():
        selected, layout = frame_value(name, value, index, length)
        report['modules'][name] = dict(full_shape=list(value.shape), dtype=str(value.dtype),
                                      layout=layout, selected_shape=list(selected.shape), values=selected.tolist())
    for name, value in predictions.items():
        if value.ndim and len(value) == len(predictions['raw_frame_ids']):
            report['prediction'][name] = value[index].tolist()
    pred, ref = predictions['predicted'][index], predictions['reference'][index]
    if pred.shape != (22,3) or ref.shape != (22,3):
        raise ValueError('Expected 22 FK joints')
    errors = np.linalg.norm(pred-ref, axis=-1)*1000
    report['joint_comparison'] = {name: dict(predicted_xyz_m=pred[i].tolist(),
        reference_xyz_m=ref[i].tolist(), error_mm=float(errors[i])) for i,name in enumerate(JOINTS)}
    report['frame_mpjpe_mm'] = float(errors.mean())
    pairs = [p for p in (run/'dataset').iterdir() if p.is_dir()]
    if len(pairs) != 1:
        raise ValueError('Expected exactly one demo pair')
    # Saved preparation arrays include raw frame 1, unlike differenced model inputs.
    with np.load(pairs[0]/'vitpose_egonet.npz', allow_pickle=False) as z:
        raw_length = z['vitpose_kp'].shape[0]
    for filename in ['vitpose_egonet.npz','bbox_egonet.npz','wearer_smplh_fk.npz','observer.npz']:
        path = pairs[0]/filename
        if not path.exists():
            report['prepared_data'][filename] = {'status': 'file missing'}
            continue
        with np.load(path, allow_pickle=False) as z:
            report['prepared_data'][filename] = {k: dict(full_shape=list(z[k].shape),
                selection='raw frame' if z[k].ndim and z[k].shape[0] == raw_length else 'static/full array',
                values=(z[k][raw_frame-1] if z[k].ndim and z[k].shape[0] == raw_length else z[k]).tolist()) for k in z.files}
    annotation = Path(protocol['sequence'])/'processed_data'/'smpl'/f'{raw_frame:05}.npy'
    report['original_annotation_path'] = str(annotation)
    # Do not unpickle arbitrary datasets: use the numeric copy already saved by inspection.
    copied = inspection/'first_frame_annotation.npz'
    if index == 0 and copied.exists():
        with np.load(copied, allow_pickle=False) as z:
            for k in z.files:
                a = z[k]
                report['original_annotation'][k] = dict(shape=list(a.shape),
                    values=a.tolist() if a.size <= 1000 else None,
                    note='Mesh/large annotation omitted; full numeric array remains in first_frame_annotation.npz' if a.size > 1000 else '')
    else:
        report['original_annotation']['status'] = 'Only first-frame annotation was copied by demo_inspect; original path supplied, no guessed values.'
    out = inspection/f'values_frame_{raw_frame:05}.txt'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('Full values saved:', out)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--raw-frame', type=int, default=2)
    export_values(**vars(p.parse_args()))
