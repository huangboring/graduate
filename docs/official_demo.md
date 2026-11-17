# Single-pair pretrained demo with DINOv2 substitution

This is separate from observer selection. It loads every official checkpoint
tensor strictly (no partially initialized weights), does no training, uses the
official EgoHumans synthetic head/wrist, floor and body-shape setup, and uses
EgoNet proposals instead of detector/ReID tracking. DINOv2 replaces DINOv3
without adapting ScoreNet: feature-distribution compatibility and paper-level
accuracy are NOT established. This is a substitution demo, not exact reproduction.

Run from graduate, in the already-working CUDA environment:

```powershell
& $gpuPython -m observer_selection.official_demo --sequence data/EgoHumans/01_tagging/001_tagging --output runs/official_demo_v2_3pt --points 3 --wearer aria01 --observer aria03 --device cuda
```

Default checkpoint: `external/EgoExoMoCap/model_zoo/egohumans_syn_tracker_3pt/model.pth`.
Requires existing neutral SMPL-H, ViTPose HF, and DINOv2 HF checkpoints. No gated
DINOv3 download is attempted. No additional detector or manual identity steps.
Outputs are isolated; reruns require a fresh output directory.

The wrapper calls upstream stage-1 extraction for one wearer/observer pair,
then official sparse-input normalization, EgoNet, proposal geometry, data loader,
fusion, and prediction/reference decoding. It uses per-frame rectification with
rigid-normalized similarity alignment and upright diagonal K. It invokes the
loader entry directly after extracting EgoNet-crop features because upstream
dataset discovery checks unrelated GT-crop filenames even with egonet crops.
These deviations are recorded in `demo_protocol.json`, along with hashes,
evaluated frame count, omitted tail, crop availability and measured MPJPE.
Official non-overlapping 96-frame windows are retained; no smoothing is added.

Open `viewer.html` for offline interactive skeleton playback, `demo.mp4` for
observer RGB plus two skeleton projections, and PNG/PDF error and pose figures
for reporting. CSV and `predictions.npz` retain measured errors and predictions.
All plots are labeled DINOv2 substitution, with GT-derived synthetic inputs.
Pose display is centered at the reference pelvis; numeric MPJPE is calculated
before display centering, without Procrustes alignment.

Local checks cover strict checkpoint rejection and visual/metric export using
explicit fixtures. Real checkpoint inference and dataset geometry still require
execution on the remote machine; local tests are not evidence of paper accuracy.
