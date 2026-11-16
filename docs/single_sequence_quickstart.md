# Remote single-sequence training (DINOv2)

This path uses actual EgoHumans images and fitted SMPL reference labels. It is
not a synthetic demo. With only one sequence it measures fitting on the training
sequence, not generalization. Formal scene-disjoint protocol commands remain
unchanged. Do not present the in-sample policy comparison as held-out accuracy.

## Update on the remote Windows host

```powershell
Set-Location D:\Ray\graduate
$gpuPython = 'D:\Ray\venvs\graduate-research\Scripts\python.exe'
$env:PIP_CONSTRAINT = 'D:\Ray\graduate-gpu-constraints.txt'
$env:TEMP = 'D:\Ray\pip-temp'
$env:TMP = 'D:\Ray\pip-temp'
git pull --ff-only
& $gpuPython -m pip install --no-cache-dir -e '.[reports,features]'
& $gpuPython -m pip check
```

## Prepare raw images

```powershell
& $gpuPython -m observer_selection prepare-raw --sequence data/EgoHumans/01_tagging/001_tagging --output data/prepared/tagging001_v2 --target aria01 --requester aria02 --sources aria03 aria04 --device cuda
```

The baseline is torchvision Faster R-CNN ResNet50-FPN COCO_V1, person score
threshold 0.7. Its weights download into the output directory's model_cache
on D:, not the default user profile. ViTPose/DINOv2 use the existing local
weights. The generated crop is square, padded outside the image exactly as
in the existing feature preparation API. Model/config/source hashes are saved.

For each observer, open the printed `identity_frame_XXXXX.jpg` in VS Code.
Enter the box index of the actual target aria01, not an arbitrary person.
Dataset identity must be established by the operator; a detector box number
does not itself identify an Aria participant. Enter `n` if the target is absent;
the next initialization question is at least 20 frames later. Earlier frames
remain missing; we do not fill them using later observations. `q` aborts.

Tracking is a deliberately disclosed conservative baseline: maximum IoU >=0.2,
runner-up margin >=0.1. After 5 consecutive unmatched frames it does not
automatically reacquire another person. There is no GT-based assignment and
no bidirectional smoothing. This can leave long missing intervals and needs
review on fast motion/crossings; it is not claimed as a robust identity tracker.
Each source produces `tracking_preview.mp4`, green target boxes, and coverage
statistics in `preparation_summary.json`. Inspect these before trusting labels.
If identity switches occur, do not report results as correct tracking.

Camera rectification follows the official EgoHumans rotation sequence:
https://github.com/rawalkhirodkar/egohumans/blob/main/egohumans/lib/datasets/aria_camera.py
RGB upright -> CCW native -> radtanthinprism map -> CW upright. The same CW
rotation is applied to camera axes, retaining diagonal pinhole K. Exact frame
IDs must agree; non-rigid extrinsics or missing target fitted parameters fail
instead of silently truncating or filling zeros. Intrinsics are checked every
frame and maps recomputed when they change. Parameters/labels are loaded after
all observation extraction; fitted SMPL first21 body rotations plus root are
decoded using SMPL-H with per-frame supplied 10 betas. Network reconstruction
uses neutral shape and first22 root-relative joints, without GT body shape.

The confirmed sequence has 601 frames, 20 Hz. For A=aria02, B=aria01, helpers
aria03/aria04, three views require 1,803 image-frame passes for detection and
another 1,803 rectification passes for features. Runtime includes manual ID
initialization and cannot be inferred from GPU name alone. Progress is printed
every 50 frames; final actual wall time is recorded. Folders must be new;
interrupted preparation does not currently resume.

## Train both stages and generate results

```powershell
& $gpuPython -m observer_selection fit-sequence --data data/prepared/tagging001_v2/replay.npz --output runs/tagging001_v2 --body-model external/EgoExoMoCap/third_party/body_models/smplh/neutral/model.npz --device cuda --recon-epochs 8 --selector-epochs 8
```

This performs actual weight updates, not a dry run. It freshly trains the
adapted local-aux reconstruction model, freezes it, then trains expected-MPJPE
selection with k=1 from two helpers and periodic mandatory refresh. With k=2
and only two helpers every source would be acquired, leaving no selection task.
The selection rule supports larger k when more helper candidates are supplied
through the formal train command. Original EgoExoMoCap weights are not loaded.

No validation is performed and no `best.pt` is invented. The reconstructor uses
`last.pt`, selector uses `selector.pt`. Eight epochs is an initial run length,
not a claim of convergence. Exact fixed-window reconstruction cost is used to
train selection; no labels identifying the 'best observer' are fabricated.

Outputs:

- `runs/tagging001_v2/reconstructor/figures/metrics_curves.png` and PDF/CSV
- `runs/tagging001_v2/selector/figures/training_curves.png` and PDF/CSV
- `runs/tagging001_v2/training_results/figures/`: policy/error/selection charts
- `runs/tagging001_v2/training_results/viewer.html`: interactive pose points

Training-only status is printed on figures and the viewer. The reference is
dataset fitted pose, not newly measured ground-truth mocap. Held-out evaluation
continues to reject the training sequence even for these checkpoints.

Local checks cover mathematical geometry, ambiguity rejection, explicit
train-only checkpoint semantics, fixed-backend gradients and report export.
The remote user's images/GPU are unavailable on the development host, so a
successful full real-data run and tracking accuracy have not been certified.
