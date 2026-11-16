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
& $gpuPython -m observer_selection prepare-raw --sequence data/EgoHumans/01_tagging/001_tagging --output data/prepared/tagging001_v3 --target aria01 --requester aria02 --sources aria03 aria04 --device cuda --model-cache data/prepared/tagging001_v2/model_cache --resume
```

The baseline is torchvision Faster R-CNN ResNet50-FPN COCO_V1, person score
threshold 0.7. The command reuses the previously downloaded weights through
`--model-cache`; omit that argument on a fresh installation to use the output
directory's model_cache. ViTPose/DINOv2 use the existing local weights. Pose
features use the existing square padded crop; tracking appearance uses each
detector rectangle. Model/config/source/input hashes are saved.

For each observer, open the printed `identity_frame_XXXXX.jpg` in VS Code.
Enter the box index of the actual target aria01, not an arbitrary person.
Dataset identity must be established by the operator; a detector box number
does not itself identify an Aria participant. Enter `n` if the target is absent;
the next initialization question is at least 20 frames later. Earlier frames
remain missing; we do not fill them using later observations. `q` stops while
preserving candidate caches and previous identity choices. Later observers'
identity sheets show a reference crop from the first observer. This helps
manual matching; it does not prove that the person corresponds to aria01.

Tracking now uses causal DINOv2 appearance plus IoU. Normal association uses
0.7*cosine_similarity + 0.3*IoU, requires similarity >=0.55, IoU >=0.05 and a
best-versus-runner-up score margin >=0.08. Global reacquisition is attempted
every frame, regardless of the length of the missing interval: similarity
>=0.75 and appearance margin >=0.08. The initial descriptor stays in a bounded
gallery; only confident recent matches extend it (>=0.85, interval 10 frames,
maximum 8 templates). These are engineering defaults, not validated optimum
thresholds. DINOv2 is a generic feature encoder, not a trained person ReID model.
Similar clothing, crossings or occlusion can still cause missing/wrong tracks.
No GT, future image, remote camera geometry or cross-view features enter this
association. Initial manual identity is explicitly disclosed; no later manual
recovery annotations are silently introduced. The obsolete permanent-stop
behaviour is removed. Old v2 observations are NOT reused as v3 tracking.
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
every 50 frames. Candidate detections and appearance features are saved per
frame before prompting. With `--resume`, identical raw files/models/settings
reuse completed views and cached GPU inference. Partial views replay their
cached past in chronological order to rebuild tracker state; no later frame
fills earlier missing observations. An interrupted feature stage is archived
and recomputed; completed view features are reused. The summary records wall
time for the current invocation, not total time across previous interruptions.

To correct a mistaken initial identity, append `--reset-view aria02` to the
same resume command. Generated outputs are archived, and cached candidate
inference is retained. If the target person was wrong across all views, use
`--reset-view aria02 aria03 aria04`. Old replay outputs are archived so they
cannot be mistaken for a newly corrected export. A changed raw input/model or
algorithm contract requires a new output folder, while model weights may be
shared. Do not run two writers against the same output directory concurrently.

Exception: if failure occurred during camera setup, before any detections,
rerun the same `prepare-raw` command with `--restart-setup`. It retains model
weights and checks that existing metadata/frame grids match. It refuses to
overwrite identity images, tracking videos, detector outputs or features.

Aria-to-COLMAP alignment includes positive uniform scale. The adapter checks
the similarity structure and divides the full composed camera extrinsic's
top 3x4 block by that scale, including its translation. Projection and world
camera centre are unchanged; the anchor world and GT are not rescaled.
Reflection, shear and non-uniform scale remain errors. Per-camera alignment
scales are saved in export provenance. All camera frames are validated before
loading feature models or downloading the detector weights.

## Correct a mistaken initial identity

### v4 preparation after persistent appearance mismatches

The v2 association algorithm uses both the last accepted appearance and the
conservative seed/gallery for short-gap matching. Its thresholds are unchanged;
long-gap automatic recovery still requires the gallery evidence. This addresses
gradual appearance changes, but can still drift and is not certified ReID.

For explicit **manually assisted tracking**, use a fresh output and import
only candidate detections/descriptors from v3:

```powershell
& $gpuPython -m observer_selection prepare-raw --sequence data/EgoHumans/01_tagging/001_tagging --output data/prepared/tagging001_v4 --target aria01 --requester aria02 --sources aria03 aria04 --device cuda --model-cache data/prepared/tagging001_v2/model_cache --reuse-candidates data/prepared/tagging001_v3 --review-tracking --resume
```

Candidate import verifies the original input/model contract and array shapes;
old identity choices, boxes and extracted joint features are not imported.
Corrected tracks require ViTPose and crop-feature extraction again, but detector
and candidate DINO inference is reused. Old v3 files are preserved.

`--review-tracking` asks for a current-frame identity annotation after five
consecutive misses and every 100 frames since the last review (when there are
candidate boxes). Following a review it waits at least 20 frames before asking
again. `n` means no matching candidate, not proof the person is physically absent;
it clears the active association. `q` preserves caches/decisions. Rerun the same
command to resume. `identity_current.jpg` always contains the latest prompt;
the timestamped file is retained too. The user must select the previously
verified aria01, not detection index 1. Do not select a different person merely
to increase coverage. If the detector never boxes the visible target, this mode
cannot recover it: retain missing observations and inspect the detector failure.

Additional annotations are recorded in `recovery_choices.json`, digested in
receipts, and counted as `manual_review_count`. They only affect the annotated
frame and subsequent tracking, never earlier missing frames. Declare this as
**annotation-assisted preprocessing**, not an autonomous deployed system.
Periodic review cannot prove or repair identity switches between reviews;
inspect the final preview. `tracking_diagnostics.json` records per-candidate
gallery/recent cosine similarity, IoU, gap, and the automatic decision before
manual intervention. Coverage is not an identity-accuracy metric.

Do not replace `identity_selected.jpg` by hand: saved decisions and observations
would still describe the old selection. Detector index 0 does not mean aria01.
First review the supplied dataset identity labels in EXO images:

```powershell
& $gpuPython -m observer_selection audit-identity --sequence data/EgoHumans/01_tagging/001_tagging --output data/identity_audit/tagging001_frame1
Start-Process data/identity_audit/tagging001_frame1/index.html
```

This independent CPU command renders official `human_name`/`bbox` records from
`processed_data/poses2d/camXX/rgb/00001.npy` over the matching EXO images. It
does not change preparation contracts, cached detections, or training inputs.
It rejects records lacking explicit identities rather than inferring them from
list order. Orange boxes flag invalid or zero-confidence pose evidence; even
green annotations require visual review. Missing annotations/images are reported.
Official schema references:
https://github.com/rawalkhirodkar/egohumans/blob/main/egohumans/lib/datasets/ego_exo_scene.py
https://github.com/rawalkhirodkar/egohumans/blob/main/egohumans/lib/models/pose_estimator.py

Disclose this as **dataset-annotation-assisted manual identity initialization**,
not autonomous identity discovery. The audit reads first-frame annotations;
later-frame audits must not be represented as causal first-frame initialization.
No annotation boxes/keypoints are exported to model inputs. Retain the audit
JSON with experiment records. If IDs cannot be verified, do not train mismatched
images and labels. Once aria01 is identified, run the same prepare command with
`--resume --reset-view aria02 aria03 aria04` to correct all initial selections.
This archives old derived outputs and preserves per-frame detector/DINO caches.
Omit `--reset-view` on subsequent ordinary resumes or it will reset them again.

## Train both stages and generate results

If all three observation stages completed but export failed with confidence
outside `[0,1]`, preserve the original folder and run:

```powershell
& $gpuPython -m observer_selection finalize-raw --prepared data/prepared/tagging001_v3 --output data/prepared/tagging001_v3_export
```

This verifies completed-view hashes and original raw inputs, then re-exports
without detection, feature inference, or FK recomputation. Use the new
`tagging001_v3_export/replay.npz` as `--data`. Original observations/receipts
remain unchanged. This path also avoids rerunning old preparation contracts
after the observation reader code changed. Missing stages are errors.

Local-aux exports explicitly apply `clip_heatmap_v1`: clip finite ViTPose
heatmap scores to `[0,1]` as a bounded feature, not a calibrated probability.
Raw feature files preserve all amplitudes. The export sidecar records per-view
raw min/max, counts below zero/above one, the policy and code hashes. NaN/Inf
and invalid availability flags remain errors. This does not change bbox
validity, identity assignments or repair poor tracking. A completed export
with only one tracked frame is not thereby adequate for research training.

```powershell
& $gpuPython -m observer_selection fit-sequence --data data/prepared/tagging001_v3/replay.npz --output runs/tagging001_v3 --body-model external/EgoExoMoCap/third_party/body_models/smplh/neutral/model.npz --device cuda --recon-epochs 8 --selector-epochs 8
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

- `runs/tagging001_v3/preparation_quality.json`: observation availability and tracking statistics
- `runs/tagging001_v3/reconstructor/figures/metrics_curves.png` and PDF/CSV
- `runs/tagging001_v3/selector/figures/training_curves.png` and PDF/CSV
- `runs/tagging001_v3/training_results/figures/`: policy/error/selection charts
- `runs/tagging001_v3/training_results/viewer.html`: interactive pose points

Training preflight rejects incomplete v3 exports and known legacy permanent
track-loss results. Availability counts are recorded but are not proof of
identity accuracy or physical visibility. Inspect each tracking preview.

Training-only status is printed on figures and the viewer. The reference is
dataset fitted pose, not newly measured ground-truth mocap. Held-out evaluation
continues to reject the training sequence even for these checkpoints.

Local checks cover mathematical geometry, ambiguity rejection, explicit
train-only checkpoint semantics, fixed-backend gradients and report export.
The remote user's images/GPU are unavailable on the development host, so a
successful full real-data run and tracking accuracy have not been certified.
