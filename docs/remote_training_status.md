# Remote DINOv2 training and report status

The verified remote environment can execute DINOv2 (768 CLS), ViTPose
(17 COCO heatmaps) and SMPL-H (52 joints). This is model execution verification,
not dataset preparation or training. The feature adapter selects the upstream
13 COCO body keypoints; reconstruction supervises the first 22 SMPL-H joints.

## Available commands

From `D:\Ray\graduate`, using the research virtual environment:

```powershell
$gpuPython = 'D:\Ray\venvs\graduate-research\Scripts\python.exe'
$env:PIP_CONSTRAINT = 'D:\Ray\graduate-gpu-constraints.txt'
git pull --ff-only
& $gpuPython -m pip install --no-cache-dir -e '.[reports,features]'
& $gpuPython -m observer_selection prepare-features --help
& $gpuPython -m observer_selection report --help
```

`prepare-features` now uses DINOv2 and records exact checkpoint/config hashes.
It requires a prepared pair video and genuine image-detector boxes/receipt.
It does NOT generate these prerequisites from raw EgoHumans. Do not manufacture
receipts or rename GT-projected boxes to satisfy the contract.

## Blocking raw-data issues (not solved by installing dependencies)

- The pinned upstream stage2 code performs remapping without a 90-degree image
  rotation, while stage1 writes a rotated intrinsic matrix with zero diagonal
  focal entries. The local exporter deliberately rejects this. Verify image,
  ray and calibrated-camera coordinate consistency before real training.
- Upstream stage1 silently takes minimum calibration/SMPL counts and fills
  missing SMPL observations with zeros. Our raw adapter must instead align
  explicit frame IDs and propagate validity; do not train on these as valid GT.
- Independent detector/tracker and target identity initialization are still
  required. Candidate cameras cannot be selected using current remote data.
- Only `01_tagging/001_tagging` is currently confirmed on the remote host.
  The existing protocol requires disjoint train/validation/test scenes. One
  sequence is insufficient; do not duplicate it or relabel its scene identity.

## Training commands AFTER preparation and protocol validation

These paths are planned output paths, not files already present on the host.
Use new run directories for each experiment. With target aria01, requester
aria02 and helper cameras aria03/aria04, use k=1 to have a meaningful choice.

```powershell
$body = 'external/EgoExoMoCap/third_party/body_models/smplh/neutral/model.npz'
& $gpuPython -m observer_selection check-data --protocol data/protocol.json
& $gpuPython -m observer_selection train-reconstructor --protocol data/protocol.json --body-model $body --device cuda --max-sources 1 --epochs 8 --output runs/reconstructor_v2
& $gpuPython -m observer_selection report --run runs/reconstructor_v2 --output runs/reconstructor_v2/figures
& $gpuPython -m observer_selection train --protocol data/protocol.json --backend local-aux --checkpoint runs/reconstructor_v2/best.pt --body-model $body --device cuda --max-sources 1 --epochs 8 --output runs/selector_v2
& $gpuPython -m observer_selection report --run runs/selector_v2 --output runs/selector_v2/figures
& $gpuPython -m observer_selection evaluate --protocol data/protocol.json --backend local-aux --checkpoint runs/reconstructor_v2/best.pt --body-model $body --device cuda --model runs/selector_v2/best.pt --output runs/test_v2
& $gpuPython -m observer_selection report --run runs/test_v2 --output runs/test_v2/figures
```

Eight epochs is an initial configuration, not a convergence guarantee. The
adapted reconstructor is freshly initialized; original checkpoints are not
silently loaded. Train it, then freeze it for expected-MPJPE selector training.

Reports export 300 dpi PNG, vector PDF and numeric CSV: reconstruction MPJPE,
selector expected-MPJPE objective, validation MPJPE, policy comparison,
unsmoothed per-sequence error and acquisition/mandatory-refresh timelines.
The viewer.html from evaluation provides pose points and frame scrubbing.
Reports record input SHA256. All-view has a different acquisition budget and
is not a guaranteed accuracy bound. No single-run confidence intervals or
claims of generalization are generated.
