param(
    [string]$Python = 'D:\Ray\venvs\graduate-research\Scripts\python.exe',
    [string]$Sequence = 'data/EgoHumans/01_tagging/001_tagging',
    [string]$Prepared = 'data/prepared/tagging001_triangulated_annotated',
    [string]$Run = 'runs/tagging001_triangulated_annotated',
    [string]$Repo = 'external/EgoExoMoCap',
    [string[]]$Sources = @('aria03','aria04','cam01','cam02','cam03'),
    [string]$ModelCache = 'data/prepared/tagging001_v2/model_cache',
    [int]$Epochs = 8,
    [int]$Budget = 2,
    [switch]$PreparedOnly,
    [switch]$SkipPrepare,
    [switch]$TrackBoxes
)

$ErrorActionPreference = 'Stop'
if ($Epochs -lt 1 -or $Budget -lt 1) { throw 'Epochs and budget must be positive.' }

function Invoke-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Python exited with status $LASTEXITCODE" }
}

Push-Location $PSScriptRoot
try {
    if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
    if (!(Test-Path -LiteralPath $Sequence -PathType Container)) { throw "Sequence not found: $Sequence" }
    $checkpoint = Join-Path $Repo 'model_zoo/egohumans_syn_tracker_3pt/model.pth'
    foreach ($path in @($checkpoint, (Join-Path $Repo 'third_party/body_models/smplh/neutral/model.npz'))) {
        if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file missing: $path" }
    }
    Invoke-Python -c 'import cv2, matplotlib, numpy, torch; from human_body_prior.body_model.body_model import BodyModel; assert torch.cuda.is_available(), "CUDA unavailable"; v=cv2.VideoWriter.__name__; print(torch.cuda.get_device_name(0))'

    if (!$SkipPrepare -and !(Test-Path -LiteralPath (Join-Path $Prepared 'preparation_complete.json'))) {
        $prep = @('-m','observer_selection','prepare-raw','--sequence',$Sequence,'--output',$Prepared,
                  '--requester','aria02','--target','aria01','--repo',$Repo,'--device','cuda','--sources') + $Sources
        if (!$TrackBoxes) { $prep += '--annotation-boxes' }
        if ($ModelCache -and (Test-Path -LiteralPath $ModelCache -PathType Container)) {
            $prep += @('--model-cache',$ModelCache)
        }
        if (Test-Path -LiteralPath (Join-Path $Prepared 'preparation_run.json')) {
            $prep += '--resume'
        }
        Invoke-Python @prep
    }
    if ($PreparedOnly) { return }

    $common = @('-m','observer_selection.fit_triangulated','--sequence',$Sequence,'--prepared',$Prepared,
                '--repo',$Repo,'--budget',"$Budget",'--device','cuda')
    $smoke = $Run + '_smoke'
    if (!(Test-Path -LiteralPath (Join-Path $smoke 'metrics.json'))) {
        $smokeArgs = $common + @('--output',$smoke,'--epochs','1','--max-windows','1','--no-video')
        if (Test-Path -LiteralPath (Join-Path $smoke 'experiment.json')) { $smokeArgs += '--resume' }
        Invoke-Python @smokeArgs
    }
    $fitArgs = $common + @('--output',$Run,'--epochs',"$Epochs")
    if (Test-Path -LiteralPath (Join-Path $Run 'experiment.json')) { $fitArgs += '--resume' }
    Invoke-Python @fitArgs
    Write-Host "Results: $Run"
}
finally { Pop-Location }
