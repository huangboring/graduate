param(
    [string]$Python = 'D:\Ray\venvs\graduate-research\Scripts\python.exe',
    [string]$From = 'runs/tagging001_triangulated_vitpose_none',
    [string]$Run = 'runs/tagging001_subset_none',
    [int]$Epochs = 30,
    [double]$Cost = 0.002
)
$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    if ($Epochs -lt 1 -or $Cost -lt 0 -or [double]::IsNaN($Cost) -or [double]::IsInfinity($Cost)) {
        throw 'Use positive epochs and finite nonnegative cost.'
    }
    if (!(Test-Path -LiteralPath (Join-Path $From 'checkpoint.pt'))) {
        throw "Frozen reconstruction checkpoint missing: $From/checkpoint.pt. Finish the none run first."
    }
    $costText=$Cost.ToString('R',[System.Globalization.CultureInfo]::InvariantCulture)
    $fit=@('-m','observer_selection.fit_triangulated',
        '--prepared','data/prepared/tagging001_triangulated_annotated',
        '--sequence','data/EgoHumans/01_tagging/001_tagging',
        '--output',$Run,'--appearance','none','--joint-weights','vitpose',
        '--ranking-from',$From,'--subset-training','--label-stride','8','--device','cuda',
        '--epochs',"$Epochs",'--cost',$costText)
    if (Test-Path -LiteralPath (Join-Path $Run 'experiment.json')) { $fit+='--resume' }
    & $Python @fit
    if ($LASTEXITCODE -ne 0) { throw "Training exited with status $LASTEXITCODE" }
    Get-Content -LiteralPath (Join-Path $Run 'metrics.json')
    Write-Host "Video: $Run/multiview_selection.mp4"
}
finally { Pop-Location }
