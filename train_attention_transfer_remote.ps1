param(
    [string]$Python = 'D:\Ray\venvs\graduate-research\Scripts\python.exe',
    [string]$Sequence = 'data/EgoHumans/01_tagging/001_tagging',
    [string]$Prepared = 'data/prepared/tagging001_triangulated_annotated',
    [string]$TestPrepared = 'data/prepared/tagging001_transfer_34',
    [string]$Root = 'runs/tagging001_camera_ablation',
    [string]$ModelCache = 'data/prepared/tagging001_v2/model_cache',
    [ValidateSet(50,100)][int]$Epochs = 100,
    [int]$AdapterEpochs = 16,
    [double]$Cost = 0.002,
    [switch]$SkipTransfer
)
$ErrorActionPreference='Stop'
function Invoke-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Python exited with status $LASTEXITCODE" }
}
Push-Location $PSScriptRoot
try {
    $common=@('-m','observer_selection.fit_triangulated','--sequence',$Sequence,
              '--device','cuda','--appearance','none','--joint-weights','vitpose','--no-video')
    $trainSources=@('aria03','aria04','cam01','cam02')
    $testSources=@('aria03','aria04','cam03','cam04')
    if (!$SkipTransfer) {
        foreach ($camera in @('cam03','cam04')) {
            if (!(Test-Path -LiteralPath (Join-Path $Sequence "exo/$camera"))) { throw "Camera missing: $camera" }
        }
    }
    $base=Join-Path $Root 'adapter_train12'
    # Refit the residual adapter WITHOUT cam03/cam04. Never reuse the old all-camera adapter.
    $baseArgs=$common+@('--prepared',$Prepared,'--output',$base,'--epochs',"$AdapterEpochs",'--sources')+$trainSources
    if (Test-Path -LiteralPath (Join-Path $base 'experiment.json')) { $baseArgs+='--resume' }
    Invoke-Python @baseArgs
    $costText=$Cost.ToString('R',[System.Globalization.CultureInfo]::InvariantCulture)
    foreach ($scorer in @('mlp','attention')) {
        $run=Join-Path $Root $scorer
        $fit=$common+@('--prepared',$Prepared,'--output',$run,'--scorer',$scorer,
             '--ranking-from',$base,'--subset-training','--label-stride','8',
             '--cost',$costText,'--epochs',"$Epochs",'--save-every','25','--sources')+$trainSources
        if ($scorer -eq 'attention') { $fit+=@('--target-cache-from',(Join-Path $Root 'mlp')) }
        if (Test-Path -LiteralPath (Join-Path $run 'experiment.json')) { $fit+='--resume' }
        Invoke-Python @fit
    }
    if (!$SkipTransfer) {
        if (!(Test-Path -LiteralPath (Join-Path $TestPrepared 'preparation_complete.json'))) {
            $prep=@('-m','observer_selection','prepare-raw','--sequence',$Sequence,'--output',$TestPrepared,
                    '--target','aria01','--requester','aria02','--device','cuda','--annotation-boxes','--sources')+$testSources
            if (Test-Path -LiteralPath $ModelCache) { $prep+=@('--model-cache',$ModelCache) }
            if (Test-Path -LiteralPath (Join-Path $TestPrepared 'preparation_run.json')) { $prep+='--resume' }
            Invoke-Python @prep
        }
        foreach ($scorer in @('mlp','attention')) {
            foreach ($epoch in @(50,100) | Where-Object { $_ -le $Epochs }) {
                $milestone=Join-Path (Join-Path $Root $scorer) ('epoch_{0:d4}' -f $epoch)
                $out=Join-Path $milestone 'transfer34'
                if (Test-Path -LiteralPath (Join-Path $out 'camera_transfer.json')) { continue }
                $evaluation=$common+@('--prepared',$TestPrepared,'--output',$out,'--scorer',$scorer,
                                    '--evaluate-from',$milestone,'--sources')+$testSources
                Invoke-Python @evaluation
            }
        }
    }
    Invoke-Python -m observer_selection.scorer_report --root $Root
    Write-Host "Results: $Root/comparison.csv and $Root/comparison.png"
}
finally { Pop-Location }
