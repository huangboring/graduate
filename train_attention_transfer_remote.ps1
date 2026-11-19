param(
    [string]$Python = 'D:\Ray\venvs\graduate-research\Scripts\python.exe',
    [string]$Sequence = 'data/EgoHumans/01_tagging/001_tagging',
    [string]$Prepared = 'data/prepared/tagging001_triangulated_annotated',
    [string]$TestPrepared = 'data/prepared/tagging001_transfer_34',
    [string]$Root = '',
    [ValidateSet('legacy','compact')][string]$Encoding = 'legacy',
    [string]$From = '',
    [string]$TargetCacheFrom = '',
    [string]$ModelCache = 'data/prepared/tagging001_v2/model_cache',
    [ValidateSet(50,100)][int]$Epochs = 100,
    [int]$AdapterEpochs = 16,
    [double]$Cost = 0.002,
    [switch]$SkipTransfer
)
$ErrorActionPreference='Stop'
if (!$Root) {
    $Root=if ($Encoding -eq 'compact') {'runs/tagging001_compact_ablation'} else {'runs/tagging001_camera_ablation'}
}
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
    if ($From) {
        $base=$From
        foreach ($name in @('checkpoint.pt','experiment.json')) {
            if (!(Test-Path -LiteralPath (Join-Path $base $name))) { throw "Frozen run is incomplete: $base/$name" }
        }
        Write-Host "Reusing frozen adapter: $base"
    }
    else {
        $base=Join-Path $Root 'adapter_train12'
        # Adapter preparation stays legacy so the scorer encoding is the only ablation.
        $baseArgs=$common+@('--prepared',$Prepared,'--output',$base,'--epochs',"$AdapterEpochs",'--sources')+$trainSources
        if (Test-Path -LiteralPath (Join-Path $base 'experiment.json')) { $baseArgs+='--resume' }
        Invoke-Python @baseArgs
    }
    $costText=$Cost.ToString('R',[System.Globalization.CultureInfo]::InvariantCulture)
    foreach ($scorer in @('mlp','attention')) {
        $run=Join-Path $Root $scorer
        $fit=$common+@('--prepared',$Prepared,'--output',$run,'--scorer',$scorer,'--encoding',$Encoding,
             '--ranking-from',$base,'--subset-training','--label-stride','8',
             '--cost',$costText,'--epochs',"$Epochs",'--save-every','25','--sources')+$trainSources
        if ($scorer -eq 'mlp' -and $TargetCacheFrom) { $fit+=@('--target-cache-from',$TargetCacheFrom) }
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
                $evaluation=$common+@('--prepared',$TestPrepared,'--output',$out,'--scorer',$scorer,'--encoding',$Encoding,
                                    '--evaluate-from',$milestone,'--sources')+$testSources
                Invoke-Python @evaluation
            }
        }
    }
    Invoke-Python -m observer_selection.scorer_report --root $Root
    Write-Host "Results: $Root/comparison.csv and $Root/comparison.png"
}
finally { Pop-Location }
