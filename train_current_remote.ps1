param(
    [string]$Python = 'D:\Ray\venvs\graduate-research\Scripts\python.exe',
    [string]$Sequence = 'data/EgoHumans/01_tagging/001_tagging',
    [string]$Prepared = 'data/prepared/tagging001_triangulated_annotated',
    [string]$TestPrepared = 'data/prepared/tagging001_transfer_34',
    [string]$Root = 'runs/tagging001_current_e_v2',
    [string]$From = '',
    [string]$ModelCache = 'data/prepared/tagging001_v2/model_cache',
    [ValidateRange(1,10000)][int]$Epochs = 100,
    [ValidateRange(1,10000)][int]$AdapterEpochs = 16,
    [ValidateRange(0,1)][double]$Cost = 0.002,
    [switch]$SkipTransfer,
    [switch]$Video
)
$ErrorActionPreference='Stop'
function Invoke-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Python exited with status $LASTEXITCODE" }
}
Push-Location $PSScriptRoot
try {
    if (!(Test-Path -LiteralPath $Python)) { throw "Python missing: $Python" }
    if (!(Test-Path -LiteralPath (Join-Path $Prepared 'manifest.json'))) { throw "Prepared input missing: $Prepared" }
    $common=@('-m','observer_selection.fit_triangulated','--sequence',$Sequence,'--device','cuda')
    $trainSources=@('aria03','aria04','cam01','cam02')
    $testSources=@('aria03','aria04','cam03','cam04')
    if (!$SkipTransfer) {
        foreach ($camera in @('cam03','cam04')) {
            if (!(Test-Path -LiteralPath (Join-Path $Sequence "exo/$camera"))) { throw "Camera missing: $camera; use -SkipTransfer for training only" }
        }
    }
    $base=if ($From) {$From} else {Join-Path $Root 'adapter_train12'}
    if (!$From) {
        $fit=$common+@('--stage','adapter','--prepared',$Prepared,'--output',$base,
                       '--epochs',"$AdapterEpochs",'--no-video','--sources')+$trainSources
        if (Test-Path -LiteralPath (Join-Path $base 'experiment.json')) { $fit+='--resume' }
        Invoke-Python @fit
    }
    $costText=$Cost.ToString('R',[System.Globalization.CultureInfo]::InvariantCulture)
    foreach ($scorer in @('mlp','attention')) {
        $run=Join-Path $Root $scorer
        $fit=$common+@('--stage','selector','--prepared',$Prepared,'--output',$run,
            '--adapter-from',$base,'--scorer',$scorer,'--cost',$costText,
            '--epochs',"$Epochs",'--save-every','25','--sources')+$trainSources
        if (!$Video) { $fit+='--no-video' }
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
        $milestones=@(@(50,100) | Where-Object {$_ -le $Epochs}) + @($Epochs) | Sort-Object -Unique
        foreach ($scorer in @('mlp','attention')) {
            foreach ($epoch in $milestones) {
                $milestone=Join-Path (Join-Path $Root $scorer) ('epoch_{0:d4}' -f $epoch)
                $out=Join-Path $milestone 'transfer34'
                if (Test-Path -LiteralPath (Join-Path $out 'camera_transfer.json')) { continue }
                $evaluation=$common+@('--prepared',$TestPrepared,'--output',$out,'--scorer',$scorer,
                    '--evaluate-from',$milestone,'--no-video','--sources')+$testSources
                Invoke-Python @evaluation
            }
        }
    }
    Invoke-Python -m observer_selection.scorer_report --root $Root
    Write-Host "Results: $Root/comparison.png ; $Root/mlp/metrics.json ; $Root/attention/metrics.json"
}
finally { Pop-Location }
