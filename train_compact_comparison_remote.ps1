param(
    [string]$Python = 'D:\Ray\venvs\graduate-research\Scripts\python.exe',
    [string]$Root = 'runs/tagging001_compact_ablation',
    [string]$BaselineRoot = 'runs/tagging001_camera_ablation',
    [string]$From = '',
    [ValidateSet(50,100)][int]$Epochs = 100,
    [double]$Cost = 0.002,
    [switch]$SkipTransfer
)
$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    if (!$From) {
        $oldBase=Join-Path $BaselineRoot 'adapter_train12'
        if ((Test-Path -LiteralPath (Join-Path $oldBase 'checkpoint.pt')) -and
            (Test-Path -LiteralPath (Join-Path $oldBase 'experiment.json')) -and
            (Test-Path -LiteralPath (Join-Path $oldBase 'metrics.json'))) { $From=$oldBase }
    }
    $options=@{Python=$Python;Root=$Root;Encoding='compact';Epochs=$Epochs;Cost=$Cost;SkipTransfer=$SkipTransfer}
    if ($From) {
        $options.From=$From
        $oldMlp=Join-Path $BaselineRoot 'mlp'
        $oldContract=Join-Path $oldMlp 'experiment.json'
        if ((Test-Path -LiteralPath $oldContract) -and (Test-Path -LiteralPath (Join-Path $oldMlp 'subset_cost_cache'))) {
            $old=Get-Content -LiteralPath $oldContract -Raw | ConvertFrom-Json
            $frozenHash=(Get-FileHash -LiteralPath (Join-Path $From 'checkpoint.pt') -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($old.frozen_sha256 -eq $frozenHash) { $options.TargetCacheFrom=$oldMlp }
        }
    }
    & (Join-Path $PSScriptRoot 'train_attention_transfer_remote.ps1') @options
    & $Python -m observer_selection.scorer_report --root $Root --baseline-root $BaselineRoot
    if ($LASTEXITCODE -ne 0) { throw "Report failed: $LASTEXITCODE" }
}
finally { Pop-Location }
