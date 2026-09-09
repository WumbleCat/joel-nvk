<#
.SYNOPSIS
    Run the MATH -> MMLU forgetting pipeline from PowerShell.

.DESCRIPTION
    A thin wrapper over `uv run scripts/run_pipeline.py` with a switch per stage,
    so single stages can be run and re-run without remembering the stage names.

    -LowMemory runs each stage in its own process, which is what you want on a
    machine that cannot hold the model plus everything else at once: memory is
    released between stages instead of accumulating.

.PARAMETER Env
    Config in configs/ to load: smoke (tiny, "does it run"), pilot (first real
    pass), default, dev, production.

.PARAMETER RunId
    Reuse an existing run: stages pick up its adapters and append to its results.
    Required when running stages separately across several invocations.

.EXAMPLE
    .\scripts\run_pipeline.ps1 -All -LowMemory -Summarize
    Every stage, one process each, then print the tables.

.EXAMPLE
    .\scripts\run_pipeline.ps1 -Kl -RunId smoke-cpu-190226
    Re-run only the KL probe against an existing run's adapters.

.EXAMPLE
    .\scripts\run_pipeline.ps1 -Env pilot -TrainMath -TrainMmlu
    Just the two training stages, at pilot sizes.
#>

[CmdletBinding()]
param(
    [ValidateSet('smoke', 'dev', 'default', 'pilot', 'production')]
    [string]$Env = 'smoke',

    # One switch per stage; combine freely. They always execute in pipeline order.
    [switch]$All,
    [switch]$Prepare,
    [switch]$Baseline,
    [switch]$TrainMath,
    [switch]$EvalMath,
    [switch]$TrainMmlu,
    [switch]$EvalMmlu,
    [switch]$Kl,

    [string]$RunId,
    [int]$Seed,

    # Each stage in its own process, so memory is released between them.
    [switch]$LowMemory,
    # Print the accuracy / forgetting / KL tables when the run finishes.
    [switch]$Summarize,
    # This machine's TLS is inspected; uv needs the OS trust store. Pass
    # -NoNativeTls if your network does not do that.
    [switch]$NoNativeTls
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

if (-not $NoNativeTls) { $env:UV_NATIVE_TLS = '1' }

# Stage order is the pipeline's own order, never the order the switches were typed.
$stageSwitches = [ordered]@{
    'prepare'    = $Prepare
    'baseline'   = $Baseline
    'train_math' = $TrainMath
    'eval_math'  = $EvalMath
    'train_mmlu' = $TrainMmlu
    'eval_mmlu'  = $EvalMmlu
    'kl'         = $Kl
}

if ($All) {
    $stages = @($stageSwitches.Keys)
}
else {
    $stages = @($stageSwitches.Keys | Where-Object { $stageSwitches[$_] })
}

if ($stages.Count -eq 0) {
    Write-Host 'No stage selected. Pass -All, or one or more of:' -ForegroundColor Yellow
    Write-Host '  -Prepare -Baseline -TrainMath -EvalMath -TrainMmlu -EvalMmlu -Kl'
    Write-Host "`nExamples:"
    Write-Host '  .\scripts\run_pipeline.ps1 -All -LowMemory -Summarize'
    Write-Host '  .\scripts\run_pipeline.ps1 -Kl -RunId <existing-run-id>'
    exit 1
}

# Stages must share a run id to see each other's adapters and results.
if (-not $RunId) {
    $RunId = 'run-{0}-{1}' -f $Env, (Get-Date -Format 'yyyyMMdd-HHmmss')
}

Write-Host "env    : $Env"
Write-Host "run id : $RunId"
Write-Host ("stages : {0}" -f ($stages -join ', '))
Write-Host ("mode   : {0}" -f $(if ($LowMemory) { 'one process per stage' } else { 'single process' }))
Write-Host ''

function Invoke-Pipeline {
    param([string[]]$StageList)

    $arguments = @('run', 'scripts/run_pipeline.py', '--env', $Env, '--run-id', $RunId, '--stages')
    $arguments += $StageList
    if ($PSBoundParameters.ContainsKey('Seed') -or $Seed) { $arguments += @('--seed', $Seed) }

    & uv @arguments
    return $LASTEXITCODE
}

if ($LowMemory) {
    foreach ($stage in $stages) {
        Write-Host "########## $stage ##########" -ForegroundColor Cyan
        $code = Invoke-Pipeline -StageList @($stage)
        if ($code -ne 0) {
            Write-Host "stage '$stage' failed (exit $code)" -ForegroundColor Red
            exit $code
        }
    }
}
else {
    $code = Invoke-Pipeline -StageList $stages
    if ($code -ne 0) {
        Write-Host "pipeline failed (exit $code)" -ForegroundColor Red
        exit $code
    }
}

Write-Host ''
Write-Host "run id: $RunId" -ForegroundColor Green

if ($Summarize) {
    Write-Host ''
    & uv run scripts/summarize_run.py $RunId
}
else {
    Write-Host "summarise with: uv run scripts/summarize_run.py $RunId"
}
