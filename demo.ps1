<#
.SYNOPSIS
    One-click render of the triple inverted pendulum.

.DESCRIPTION
    Generates the articulation asset if it is missing, then opens the Isaac Sim viewport.
    With no arguments, runs the swing-up policy from an exactly downward pose,
    with a 12 N / 200 ms impulse every 25 seconds and no timed reset.
    Any extra arguments are forwarded to scripts/demo.py.

.EXAMPLE
    .\demo.ps1
    .\demo.ps1 --initial_pose random
    .\demo.ps1 --disturb 0
    .\demo.ps1 --task balance
#>
# No param() block on purpose; see the note in run.ps1.
$Forward = $args

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot
$Swingup = -not (($Forward -contains 'balance') -or ($Forward -contains '--task=balance'))
$Asset = Join-Path $RepoRoot ($(if ($Swingup) { 'assets\triple_cartpole_swingup.usd' } else { 'assets\triple_cartpole_physics.usd' }))

if (-not (Test-Path $Asset)) {
    Write-Host '[demo] articulation asset missing, generating it first...' -ForegroundColor Yellow
    if ($Swingup) {
        & (Join-Path $RepoRoot 'run.ps1') (Join-Path $RepoRoot 'scripts\prepare_asset.py') '--output' $Asset '--slider-limit-m' '12'
    } else {
        & (Join-Path $RepoRoot 'run.ps1') (Join-Path $RepoRoot 'scripts\prepare_asset.py')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Asset preparation failed.' }
}

$DemoArgs = @((Join-Path $RepoRoot 'scripts\demo.py')) + $Forward
& (Join-Path $RepoRoot 'run.ps1') @DemoArgs
exit $LASTEXITCODE
