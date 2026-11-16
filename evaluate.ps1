<# Run three continuous swing-up / 10 s hold / impulse / 10 s hold evaluations. #>
param([int]$NumEnvs = 0, [string]$OutputDir = 'reports')
$ErrorActionPreference = 'Stop'
if ($NumEnvs -lt 0) { throw 'NumEnvs must be nonnegative; 0 uses the reference trial counts.' }
$RepoRoot = $PSScriptRoot
if (-not [IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $RepoRoot $OutputDir }
$Tests = @(
    @{ Initial = 'down_exact'; Seed = 2; Horizon = 60; Count = 128; Name = 'recovery_exact_seed2' },
    @{ Initial = 'down'; Seed = 0; Horizon = 60; Count = 128; Name = 'recovery_down_seed0' },
    @{ Initial = 'random'; Seed = 1; Horizon = 70; Count = 256; Name = 'recovery_random_seed1' }
)
foreach ($Test in $Tests) {
    $Output = Join-Path $OutputDir ($Test.Name + '.json')
    $Count = if ($NumEnvs -eq 0) { $Test.Count } else { $NumEnvs }
    $Started = Get-Date
    & (Join-Path $RepoRoot 'run.ps1') (Join-Path $RepoRoot 'scripts\eval_swingup.py') `
        '--headless' '--initial' $Test.Initial '--num_envs' $Count '--episode_s' $Test.Horizon `
        '--push_force' '0' '--recovery_force' '12' '--recovery_ms' '200' '--seed' $Test.Seed '--output' $Output
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Output)) { throw "Evaluation failed: $($Test.Name)" }
    if ((Get-Item -LiteralPath $Output).LastWriteTime -lt $Started) { throw "No new result produced: $($Test.Name)" }
    $Result = Get-Content -LiteralPath $Output -Raw | ConvertFrom-Json
    Write-Host ("{0}: {1}/{2} complete sequences" -f $Test.Name, $Result.recovery_sequence_successes, $Result.num_trials)
}
