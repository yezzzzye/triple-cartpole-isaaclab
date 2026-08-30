$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot
$Forward = @($args)
$Command = if ($Forward.Count -gt 0) { [string]$Forward[0] } else { 'policy' }
$ExtraArgs = if ($Forward.Count -gt 1) { @($Forward[1..($Forward.Count - 1)]) } else { @() }

$IsaacSim = $env:ISAACSIM_PATH
$IsaacLab = $env:ISAACLAB_PATH

if (-not $IsaacSim) {
    throw 'ISAACSIM_PATH is not set. Point it to the Isaac Sim 5.1 installation directory.'
}
if (-not $IsaacLab) {
    throw 'ISAACLAB_PATH is not set. Point it to the IsaacLab 2.3.2 checkout.'
}

$Python = Join-Path $IsaacSim 'python.bat'
if (-not (Test-Path $Python)) {
    throw "Isaac Sim python not found at '$Python'. Set ISAACSIM_PATH to your Isaac Sim 5.1 install."
}
if (-not (Test-Path (Join-Path $IsaacLab 'source\isaaclab'))) {
    throw "Isaac Lab source not found under '$IsaacLab'. Set ISAACLAB_PATH to your Isaac Lab checkout."
}

$LabPackages = @(
    'isaaclab',
    'isaaclab_tasks',
    'isaaclab_assets',
    'isaaclab_rl',
    'isaaclab_mimic',
    'isaaclab_contrib'
) | ForEach-Object { Join-Path $IsaacLab "source\$_" } | Where-Object { Test-Path $_ }

$env:PYTHONPATH = (@($RepoRoot) + $LabPackages + @($env:PYTHONPATH) | Where-Object { $_ }) -join ';'

$env:PYTHONUNBUFFERED = '1'

function Invoke-IsaacScript {
    param(
        [string]$Script,
        [string[]]$PresetArgs = @(),
        [string[]]$UserArgs = @()
    )
    & $Python $Script @PresetArgs @UserArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Demo {
    param([string[]]$PresetArgs = @(), [string[]]$UserArgs = @())
    $Asset = Join-Path $RepoRoot 'assets\triple_cartpole_physics.usd'
    if (-not (Test-Path $Asset)) {
        Write-Host '[run] articulation asset missing, generating it first...' -ForegroundColor Yellow
        Invoke-IsaacScript (Join-Path $RepoRoot 'scripts\prepare_asset.py')
    }
    Invoke-IsaacScript (Join-Path $RepoRoot 'scripts\demo.py') $PresetArgs $UserArgs
}

switch ($Command) {
    'help' {
        Write-Host @'
Usage:
  .\run.ps1                         # policy showcase
  .\run.ps1 check                   # verify asset and physics
  .\run.ps1 physics                 # uncontrolled fall
  .\run.ps1 policy                  # policy recovery
  .\run.ps1 disturb                 # policy + manual/automatic kicks
  .\run.ps1 eval                    # 30 s clean evaluation
  .\run.ps1 robust                  # 30 s evaluation with 2 N kicks
  .\run.ps1 train                   # full PPO training
  .\run.ps1 demo [demo.py options]  # custom rendering
  .\run.ps1 scripts\<name>.py ...  # run any project script
'@
    }
    'check'   { Invoke-IsaacScript (Join-Path $RepoRoot 'scripts\check_env.py') @('--headless') $ExtraArgs }
    'physics' { Invoke-Demo @('--mode', 'free', '--initial_tilt_deg', '2', '--initial_hold_s', '1') $ExtraArgs }
    'policy'  { Invoke-Demo @('--showcase', '--disturb', '0') $ExtraArgs }
    'disturb' { Invoke-Demo @('--showcase', '--disturb', '2', '--manual_force') $ExtraArgs }
    'eval'    { Invoke-IsaacScript (Join-Path $RepoRoot 'scripts\eval.py') @('--headless', '--episode_s', '30') $ExtraArgs }
    'robust'  { Invoke-IsaacScript (Join-Path $RepoRoot 'scripts\eval.py') @('--headless', '--episode_s', '30', '--push_force', '2') $ExtraArgs }
    'train'   { Invoke-IsaacScript (Join-Path $RepoRoot 'scripts\train.py') @('--headless') $ExtraArgs }
    'demo'    { Invoke-Demo @() $ExtraArgs }
    default {
        $Script = if ([IO.Path]::IsPathRooted($Command)) { $Command } else { Join-Path $RepoRoot $Command }
        if (Test-Path -LiteralPath $Script -PathType Leaf) {
            Invoke-IsaacScript $Script @() $ExtraArgs
        } elseif ($Command.StartsWith('-')) {
            & $Python @Forward
        } else {
            throw "Unknown command '$Command'. Run '.\run.ps1 help' to see the available commands."
        }
    }
}

exit $LASTEXITCODE
