<#
.SYNOPSIS
    Runs a script inside the Isaac Sim 5.1 python interpreter with Isaac Lab on PYTHONPATH.

.DESCRIPTION
    Isaac Lab is used here from a source checkout rather than a pip install, so both the
    Isaac Lab packages and this repository have to be put on PYTHONPATH before Isaac
    Sim's bundled python is invoked.

    Override the two install locations with the ISAACSIM_PATH / ISAACLAB_PATH environment
    variables if your layout differs from the defaults below.

.EXAMPLE
    .\run.ps1 scripts/prepare_asset.py
    .\run.ps1 scripts/train.py --headless --num_envs 4096
#>
# No param() block on purpose. A declared parameter turns this into an advanced script,
# which then owns the common parameters (-OutVariable, -Verbose, -Debug, ...) and
# prefix-matches flags such as --out against them instead of forwarding them.
$Forward = $args

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot

if ($env:ISAACSIM_PATH) {
    $IsaacSim = $env:ISAACSIM_PATH
} else {
    $IsaacSim = 'D:\project-rl\issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64'
}
if ($env:ISAACLAB_PATH) {
    $IsaacLab = $env:ISAACLAB_PATH
} else {
    $IsaacLab = 'D:\project-rl\issac_sim\IsaacLab'
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

if (-not $Forward -or $Forward.Count -eq 0) {
    throw 'Usage: .\run.ps1 <script.py> [args...]'
}

& $Python @Forward
exit $LASTEXITCODE
