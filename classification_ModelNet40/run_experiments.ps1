# AdaGAM-3D experiment matrix for Windows PowerShell.
#   .\run_experiments.ps1 train -Epochs 100            # base, adagam, base_aug, adagam_aug (sequential)
#   .\run_experiments.ps1 eval  -Epochs 100            # ModelNet-C (last ckpt) -> results.md
#   .\run_experiments.ps1 train -DryRun                # print the commands only
#   .\run_experiments.ps1 train -Ablations             # also the optional ablations
# If scripts are blocked: powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 train
param(
    [Parameter(Mandatory = $true, Position = 0)][ValidateSet("train", "eval", "all")][string]$Mode,
    [int]$Epochs = 100,
    [int]$Gpu = 0,
    [int]$R = 8,
    [string]$Model = "pointMLPElite",
    [int]$Batch = 32,
    [int]$Amp = 1,
    [switch]$DryRun,
    [switch]$Ablations,
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if ($Python -eq "") {
    $default = "C:\Users\Ankur\AppData\Local\Programs\Python\Python310\python.exe"
    if (Test-Path $default) { $Python = $default }
    elseif ($env:PYTHON) { $Python = $env:PYTHON }
    else { $Python = "python" }
}
Write-Host "Using Python: $Python"

$env:GPU = "$Gpu"; $env:EPOCHS = "$Epochs"; $env:R = "$R"; $env:MODEL = $Model
$env:BATCH = "$Batch"; $env:AMP = "$Amp"; $env:WORKERS = "0"

$argsList = @("run_experiments.py", $Mode)
if ($DryRun) { $argsList += "--dry_run" }
if ($Ablations) { $argsList += "--ablations" }
& $Python @argsList
exit $LASTEXITCODE
