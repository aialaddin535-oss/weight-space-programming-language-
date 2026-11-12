# The real experiment on Windows PowerShell.
#   $env:NVIDIA_API_KEY="nvapi-..."; $env:NVIDIA_MODEL="qwen/qwen3-next-80b-a3b-instruct"
#   $env:ALADDIN_MODEL_PATH="D:\models\Qwen3.5-0.8B-weight-space-programming"
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -Steps 10 -PerCategory 3
param(
  [int]$Steps = 10,
  [int]$PerCategory = 3,
  [string]$Exp = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
if (-not $Exp) { $Exp = "qwen35_nvidia_" + (Get-Date -Format "yyyyMMdd-HHmmss") }

if (-not $env:NVIDIA_API_KEY) { throw "set `$env:NVIDIA_API_KEY first" }
if (-not $env:NVIDIA_MODEL)   { throw "set `$env:NVIDIA_MODEL (check with: python -m aladdin.cli nvidia-models)" }
if (-not $env:ALADDIN_MODEL_PATH) { throw "set `$env:ALADDIN_MODEL_PATH to the downloaded model dir" }
if (-not $env:ALADDIN_BACKEND) { $env:ALADDIN_BACKEND = "hf_torch" }

Write-Host "== 0. verify the model id =="
python -m aladdin.cli nvidia-models | Select-Object -Last 5

Write-Host "== 1. inspect the real tensors =="
python -m aladdin.cli inspect --layers 6

Write-Host "== 2. baseline (dev) =="
python -m aladdin.cli baseline --split dev --limit-per-category $PerCategory

Write-Host "== 3. search =="
python -m aladdin.cli run --agent nvidia --experiment $Exp --steps $Steps --limit-per-category $PerCategory

Write-Host "== 4. held-out report =="
python -m aladdin.cli evaluate --split heldout --experiment $Exp --out "experiments/$Exp/heldout.json"
python -m aladdin.cli report --experiment $Exp

Write-Host "== 5. audit =="
python -m aladdin.cli edits list
Write-Host "done -> experiments/$Exp/"
