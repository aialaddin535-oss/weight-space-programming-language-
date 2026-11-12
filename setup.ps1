# Windows PowerShell setup. Run:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

Write-Host "== python =="
python --version
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

if ($env:WITH_GPU -eq "1") {
  Write-Host "== installing torch/transformers (this is a large download) =="
  pip install -q torch transformers accelerate huggingface_hub safetensors
}

New-Item -ItemType Directory -Force -Path checkpoints, experiments, logs, .cache, models | Out-Null
if (-not (Test-Path .env)) { Copy-Item .env.example .env }

python scripts\build_datasets.py
python -m aladdin.cli doctor

Write-Host ""
Write-Host "Next:"
Write-Host "  1) edit .env  (NVIDIA_API_KEY, NVIDIA_MODEL, ALADDIN_MODEL_PATH)"
Write-Host "  2) python -m aladdin.cli download --model Qwen/Qwen3.5-0.8B --dest D:\models\Qwen3.5-0.8B"
Write-Host "  3) python -m aladdin.cli baseline --split dev"
Write-Host "  4) python -m aladdin.cli run --agent nvidia --steps 10 --limit-per-category 3"
Write-Host ""
Write-Host "Environment variables for this shell:"
Write-Host '  $env:NVIDIA_API_KEY="nvapi-..."'
Write-Host '  $env:NVIDIA_MODEL="qwen/qwen3-next-80b-a3b-instruct"'
Write-Host '  $env:ALADDIN_BACKEND="hf_torch"'
Write-Host '  $env:ALADDIN_MODEL_PATH="D:\models\Qwen3.5-0.8B"'
