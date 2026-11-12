#!/usr/bin/env bash
# One-shot setup for Linux / cloud GPU. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
echo "== python: $($PY -V) =="
$PY -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ "${WITH_GPU:-0}" = "1" ]; then
  echo "== installing torch/transformers (CUDA build; this is a large download) =="
  pip install -q torch transformers accelerate huggingface_hub safetensors
fi

mkdir -p checkpoints experiments logs .cache models
[ -f .env ] || cp .env.example .env

python scripts/build_datasets.py
echo
echo "== versions =="
python -m aladdin.cli doctor || true
echo
echo "Next:"
echo "  1) edit .env  (NVIDIA_API_KEY, NVIDIA_MODEL, ALADDIN_MODEL_PATH)"
echo "  2) python -m aladdin.cli download   # RikiyaT/Qwen3.5-0.8B-weight-space-programming"
echo "  3) python -m aladdin.cli baseline --split dev"
echo "  4) python -m aladdin.cli run --agent nvidia --steps 10 --limit-per-category 3"
