#!/usr/bin/env bash
# The real experiment, end to end, on the required WSP checkpoint:
#   RikiyaT/Qwen3.5-0.8B-weight-space-programming
#   NVIDIA_API_KEY=... NVIDIA_MODEL=... bash scripts/run_experiment.sh 10 3
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS="${1:-10}"
PER_CAT="${2:-3}"
EXP="${3:-qwen35_nvidia_$(date +%Y%m%d-%H%M%S)}"

: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY first (see .env.example)}"
: "${NVIDIA_MODEL:?set NVIDIA_MODEL to an id your key can call; check with: python -m aladdin.cli nvidia-models}"
: "${ALADDIN_MODEL_PATH:?set ALADDIN_MODEL_PATH to the downloaded RikiyaT/Qwen3.5-0.8B-weight-space-programming directory}"

export ALADDIN_BACKEND="${ALADDIN_BACKEND:-hf_torch}"

echo "== 0. verify the model id =="
python -m aladdin.cli nvidia-models | tail -5

echo "== 1. inspect the real tensors =="
python -m aladdin.cli inspect --layers 6

echo "== 2. baseline (dev) =="
python -m aladdin.cli baseline --split dev --limit-per-category "$PER_CAT"

echo "== 3. search: $STEPS candidate edits =="
python -m aladdin.cli run --agent nvidia --experiment "$EXP" --steps "$STEPS" \
  --limit-per-category "$PER_CAT"

echo "== 4. held-out report (touched once) =="
python -m aladdin.cli evaluate --split heldout --experiment "$EXP" \
  --out "experiments/$EXP/heldout.json"
python -m aladdin.cli report --experiment "$EXP"

echo "== 5. audit the deltas =="
python -m aladdin.cli edits list
echo "done -> experiments/$EXP/"
