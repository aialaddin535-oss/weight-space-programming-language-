#!/usr/bin/env bash
# Download the REQUIRED edited WSP checkpoint (never the base model).
#
#   bash scripts/download_model.sh                                   # -> models/<name>
#   bash scripts/download_model.sh "" /data/models/wsp-qwen35
#   bash scripts/download_model.sh RikiyaT/Qwen3.5-0.8B-weight-space-programming /models/wsp
#
# Writes aladdin_checkpoint.json next to the weights so later loads can verify
# the repository id and revision instead of trusting the directory name.
set -euo pipefail
MODEL_ID="${1:-RikiyaT/Qwen3.5-0.8B-weight-space-programming}"
DEST="${2:-models/$(basename "$MODEL_ID")}"
NAME="$(basename "$MODEL_ID")"
DEST="${2:-models/$NAME}"
mkdir -p "$(dirname "$DEST")"
python -m aladdin.cli download --model "$MODEL_ID" --dest "$DEST"
echo
echo "Now point the project at it:"
echo "  export ALADDIN_MODEL_PATH=\"$(cd "$DEST" && pwd)\""
echo "  export ALADDIN_BACKEND=hf_torch"
echo "  python -m aladdin.cli verify-model"
