#!/usr/bin/env bash
# Download the genuine target weights. Nothing is bundled with this project.
#   bash scripts/download_model.sh [MODEL_ID] [DEST]
set -euo pipefail
MODEL_ID="${1:-Qwen/Qwen3.5-0.8B}"
DEST="${2:-$(pwd)/models/$(basename "$MODEL_ID")}"

python - <<'PY'
import sys
try:
    import huggingface_hub  # noqa: F401
except ImportError:
    sys.exit("huggingface_hub missing: put it in requirements-gpu or run "
             "`pip install huggingface_hub` (this is a large download, so it is opt-in)")
PY

echo "model : $MODEL_ID"
echo "dest  : $DEST"
python -m aladdin.cli download --model "$MODEL_ID" --dest "$DEST"

cat <<EOF

Now point the project at the weights:

  export ALADDIN_BACKEND=hf_torch
  export ALADDIN_MODEL_PATH="$DEST"
  export ALADDIN_DEVICE=cuda          # or cpu

Then confirm the architecture was read correctly (this prints the real tensor table):

  python -m aladdin.cli inspect
EOF
