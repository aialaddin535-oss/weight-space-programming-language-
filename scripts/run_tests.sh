#!/usr/bin/env bash
# Full test suite. Integration tests for the real checkpoint run only when
# ALADDIN_MODEL_PATH points at a downloaded snapshot; otherwise they are
# SKIPPED and reported as not run (never as passed).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m compileall -q aladdin scripts tests
python -m pytest -q tests "$@"
