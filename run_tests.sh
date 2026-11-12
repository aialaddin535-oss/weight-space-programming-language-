#!/usr/bin/env bash
# Everything that can be checked without a GPU or an API key.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
fail=0
step() { echo; echo "== $* =="; }

step "1. byte-compile every module"
$PY -m compileall -q aladdin scripts tests && echo "OK" || fail=1

step "2. unit + integration tests"
$PY -m pytest -q tests || fail=1

step "3. datasets are valid and disjoint"
$PY scripts/build_datasets.py || fail=1

step "4. environment report"
$PY -m aladdin.cli doctor || fail=1

step "5. tensor inspection (tiny backend)"
$PY -m aladdin.cli inspect --layers 2 | head -12 || fail=1

step "6. real mini experiment (scripted agent, tiny model)"
$PY -m aladdin.cli run --agent scripted --experiment verify_run --steps 2 \
   --limit-per-category 1 | tail -12 || fail=1

step "7. delta audit"
$PY -m aladdin.cli edits list || fail=1

step "8. no training / no secrets audit"
if grep -rn --include=*.py -E "\.backward\(|torch\.optim|LoraConfig|SFTTrainer|get_peft_model" \
     aladdin scripts | grep -v "test_no_training" ; then
  echo "FAIL: training API found"; fail=1
else
  echo "OK: no training API in aladdin/ or scripts/"
fi

echo
if [ "$fail" = "0" ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; fi
exit "$fail"
