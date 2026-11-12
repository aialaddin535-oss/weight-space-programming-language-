# Everything checkable without a GPU or key, on Windows.
#   powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$fail = 0

Write-Host "== 1. byte-compile =="
python -m compileall -q aladdin scripts tests; if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host "== 2. tests =="
python -m pytest -q tests; if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host "== 3. datasets =="
python scripts\build_datasets.py; if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host "== 4. doctor =="
python -m aladdin.cli doctor

Write-Host "== 5. inspect =="
python -m aladdin.cli inspect --layers 2 | Select-Object -First 12

Write-Host "== 6. mini experiment =="
python -m aladdin.cli run --agent scripted --experiment verify_run --steps 2 --limit-per-category 1 | Select-Object -Last 12

Write-Host "== 7. delta audit =="
python -m aladdin.cli edits list

Write-Host "== 8. no-training audit =="
$hits = Select-String -Path aladdin\*.py, aladdin\*\*.py, scripts\*.py -Pattern '\.backward\(|torch\.optim|LoraConfig|SFTTrainer|get_peft_model' -ErrorAction SilentlyContinue
if ($hits) { Write-Host "FAIL: training API found"; $hits; $fail = 1 } else { Write-Host "OK: none found" }

if ($fail -eq 0) { Write-Host "ALL CHECKS PASSED" } else { Write-Host "SOME CHECKS FAILED" }
exit $fail
