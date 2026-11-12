# Results — what ran in the build environment (2026-11-11)

Build box: Linux sandbox, Python 3.12.3, 2 vCPU, no GPU (`nvidia-smi` absent),
**no `NVIDIA_API_KEY`**, no model weights. Everything below is the verbatim outcome of
commands that actually executed there. Anything that could not run is listed as
NOT RUN and is not counted as a result.

---

## 1. Test suite

```
$ python -m compileall -q aladdin scripts tests   → exit 0
$ python -m pytest -q tests
74 passed, 4 warnings in 1.01s
```

| test file | tests | covers |
|---|---|---|
| `test_backend_tiny.py` | 7 | forward shapes, **causal masking**, determinism, activation capture, tensor write/hash |
| `test_tensor_editor.py` | 13 | validation, bounds, frozen refusal, uniform-rescale refusal, op purity, apply+verify |
| `test_sparse_deltas.py` | 7 | bf16 round-trip & ties-to-even, sub-precision edits ignored, exact reconstruction, wrong-base detection, store status history |
| `test_checkpoints.py` | 7 | immutable base, snapshot restore, **base+delta replay reproduces exact bytes**, wrong-order failure, diff, state round-trip |
| `test_evaluation.py` | 14 | scoring dims, translation/dodge/empty penalties, bullets & length, fact bans, loop penalty, dev/held-out disjointness, accept & reject |
| `test_forward_and_tools.py` | 6 | budget enforcement, failed call is free, teacher-forced logprobs, tool-call parsing (3 shapes), apply/revert flow |
| `test_end_to_end.py` | 4 | full loop, rejection→rollback, resume, baseline drift detection |
| `test_no_training.py` | 9 | **no training API anywhere**, AST check of the editing path, no secret literals, env-only key, .env.example empty, .gitignore |
| `test_config_cli.py` | 7 | config merge, env overrides, redaction, JSON validity, every documented CLI command parses |

The two tests that matter most for the project's promise:

```
test_no_training_api_anywhere        PASSED   (greps every .py for .backward(), torch.optim,
                                                Trainer, LoraConfig, SFTTrainer, get_peft_model,
                                                PeftModel, optimizer.step, AdamW, SGD, ...)
test_no_gradient_descent_semantics…  PASSED   (AST-inspects engine.py/ops.py/store.py/cast.py/
                                                manager.py: no optim imports, no .backward()/.step() calls)
```

## 2. Environment report (`aladdin doctor`)

```
python 3.12.3 · numpy 2.5.4 · requests 2.34.2 · pytest 9.2.0
torch null · transformers null · huggingface_hub null · cuda null
nvidia: {api_key_present: false, model: "<unset>", configured: false}
backend: tiny_numpy
```

## 3. Tensor inspection (`aladdin inspect`)

```
tiny model: 21 tensors, 139,584 parameters, float32
baseline manifest: .cache/baseline_manifest.json (created=True, pristine=True)
editable candidates:
  model.layers.0.mlp.down_proj.weight   [64, 128]   entries=8,192
  model.layers.1.mlp.down_proj.weight   [64, 128]   entries=8,192
```

## 4. Baseline (tiny model, 1 item per category)

```
$ python -m aladdin.cli baseline --split dev --limit-per-category 1
n=10  composite=0.5271
context_followup 0.6267 · devanagari_hindi 0.4267 · factuality 0.4625
instruction_following 0.3267 · mixed_code_switch 0.6267 · natural_response 0.6267
question_answering 0.4947 · roman_hinglish 0.6267 · spelling_robustness 0.4267
technical_hindi 0.6267
dims: intent 0.00 · correctness 0.00 · script 0.60 · compliance 0.95
      relevance 1.00 · naturalness 0.70 · factual 1.00
```

**Read this correctly:** `intent` and `correctness` are 0 because the tiny model is
randomly initialised — it emits byte noise (`"\ufffd>kQ>kQ\u0013h\ufffd"`), so it never
contains the rubric tokens. The non-zero composite comes only from the
"not-empty / not-a-dodge / no-loop / right length" dimensions. **This number is a
pipeline check, not a language-quality score** — and it is why the tiny backend is
documented as a test fixture, never as a result.

## 5. End-to-end experiment (`aladdin run --agent scripted`, tiny model)

```
$ python -m aladdin.cli run --agent scripted --experiment demo_run --steps 3 --limit-per-category 1

=== step 1/3 ===        (candidate 1: layer 0 down_proj, 10 highest-L2-norm columns, alpha=+0.125)
[reject +0.0000] rolled back - rejected: gain 0.0000 (needed 0.0100), regressions=0
=== step 2/3 ===        (layer 1 down_proj, alpha=+0.0875)
[reject +0.0000] rolled back - rejected: gain 0.0000 (needed 0.0100), regressions=0
=== step 3/3 ===
[agent] no edit proposed: plan exhausted; no further candidates

status finished · step 3/3 · baseline 0.5271 · best 0.5271 · kept 0 · rejected 2
forward_passes: {budget 12, used 2, remaining 10, by_kind {generate: 2}}
verify: {ok: true, changed_tensors: [], n_tensors: 21}      ← weights back to pristine
```

Artifacts actually written: `experiments/demo_run/{state.json, baseline_dev.json,
history.jsonl, step_000.json, step_001.json, forward_passes.jsonl, report.md,
final.json}`.

## 6. The deltas that were produced (and then rejected)

| edit_id | tensor | entries changed | relative change | dtype | status |
|---|---|---|---|---|---|
| `e_20261111-093341_26b729` | `model.layers.0.mlp.down_proj.weight` | 640 | 0.04075 | float32 | **rejected** |
| `e_20261111-093342_847e5e` | `model.layers.1.mlp.down_proj.weight` | 640 | 0.02834 | float32 | **rejected** |

Each is a real, hash-verified direct tensor edit: 640 entries of a `[64,128]` tensor
were multiplied by 1.125 in place, the tensor re-hashed and compared with
`edited_sha256`, the sparse delta written as `edit.npz` (`t0_idx` int64 + `t0_val`)
plus a manifest, and then rolled back after the evaluation showed no gain. **Zero
edits were kept**, because on a random model nothing can improve.

## 7. What did NOT run — and why

| Item | Status | Reason |
|---|---|---|
| Qwen3.5-0.8B load / inspect | **NOT RUN** | no weights on the box; download is ~1.6 GB and there is no GPU here |
| Qwen3.5-0.8B baseline Hindi/Hinglish score | **NOT RUN** | same |
| Before/after evaluation on Qwen | **NOT RUN** | same — **no improvement on Qwen is claimed anywhere in this project** |
| NVIDIA API agent loop against the real endpoint | **NOT RUN** | `NVIDIA_API_KEY` absent in this environment; only its parser/dispatch/error paths are unit tested |
| NVIDIA model-id verification via `GET /models` | **NOT RUN** (key needed) | catalogue ids in `docs/NVIDIA_MODEL.md` were read from NVIDIA's public docs, dated 2026-11-11 |
| GPU-only paths (`cuda` device, bf16 checkpoint load) | **NOT RUN** | no CUDA device |
| `apply_edit` on real Qwen bf16 tensors | **NOT RUN** on real weights; bf16 logic covered by unit tests with synthetic bf16 arrays |

## 8. Honest summary

What is proven here: the whole chain — inspect → hypothesise → bounded edit → sparse
delta → hash verify → evaluate → reject → rollback → resume — works, is reproducible,
and contains no training step. What is not proven: that this improves Qwen3.5-0.8B's
Hindi/Hinglish. That experiment needs a GPU box, the weights, and an NVIDIA key; the
code path for it is written and unit-tested, and the exact command is
`bash scripts/run_experiment.sh 10 3`.
