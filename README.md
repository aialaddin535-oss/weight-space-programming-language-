# Aladdin WSP — Weight-Space Programming for Hindi & Hinglish

Aladdin WSP improves a small language model's Hindi/Hinglish behaviour by
**editing its weights directly**. You inspect tensors, form a hypothesis, apply a
small bounded arithmetic edit, measure it on a held-out protocol, and keep or
revert it.

It runs on the MIT WSP researcher's already-edited checkpoint:

> **`RikiyaT/Qwen3.5-0.8B-weight-space-programming`**
> <https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming>

That is the default model everywhere (config, loader, CLI, docs, tests). The base
`Qwen/Qwen3.5-0.8B` release is **not** used as a default and is never silently
substituted.

```
inspect tensors ─► evidence ─► hypothesis ─► bounded edit ─► sparse delta
        ─► measure on dev ─► accept/reject ─► exact rollback ─► checkpoint
```

**No training.** No LoRA, QLoRA, SFT, fine-tuning, gradient descent, optimizer,
backward pass or `Trainer`. The test suite greps and AST-inspects the whole tree
to enforce that (`tests/test_no_training.py`). See [docs/METHOD.md](docs/METHOD.md).

---

## 1. Install

Python 3.9+.

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt      # numpy + requests + pytest
```

For the real checkpoint (text tower through `Qwen3_5ForCausalLM`):

```bash
pip install torch 'transformers>=4.57' accelerate safetensors huggingface_hub
```

`transformers` must be >= 4.57: the checkpoint declares
`transformers_version: 4.57.0.dev0` and the loader refuses older releases with an
install instruction instead of failing deep inside `from_pretrained`.

Or run the helper: `bash scripts/setup.sh` (`WITH_GPU=1` to include the HF stack).

## 2. Configure

Copy `.env.example` to `.env` and load it:

```bash
cp .env.example .env
set -a; . ./.env; set +a
```

| variable | meaning |
|---|---|
| `ALADDIN_MODEL_ID` | HF repo id; default `RikiyaT/Qwen3.5-0.8B-weight-space-programming` |
| `ALADDIN_MODEL_PATH` | local snapshot directory of that checkpoint |
| `ALADDIN_BACKEND` | `hf_torch` for the real model, `tiny_numpy` for the CPU test fixture |
| `ALADDIN_DEVICE` | `cpu` or `cuda` |
| `ALADDIN_LOAD_MODE` | `text` (default) or `multimodal` |
| `NVIDIA_API_KEY` | only for `--agent nvidia`; read from the environment, never from a file |
| `NVIDIA_MODEL` | an NVIDIA model id your key can call (see `aladdin nvidia-models`) |

Secrets are never written to config, logs, experiments or checkpoints.
`NVIDIA_API_KEY` stays empty in the shipped example and `.env` is git-ignored.

## 3. Download the required checkpoint

```bash
python -m aladdin.cli download         # RikiyaT/Qwen3.5-0.8B-weight-space-programming
export ALADDIN_MODEL_PATH="$PWD/models/Qwen3.5-0.8B-weight-space-programming"
export ALADDIN_BACKEND=hf_torch
python -m aladdin.cli model-info       # repo id, revision, architecture, geometry
python -m aladdin.cli verify-model     # live identity + tensor-geometry verification
```

`download` writes `aladdin_checkpoint.json` next to the weights: repository id,
resolved commit sha, file list/sizes, `config.json` hash. Later loads verify that
metadata, so a directory containing some other model cannot masquerade as the
edited WSP checkpoint.

The downloaded snapshot is **never written to**. Edits happen in memory; deltas
and checkpoints go to `checkpoints/` and `experiments/` (both git-ignored).

### What the loader actually does

The checkpoint is natively multimodal (`Qwen3_5ForConditionalGeneration`) and
stores 488 tensors:

| namespace | count | use |
|---|---|---|
| `model.language_model.*` | 320 | text tower — **loaded** |
| `model.visual.*` | 153 | vision tower — skipped for text-only WSP |
| `mtp.*` | 15 | multi-token-prediction head — skipped |

For text-only generation the official class is `Qwen3_5ForCausalLM` with
`Qwen3_5TextConfig` (not `AutoModelForImageTextToText`). Its key layout is
`model.*`, so the loader remaps `model.language_model.X → model.X` explicitly,
skips the vision/MTP tensors, and reports counts. `lm_head.weight` is absent
because `tie_word_embeddings` is true; that is the only permitted missing tensor.
The full multimodal path is available explicitly with `--load-mode multimodal`.

Verification checks the class, config geometry (hidden 1024, 24 layers, vocab
248320, 3584 intermediate), the **live** layer count and embedding/head shapes,
the loaded tensor count, the tokenizer (ids fit the vocabulary, chat template,
EOS), and the recorded repository identity. Any mismatch fails loudly.

## 4. Generate text

```bash
python -m aladdin.cli --backend hf_torch --model-path "$ALADDIN_MODEL_PATH" \
  baseline --split dev --limit-per-category 2
```

This runs real text-only generation under a fixed protocol
(`temperature=0`, `seed=0`) and writes
`logs/baseline_dev/{results.json,summary.csv,generations.jsonl}`. For a quick
single-prompt smoke test use the Python API:

```python
from aladdin.config import load_config
from aladdin.model.loader import load_model

cfg = load_config("config/qwen35_cpu.json")
cfg["backend"]["model_path"] = "/path/to/Qwen3.5-0.8B-weight-space-programming"
backend, desc = load_model(cfg)
print(desc["model_id"], desc["hidden_size"], desc["num_hidden_layers"])
print(backend.generate("iska kya matlab hai", max_new_tokens=16))
```

## 5. Inspect weights

```bash
python -m aladdin.cli inspect --layers 6
python -m aladdin.cli inspect --stats model.layers.12.mlp.down_proj.weight
```

`inspect` prints the model description, records the immutable baseline hash
manifest, and ranks editable 2-D tensors. Tensor stats include norms, sparsity,
top rows/columns and an effective-rank probe. Read-only.

## 6. Perform a direct WSP edit

Edits are declarative JSON operations on one 2-D tensor:

```json
{"kind": "scale_columns", "tensor": "model.layers.12.mlp.down_proj.weight",
 "columns": [37, 88, 102], "alpha": 0.04,
 "reason": "these output columns carry the largest Hinglish residual"}
```

```python
from aladdin.config import load_config
from aladdin.model.loader import load_model
from aladdin.sparse_deltas.store import DeltaStore
from aladdin.tensor_editor import TensorEditor

cfg = load_config("config/qwen35_cpu.json")
cfg["backend"]["model_path"] = "/path/to/Qwen3.5-0.8B-weight-space-programming"
backend, _ = load_model(cfg)
store = DeltaStore("checkpoints/deltas")
editor = TensorEditor(backend, cfg, store)

op = {"kind": "scale_columns", "tensor": "model.layers.12.mlp.down_proj.weight",
      "columns": [37, 88, 102], "alpha": 0.04, "reason": "Hinglish column probe"}
print(editor.preview(op))          # validate + measure, no write
result = editor.apply(op, reason=op["reason"])
print(result["edit_id"], result["changed_entries"], result["relative_change"])
```

Guards: known tensor, 2-D shape, memory dtype, non-frozen, `|alpha|` cap, entry
cap, no whole-tensor rescale, and a base-hash check before writing. After the
write the tensor is re-hashed and compared with `edited_sha256`; a mismatch
reverts it immediately.

Four op kinds only: `scale_columns`, `shift_columns`, `scale_rows`,
`scale_block`.

## 7. Sparse deltas

`editor.apply` writes `checkpoints/deltas/<edit_id>/`:

```
edit.npz        t0_idx (int64 flat indices) + t0_val (new storage bit patterns)
manifest.json   tensor names/shapes/dtype, changed-entry counts, relative change,
                base_sha256, edited_sha256, operation, reason, status history
```

A delta stores only changed entries. Rebuilding is
`base_bits.flat[idx] = val` — exact, hash-checked, no accumulation. bf16 values
are stored as `uint16` bit patterns so edits below bf16 resolution are recorded
as zero changed entries rather than pretending to work.

## 8. Run an experiment (hypothesis → accept/reject)

```bash
# deterministic policy, no API key, CPU fixture model
python -m aladdin.cli run --agent scripted --steps 3 --limit-per-category 1

# the real loop with the NVIDIA reasoning agent on the required checkpoint
python -m aladdin.cli --config config/qwen35.json \
  run --agent nvidia --steps 10 --limit-per-category 3
```

Each step: baseline dev score → agent evidence/hypothesis → one bounded edit →
dev evaluation → accept/reject. Accepting keeps every edit of the group, marks
the deltas `accepted`, and saves a checkpoint. Rejecting marks them `rejected`
and rebuilds the weights from base + kept deltas (hash-verified).

Artifacts per experiment: `state.json`, `history.jsonl`, `step_NNN.json`,
`baseline_dev.json`, `forward_passes.jsonl`, `final.json`, `report.md`, and an
`edit_ledger/` with one directory per transaction.

## 9. Accept, reject, revert

* **Accept/reject** happens in the loop (`acceptance.*` in the config:
  `min_composite_gain`, `max_category_regression`).
* `revert(n)` undoes the newest `n` **edits** (not transactions), restoring each
  tensor to the exact pre-edit bits and verifying the hash.
* `revert_transaction(txn_id)` undoes an **entire transaction, including one that
  was already committed**, atomically. Reverting an older transaction while a
  newer transaction still holds a value of the same tensor is refused with
  `blocked_by`; `cascade=True` reverts the dependents newest-first.
* A failed restore changes nothing: journals are validated and hashed before any
  write, live values are captured, and a failure midway restores them. Edit IDs
  and ledger statuses only change after a verified restore.

Full contract and response fields: [docs/REVERT.md](docs/REVERT.md).

```python
from aladdin.agent.tools import ToolRuntime
rt = ToolRuntime(backend, cfg, editor, fwd, dev_dataset, "experiments/demo")
rt.call("apply_edit", {"operation": op, "reason": "probe"})
rt.call("revert", {"n": 1})                                  # newest edit
rt.call("revert_transaction", {"txn_id": "txn_...", "cascade": True})
```

## 10. Checkpoints and recovery

```bash
python -m aladdin.cli rollback --keep <edit_id> <edit_id>     # base + kept deltas
python -m aladdin.cli rollback --experiment <id>              # use its kept edits
python -m aladdin.cli verify                                  # compare to baseline hashes
python -m aladdin.cli report --experiment <id>
```

`CheckpointManager` stores small models in `full` mode and large models in
`modified_only` mode (exact `.npy` snapshots of the tensors that differ from a
hash-verified base; everything else is restored from a pristine reload). Restore
is staged, hash-verified before any write, journalled, and rolled back on failure.
The `base` checkpoint is immutable.

## 11. Hindi / Hinglish evaluation

```bash
python -m aladdin.cli baseline --split dev
python -m aladdin.cli evaluate --split heldout --experiment <id> --out heldout.json
```

Two disjoint sets, enforced in code: `datasets/dev.jsonl` (75 items) for the
search and `datasets/heldout.jsonl` (45 items) for the final report only. Ten
categories cover Roman Hinglish, Devanagari Hindi, code-switching, context
follow-up, instructions, QA, naturalness, spelling/typo robustness, technical
Hindi and factuality, plus the exact probes named in the project brief
(`mujhe ye samjha`, `muje ye samja`, `iska kya matlab hai`, `ab isko thik karo`,
`kal wala kaam continue karo`, `मुझे यह समझाओ`, …).

`evaluate` measures the **same required checkpoint before and after** the kept
WSP edits: baseline = freshly loaded pristine snapshot, edited = base + kept
sparse deltas (rebuilt and verified). Details and limitations:
[docs/EVALUATION.md](docs/EVALUATION.md).

## 12. Tests

```bash
python -m pytest -q tests          # or: bash scripts/run_tests.sh
```

Real-checkpoint integration tests run only when weights are present:

```bash
export ALADDIN_MODEL_PATH="$PWD/models/Qwen3.5-0.8B-weight-space-programming"
python -m pytest -q tests/test_qwen35_real.py
```

Without the weights or the HF stack they are **skipped and reported as not run**,
never as passed. `docs/RESULTS.md` records exactly what ran in which environment.

## 13. Layout

```
aladdin/
  model_identity.py     required checkpoint id, revision and verified geometry
  config.py  cli.py  reporting.py  edit_ledger.py
  backends/             base contract, tiny CPU fixture, HF torch backend,
                        qwen_loader.py (text-only Qwen3.5 loading + verification)
  model/loader.py       backend construction, identity report, baseline manifest, download
  inspection/           tensor statistics, activation capture/contrast
  tensor_editor/        bounded ops + the edit engine
  sparse_deltas/        dtype-exact deltas, rebuild
  checkpoints/          checkpoint manager, materialised checkpoints
  forward_pass/         budgeted generation / teacher-forced / activation passes
  evaluation/           datasets, scorer, runner, optional NVIDIA judge
  experiment/           loop + resumable state
  agent/                NVIDIA agent, deterministic scripted agent, tool runtime, prompt
config/   default.json, qwen35.json (GPU), qwen35_cpu.json
datasets/ dev.jsonl, heldout.jsonl
scripts/  setup, download_model, run_experiment, run_tests, build_datasets
docs/     METHOD, EVALUATION, QWEN3_SETUP, REVERT, NVIDIA_MODEL, RESULTS
tests/    unit + regression + real-checkpoint integration
```

## 14. Honesty

Scores are heuristic rubric scores over real model output, not human ratings.
No Hindi/Hinglish improvement is claimed unless `docs/RESULTS.md` reports a
measured before/after run on the required checkpoint. The tiny CPU backend is a
randomly initialised fixture for pipeline tests: its numbers prove the machinery,
never language quality.

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
