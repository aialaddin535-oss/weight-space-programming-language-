# Method — direct weight-space programming, no training

## 1. The one rule

A weight may change only through an arithmetic edit that a human or agent can read,
bound and reproduce. Concretely, in this codebase:

* nothing computes an objective, a derivative, or an update rule from data;
* there is no `backward()`, no `torch.optim`, no `Trainer`, no PEFT/LoRA anywhere
  (`tests/test_no_training.py` greps the tree and fails the build);
* the only write into a model is `Backend.set_tensor(name, values)`, called from
  `TensorEditor.apply` after a validated, bounded operation;
* every write is followed by a re-hash and compared with the recorded
  `edited_sha256`; a mismatch reverts that tensor and raises.

Editing is therefore *programming*: you state a hypothesis about a pathway
("these MLP output columns under-serve Hinglish"), express it as `W[:, c] *= 1.05`,
and measure.

## 2. The loop

```
load model + record baseline tensor hashes (immutable manifest)
baseline dev evaluation  ──►  failure digest (lowest-scoring items, with notes)
        │
        ▼
agent step:  evidence ─► hypothesis ─► one bounded op
        │        (forward_generate / activation_contrast / tensor_stats)
        ▼
TensorEditor.validate  ─►  preview (changed entries, relative change)
        │                                   │
        │  refused ─────────────────────────┘ (reason returned to the agent)
        ▼
apply: build sparse record from copies ─► save delta ─► write tensor ─► re-hash
        ▼
dev evaluation  ─►  compare(baseline, candidate)
        │
   accept: gain ≥ min_composite_gain AND no category drop > max_category_regression
        │            └─► delta status=accepted, checkpoint saved, candidate becomes the new baseline
   reject: ─────────┴─► reset to base weights + replay only kept deltas
        ▼
state.json updated atomically; loop continues or resumes after an interruption
```

## 3. Why sparse deltas instead of "undo" files

A delta stores new *bit patterns* for changed entries, plus `base_sha256` and
`edited_sha256`. Rollback = base weights + replay of kept replays. Three properties
follow, and all three are tested:

1. **exactness** — `base_bits.flat[idx] = val` is a substitution, not an
   accumulation, so replaying N edits reproduces the same bytes as the original run;
2. **detection** — if the base tensor no longer hashes to `base_sha256`, the replay
   stops with an error instead of silently corrupting the model (wrong order,
   wrong checkpoint, hand-edited weights);
3. **cost** — a checkpoint is ~100 KB instead of 1.6 GB, so dozens fit in a repo.

For bf16 targets the stored values are `uint16` bit patterns and comparisons happen
in bf16 space, so an edit that rounds back to the original value is recorded as
"0 entries changed" rather than pretending to act.

## 4. Where the hypotheses come from

`inspection/activations.py::group_contrast` runs the model on N Hindi/Hinglish
prompts and N English controls, captures per-layer hidden states, and ranks layers
by `mean|activation|(Hindi) / mean|activation|(English)`, then ranks channels inside
each tensor by the same contrast. Layers that fire much harder on the Hindi side are
the plausible sites for a bounded gain edit. `logprob_of` gives a cheap
teacher-forced signal (e.g. "does the model find a Devanagari reply likely?").

This is a *search-space reducer*, not an oracle: the agent still has to pick a
tensor, justify it, and survive evaluation. Nothing in this project edits a tensor
because it is large or because it is layer 16 — only because a measurement pointed
there.

## 5. Guard rails (and why each exists)

| Guard | Failure it prevents |
|---|---|
| frozen suffixes (embedding, final norm, `lm_head`) | breaking tokenization/output projection in a way no dev item can catch |
| `max_entries_per_tensor` | "edits" so large they are really a rewritten tensor |
| `max_relative_change` | step sizes that destroy the model in one shot |
| refusal of a whole-tensor rescale | the trivial global-temperature hack the reference experiment also blocked |
| shape + dtype + base-hash check | applying a delta to the wrong tensor or the wrong checkpoint |
| post-write hash re-check | silent casting bugs (bf16 vs f32) |
| dev/held-out separation, asserted in code | optimising against the test set |
| category-regression cap | buying a Hindi gain by wrecking factuality |

## 6. What is deliberately out of scope

Rank-1 factor edits, per-head attention scaling, activation steering at inference,
and multi-tensor coordinated edits are all expressible in principle but not
implemented. The four ops in `tensor_editor/ops.py` are the whole surface: small
enough that every experiment can be audited by reading one JSON object.

## V2 bugfix notes (checkpoint / revert / multi-edit)

- **CheckpointManager** now takes `backend_factory=` at every construction site
  (`experiment/loop.py`, `cli.py`). Large models no longer store hashes-only
  checkpoints: checkpoints save exact `.npy` snapshots of the tensors that differ
  from the verified base (`mode: modified_only`); restoration reloads the pristine
  weights via `backend_factory`, verifies them against the recorded base SHA-256
  hashes BEFORE any write, hash-verifies the staged snapshot data, and journals
  the live values first so a failed restore can never leave partial writes.
- **`t_revert()`** now restores the real tensors: exact pre-transaction bits are
  journaled to disk before the first write per tensor, `rollback_transaction()`
  writes them back and verifies every restored tensor's SHA-256, and the edit
  ledger is updated only after a verified restore. Removing edit IDs from a list
  is no longer used as a rollback anywhere.
- **Multi-edit tracking**: one `EditTransaction` per candidate interaction
  (`aladdin/edit_ledger.py`), a unique edit ID per successful edit, a shared
  `txn_id` linking every edit to its checkpoint, and the experiment loop records
  ALL edit IDs (`edit_ids`, `n_edits`, `txn_id`) in `history.jsonl` and
  `state.json`, accepting/rejecting every edit of the group. Rejection (or an
  exception mid-transaction) restores every touched tensor to its exact
  pre-transaction state, verified byte-for-byte.


## 7. Revert semantics (V3)

`revert(n)` undoes the last `n` EDITS, newest first, restoring each tensor to
the exact pristine bits captured before that edit and verifying the SHA-256
after the write. An edit ID is reported as `undone` only after verified
restoration; `remaining` reports the still-applied edit IDs; failures never
remove IDs from active tracking. `revert_transaction()` undoes an entire
transaction atomically. Full contract: `docs/REVERT.md`.

Reverting a *rejected candidate* at the loop level is unchanged: base weights +
replay of kept deltas, hash-verified end to end
(`CheckpointManager.rebuild_chain_after_reset`).
