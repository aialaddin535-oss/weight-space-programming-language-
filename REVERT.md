# Revert & rollback — V3 contract

## Tool `revert` (per-edit)

`revert(n)` where **n counts EDITS** (not transactions): the most recently
applied n edits are undone newest-first. Each edit's tensor is restored to the
exact pristine bits saved before that edit was applied, and the SHA-256 is
verified AFTER the write. Only then are edit IDs reported as `undone`.

Response contract (all fields always present):

```json
{
  "ok": true,
  "undone":   ["e_20261111-101500_ab12cd"],   // REAL edit IDs, never tensor names
  "restored": ["model.layers.0.mlp.down_proj.weight"],  // tensor names
  "remaining": ["e_20261111-101530_ef34ab"],  // edit IDs still applied
  "verified": true,
  "txn_id":   "txn_20261111-101500_ab12cd",
  "error":    null
}
```

Hard guarantees (enforced in `aladdin/agent/tools.py`, tests in
`tests/test_v3_revert_and_loader.py`):

1. An edit ID appears in `undone` **only after** its tensor was actually
   restored AND hash-verified. A restored tensor with a wrong hash is a failed
   revert (`ok: false`), never a success.
2. A **failed** revert never removes IDs from active tracking
   (`ToolRuntime.applied`, the NVIDIA agent's `applied_here`, the ledger): the
   original error detail is preserved in `error`, and `remaining` lists what is
   still applied.
3. A partial revert that would break a same-tensor dependency (reverting an
   older edit while a newer edit on the SAME tensor is still applied) is
   rejected with a clear error and **no model modification**.
4. Repeated revert is idempotent: `undone: []`, `ok: true`.
5. `n` larger than the number of live edits is rejected (`ok: false`) instead
   of silently reverting everything.

## Tool `revert_transaction` (whole transaction)

`revert_transaction(txn_id=None)` atomically undoes an ENTIRE transaction —
every one of its edits, in one verified operation. `txn_id=None` targets the
currently open transaction. Use this instead of many `revert(n=1)` calls when
you want all-or-nothing semantics.

## Agent synchronization

`NvidiaAgent.run_step` removes IDs from `applied_here` **only** from the
`undone` list of a successful (`ok: true`, `verified: true`) revert result, and
logs the removal (`event: revert_applied`) with what is still applied. A failed
revert leaves `applied_here` untouched. Reverted edit IDs are also marked
`reverted` in the durable `edit_ledger` journal, so the ledger, the agent
tracking, the experiment history and the checkpoint chain agree.

## Experiment rollback (loop level)

Unchanged from V2 and still enforced: a rejected candidate resets the model to
base weights and replays only the kept deltas
(`CheckpointManager.rebuild_chain_after_reset`), hash-verified end to end; the
run aborts loudly if verification fails.
