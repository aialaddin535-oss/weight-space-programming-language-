# Revert & rollback — V4 contract

Three complementary mechanisms, each with a different scope:

| mechanism | scope | storage |
|---|---|---|
| `revert(n)` | newest `n` **edits** | per-edit pre-images (hash-keyed) |
| `revert_transaction(txn_id)` | one whole transaction, **committed or open** | per-transaction pristine journal on disk |
| `CheckpointManager.rebuild_chain_after_reset` | the whole experiment | base weights + kept sparse deltas |

## 1. `revert(n)` — per-edit

`n` counts **edits**, not transactions. The newest `n` live edits are undone
newest-first. Each tensor is restored to the exact bits captured immediately
before that edit and the SHA-256 is verified after the write.

```json
{
  "ok": true,
  "undone":   ["e_20261112-101500_ab12cd"],
  "restored": ["model.layers.0.mlp.down_proj.weight"],
  "remaining": ["e_20261112-101530_ef34ab"],
  "verified": true,
  "txn_id":   "txn_20261112-101500_ab12cd",
  "error":    null
}
```

* `undone` holds real **edit IDs**, only after a verified restore.
* `restored` holds **tensor names**.
* `remaining` holds the edit IDs still applied.
* A failed restore returns `ok: false`, `undone: []`, keeps every ID active, and
  preserves the original error text.
* An out-of-order revert (older edit while a newer edit on the same tensor is
  still applied) is rejected before any write (`RevertConflict`).
* Repeating a revert is an idempotent no-op (`ok: true`, `undone: []`).
* `n` larger than the number of live edits is rejected rather than silently
  reverting everything.

## 2. `revert_transaction(txn_id)` — whole transaction

Reverts **every edit of one transaction atomically**. `txn_id=None` targets the
currently open transaction; a committed transaction ID is equally valid, because
its pristine pre-images remain on disk in `edit_ledger/<txn_id>/*.npy`.

```json
{
  "ok": true,
  "undone": [...], "restored": [...], "remaining": [...],
  "verified": true, "txn_id": "txn_...",
  "atomic": true,
  "journal": {"journal_present": [...], "journal_missing": [], "revertible": true},
  "cascaded": [], "state_changed": true
}
```

### Dependency rule (one consistent design, documented)

If a **later** transaction still has edits on one of the same tensors, reverting
this one alone would restore an older value and destroy the later edit's
pre-image. The runtime therefore:

* **refuses** the standalone revert, changes **no tensor and no ledger entry**,
  and returns the blocking transactions:

  ```json
  {"ok": false, "state_changed": false,
   "blocked_by": [{"txn_id": "txn_B", "edit_ids": [...], "tensors": ["..."]}],
   "error": "refusing to revert transaction 'txn_A' in isolation: later transaction(s) ['txn_B'] still have edits on the same tensor(s) [...]"}
  ```

* or, with `cascade=True`, reverts the dependents **newest-first**, then the
  target. The response lists them in `cascaded`.

If the cascade fails partway, the already-reverted dependents stay consistently
reverted (their tensors and ledger agree) and the target is left untouched; the
error names the transaction that failed.

### Safety properties

1. **Pre-validation.** Every journal file must exist and hash-match before
   anything is written. Missing/corrupt journal → error, zero writes.
2. **Atomic restore.** Live values of every tensor about to change are captured
   first. If a write or its verification fails, the captured values are restored
   and `RestoreError` is raised with `recovered: true`; the ledger is untouched.
3. **Ledger after restore only.** Edit statuses, `reverted_edit_ids` and agent
   tracking change only after the verified restore.
4. **Ownership.** Each edit's status is written to the transaction that **owns**
   it (`edit_id → txn_id`), never to "the latest transaction". Reverting edits
   from A and B updates A and B separately.
5. **Idempotent.** Repeating the same revert returns `undone: []`, `ok: true`.
6. **Clear errors.** Unknown IDs list the known transactions; incomplete
   journals explain how to recover through the checkpoint chain.
7. **Restart-safe.** A new `ToolRuntime` on the same ledger directory recovers
   active edit IDs, owning transactions, and journal pre-images, so committed
   transactions can still be reverted after a restart.

## 3. Experiment-level rollback

A rejected candidate resets the model to base weights and replays only the kept
deltas (`CheckpointManager.rebuild_chain_after_reset`), hash-verified end to end.
The pre-rollback values of changed tensors are journalled to disk first, so a
failure mid-rebuild restores them and the run aborts loudly instead of continuing
on partially rolled-back weights.

## 4. Ledger layout

```
experiments/<id>/edit_ledger/
  ledger_index.json                 txn_id -> {status, outcome, accepted, seq,
                                              edit_ids, reverted_edit_ids, tensors}
  txn_.../ledger.json               that transaction's edits + their statuses
  txn_.../<tensor__name>.npy        exact pre-transaction bits for that tensor
```

`ledger_index.json` status values: `committed`, `accepted`, `rejected`, `failed`,
`partially_reverted`, `reverted`. `outcome` preserves the original outcome.
Invariant enforced by tests: `reverted_edit_ids ⊆ edit_ids` for every
transaction, and the union of reverted IDs matches the tensors actually restored.

## 5. Test coverage

`tests/test_v4_wsp.py` (C/D/E groups) and `tests/test_v3_revert_and_loader.py`
cover: two and three committed transactions on one tensor, `revert(n=2)` ledger
ownership, committed `revert_transaction`, dependency refusal, cascading revert,
failed-restore atomicity with first-write recovery, missing journal, idempotency,
unknown IDs, restart recovery, and agent synchronization (including a failed
revert leaving the edit active).
