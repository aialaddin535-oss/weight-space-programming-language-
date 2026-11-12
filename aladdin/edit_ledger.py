"""Transaction-scoped edit ledger: the single source of truth for edit state.

One EditTransaction covers one candidate experiment (which may contain several
tensor edits, possibly on the same tensor). For every tensor touched by the
transaction, the EXACT pre-transaction bits are captured once, before the first
edit in that transaction writes to it, and kept on disk.

Two hard invariants:

1. **Ownership is per edit, never "the latest transaction".** Every edit record
   carries the `txn_id` of the transaction that created it; status changes are
   written back to THAT transaction's ledger file and index entry. Reverting
   edits from transactions A and B updates A and B separately.

2. **A restore is atomic and verified.** All journals are validated and hashed
   before anything is written; the live values that are about to change are
   captured first; a failure midway restores those captured values and raises
   loudly. Only after a verified restore may statuses change.

A failed rollback therefore cannot leave tensor values and transaction history
inconsistent: either the tensors are back and the ledger says so, or nothing
was written and the ledger is untouched.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .sparse_deltas.cast import cast_to_dtype
from .sparse_deltas.store import sha256_arr

# Edit statuses that mean "this edit is not on the model right now".
INACTIVE_EDIT_STATUSES = ("reverted", "rejected", "failed")
# Transaction statuses that mean "none of its edits are on the model right now".
INACTIVE_TXN_STATUSES = ("rejected", "failed", "reverted")


class EditRecord(dict):
    """One applied tensor edit, always tagged with its owning transaction."""


class LedgerError(RuntimeError):
    pass


class RestoreError(LedgerError):
    """A verified tensor restore failed; the ledger was not changed."""

    def __init__(self, message: str, *, problems: Optional[List[str]] = None,
                 recovered: bool | None = None, restored: Optional[List[str]] = None):
        super().__init__(message)
        self.problems = problems or []
        self.recovered = recovered
        self.restored = restored or []


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json_atomic(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


class EditTransaction:
    """All edits from one agent interaction / candidate, as one atomic unit."""

    def __init__(self, backend, root: str | Path, txn_id: str | None = None):
        self.backend = backend
        self.txn_id = txn_id or f"txn_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.root = Path(root) / self.txn_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.edits: List[EditRecord] = []
        self.status = "open"                       # open | committed | rejected | failed
        self.error: Optional[str] = None
        self.created_utc = _utc()
        self._saved: Dict[str, str] = {}           # tensor -> sha256 of pristine bits

    # ------------------------------------------------------------ journaling
    def _journal_path(self, name: str) -> Path:
        return self.root / (name.replace(".", "__") + ".npy")

    def capture_pristine(self, name: str) -> None:
        """Save the tensor's exact current bits once per transaction."""
        if name in self._saved:
            return
        p = self._journal_path(name)
        arr = self.backend.get_tensor(name)
        np.save(p, arr)
        self._saved[name] = sha256_arr(cast_to_dtype(arr, self.backend.dtype_of(name)))

    def record_edit(self, edit_id: str, tensor: str, rec: dict, meta: dict) -> EditRecord:
        entry = EditRecord({
            "edit_id": edit_id,
            "txn_id": self.txn_id,                 # owned by THIS transaction
            "tensor": tensor,
            "tensor_sha256_after": rec["edited_sha256"],
            "tensor_sha256_before": rec["base_sha256"],
            "shape": list(rec["shape"]),
            "dtype": rec["dtype"],
            "changed_entries": rec["changed_entries"],
            "relative_change": rec["relative_change"],
            "order": len(self.edits),
            # strict execution order, used by per-edit (partial) revert
            "txn_order": len(self.edits),
            "utc": _utc(),
            "status": "applied",
            **meta,
        })
        self.edits.append(entry)
        self._write_ledger()
        return entry

    # -------------------------------------------------------------- restore
    def pristine_sha(self, name: str) -> str:
        return self._saved[name]

    def _preimage(self, name: str) -> np.ndarray:
        """The exact saved pristine float32 values of one tensor."""
        return np.load(self._journal_path(name))

    def tensors_touched(self) -> List[str]:
        return list(dict.fromkeys(e["tensor"] for e in self.edits))

    def edit_ids(self) -> List[str]:
        return [e["edit_id"] for e in self.edits]

    def live_edit_ids(self) -> List[str]:
        return [e["edit_id"] for e in self.edits
                if e.get("status") not in INACTIVE_EDIT_STATUSES]

    def mark_edits_reverted(self, edit_ids: List[str], error: str | None = None) -> bool:
        """Flip individual edits to 'reverted' in this txn's durable ledger.

        Only called AFTER the tensors were actually restored and hash-verified;
        idempotent.
        """
        ids = set(edit_ids)
        changed = False
        for e in self.edits:
            if e["edit_id"] in ids and e.get("status") != "reverted":
                e["status"] = "reverted"
                e["reverted_utc"] = _utc()
                if error:
                    e["revert_error"] = error
                changed = True
        if changed:
            self._write_ledger()
        return changed

    def _restore_plan(self) -> Dict[str, str]:
        """tensor -> expected pre-transaction hash, for every journaled tensor."""
        expected = dict(self._saved)
        for e in self.edits:
            expected.setdefault(e["tensor"], e["tensor_sha256_before"])
        return expected

    def restore_pristine(self, verify: bool = True) -> dict:
        """Undo the WHOLE transaction atomically.

        1. Pre-validate: every journal file exists and its hash matches.
        2. Capture the live values of the tensors that will change.
        3. Write the pristine values, hash-verify each write.
        4. On any failure, restore the captured values and raise RestoreError.

        Idempotent: restoring an already-pristine model verifies and succeeds.
        """
        expected = self._restore_plan()
        if not expected:
            return {"txn_id": self.txn_id, "restored": [], "drift": [], "ok": True,
                    "atomic": True, "note": "transaction touched no tensors"}

        # ---- 1. validate EVERYTHING before touching the model -------------
        arrays: Dict[str, np.ndarray] = {}
        problems: List[str] = []
        for name, want in expected.items():
            p = self._journal_path(name)
            if not p.exists():
                problems.append(f"{name}: missing pristine journal {p.name}")
                continue
            arr = np.load(p)
            if verify:
                have = sha256_arr(cast_to_dtype(arr, self.backend.dtype_of(name)))
                if have != want:
                    problems.append(
                        f"{name}: pristine journal hash mismatch "
                        f"({have[:12]} != {want[:12]})")
                    continue
            arrays[name] = arr
        if problems:
            raise RestoreError(
                "pre-validation failed; NO tensor was modified: "
                + "; ".join(problems), problems=problems, recovered=None)

        # ---- 2. capture what is about to change ---------------------------
        backup = {n: self.backend.get_tensor(n) for n in arrays}

        # ---- 3. write in reverse edit order (newest edit undone first) -----
        order: List[str] = []
        for e in reversed(self.edits):
            if e["tensor"] in arrays and e["tensor"] not in order:
                order.append(e["tensor"])
        order += [n for n in arrays if n not in order]
        written: List[str] = []
        try:
            for name in order:
                self.backend.set_tensor(name, arrays[name])
                written.append(name)
                if verify and self.backend.tensor_hash(name) != expected[name]:
                    raise RestoreError(
                        f"{name}: post-restore hash mismatch; rollback NOT verified",
                        problems=[name], restored=written)
        except Exception as ex:
            # ---- 4. recover: put every touched tensor back -----------------
            recovery_errors = []
            for name, arr in backup.items():
                try:
                    self.backend.set_tensor(name, arr)
                except Exception as rex:              # pragma: no cover - hw failure
                    recovery_errors.append(f"{name}: {type(rex).__name__}: {rex}")
            msg = f"restore failed ({type(ex).__name__}: {ex})"
            if recovery_errors:
                msg += ("; PRE-RESTORE VALUES COULD NOT BE RECOVERED: "
                        + "; ".join(recovery_errors))
                raise RestoreError(msg, problems=recovery_errors,
                                   recovered=False, restored=[]) from ex
            raise RestoreError(msg + "; all touched tensors were restored to their "
                              "pre-rollback values", recovered=True,
                              restored=[]) from ex

        drift = [n for n in order
                 if verify and self.backend.tensor_hash(n) != expected[n]]
        return {"txn_id": self.txn_id, "restored": written, "drift": drift,
                "ok": not drift and len(written) == len(arrays), "atomic": True,
                "verified": verify and not drift}

    # --------------------------------------------------------------- ledger
    def _ledger_file(self) -> Path:
        return self.root / "ledger.json"

    def _write_ledger(self) -> None:
        _write_json_atomic(self._ledger_file(), {
            "txn_id": self.txn_id, "status": self.status, "error": self.error,
            "created_utc": self.created_utc, "edits": self.edits,
            "tensors": self.tensors_touched(), "updated_utc": _utc()})

    def mark(self, status: str, error: str | None = None) -> None:
        """Close the transaction. Per-edit 'reverted' statuses are preserved."""
        self.status = status
        self.error = error
        for e in self.edits:
            if e.get("status") in ("applied", None, ""):
                if status in ("rejected", "failed"):
                    e["status"] = status
        self._write_ledger()

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, root: str | Path, txn_id: str, backend=None) -> "EditTransaction":
        """Rehydrate a transaction (committed or not) from its durable journal.

        This is what makes `revert_transaction(txn_id)` work after a restart:
        the pristine pre-images live in `<root>/<txn_id>/*.npy`, not in memory.
        """
        tdir = Path(root) / txn_id
        lf = tdir / "ledger.json"
        if not lf.exists():
            raise LedgerError(f"no durable ledger for transaction {txn_id!r} at {lf}")
        data = json.load(open(lf, "r", encoding="utf-8"))
        obj = cls.__new__(cls)
        obj.backend = backend
        obj.txn_id = data.get("txn_id", txn_id)
        obj.root = tdir
        obj.edits = [EditRecord(e) for e in data.get("edits", [])]
        obj.status = data.get("status", "committed")
        obj.error = data.get("error")
        obj.created_utc = data.get("created_utc")
        obj._saved = {}
        for e in obj.edits:
            name = e["tensor"]
            p = obj._journal_path(name)
            if p.exists() and name not in obj._saved:
                obj._saved[name] = e["tensor_sha256_before"]
        return obj

    def journal_report(self) -> dict:
        expected = self._restore_plan()
        present, missing = [], []
        for name in expected:
            (present if self._journal_path(name).exists() else missing).append(name)
        return {"txn_id": self.txn_id, "tensors": sorted(expected),
                "journal_present": sorted(present), "journal_missing": sorted(missing),
                "revertible": not missing}


class EditLedger:
    """Durable registry of all transactions of an experiment run."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root / "ledger_index.json"
        self.index: Dict[str, dict] = {}
        if self.index_file.exists():
            try:
                self.index = json.load(open(self.index_file, "r", encoding="utf-8"))
            except json.JSONDecodeError:
                self.index = {}
        self._seq = max([int(v.get("seq", 0)) for v in self.index.values()] or [0])

    # ------------------------------------------------------------- registry
    def _save_index(self) -> None:
        _write_json_atomic(self.index_file, self.index)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def register(self, txn: EditTransaction, status: str, accepted: bool | None,
                 error: str | None = None) -> None:
        prev = self.index.get(txn.txn_id, {})
        self.index[txn.txn_id] = {
            "txn_id": txn.txn_id,
            "status": status,
            "outcome": status,
            "accepted": accepted,
            "error": error,
            "edit_ids": txn.edit_ids(),
            "n_edits": len(txn.edits),
            "tensors": txn.tensors_touched(),
            "utc": _utc(),
            "seq": prev.get("seq") or self._next_seq(),
            "reverted_edit_ids": sorted(prev.get("reverted_edit_ids") or []),
            "edits_reverted": len(prev.get("reverted_edit_ids") or []),
        }
        self._save_index()

    def transactions(self) -> List[dict]:
        return [v for _, v in sorted(self.index.items(),
                                     key=lambda kv: (int(kv[1].get("seq", 0)),
                                                     kv[1].get("utc") or ""))]

    def get(self, txn_id: str) -> dict | None:
        return self.index.get(txn_id)

    def has(self, txn_id: str) -> bool:
        return txn_id in self.index or (self.root / txn_id / "ledger.json").exists()

    def transaction(self, txn_id: str, backend=None) -> EditTransaction:
        return EditTransaction.load(self.root, txn_id, backend=backend)

    def owner_of(self, edit_id: str) -> str | None:
        """The transaction that created `edit_id` (never inferred from recency)."""
        for t in self.transactions():
            if edit_id in (t.get("edit_ids") or []):
                return t["txn_id"]
        return None

    # ------------------------------------------------------- status updates
    def _recompute(self, txn_id: str, edits: List[dict]) -> dict:
        idx = self.index.setdefault(txn_id, {"txn_id": txn_id, "edit_ids": [],
                                             "seq": self._next_seq(),
                                             "reverted_edit_ids": []})
        reverted = [e["edit_id"] for e in edits if e.get("status") == "reverted"]
        idx["edit_ids"] = [e["edit_id"] for e in edits]
        idx["n_edits"] = len(edits)
        idx["reverted_edit_ids"] = sorted(set(idx.get("reverted_edit_ids") or [])
                                          | set(reverted))
        idx["edits_reverted"] = len(idx["reverted_edit_ids"])
        outcome = idx.get("outcome") or idx.get("status") or "committed"
        if edits and len(reverted) == len(edits):
            idx["status"] = "reverted"
        elif reverted:
            idx["status"] = "partially_reverted"
        else:
            idx["status"] = outcome
        idx["ledger_updated_utc"] = _utc()
        return idx

    def mark_edits_reverted(self, txn_id: str, edit_ids: List[str],
                            error: str | None = None) -> dict:
        """Write per-edit revert status into the OWNING transaction only.

        Called after a verified tensor restore; idempotent. Never touches any
        other transaction's edit IDs.
        """
        ids = set(edit_ids)
        known = set(self.index.get(txn_id, {}).get("edit_ids") or [])
        if known and not ids <= known:
            stray = sorted(ids - known)
            raise LedgerError(
                f"refusing to mark {stray[:5]} as reverted in transaction {txn_id!r}: "
                "those edit IDs belong to a different transaction")
        lf = self.root / txn_id / "ledger.json"
        edits: List[dict] = []
        if lf.exists():
            d = json.load(open(lf, "r", encoding="utf-8"))
            for e in d.get("edits", []):
                if e.get("edit_id") in ids and e.get("status") != "reverted":
                    e["status"] = "reverted"
                    e["reverted_utc"] = _utc()
                    if error:
                        e["revert_error"] = error
            d["updated_utc"] = _utc()
            _write_json_atomic(lf, d)
            edits = d.get("edits", [])
        if not edits:                       # txn dir gone: index is the only record
            for eid in ids:
                if eid in known:
                    idx = self.index[txn_id]
                    idx.setdefault("reverted_edit_ids", []).append(eid)
            idx = self.index.get(txn_id)
            if idx is not None:
                idx["reverted_edit_ids"] = sorted(set(idx["reverted_edit_ids"]))
                self._save_index()
            return {"txn_id": txn_id, "edit_ids": sorted(ids),
                    "status": (self.index.get(txn_id) or {}).get("status")}
        idx = self._recompute(txn_id, edits)
        self._save_index()
        return {"txn_id": txn_id, "edit_ids": sorted(ids), "status": idx["status"]}

    def mark_edits_reverted_bulk(self, by_txn: Dict[str, List[str]],
                                 error: str | None = None) -> List[dict]:
        """Apply revert statuses grouped by OWNING transaction.

        This is the fix for the cross-transaction ledger bug: reverting edits
        from A and B updates A and B separately instead of dumping both IDs on
        the most recent transaction.
        """
        return [self.mark_edits_reverted(txn_id, ids, error)
                for txn_id, ids in sorted(by_txn.items()) if ids]

    # ------------------------------------------------------------ queries
    def edits_in_order(self) -> List[dict]:
        """Every edit record across transactions, in execution order."""
        out: List[dict] = []
        for t in self.transactions():
            lf = self.root / t["txn_id"] / "ledger.json"
            if not lf.exists():
                continue
            try:
                d = json.load(open(lf, "r", encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for e in d.get("edits", []):
                rec = EditRecord(e)
                rec.setdefault("txn_id", t["txn_id"])
                rec["txn_seq"] = t.get("seq", 0)
                out.append(rec)
        return out

    def live_edits(self) -> List[dict]:
        return [e for e in self.edits_in_order()
                if e.get("status") not in INACTIVE_EDIT_STATUSES]

    def dependents(self, txn_id: str) -> List[dict]:
        """Later transactions that still have edits on this transaction's tensors.

        Reverting an older transaction while a later one keeps a modified value
        of the same tensor would destroy the later edit's pre-image, so callers
        must refuse or cascade (reverse chronological order).
        """
        target = self.index.get(txn_id)
        if target is None:
            return []
        tensors = set(target.get("tensors") or [])
        if not tensors:
            return []
        seq = int(target.get("seq", 0))
        out = []
        for t in self.transactions():
            if t["txn_id"] == txn_id or int(t.get("seq", 0)) <= seq:
                continue
            if t.get("status") in INACTIVE_TXN_STATUSES:
                continue
            shared = sorted(tensors & set(t.get("tensors") or []))
            if not shared:
                continue
            reverted = set(t.get("reverted_edit_ids") or [])
            live = [e for e in (t.get("edit_ids") or []) if e not in reverted]
            if live:
                out.append({"txn_id": t["txn_id"], "edit_ids": live,
                            "tensors": shared, "status": t.get("status")})
        return out

    def status_report(self) -> dict:
        """Ledger-wide view used by the agent and by tests."""
        txns = self.transactions()
        return {
            "n_transactions": len(txns),
            "n_edits": sum(len(t.get("edit_ids") or []) for t in txns),
            "n_reverted": sum(len(t.get("reverted_edit_ids") or []) for t in txns),
            "transactions": [{"txn_id": t["txn_id"], "status": t.get("status"),
                              "outcome": t.get("outcome"),
                              "edit_ids": t.get("edit_ids"),
                              "reverted_edit_ids": t.get("reverted_edit_ids")}
                             for t in txns],
        }
