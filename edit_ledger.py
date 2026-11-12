"""Transaction-scoped edit ledger: the single source of truth for multi-edit state.

One EditTransaction covers one candidate experiment (which may contain several
tensor edits, possibly on the same tensor). For every tensor touched by the
transaction, the EXACT pre-transaction bits are captured once, before the first
edit in that transaction writes to it, and kept on disk. Rejecting or failing the
transaction therefore restores the exact original model state with byte-level
fidelity - verified by SHA-256 over the raw storage representation - and the live
ledger is only updated AFTER a verified restore.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .sparse_deltas.cast import cast_to_dtype
from .sparse_deltas.store import sha256_arr


class EditRecord(dict):
    pass


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
        self._saved: Dict[str, str] = {}           # tensor -> sha256 of saved pristine bits

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
            "txn_id": self.txn_id,
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
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
        """The exact saved pristine float32 values of one tensor (for per-edit revert)."""
        return np.load(self._journal_path(name))

    def tensors_touched(self) -> List[str]:
        return list(dict.fromkeys(e["tensor"] for e in self.edits))

    def edit_ids(self) -> List[str]:
        return [e["edit_id"] for e in self.edits]

    def mark_edits_reverted(self, edit_ids: List[str], error: str | None = None) -> None:
        """Flip individual edits to 'reverted' in the durable ledger (idempotent).

        Only called AFTER the tensors were actually restored and hash-verified.
        """
        ids = set(edit_ids)
        changed = False
        for e in self.edits:
            if e["edit_id"] in ids and e.get("status") != "reverted":
                e["status"] = "reverted"
                e["reverted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                if error:
                    e["revert_error"] = error
                changed = True
        if changed:
            self._write_ledger()

    def restore_pristine(self, verify: bool = True) -> dict:
        """Undo the WHOLE transaction: every tensor captured in the journal (whether
        or not its edit completed) goes back to its exact pre-transaction bits.
        Verified; idempotent; raises on failure."""
        touched = list(self._saved) + [n for n in self.tensors_touched()
                                       if n not in self._saved]
        restored, drift = [], []
        for name in touched:
            p = self._journal_path(name)
            if not p.exists():
                raise FileNotFoundError(f"missing pristine journal tensor {p}")
            arr = np.load(p)
            if verify:
                h = sha256_arr(cast_to_dtype(arr, self.backend.dtype_of(name)))
                if h != self._saved.get(name):
                    raise RuntimeError(
                        f"{name}: pristine journal hash mismatch ({h[:12]} != "
                        f"{(self._saved.get(name) or '')[:12]}); tensor NOT modified")
            self.backend.set_tensor(name, arr)
            if verify and self.backend.tensor_hash(name) != self._saved[name]:
                drift.append(name)
            else:
                restored.append(name)
        return {"txn_id": self.txn_id, "restored": restored, "drift": drift,
                "ok": not drift and len(restored) == len(touched)}

    # --------------------------------------------------------------- ledger
    def _ledger_file(self) -> Path:
        return self.root / "ledger.json"

    def _write_ledger(self) -> None:
        tmp = self._ledger_file().with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"txn_id": self.txn_id, "status": self.status,
                       "error": self.error, "edits": self.edits}, f, indent=1,
                      ensure_ascii=False)
        tmp.replace(self._ledger_file())

    def mark(self, status: str, error: str | None = None) -> None:
        self.status = status
        self.error = error
        for e in self.edits:
            # never overwrite a per-edit status set by a verified per-edit revert
            if e.get("status") in ("applied", None, ""):
                e["status"] = status if status in ("rejected", "failed") else e["status"]
        self._write_ledger()


class EditLedger:
    """Durable registry of all transactions of an experiment run."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root / "ledger_index.json"
        self.index: Dict[str, dict] = {}
        if self.index_file.exists():
            self.index = json.load(open(self.index_file, "r", encoding="utf-8"))

    def register(self, txn: EditTransaction, status: str, accepted: bool | None,
                 error: str | None = None) -> None:
        self.index[txn.txn_id] = {
            "txn_id": txn.txn_id, "status": status, "accepted": accepted,
            "error": error,
            "edit_ids": txn.edit_ids(),
            "tensors": txn.tensors_touched(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=1, ensure_ascii=False)

    def transactions(self) -> List[dict]:
        return [v for _, v in sorted(self.index.items())]

    def mark_edits_reverted(self, txn_id: str, edit_ids: List[str],
                            error: str | None = None) -> None:
        """Record per-edit revert status in the txn's durable ledger.json and in
        the index. Only called after a verified tensor restore; idempotent."""
        lf = self.root / txn_id / "ledger.json"
        ids = set(edit_ids)
        if lf.exists():
            d = json.load(open(lf, "r", encoding="utf-8"))
            for e in d.get("edits", []):
                if e.get("edit_id") in ids and e.get("status") != "reverted":
                    e["status"] = "reverted"
                    e["reverted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    if error:
                        e["revert_error"] = error
            with open(lf, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1, ensure_ascii=False)
        idx = self.index.get(txn_id)
        if idx is not None:
            idx["reverted_edit_ids"] = sorted(set(idx.get("reverted_edit_ids", [])) | ids)
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=1, ensure_ascii=False)

    def edits_in_order(self) -> List[dict]:
        """Every edit record across transactions, in execution order."""
        out: List[dict] = []
        for t in self.transactions():
            lf = self.root / t["txn_id"] / "ledger.json"
            if lf.exists():
                d = json.load(open(lf, "r", encoding="utf-8"))
                out.extend(d.get("edits", []))
        return out
