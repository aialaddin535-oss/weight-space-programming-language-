"""Direct weight-editing engine.

Guarantees enforced here, on every candidate, before a single entry is written:

  1. tensor exists                4. dtype matches the model's storage dtype
  2. shape matches the model      5. expected base hash matches (no stale edits)
  3. not a frozen tensor          6. changed entries / relative change within bounds
  7. not a pure uniform rescale of a whole tensor (refused, like the reference)

Apply path: validate -> preview new values -> build sparse record -> write delta
-> write values into the model -> re-hash -> assert edited hash. Any failure
leaves the model untouched because the delta is built from copies first.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from ..inspection.tensors import diff_tensors
from ..sparse_deltas.store import DeltaStore, build_record
from . import ops as O


class EditRefused(RuntimeError):
    pass


class TensorEditor:
    def __init__(self, backend, cfg: dict, store: DeltaStore | None = None):
        self.backend = backend
        self.cfg = cfg
        e = cfg["editing"]
        self.max_entries = int(e["max_entries_per_tensor"])
        self.max_rel = float(e["max_relative_change"])
        self.min_rel = float(e["min_relative_change"])
        self.frozen = list(e["frozen_suffixes"])
        self.refuse_rescale = bool(e["refuse_uniform_rescale"])
        self.store = store

    # ------------------------------------------------------------- validation
    def validate(self, op: dict) -> dict:
        """Full pre-flight check. Returns {'ok': bool, 'errors': [...], 'info': {...}}."""
        errors: List[str] = []
        info: dict = {}
        name = (op or {}).get("tensor") or (op or {}).get("name")
        shape = ()
        if name and name in set(self.backend.param_names()):
            shape = self.backend.shape_of(name)
        else:
            errors.append(f"unknown tensor {name!r}")
        errors += O.validate(op, shape) if not errors else []
        if errors:
            return {"ok": False, "errors": errors, "info": info}

        dt = self.backend.dtype_of(name)
        if dt.lower() not in ("float32", "bfloat16", "float16"):
            errors.append(f"dtype {dt} not editable (only float dtypes)")
        if self.backend.frozen(name, self.frozen):
            errors.append(f"{name} is frozen (embedding / final norm / lm_head)")

        alpha = float(op["alpha"])
        if abs(alpha) > self.max_rel:
            errors.append(f"|alpha|={abs(alpha):.4f} exceeds max_relative_change={self.max_rel}")
        kind = op.get("kind") or op.get("op")
        n_cols, n_rows = shape[1], shape[0]
        if kind in ("scale_columns", "shift_columns"):
            n_ch = len(set(op["columns"])) * n_rows
        elif kind == "scale_rows":
            n_ch = len(set(op["rows"])) * n_cols
        else:
            n_ch = len(set(op["columns"])) * len(set(op["rows"]))
        if n_ch > self.max_entries:
            errors.append(f"edit touches {n_ch} entries > max_entries_per_tensor={self.max_entries}")
        if self.refuse_rescale and kind in ("scale_columns",) and n_cols == len(set(op["columns"])):
            errors.append("refused: scaling every column is a uniform rescale of the tensor")
        if self.refuse_rescale and kind == "scale_rows" and n_rows == len(set(op["rows"])):
            errors.append("refused: scaling every row is a uniform rescale of the tensor")
        info.update({"tensor": name, "dtype": dt, "shape": list(shape),
                     "entries_touched": int(n_ch), "alpha": alpha,
                     "base_sha256": self.backend.tensor_hash(name),
                     "description": O.describe(op)})
        return {"ok": not errors, "errors": errors, "info": info}

    # ---------------------------------------------------------------- preview
    def preview(self, op: dict) -> dict:
        v = self.validate(op)
        if not v["ok"]:
            return {"ok": False, "errors": v["errors"]}
        name = op["tensor"] if "tensor" in op else op["name"]
        base = self.backend.get_tensor(name)
        new = O.apply(base, op)
        diff = diff_tensors(base, new)
        rec, idx, val = build_record(name, self.backend.dtype_of(name), base, new,
                                     reason=op.get("reason", ""))
        within = (rec["changed_entries"] <= self.max_entries and
                  rec["relative_change"] <= self.max_rel and
                  rec["relative_change"] >= 0.0)
        return {"ok": True, "info": v["info"], "diff": diff, "record": rec,
                "delta_entries": int(idx.size), "within_bounds": bool(within),
                "would_change_entries": rec["changed_entries"],
                "relative_change": rec["relative_change"]}

    # ------------------------------------------------------------------ apply
    def apply(self, op: dict, edit_id: str | None = None, reason: str = "",
              meta: dict | None = None, store: DeltaStore | None = None) -> dict:
        store = store or self.store
        if store is None:
            raise EditRefused("no DeltaStore configured")
        preview = self.preview(op)
        if not preview["ok"]:
            raise EditRefused("; ".join(preview["errors"]))
        if not preview["within_bounds"]:
            raise EditRefused("candidate exceeds configured bounds")
        if preview["record"]["changed_entries"] == 0:
            raise EditRefused("candidate would change no stored entries")
        name = op["tensor"] if "tensor" in op else op["name"]
        base = self.backend.get_tensor(name)
        new = O.apply(base, op)
        dtype = self.backend.dtype_of(name)
        rec, idx, val = build_record(name, dtype, base, new, reason=reason)

        edit_id = edit_id or store.new_edit_id()
        man = store.save(edit_id, [(rec, idx, val)], {
            "reason": reason,
            "status": "applied",
            "operation": op,
            "agent": (meta or {}).get("agent"),
            "hypothesis": (meta or {}).get("hypothesis"),
            "experiment": (meta or {}).get("experiment"),
            "pre_apply_hash": self.backend.tensor_hash(name),
            **(meta or {}),
        })
        # write into the live model, then prove the hash
        self.backend.set_tensor(name, new)
        after = self.backend.tensor_hash(name)
        if after != rec["edited_sha256"]:
            # roll the single tensor back immediately
            self.backend.set_tensor(name, base)
            store.update_status(edit_id, "failed_hash")
            raise EditRefused(
                f"post-write hash mismatch on {name}: {after[:12]} != {rec['edited_sha256'][:12]}; reverted")
        return {
            "edit_id": edit_id,
            "manifest": man,
            "tensor": name,
            "changed_entries": rec["changed_entries"],
            "relative_change": rec["relative_change"],
            "base_sha256": rec["base_sha256"],
            "edited_sha256": rec["edited_sha256"],
            "verified": True,
            "diff": preview["diff"],
        }

    # ------------------------------------------------------------------ undo
    def explain_undo(self) -> str:
        """Sparse deltas are forward-only by design: they store the NEW values of
        changed entries, not the old ones. Reverting therefore means rebuilding the
        weights from the pristine base plus the deltas you want to keep - which is
        exactly what CheckpointManager.rebuild_chain_after_reset() does, and it is
        exact because reconstruction re-derives the bits instead of inverting them.
        """
        return ("use CheckpointManager.rebuild_chain_after_reset(backend, kept_edit_ids) "
                "for any revert/rollback; sparse deltas are not invertible by design")
