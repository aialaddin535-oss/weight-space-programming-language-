"""Checkpoint + rollback manager.

Layout
    checkpoints/checkpoints.json     registry (hashes + metrics per checkpoint)
    checkpoints/base/                immutable marker + pristine reference
    checkpoints/<name>/              .npy snapshots (see modes below)
    checkpoints/deltas/<edit_id>/    sparse edits (see sparse_deltas.store)

Rollback is *chain based*: reset the live model to the pristine base weights, then
re-apply only the kept sparse deltas in order, verifying every tensor's recorded
base hash. That is exact for forward-only sparse deltas - no inverse edit and no
accumulated rounding - which is why the reference experiment stores deltas rather
than "undo" files.

Recoverable checkpoint modes (BUG 1 fix)
    full           small models: every tensor saved as .npy.
    modified_only  large models: ONLY the tensors whose hash differs from the
                   verified base checkpoint are saved. The untouched majority is
                   restored from a pristine reload via backend_factory (verified
                   against the recorded base hashes BEFORE anything is written).
                   A hash alone is never treated as recoverable weight data.

Every checkpoint therefore contains enough information to restore the exact model
state without cloning the whole model into memory per checkpoint.

Restoration is verified + atomic: staged data is hash-checked before any write,
the live values of tensors that are about to change are captured first, and any
failure restores that captured state and raises loudly - a failed or interrupted
restoration never silently leaves partially modified weights.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from ..sparse_deltas.cast import cast_to_dtype
from ..sparse_deltas.store import DeltaStore, sha256_arr

SNAPSHOT_PARAM_LIMIT = 8_000_000      # ~32 MB of float32 .npy; above this, modified-tensors mode


class CheckpointManager:
    def __init__(self, backend, root: str | Path, deltas_root: str | Path | None = None,
                 backend_factory: Callable[[], object] | None = None):
        self.backend = backend
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.deltas_root = Path(deltas_root) if deltas_root else self.root / "deltas"
        self.deltas_root.mkdir(parents=True, exist_ok=True)
        self.store = DeltaStore(self.deltas_root)
        self.backend_factory = backend_factory
        self.manifest = self.root / "checkpoints.json"
        self._data: Dict[str, dict] = {}
        if self.manifest.exists():
            self._data = json.load(open(self.manifest, "r", encoding="utf-8"))

    # ---------------------------------------------------------------- helpers
    def _save_manifest(self) -> None:
        with open(self.manifest, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=1)

    @staticmethod
    def _snap_file(name: str) -> str:
        return name.replace(".", "__") + ".npy"

    def list(self) -> List[dict]:
        return [{"name": k, **{kk: vv for kk, vv in v.items() if kk != "hashes"}}
                for k, v in sorted(self._data.items())]

    # ------------------------------------------------------------- snapshots
    def save(self, name: str, note: str = "", parent: str | None = None,
             metrics: dict | None = None, immutable: bool = False,
             txn: str | None = None) -> dict:
        """Save a checkpoint.

        Small models -> mode 'full' (every tensor). Large models -> mode
        'modified_only': exact snapshots of exactly the tensors that differ from
        the verified base checkpoint. The base checkpoint itself stores no tensor
        copies in this mode (the pristine weights live on disk and are hash-
        verified on every restore).
        """
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        hashes = self.backend.snapshot_hashes()
        n_par = self.backend.n_params()
        if n_par <= SNAPSHOT_PARAM_LIMIT:
            mode = "full"
            changed = sorted(hashes)
            for n in changed:
                np.save(d / self._snap_file(n), self.backend.get_tensor(n).astype(np.float32))
        else:
            mode = "modified_only"
            if name == "base":
                changed = []                       # the base equals itself by definition
            else:
                base_hashes = (self._data.get("base") or {}).get("hashes", {})
                changed = sorted(n for n, h in hashes.items() if base_hashes.get(n) != h)
            for n in changed:
                np.save(d / self._snap_file(n), self.backend.get_tensor(n).astype(np.float32))
        entry = {
            "name": name, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": note, "parent": parent, "mode": mode,
            "changed_from_base": changed,
            "n_tensors": len(hashes), "n_params": int(n_par),
            "metrics": metrics or {}, "immutable": bool(immutable), "txn": txn,
            "hashes": hashes,
        }
        self._data[name] = entry
        self._save_manifest()
        with open(d / "checkpoint.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in entry.items() if k != "hashes"}, f, indent=1)
        return entry

    def save_base(self) -> dict:
        if "base" in self._data and self._data["base"].get("immutable"):
            return self._data["base"]
        return self.save("base", note="pristine model before any weight edit", immutable=True)

    # ------------------------------------------------------- staged restore
    def _fresh_verified_backend(self) -> object:
        """Reload pristine weights from disk and verify them against the base
        checkpoint hashes BEFORE any restoration touches the live model."""
        if self.backend_factory is None:
            raise RuntimeError(
                "no backend_factory configured; cannot reload pristine weights for "
                "restoration. Pass backend_factory= to CheckpointManager.")
        base = self._data.get("base")
        if base is None:
            raise RuntimeError("no base checkpoint recorded; call save_base() first")
        fresh = self.backend_factory()
        mismatch = [n for n, h in base["hashes"].items() if fresh.tensor_hash(n) != h]
        if mismatch:
            raise RuntimeError(
                "base weights on disk do NOT match the verified base checkpoint hashes "
                f"(first mismatches: {mismatch[:5]}); refusing to restore on top of an "
                "unverified base")
        return fresh

    def _staged_values(self, e: dict, d: Path, fresh) -> Dict[str, np.ndarray]:
        """Collect the exact float32 values for every tensor and hash-verify the
        staged data BEFORE the live model is touched."""
        if e["mode"] == "full":
            wanted = set(e["hashes"])
        elif e["mode"] == "modified_only":
            wanted = set(e.get("changed_from_base") or [])
        else:
            raise RuntimeError(
                f"checkpoint {e['name']!r} uses legacy mode {e['mode']!r}: it stores "
                "hashes only, which are not recoverable weight data. Use "
                "reset_to_base()/rebuild_chain() instead.")
        arrays: Dict[str, np.ndarray] = {}
        for n in self.backend.param_names():
            if n in wanted:
                f = d / self._snap_file(n)
                if not f.exists():
                    raise FileNotFoundError(f"missing snapshot tensor {f}")
                arrays[n] = np.load(f)
            else:
                arrays[n] = fresh.get_tensor(n)
        bad = []
        for n, a in arrays.items():
            bits = cast_to_dtype(a, self.backend.dtype_of(n))
            if sha256_arr(bits) != e["hashes"].get(n):
                bad.append(n)
        if bad:
            raise RuntimeError(
                f"staged checkpoint data failed hash verification for {bad[:5]}; "
                "the live model was NOT modified")
        return arrays

    def restore(self, name: str) -> dict:
        """Restore an exact checkpoint state, verified, without partial writes."""
        if name not in self._data:
            raise KeyError(f"no checkpoint {name!r}; have {sorted(self._data)}")
        e = self._data[name]
        d = self.root / name
        fresh = None
        if e["mode"] != "full":
            fresh = self._fresh_verified_backend()
        arrays = self._staged_values(e, d, fresh)
        # undo journal: current values of every tensor that is about to change
        journal = {n: self.backend.get_tensor(n) for n, a in arrays.items()
                   if self.backend.tensor_hash(n) != e["hashes"].get(n)}
        try:
            for n, a in arrays.items():
                self.backend.set_tensor(n, a)
        except Exception:
            for n, a in journal.items():        # best-effort rollback of the writes
                try:
                    self.backend.set_tensor(n, a)
                except Exception:
                    pass
            raise
        drift = [n for n, h in e["hashes"].items() if self.backend.tensor_hash(n) != h]
        return {"restored": len(arrays), "drift": drift, "ok": not drift,
                "mode": e["mode"], "verified": not drift}

    # ------------------------------------------------------------- rollback
    def reset_to_base(self) -> dict:
        """Put the LIVE backend back to pristine weights, in place.

        Small model  -> copy the saved base .npy files in.
        Big model    -> verify a pristine reload against the recorded base hashes,
                        then copy its tensors into the live object (so every held
                        reference stays valid).
        """
        if "base" not in self._data:
            raise RuntimeError("no base checkpoint recorded; call save_base() first")
        base = self._data["base"]
        if base["mode"] == "full":
            r = self.restore("base")
            return {"mode": "snapshot", **r}
        r = self.restore("base")
        return {"mode": "reloaded_from_disk", **r}

    def rebuild_chain(self, edit_ids: List[str], verify: bool = True) -> dict:
        """Apply deltas in order onto the CURRENT state (used when resuming)."""
        applied, failed = [], None
        for eid in edit_ids:
            try:
                names = self.store.rebuild_tensors(eid, self.backend, verify=verify)
                applied.append({"edit_id": eid, "tensors": names})
            except Exception as ex:
                failed = {"edit_id": eid, "error": str(ex)}
                break
        return {"applied": applied, "failed": failed, "ok": failed is None}

    def rebuild_chain_after_reset(self, backend, keep: List[str],
                                  verify: bool = True) -> dict:
        """Full rollback: base weights, then only the kept deltas.

        The pre-rollback values of every tensor that differs from the base are
        journalled to disk first; if the rebuild fails midway, that journal is
        restored so the model is never left half-rolled-back, and the failure is
        reported loudly.
        """
        base_hashes = (self._data.get("base") or {}).get("hashes", {})
        j_names = [n for n in backend.param_names()
                   if base_hashes.get(n) and backend.tensor_hash(n) != base_hashes[n]]
        journal_path = self.root / "rollback_journal.npz"
        if j_names:
            np.savez_compressed(journal_path,
                                **{n.replace(".", "__"): backend.get_tensor(n)
                                   for n in j_names})
        reset = self.reset_to_base()
        if not reset.get("ok"):
            return {"ok": False, "stage": "reset", "reset": reset}
        rep = self.rebuild_chain(list(keep), verify=verify)
        if not rep["ok"]:
            recovered = None
            if journal_path.exists():
                try:
                    z = np.load(journal_path)
                    for k in z.files:
                        backend.set_tensor(k.replace("__", "."), z[k])
                    recovered = True
                except Exception as ex:
                    recovered = False
                    rep["recovery_error"] = str(ex)
            rep["recovered_pre_rollback_state"] = recovered
            rep["stage"] = "rebuild"
            return {"ok": False, **rep}
        if journal_path.exists():
            journal_path.unlink()
        return {"ok": True, "reset": reset, **rep}

    def diff(self, a: str, b: str) -> dict:
        if a not in self._data or b not in self._data:
            raise KeyError("unknown checkpoint name")
        ha, hb = self._data[a]["hashes"], self._data[b]["hashes"]
        changed = sorted(n for n in ha if hb.get(n) != ha[n])
        return {"a": a, "b": b, "n_changed": len(changed), "changed": changed,
                "metrics_a": self._data[a].get("metrics", {}),
                "metrics_b": self._data[b].get("metrics", {})}
