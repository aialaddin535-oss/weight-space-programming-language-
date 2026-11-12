"""Sparse, hash-verified delta store (format mirrors the reference experiment).

For each edited tensor we store only the entries whose *stored bit pattern*
changed:

    t<i>_idx   int64   flat indices into the tensor
    t<i>_val   uint16  new bit patterns (bfloat16)  |  float32 (float32 models)

plus `manifest.json` with shape, dtype, changed-entry counts, relative change and
the sha256 of both the base and the edited tensor. Rebuilding a tensor is
`base_bits.flat[idx] = val`, which is exact - no accumulation, no drift - and is
verified against `edited_sha256` before it is ever written into a model.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .cast import bits_to_values, cast_to_dtype, to_dtype_bytes


def sha256_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def build_record(name: str, dtype: str, base_f32: np.ndarray, new_f32: np.ndarray,
                 reason: str = "") -> Tuple[dict, np.ndarray, np.ndarray]:
    """Compare base and candidate values in storage precision -> (record, idx, val)."""
    if base_f32.shape != new_f32.shape:
        raise ValueError(f"{name}: shape mismatch {base_f32.shape} vs {new_f32.shape}")
    base_bits = cast_to_dtype(base_f32, dtype)
    new_bits = cast_to_dtype(new_f32, dtype)
    if base_bits.dtype != new_bits.dtype:
        raise ValueError(f"{name}: dtype mismatch {base_bits.dtype} vs {new_bits.dtype}")
    flat_b, flat_n = base_bits.reshape(-1), new_bits.reshape(-1)
    mask = flat_b != flat_n
    idx = np.flatnonzero(mask).astype(np.int64)
    val = flat_n[idx].copy()
    rec = {
        "name": name,
        "shape": [int(x) for x in base_f32.shape],
        "dtype": dtype,
        "form": "sparse",
        "changed_entries": int(idx.size),
        "changed_entries_total": int(flat_b.size),
        "changed_fraction": float(idx.size / flat_b.size) if flat_b.size else 0.0,
        "relative_change": float(np.linalg.norm((new_f32 - base_f32).astype(np.float64)) /
                                 (np.linalg.norm(base_f32.astype(np.float64)) + 1e-30)),
        "base_sha256": sha256_arr(base_bits),
        "edited_sha256": sha256_arr(new_bits),
        "reason": reason,
    }
    if base_f32.ndim == 2:
        m2 = mask.reshape(base_f32.shape)
        rec["changed_rows"] = int(m2.any(axis=1).sum())
        rec["changed_columns"] = int(m2.any(axis=0).sum())
    return rec, idx, val


def reconstruct(record: dict, idx: np.ndarray, val: np.ndarray, base_f32: np.ndarray) -> np.ndarray:
    """base + sparse patch -> float32 values of the edited tensor (exact)."""
    dtype = record["dtype"]
    base_bits = cast_to_dtype(base_f32, dtype)
    if sha256_arr(base_bits) != record["base_sha256"]:
        raise ValueError(
            f"{record['name']}: base tensor sha256 does not match the recorded base "
            f"({sha256_arr(base_bits)[:12]} != {record['base_sha256'][:12]}); "
            "the edit was made on different weights"
        )
    out = base_bits.copy().reshape(-1)
    out[idx] = val
    shaped = out.reshape(tuple(record["shape"]))
    if sha256_arr(shaped) != record["edited_sha256"]:
        raise ValueError(f"{record['name']}: reconstructed bits do not match edited_sha256")
    return bits_to_values(shaped, dtype).astype(np.float32)


class DeltaStore:
    """Writes/reads `edit.npz` + `manifest.json` directories."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def new_edit_id(self, tag: str = "e") -> str:
        return f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"

    def dir_of(self, edit_id: str) -> Path:
        return self.root / edit_id

    def save(self, edit_id: str, tensors: List[Tuple[dict, np.ndarray, np.ndarray]],
             meta: dict) -> dict:
        """tensors: [(record, idx, val)] - written as one npz + manifest."""
        d = self.dir_of(edit_id)
        d.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        recs = []
        for i, (rec, idx, val) in enumerate(tensors):
            arrays[f"t{i}_idx"] = np.asarray(idx, dtype=np.int64)
            arrays[f"t{i}_val"] = np.asarray(val)
            recs.append(rec)
        np.savez_compressed(d / "edit.npz", **arrays)
        man = {
            "edit_id": edit_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": meta.get("status", "applied"),
            "status_history": [{"status": meta.get("status", "applied"),
                                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}],
            "tensors": recs,
            "n_tensors": len(recs),
            "changed_entries_total": int(sum(r["changed_entries"] for r in recs)),
            **meta,
        }
        with open(d / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(man, f, indent=1, ensure_ascii=False)
        return man

    def load(self, edit_id: str) -> Tuple[dict, Dict[str, np.ndarray]]:
        d = self.dir_of(edit_id)
        man = json.load(open(d / "manifest.json", "r", encoding="utf-8"))
        z = np.load(d / "edit.npz")
        return man, {k: z[k] for k in z.files}

    def has(self, edit_id: str) -> bool:
        return (self.dir_of(edit_id) / "manifest.json").exists()

    def list_edits(self) -> List[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            f = d / "manifest.json"
            if f.is_file():
                try:
                    m = json.load(open(f, "r", encoding="utf-8"))
                except Exception:
                    continue
                out.append({
                    "edit_id": m.get("edit_id", d.name),
                    "created_utc": m.get("created_utc"),
                    "n_tensors": m.get("n_tensors"),
                    "changed_entries": m.get("changed_entries_total"),
                    "status": m.get("status", "unknown"),
                    "reason": (m.get("reason") or "")[:160],
                    "tensors": [t["name"] for t in m.get("tensors", [])],
                    "relative_change": max([t.get("relative_change", 0.0)
                                            for t in m.get("tensors", [])] or [0.0]),
                })
        return out

    def update_status(self, edit_id: str, status: str, extra: dict | None = None) -> dict:
        d = self.dir_of(edit_id)
        man = json.load(open(d / "manifest.json", "r", encoding="utf-8"))
        man["status"] = status
        man.setdefault("status_history", []).append(
            {"status": status, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if extra:
            man.update(extra)
        with open(d / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(man, f, indent=1, ensure_ascii=False)
        return man

    # -------------------------------------------------------------- rebuild
    def rebuild_tensors(self, edit_id: str, backend, verify: bool = True) -> List[str]:
        """Write an edit's tensors into the live backend (base must match)."""
        man, arrays = self.load(edit_id)
        names = []
        for i, rec in enumerate(man["tensors"]):
            name = rec["name"]
            if backend.shape_of(name) != tuple(rec["shape"]):
                raise ValueError(f"{name}: live shape {backend.shape_of(name)} != recorded {rec['shape']}")
            if backend.dtype_of(name).lower() != rec["dtype"].lower():
                raise ValueError(f"{name}: live dtype {backend.dtype_of(name)} != recorded {rec['dtype']}")
            base = backend.get_tensor(name)
            new = reconstruct(rec, arrays[f"t{i}_idx"], arrays[f"t{i}_val"], base)
            backend.set_tensor(name, new)
            if verify:
                h = backend.tensor_hash(name)
                if h != rec["edited_sha256"]:
                    raise ValueError(
                        f"{name}: after writing, tensor hash {h[:12]} != expected "
                        f"{rec['edited_sha256'][:12]}")
            names.append(name)
        return names
