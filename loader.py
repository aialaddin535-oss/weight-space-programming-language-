"""Model loading, inspection helpers and the immutable baseline manifest."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..backends import build_backend
from ..config import ensure_dirs, path_of


def load_model(cfg: dict):
    """Build the configured backend and return (backend, describe dict)."""
    backend = build_backend(cfg)
    return backend, backend.describe()


def describe_model(backend) -> dict:
    d = backend.describe()
    d["tensor_count"] = len(backend.param_names())
    d["total_parameters"] = int(backend.n_params())
    return d


def tensor_table(backend, limit: int = 400) -> List[dict]:
    """Per-tensor shape/dtype/param-count/hash, sorted by name."""
    rows = []
    for name in backend.param_names():
        shp = backend.shape_of(name)
        rows.append({
            "name": name,
            "shape": list(shp),
            "dtype": backend.dtype_of(name),
            "n_params": int(np.prod(shp)) if shp else 1,
            "sha256": backend.tensor_hash(name),
        })
        if len(rows) >= limit:
            break
    return rows


def baseline_manifest_path(cfg: dict) -> Path:
    return path_of(cfg, "cache") / "baseline_manifest.json"


def write_or_check_baseline(backend, cfg: dict) -> dict:
    """Record every tensor hash once; later runs verify against it.

    This is the guard that makes 'never overwrite the original weights' real:
    any drift from the recorded baseline is detected and reported.
    """
    ensure_dirs(cfg)
    p = baseline_manifest_path(cfg)
    current = backend.snapshot_hashes()
    if p.exists():
        prev = json.load(open(p, "r", encoding="utf-8"))
        drift = [n for n, h in prev.get("hashes", {}).items()
                 if n in current and current[n] != h]
        missing = [n for n in prev.get("hashes", {}) if n not in current]
        return {"created": False, "path": str(p), "n_tensors": len(current),
                "drift": drift, "missing": missing,
                "pristine": (not drift and not missing)}
    payload = {
        "created_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "backend": backend.kind,
        "describe": backend.describe(),
        "hashes": current,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    return {"created": True, "path": str(p), "n_tensors": len(current),
            "drift": [], "missing": [], "pristine": True}


def verify_against_baseline(backend, cfg: dict) -> dict:
    p = baseline_manifest_path(cfg)
    if not p.exists():
        return {"ok": False, "reason": "no baseline manifest recorded yet"}
    prev = json.load(open(p, "r", encoding="utf-8"))["hashes"]
    cur = backend.snapshot_hashes()
    changed = sorted(n for n, h in prev.items() if cur.get(n) != h)
    return {"ok": not changed, "changed_tensors": changed, "n_tensors": len(cur)}


# --------------------------------------------------------------------- download
def download_model(model_id: str, dest: str, revision: str = "") -> dict:
    """Documented, resumable download of the genuine weights (needs network+disk)."""
    os.makedirs(dest, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        raise SystemExit(
            "huggingface_hub not installed. Run:\n"
            "  pip install huggingface_hub\n"
            "  huggingface-cli login   # if the repo is gated"
        )
    kw = dict(repo_id=model_id, local_dir=dest, resume_download=True)
    if revision:
        kw["revision"] = revision
    path = snapshot_download(**kw)
    return {"model_id": model_id, "path": path, "revision": revision or "main"}
