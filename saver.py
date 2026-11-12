"""Writing edited checkpoints without touching the original model directory.

Two output modes:
  * delta mode (always): a sparse .npz + manifest holding only changed entries.
    Reconstruct any checkpoint later as base + delta, hash-verified.
  * materialised mode (optional, needs disk): a copy of the HF config/tokenizer
    plus edited tensors written to .npy files that `apply_delta` can load; used
    when you want to hand the checkpoint straight to transformers/vLLM.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


def materialise(backend, out_dir: str | Path, names=None) -> dict:
    """Write edited tensors as .npy (float32) + a pointer json. Never overwrites base."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = list(names or backend.param_names())
    written = []
    for n in names:
        arr = backend.get_tensor(n)          # float32 copy
        f = out / (n.replace(".", "__") + ".npy")
        np.save(f, arr.astype(np.float32))
        written.append({"name": n, "file": f.name, "shape": list(arr.shape)})
    ptr = {
        "backend": backend.kind,
        "describe": backend.describe(),
        "base_model_path": backend.cfg["backend"].get("model_path", ""),
        "tensors": written,
        "note": "float32 .npy files; cast to the model dtype when loading",
    }
    with open(out / "checkpoint.json", "w", encoding="utf-8") as fh:
        json.dump(ptr, fh, indent=1)
    return {"out": str(out), "n_tensors": len(written)}


def copy_model_shell(src_model_dir: str, out_dir: str | Path) -> dict:
    """Copy only config/tokenizer files (not the weights) for a loadable checkpoint."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(src_model_dir)
    if not src.exists():
        return {"copied": [], "reason": "source model dir not found (tiny backend?)"}
    pats = ("*.json", "*.jinja", "*.txt", "*.model", "merges.txt", "vocab.json")
    copied = []
    for pat in pats:
        for f in src.glob(pat):
            if f.stat().st_size > 40_000_000:
                continue
            shutil.copy2(f, out / f.name)
            copied.append(f.name)
    return {"copied": sorted(set(copied)), "out": str(out)}
