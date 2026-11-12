"""Tensor inspection: shapes, dtypes, norms, sparsity, column/row statistics.

Everything here only READS tensors (float32 copies) plus the exact stored bytes
for hashing. No in-place mutation happens in this module.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def tensor_stats(backend, name: str, top_k: int = 8) -> dict:
    w = backend.get_tensor(name)
    flat = w.reshape(-1)
    n = int(flat.size)
    nz = int(np.count_nonzero(flat))
    st = {
        "name": name,
        "shape": list(w.shape),
        "ndim": int(w.ndim),
        "dtype": backend.dtype_of(name),
        "n_entries": n,
        "nonzero": nz,
        "zero_frac": float(1.0 - nz / n) if n else 1.0,
        "mean": float(flat.mean()) if n else 0.0,
        "std": float(flat.std()) if n else 0.0,
        "abs_mean": float(np.abs(flat).mean()) if n else 0.0,
        "abs_max": float(np.abs(flat).max()) if n else 0.0,
        "fro_norm": float(np.linalg.norm(flat)),
        "sha256": backend.tensor_hash(name),
    }
    if w.ndim == 2:
        col = np.linalg.norm(w, axis=0)
        row = np.linalg.norm(w, axis=1)
        st.update({
            "rows": int(w.shape[0]),
            "cols": int(w.shape[1]),
            "col_norm_mean": float(col.mean()),
            "col_norm_max": float(col.max()),
            "row_norm_mean": float(row.mean()),
            "row_norm_max": float(row.max()),
            "top_cols": [int(i) for i in np.argsort(-col)[:top_k]],
            "top_col_norms": [float(col[i]) for i in np.argsort(-col)[:top_k]],
            "top_rows": [int(i) for i in np.argsort(-row)[:top_k]],
            "top_row_norms": [float(row[i]) for i in np.argsort(-row)[:top_k]],
            # effective rank via singular values (cost limited on huge tensors)
            "svd_rank_99": _rank99(w),
        })
    return st


def _rank99(w: np.ndarray) -> int | None:
    if w.size > 8_000_000:
        return None                      # skip: too big for a cheap probe
    try:
        s = np.linalg.svd(w, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    if s.size == 0 or s.sum() == 0:
        return 0
    c = np.cumsum(s) / s.sum()
    return int(np.searchsorted(c, 0.99) + 1)


def list_tensors(backend, pattern: str = "", limit: int = 200) -> List[dict]:
    out = []
    for n in backend.param_names():
        if pattern and pattern not in n:
            continue
        shp = backend.shape_of(n)
        out.append({"name": n, "shape": list(shp), "dtype": backend.dtype_of(n),
                    "n_entries": int(np.prod(shp)) if shp else 1})
        if len(out) >= limit:
            break
    return out


def suggest_editable(backend, cfg: dict, prefer: str = "mlp.down_proj",
                     max_layers: int = 6) -> List[dict]:
    """Rank candidate tensors for editing: 2-D, not frozen, moderate size.

    This is a *search-space reducer*, not a decision: the agent still has to pick
    and justify one, and every edit is gated by shape/dtype/hash/bounds checks.
    """
    frozen = cfg["editing"]["frozen_suffixes"]
    max_entries = int(cfg["editing"].get("max_entries_per_tensor", 2048)) * 64
    rows = []
    for n in backend.param_names():
        if backend.frozen(n, frozen):
            continue
        shp = backend.shape_of(n)
        if len(shp) != 2:
            continue
        n_ent = int(np.prod(shp))
        if n_ent > 40_000_000:
            continue
        score = 0.0
        if prefer and prefer in n:
            score += 3.0
        if "mlp" in n:
            score += 1.0
        if "proj" in n:
            score += 1.0
        if "layernorm" in n or "norm" in n and len(shp) == 1:
            score -= 2.0
        rows.append({"name": n, "shape": list(shp), "n_entries": n_ent,
                     "n_entries_ok": bool(n_ent <= max_entries), "score": score})
    rows.sort(key=lambda r: (-r["score"], r["n_entries"]))
    # keep at most max_layers distinct layer indices
    seen, keep = set(), []
    for r in rows:
        layer = _layer_of(r["name"])
        if layer is not None:
            if layer in seen:
                continue
            seen.add(layer)
        keep.append(r)
        if len(keep) >= max_layers:
            break
    return keep


def _layer_of(name: str) -> int | None:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def diff_tensors(a: np.ndarray, b: np.ndarray, limit: int = 10) -> dict:
    """Compare two same-shaped tensors, reporting what changed and how much."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    d = b - a
    mask = d != 0
    nb = np.linalg.norm(a)
    return {
        "shape": list(a.shape),
        "changed_entries": int(mask.sum()),
        "changed_frac": float(mask.mean()) if mask.size else 0.0,
        "changed_rows": int(mask.any(axis=1).sum()) if a.ndim == 2 else None,
        "changed_cols": int(mask.any(axis=0).sum()) if a.ndim == 2 else None,
        "delta_fro_norm": float(np.linalg.norm(d)),
        "relative_change": float(np.linalg.norm(d) / nb) if nb else 0.0,
        "max_abs_delta": float(np.abs(d).max()) if d.size else 0.0,
        "mean_abs_delta": float(np.abs(d).mean()) if d.size else 0.0,
        "examples": [
            {"index": [int(x) for x in np.unravel_index(i, a.shape)],
             "before": float(a.reshape(-1)[i]), "after": float(b.reshape(-1)[i])}
            for i in np.flatnonzero(mask.reshape(-1))[:limit]
        ],
    }


def summarize_hash_diff(before: Dict[str, str], after: Dict[str, str]) -> dict:
    changed = sorted(n for n in before if after.get(n) != before[n])
    return {"n_tensors": len(after), "changed": changed, "n_changed": len(changed)}
