"""Bounded, declarative tensor edit operations.

The agent may only emit these four op kinds. Each is a pure function
(values, op) -> new values, so a candidate can be previewed, measured and thrown
away without ever mutating the model. Each op is plain arithmetic on one weight array:
no objective function, no derivative, no solver, no iterative fitting.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

OP_KINDS = ("scale_columns", "shift_columns", "scale_rows", "scale_block")


class OpError(ValueError):
    pass


def _cols(op: dict, n_cols: int) -> np.ndarray:
    cols = op.get("columns")
    if not isinstance(cols, Iterable) or isinstance(cols, (str, bytes)):
        raise OpError("columns must be a list of column indices")
    idx = np.array(sorted({int(c) for c in cols}), dtype=np.int64)
    if idx.size == 0:
        raise OpError("columns is empty")
    if idx.min() < 0 or idx.max() >= n_cols:
        raise OpError(f"column index out of range 0..{n_cols - 1}")
    return idx


def _rows(op: dict, n_rows: int) -> np.ndarray:
    rows = op.get("rows")
    if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes)):
        raise OpError("rows must be a list of row indices")
    idx = np.array(sorted({int(r) for r in rows}), dtype=np.int64)
    if idx.size == 0:
        raise OpError("rows is empty")
    if idx.min() < 0 or idx.max() >= n_rows:
        raise OpError(f"row index out of range 0..{n_rows - 1}")
    return idx


def validate(op: dict, shape: tuple) -> List[str]:
    """Structural validation independent of the model. Returns error strings."""
    errs: List[str] = []
    if not isinstance(op, dict):
        return ["operation must be a JSON object"]
    kind = op.get("kind") or op.get("op")
    if kind not in OP_KINDS:
        return [f"kind must be one of {OP_KINDS}, got {kind!r}"]
    name = op.get("tensor") or op.get("name")
    if not name or not isinstance(name, str):
        errs.append("tensor name is required")
    alpha = op.get("alpha")
    if not isinstance(alpha, (int, float)) or not np.isfinite(alpha):
        errs.append("alpha must be a finite number")
    if len(shape) != 2:
        errs.append(f"only 2-D tensors can be edited, got shape {shape}")
        return errs
    try:
        if kind in ("scale_columns", "shift_columns"):
            _cols(op, shape[1])
        elif kind == "scale_rows":
            _rows(op, shape[0])
        elif kind == "scale_block":
            _cols(op, shape[1])
            _rows(op, shape[0])
            if "beta" in op and not isinstance(op["beta"], (int, float)):
                errs.append("beta must be a number when given")
    except OpError as e:
        errs.append(str(e))
    return errs


def apply(values: np.ndarray, op: dict) -> np.ndarray:
    """Return a NEW array with the op applied (input untouched)."""
    kind = op.get("kind") or op.get("op")
    w = np.array(values, dtype=np.float32, copy=True)
    if w.ndim != 2:
        raise OpError("only 2-D tensors supported")
    a = float(op["alpha"])
    if kind == "scale_columns":
        c = _cols(op, w.shape[1])
        w[:, c] *= (1.0 + a)
    elif kind == "shift_columns":
        c = _cols(op, w.shape[1])
        sd = float(np.std(w)) or 1.0
        w[:, c] += a * sd
    elif kind == "scale_rows":
        r = _rows(op, w.shape[0])
        w[r, :] *= (1.0 + a)
    elif kind == "scale_block":
        r, c = _rows(op, w.shape[0]), _cols(op, w.shape[1])
        w[np.ix_(r, c)] *= (1.0 + a)
        if "beta" in op:
            sd = float(np.std(w)) or 1.0
            w[np.ix_(r, c)] += float(op["beta"]) * sd
    else:
        raise OpError(f"unknown kind {kind!r}")
    return w


def describe(op: dict) -> str:
    kind = op.get("kind") or op.get("op")
    name = op.get("tensor") or op.get("name")
    a = op.get("alpha")
    if kind in ("scale_columns",):
        return f"scale {len(op.get('columns', []))} columns of {name} by (1{+a:+.4f})"
    if kind == "shift_columns":
        return f"shift {len(op.get('columns', []))} columns of {name} by {a:+.4f}*std"
    if kind == "scale_rows":
        return f"scale {len(op.get('rows', []))} rows of {name} by (1{+a:+.4f})"
    return f"scale block of {name} by (1{+a:+.4f})"
