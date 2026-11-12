"""Activation inspection: which channels/rows fire on Hindi vs English inputs.

This is where an evidence-based hypothesis comes from. For a given prompt we
capture per-layer activations (hidden states, attention output, MLP output) and
compare aggregate statistics between language groups. The result is a ranked
list of (layer, channel) pairs - the same quantity the reference experiment used
to decide *where* in the network to act.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def capture(backend, prompt: str) -> dict:
    ids = backend.encode(prompt)
    cap: dict = {}
    backend.forward_logits(ids, capture=cap)
    return {
        "n_tokens": len(ids),
        "tensors": {k: {"shape": list(v.shape), "abs_mean": float(np.abs(v).mean()),
                        "abs_max": float(np.abs(v).max())} for k, v in cap.items()},
        "_raw": cap,
    }


def mean_abs_by_channel(activation: np.ndarray) -> np.ndarray:
    """[T, D] -> [D] mean |x| over positions."""
    a = np.asarray(activation, dtype=np.float64)
    if a.ndim != 2:
        a = a.reshape(-1, a.shape[-1])
    return np.abs(a).mean(axis=0)


def group_contrast(backend, group_a: Sequence[str], group_b: Sequence[str],
                   layer_keys: Sequence[str] | None = None, top_k: int = 12) -> dict:
    """Mean |activation| per channel for group A minus group B, per captured tensor.

    group_a is typically the Hindi/Hinglish prompts, group_b the English control.
    A large positive contrast marks channels the model dedicates to the Hindi
    side of the distribution - the natural target for a bounded gain edit.
    """
    acc: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for tag, prompts in (("a", group_a), ("b", group_b)):
        for p in prompts:
            cap = capture(backend, p)["_raw"]
            for k, v in cap.items():
                if layer_keys and k not in layer_keys:
                    continue
                if v.ndim < 2:
                    continue
                acc.setdefault(k, {"a": [], "b": []})[tag].append(mean_abs_by_channel(v))
    out = {}
    for k, d in acc.items():
        if not d["a"] or not d["b"]:
            continue
        a = np.mean(np.stack(d["a"]), axis=0)
        b = np.mean(np.stack(d["b"]), axis=0)
        contrast = a - b
        order = np.argsort(-contrast)[:top_k]
        out[k] = {
            "dim": int(a.shape[0]),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "top_channels": [int(i) for i in order],
            "top_contrast": [float(contrast[i]) for i in order],
            "ratio": float((a.mean() + 1e-9) / (b.mean() + 1e-9)),
        }
    # global ranking of layers by how much more they fire on group A
    ranked = sorted(out.items(), key=lambda kv: -kv[1]["ratio"])
    return {"per_tensor": out, "ranked": [k for k, _ in ranked[:10]]}


def logprob_of(backend, prompt: str, continuation: str) -> float:
    """Teacher-forced mean log-prob of a continuation under the model.

    Used as a cheap, deterministic signal (e.g. 'does the model find a Devanagari
    answer likely given a Devanagari question?'). Not an evaluation score.
    """
    p_ids = backend.encode(prompt)
    c_ids = backend.encode(continuation, add_special_tokens=False)
    if not c_ids:
        return float("nan")
    ids = p_ids + c_ids
    logits = backend.forward_logits(ids)
    logits = logits[len(p_ids) - 1: len(ids) - 1].astype(np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    logz = np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    logp = logits - logz
    picked = logp[np.arange(len(c_ids)), np.asarray(c_ids, dtype=int)]
    return float(picked.mean())
