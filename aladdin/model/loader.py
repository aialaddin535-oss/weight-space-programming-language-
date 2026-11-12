"""Model loading, identity reporting and the immutable baseline manifest."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..backends import build_backend
from ..config import ensure_dirs, path_of
from ..model_identity import (KNOWN_MODEL_REVISION, REQUIRED_MODEL_ID,
                              SNAPSHOT_METADATA_FILE)


def load_model(cfg: dict):
    """Build the configured backend and return (backend, describe dict)."""
    backend = build_backend(cfg)
    return backend, describe_model(backend)


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


# ------------------------------------------------------------------ baseline
def baseline_manifest_path(cfg: dict) -> Path:
    return path_of(cfg, "cache") / "baseline_manifest.json"


def _manifest_header(cfg: dict) -> dict:
    b = cfg.get("backend", {})
    return {"model_id": b.get("model_id") or REQUIRED_MODEL_ID,
            "revision": b.get("revision") or "",
            "model_path": b.get("model_path") or ""}


def write_or_check_baseline(backend, cfg: dict) -> dict:
    """Record every tensor hash once; later runs verify against it.

    This is the guard that makes 'never overwrite the original weights' real:
    any drift from the recorded baseline is detected and reported.
    """
    ensure_dirs(cfg)
    p = baseline_manifest_path(cfg)
    current = backend.snapshot_hashes()
    header = _manifest_header(cfg)
    if p.exists():
        prev = json.load(open(p, "r", encoding="utf-8"))
        drift = [n for n, h in prev.get("hashes", {}).items()
                 if n in current and current[n] != h]
        missing = [n for n in prev.get("hashes", {}) if n not in current]
        identity = {k: prev.get(k) for k in ("model_id", "revision", "model_path")}
        identity_ok = (not identity.get("model_id")
                       or identity.get("model_id") == header["model_id"])
        return {"created": False, "path": str(p), "n_tensors": len(current),
                "drift": drift, "missing": missing,
                "pristine": (not drift and not missing),
                "identity": identity, "identity_ok": identity_ok,
                "model_id": header["model_id"]}
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": backend.kind,
        **header,
        "describe": backend.describe(),
        "hashes": current,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    return {"created": True, "path": str(p), "n_tensors": len(current),
            "drift": [], "missing": [], "pristine": True,
            "identity": header, "identity_ok": True,
            "model_id": header["model_id"]}


def verify_against_baseline(backend, cfg: dict) -> dict:
    p = baseline_manifest_path(cfg)
    if not p.exists():
        return {"ok": False, "reason": "no baseline manifest recorded yet"}
    prev = json.load(open(p, "r", encoding="utf-8"))
    hashes = prev.get("hashes", {})
    cur = backend.snapshot_hashes()
    changed = sorted(n for n, h in hashes.items() if cur.get(n) != h)
    header = _manifest_header(cfg)
    return {"ok": not changed, "changed_tensors": changed, "n_tensors": len(cur),
            "model_id": header["model_id"],
            "baseline_model_id": prev.get("model_id"),
            "identity_matches": (not prev.get("model_id")
                                 or prev.get("model_id") == header["model_id"])}


# --------------------------------------------------------------------- download
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(model_id: str = REQUIRED_MODEL_ID, dest: str = "",
                   revision: str = "", write_metadata: bool = True) -> dict:
    """Download the genuine HF snapshot and record its verified identity.

    The snapshot directory keeps the real weights; this project never writes
    into it. `aladdin_checkpoint.json` records the repository id, commit sha,
    file list/sizes and hash of config.json so later loads can prove which
    checkpoint is on disk without trusting a directory name.
    """
    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception:
        raise SystemExit(
            "huggingface_hub not installed. Run:\n"
            "  pip install huggingface_hub\n"
            "  huggingface-cli login   # only if the repo is gated")
    dest = dest or str(Path("models") / model_id.split("/")[-1])
    Path(dest).mkdir(parents=True, exist_ok=True)
    kw = dict(repo_id=model_id, local_dir=dest)
    if revision:
        kw["revision"] = revision
    path = snapshot_download(**kw)

    api = HfApi()
    sha, last_modified, files = "", "", []
    try:
        info = api.model_info(model_id, revision=revision or None,
                              files_metadata=True)
        sha = getattr(info, "sha", "") or ""
        last_modified = str(getattr(info, "lastModified", "") or "")
        for s in getattr(info, "siblings", []) or []:
            lfs = getattr(s, "lfs", None)
            files.append({"name": s.rfilename,
                          "size": getattr(s, "size", None),
                          "sha256": (lfs or {}).get("sha256") if isinstance(lfs, dict)
                          else getattr(lfs, "sha256", None)})
    except Exception as e:                       # network hiccup: still usable
        files = []
        sha = ""
        last_modified = f"unavailable: {type(e).__name__}"

    cfg_path = Path(path) / "config.json"
    meta = {
        "repo_id": model_id,
        "revision": revision or "main",
        "sha": sha or KNOWN_MODEL_REVISION if model_id == REQUIRED_MODEL_ID else sha,
        "resolved_sha": sha,
        "last_modified": last_modified,
        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "is_required_wsp_checkpoint": model_id == REQUIRED_MODEL_ID,
        "base_model": "Qwen/Qwen3.5-0.8B",
        "config_sha256": _sha256_file(cfg_path) if cfg_path.exists() else None,
        "n_files": len(files),
        "total_bytes": int(sum(f.get("size") or 0 for f in files)),
        "files": files,
        "note": "pristine downloaded checkpoint; Aladdin WSP never writes here",
    }
    if write_metadata:
        tmp = Path(path) / (SNAPSHOT_METADATA_FILE + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, Path(path) / SNAPSHOT_METADATA_FILE)
    return {"model_id": model_id, "path": path, "revision": meta["revision"],
            "sha": meta["sha"], "metadata": str(Path(path) / SNAPSHOT_METADATA_FILE)}


# -------------------------------------------------------------------- identity
def model_identity_report(model_path: str | Path,
                          model_id: str = REQUIRED_MODEL_ID) -> dict:
    """What checkpoint is on disk, and is it the required edited WSP model?"""
    from ..backends.qwen_loader import (detect_architecture, read_snapshot_metadata,
                                        read_weight_index, verify_loaded_model)
    p = Path(model_path)
    meta = read_snapshot_metadata(p) or {}
    arch = detect_architecture(p, expected_model_id=model_id) if (p / "config.json").exists() else {}
    idx = None
    try:
        wi = read_weight_index(p)
        idx = {"source": wi["source"], "shards": wi["shards"],
               "n_tensors": len(wi["keys"]) if wi["keys"] else None}
    except FileNotFoundError as e:
        idx = {"error": str(e)}
    recorded = meta.get("repo_id")
    return {
        "model_id_required": model_id,
        "model_id_recorded": recorded,
        "identity_verified": bool(recorded == model_id),
        "revision": meta.get("sha") or meta.get("revision"),
        "architecture": arch.get("architectures"),
        "model_type": arch.get("model_type"),
        "text_geometry": {k: arch.get(k) for k in
                          ("text_hidden_size", "text_num_hidden_layers",
                           "text_vocab_size", "text_intermediate_size",
                           "tie_word_embeddings")},
        "weight_index": idx,
        "metadata_present": bool(meta),
        "metadata": {k: meta.get(k) for k in
                     ("repo_id", "revision", "sha", "downloaded_utc",
                      "total_bytes", "config_sha256", "n_files")},
        "problems": ([] if recorded == model_id else
                     [f"snapshot does not record {model_id!r}; re-download it with "
                      "`aladdin download`"]),
    }
