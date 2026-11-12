"""REAL Qwen3.5-0.8B integration tests.

These tests run ONLY when the actual model weights are available on this
machine (local HF snapshot). Without the weights they are SKIPPED — a skip is
reported as "not run", never as a pass. They are clearly separated from the
mock/infrastructure tests in test_v3_revert_and_loader.py.

Enable by downloading the model first:

    bash scripts/download_model.sh            # or:
    python -m aladdin.cli download --model Qwen/Qwen3.5-0.8B --dest models/Qwen3.5-0.8B
    export ALADDIN_MODEL_PATH="$PWD/models/Qwen3.5-0.8B"

GPU variants additionally require a CUDA device; they report honestly when no
GPU exists. The pristine checkpoint is NEVER modified: edits are applied to a
disposable in-memory model instance and verified byte-exact afterwards.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from aladdin.backends.qwen_loader import (detect_architecture,
                                          check_transformers_version,
                                          _auto_class_for)

MODEL_DIR = Path(os.environ.get("ALADDIN_MODEL_PATH", ""))


def _model_available() -> bool:
    if not MODEL_DIR or not MODEL_DIR.exists():
        return False
    return (MODEL_DIR / "config.json").exists()


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _model_available(),
    reason="real Qwen3.5-0.8B weights not found (set ALADDIN_MODEL_PATH); "
           "NOT RUN, not passed")


# ------------------------------------------------------------------ loader
def test_real_architecture_detection_matches_official_config():
    a = detect_architecture(MODEL_DIR)
    assert a["is_qwen3_5"] is True
    assert a["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert a["text_model_type"] == "qwen3_5_text"
    assert a["is_multimodal"] is True
    cls, _ = _auto_class_for(a)
    assert cls == "AutoModelForImageTextToText"


def test_real_backend_loads_official_class():
    from aladdin.backends.hf_torch import HFTorchBackend
    cfg = {"backend": {"kind": "hf_torch", "model_path": str(MODEL_DIR),
                       "dtype": "bfloat16", "device": "cuda" if _gpu_available() else "cpu"}}
    be = HFTorchBackend(cfg)
    d = be.describe()
    assert d["auto_class"] == "AutoModelForImageTextToText"
    assert "ConditionalGeneration" in d["loaded_class"]
    assert d["text_model_type"] == "qwen3_5_text"
    assert d["n_tensors"] > 0


# ------------------------------------------------------- real tensor engine
@pytest.fixture(scope="module")
def real_backend():
    from aladdin.backends.hf_torch import HFTorchBackend
    cfg = {"backend": {"kind": "hf_torch", "model_path": str(MODEL_DIR),
                       "dtype": "bfloat16", "device": "cuda" if _gpu_available() else "cpu"}}
    return HFTorchBackend(cfg)


def _edit_target(backend):
    """A real, editable 2-D parameter inside the size bounds."""
    from aladdin.inspection.tensors import suggest_editable
    cfg = {"editing": {"frozen_suffixes": ["model.norm.weight",
                                           "model.embed_tokens.weight",
                                           "lm_head.weight"]}}
    rows = suggest_editable(backend, cfg)
    assert rows, "no editable tensor found in the real model"
    return rows[0]["name"]


def test_real_tensor_read_hash_write_revert(real_backend):
    """The full WSP loop on the REAL model: read -> hash -> edit -> verify ->
    revert -> byte-exact original. Uses a disposable instance (this backend is
    freshly loaded; the on-disk checkpoint is never written to)."""
    name = _edit_target(real_backend)
    original = real_backend.get_tensor(name)
    h0 = real_backend.tensor_hash(name)                     # exact original hash
    assert h0 == hashlib.sha256(real_backend.raw_bytes(name)).hexdigest()

    # small reversible edit (5% scale on one column)
    edited = original.copy()
    edited[:, 0] = edited[:, 0] * 1.05
    assert not np.array_equal(edited, original)
    real_backend.set_tensor(name, edited)
    assert real_backend.tensor_hash(name) != h0, "edit must change the tensor"

    # revert to the exact original bits
    real_backend.set_tensor(name, original)
    assert real_backend.tensor_hash(name) == h0, \
        "reverted tensor must be byte-identical to the original"
    assert np.array_equal(real_backend.get_tensor(name), original)


def test_real_full_agent_revert_cycle(real_backend, tmp_path):
    """apply_edit -> revert through ToolRuntime on the REAL model, verifying
    the undone/remaining contract with real tensors."""
    from aladdin.agent.tools import ToolRuntime
    from aladdin.sparse_deltas.store import DeltaStore
    from aladdin.tensor_editor import TensorEditor
    from aladdin.forward_pass import ForwardPassEngine

    name = _edit_target(real_backend)
    # shrink bounds so the editor's gate accepts a tiny probe edit on a huge tensor
    cfg = {"editing": {"max_tensors_per_edit": 2, "max_entries_per_tensor": 200000,
                       "max_relative_change": 0.25, "min_relative_change": 1e-9,
                       "frozen_suffixes": ["model.norm.weight",
                                           "model.embed_tokens.weight",
                                           "lm_head.weight"],
                       "refuse_uniform_rescale": True},
           "agent": {"model": "real-test"}}
    original = real_backend.get_tensor(name)
    h0 = real_backend.tensor_hash(name)
    store = DeltaStore(tmp_path / "deltas")
    ed = TensorEditor(real_backend, cfg, store)
    fwd = ForwardPassEngine(real_backend, budget=4, run_dir=tmp_path)
    rt = ToolRuntime(real_backend, cfg, ed, fwd, dataset=None, workdir=tmp_path)

    n_cols = real_backend.shape_of(name)[1]
    cols = [0, n_cols // 2]
    a = rt.call("apply_edit", {"operation": {
        "kind": "scale_columns", "tensor": name, "columns": cols, "alpha": 0.02},
        "reason": "real-model probe"})
    assert a["ok"], a
    assert real_backend.tensor_hash(name) == a["edited_sha256"]

    rev = rt.call("revert", {"n": 1})
    assert rev["ok"] and rev["verified"] is True, rev
    assert rev["undone"] == [a["edit_id"]]
    assert rev["remaining"] == []
    assert real_backend.tensor_hash(name) == h0, \
        "the REAL tensor must be restored byte-exactly"
    assert np.array_equal(real_backend.get_tensor(name), original)


def test_real_text_only_forward_pass(real_backend):
    """Text-only inference on the REAL model (no images, no video)."""
    ids = real_backend.encode("Hello")
    assert len(ids) > 0
    logits = real_backend.forward_logits(ids[:16])
    assert logits.ndim == 2
    assert logits.shape[1] == 248320, \
        f"vocab mismatch: real text_config says 248320, got {logits.shape[1]}"
    out = real_backend.generate("What is 2+2?", max_new_tokens=8)
    assert isinstance(out, str)


def test_real_gpu_roundtrip(real_backend):
    """GPU variants of load / inspect / edit / rollback (skipped honestly when
    no CUDA device exists)."""
    if not _gpu_available():
        pytest.skip("no CUDA device on this machine; NOT RUN")
    import torch
    name = _edit_target(real_backend)
    # parameters really live on the GPU
    dev = str(next(real_backend.model.parameters()).device)
    assert dev.startswith("cuda"), f"model is on {dev}, expected cuda"
    h0 = real_backend.tensor_hash(name)
    orig = real_backend.get_tensor(name)
    edited = orig.copy(); edited[:, 0] *= 1.05
    real_backend.set_tensor(name, edited)
    assert real_backend.tensor_hash(name) != h0
    real_backend.set_tensor(name, orig)
    assert real_backend.tensor_hash(name) == h0
    # reasonable memory usage: weights fit in under 8 GB
    free, total = torch.cuda.mem_get_info()
    assert (total - free) < 8 * 1024**3
