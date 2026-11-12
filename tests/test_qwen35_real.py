"""REAL-checkpoint integration tests for the required WSP model.

They run ONLY when the actual weights are present on this machine (a local
snapshot of `RikiyaT/Qwen3.5-0.8B-weight-space-programming`) AND the HF stack is
installed. Without them the tests are SKIPPED and reported as "not run", never as
a pass.

Enable:

    python -m aladdin.cli download          # the required edited checkpoint
    export ALADDIN_MODEL_PATH="$PWD/models/Qwen3.5-0.8B-weight-space-programming"
    python -m aladdin.cli verify-model

The pristine snapshot is NEVER modified: edits happen on a disposable in-memory
model instance and are verified byte-exact, then the directory is checked to be
unchanged.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from aladdin.backends.qwen_loader import (_auto_class_for, check_checkpoint_identity,
                                          check_text_class_available,
                                          check_transformers_version,
                                          detect_architecture, plan_text_only_load,
                                          read_weight_index)
from aladdin.model_identity import REQUIRED_MODEL_ID

MODEL_DIR = Path(os.environ.get("ALADDIN_MODEL_PATH", "") or ".")
MODEL_ID = os.environ.get("ALADDIN_MODEL_ID", REQUIRED_MODEL_ID)


def _model_available() -> bool:
    return bool(os.environ.get("ALADDIN_MODEL_PATH")) and \
        (MODEL_DIR / "config.json").exists()


def _stack_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _model_available(),
    reason="real checkpoint weights not found (set ALADDIN_MODEL_PATH); NOT RUN, "
           "not passed")


# ------------------------------------------------------------------ loader
def test_real_architecture_matches_the_published_config():
    a = detect_architecture(MODEL_DIR, expected_model_id=MODEL_ID)
    assert a["is_qwen3_5"] is True
    assert a["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert a["text_model_type"] == "qwen3_5_text"
    assert a["text_hidden_size"] == 1024
    assert a["text_num_hidden_layers"] == 24
    assert a["text_vocab_size"] == 248320
    assert a["is_multimodal"] is True          # the checkpoint has a vision tower
    cls, _ = _auto_class_for(a)
    assert cls == "Qwen3_5ForCausalLM"         # text-only path (not multimodal)


def test_real_checkpoint_identity_is_recorded():
    ident = check_checkpoint_identity(MODEL_DIR, MODEL_ID)
    assert ident["ok"], ident["problems"]
    assert ident["recorded_model_id"] == MODEL_ID
    assert ident["identity_verified"] is True
    assert ident["revision"]


def test_real_weight_index_layout():
    """The published checkpoint stores the text tower under model.language_model.*."""
    a = detect_architecture(MODEL_DIR, expected_model_id=MODEL_ID)
    index = read_weight_index(MODEL_DIR)
    keys = index["keys"]
    plan = plan_text_only_load(keys, a)
    assert plan["n_mapped"] == 320
    assert plan["skipped"]["vision"] == 153
    assert plan["skipped"]["mtp"] == 15
    assert all(not k.startswith(("model.visual.", "mtp.")) for k in plan["mapping"])


def test_real_stack_supports_text_class():
    stack = check_transformers_version()
    assert stack["ok"], stack["error"]
    cls = check_text_class_available()
    assert cls["ok"], cls["error"]


# ------------------------------------------------------- real model loading
@pytest.fixture(scope="module")
def real_backend():
    if not _stack_available():
        pytest.skip("torch/transformers not installed; NOT RUN")
    from aladdin.backends.hf_torch import HFTorchBackend
    cfg = {"backend": {"kind": "hf_torch", "model_path": str(MODEL_DIR),
                       "model_id": MODEL_ID, "load_mode": "text",
                       "dtype": "bfloat16",
                       "device": "cuda" if _gpu_available() else "cpu"}}
    return HFTorchBackend(cfg)


def test_real_backend_loads_the_text_tower(real_backend):
    d = real_backend.describe()
    assert d["model_id"] == MODEL_ID
    assert d["identity_verified"] is True
    assert d["auto_class"] == "Qwen3_5ForCausalLM"
    assert "Qwen3_5ForCausalLM" in d["loaded_class"]
    assert d["load_mode"] == "text"
    assert d["text_model_type"] == "qwen3_5_text"
    assert d["hidden_size"] == 1024
    assert d["num_hidden_layers"] == 24
    assert d["vocab_size"] == 248320
    assert d["n_tensors"] > 0
    assert d["weights"]["all_weights_loaded"] is True
    assert d["weights"]["skipped_visual"] == 153
    assert d["weights"]["skipped_mtp"] == 15
    # live verification against the module tree, not only the config
    v = real_backend.verify()
    assert v["ok"], v["problems"]
    assert v["num_hidden_layers"] == 24
    assert v["embedding_shape"] == [248320, 1024]


def test_real_snapshot_directory_is_unmodified(real_backend):
    """Loading and editing must never write into the downloaded snapshot."""
    before = {p.name: p.stat().st_size for p in MODEL_DIR.iterdir() if p.is_file()}
    name = _edit_target(real_backend)
    backend = real_backend
    orig = backend.get_tensor(name)
    edited = orig.copy()
    edited[:, 0] *= 1.05
    backend.set_tensor(name, edited)
    backend.set_tensor(name, orig)
    after = {p.name: p.stat().st_size for p in MODEL_DIR.iterdir() if p.is_file()}
    assert before == after


def _edit_target(backend):
    from aladdin.inspection.tensors import suggest_editable
    cfg = {"editing": {"frozen_suffixes": ["model.norm.weight",
                                           "model.embed_tokens.weight",
                                           "lm_head.weight"]}}
    rows = suggest_editable(backend, cfg)
    assert rows, "no editable tensor found in the real model"
    return rows[0]["name"]


def test_real_tensor_read_hash_write_revert(real_backend):
    name = _edit_target(real_backend)
    original = real_backend.get_tensor(name)
    h0 = real_backend.tensor_hash(name)
    assert h0 == hashlib.sha256(real_backend.raw_bytes(name)).hexdigest()

    edited = original.copy()
    edited[:, 0] = edited[:, 0] * 1.05
    real_backend.set_tensor(name, edited)
    assert real_backend.tensor_hash(name) != h0

    real_backend.set_tensor(name, original)
    assert real_backend.tensor_hash(name) == h0
    assert np.array_equal(real_backend.get_tensor(name), original)


def test_real_agent_revert_cycle(real_backend, tmp_path):
    from aladdin.agent.tools import ToolRuntime
    from aladdin.forward_pass import ForwardPassEngine
    from aladdin.sparse_deltas.store import DeltaStore
    from aladdin.tensor_editor import TensorEditor

    name = _edit_target(real_backend)
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
    a = rt.call("apply_edit", {"operation": {
        "kind": "scale_columns", "tensor": name, "columns": [0, n_cols // 2],
        "alpha": 0.02}, "reason": "real-model probe"})
    assert a["ok"], a
    assert real_backend.tensor_hash(name) == a["edited_sha256"]

    rev = rt.call("revert", {"n": 1})
    assert rev["ok"] and rev["verified"] is True, rev
    assert rev["undone"] == [a["edit_id"]]
    assert rev["remaining"] == []
    assert real_backend.tensor_hash(name) == h0
    assert np.array_equal(real_backend.get_tensor(name), original)

    # and a whole committed transaction, by ID
    rt.begin_transaction()
    b = rt.call("apply_edit", {"operation": {
        "kind": "scale_columns", "tensor": name, "columns": [1], "alpha": 0.02},
        "reason": "txn probe"})
    tx = rt.txn.txn_id
    rt.commit_transaction()
    rev2 = rt.call("revert_transaction", {"txn_id": tx})
    assert rev2["ok"] and rev2["verified"] is True, rev2
    assert rev2["undone"] == [b["edit_id"]]
    assert real_backend.tensor_hash(name) == h0


def test_real_text_only_forward_pass(real_backend):
    ids = real_backend.encode("mujhe ye samjha")
    assert len(ids) > 0
    logits = real_backend.forward_logits(ids[:16])
    assert logits.ndim == 2
    assert logits.shape[1] == 248320
    assert np.isfinite(logits).all()
    out = real_backend.generate("iska kya matlab hai", max_new_tokens=8)
    assert isinstance(out, str)


def test_real_gpu_roundtrip(real_backend):
    if not _gpu_available():
        pytest.skip("no CUDA device on this machine; NOT RUN")
    import torch
    name = _edit_target(real_backend)
    dev = str(next(real_backend.model.parameters()).device)
    assert dev.startswith("cuda"), f"model is on {dev}, expected cuda"
    h0 = real_backend.tensor_hash(name)
    orig = real_backend.get_tensor(name)
    edited = orig.copy()
    edited[:, 0] *= 1.05
    real_backend.set_tensor(name, edited)
    assert real_backend.tensor_hash(name) != h0
    real_backend.set_tensor(name, orig)
    assert real_backend.tensor_hash(name) == h0
    free, total = torch.cuda.mem_get_info()
    assert (total - free) < 8 * 1024 ** 3
