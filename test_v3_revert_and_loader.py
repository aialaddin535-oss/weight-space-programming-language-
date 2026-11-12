"""V3 regression tests: revert synchronization, n-argument semantics and the
Qwen3.5-aware loader.

Group A (revert): the t_revert response contract - 'undone' carries real edit
IDs only after a hash-verified tensor restore, 'remaining' reports what is
still active, failures never touch applied tracking, n counts EDITS.

Group B (loader): architecture detection from the real config shape, correct
auto-class selection, version/dependency guards. These use synthetic config
files - they are MOCK-level tests and never prove a real Qwen run.

Real-model tests live in test_qwen35_real.py and are skipped unless the
weights are present; they are reported separately from these.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from aladdin.agent.tools import RevertConflict, ToolRuntime
from aladdin.edit_ledger import EditTransaction
from aladdin.forward_pass import ForwardPassEngine
from aladdin.sparse_deltas.store import DeltaStore
from aladdin.tensor_editor import TensorEditor

from tests.test_bugfix_regression import (_op, _runtime, hash_of)  # reuse helpers

N1 = "model.layers.0.mlp.down_proj.weight"
N2 = "model.layers.1.mlp.down_proj.weight"


# ============================================================ A. revert contract
def test_v3_undone_contains_real_edit_ids(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    a = rt.call("apply_edit", {"operation": _op(N1, [0], 0.1), "reason": "r"})
    rev = rt.call("revert", {"n": 1})
    assert rev["ok"] and rev["verified"] is True
    assert rev["undone"] == [a["edit_id"]]
    assert rev["undone"][0] not in rev["restored"]          # restored = tensor names
    assert rev["restored"] == [N1]
    assert rev["remaining"] == []


def test_v3_applied_here_synchronized_after_revert(backend, cfg, tmp_project):
    """The exact bug from V2: undone was [] while tensors were restored."""
    rt = _runtime(backend, cfg, tmp_project)
    a = rt.call("apply_edit", {"operation": _op(N1, [0], 0.1), "reason": "r"})
    # simulate the agent bookkeeping, as nvidia_agent.run_step does
    applied_here = [a["edit_id"]]
    rev = rt.call("revert", {"n": 1})
    undone = set(rev.get("undone") or [])
    applied_here = [e for e in applied_here if e not in undone]
    assert applied_here == [], "reverted edit ID must leave agent tracking"
    assert rt.applied == []
    assert a["edit_id"] in rt._reverted_ids


def test_v3_failed_revert_keeps_tracking(backend, cfg, tmp_project, monkeypatch):
    rt = _runtime(backend, cfg, tmp_project)
    a = rt.call("apply_edit", {"operation": _op(N1, [0], 0.1), "reason": "r"})
    # simulate a genuine restoration failure: the tensor write itself explodes
    # (device error), so nothing can be claimed as restored
    def broken_set(name, value):
        raise RuntimeError("CUDA error: memcpy failed")
    monkeypatch.setattr(rt.backend, "set_tensor", broken_set)
    rev = rt.call("revert", {"n": 1})
    assert rev["ok"] is False
    assert rev["undone"] == [] and rev["restored"] == []
    assert rev["remaining"] == [a["edit_id"]]
    assert rt.applied and rt.applied[0]["edit_id"] == a["edit_id"], \
        "failed revert must not remove IDs from active tracking"
    assert a["edit_id"] not in rt._reverted_ids
    # the original error detail is preserved
    assert "CUDA error: memcpy failed" in rev["error"]
    monkeypatch.undo()
    # and the tensor still carries the edit (untouched by the failed revert)
    assert backend.tensor_hash(N1) == a["edited_sha256"]


def test_v3_partial_revert_reports_restored_and_remaining(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    a1 = rt.call("apply_edit", {"operation": _op(N1, [0], 0.1), "reason": "1"})
    a2 = rt.call("apply_edit", {"operation": _op(N2, [0], 0.1), "reason": "2"})
    a3 = rt.call("apply_edit", {"operation": _op(N2, [1], 0.12), "reason": "3"})
    rev = rt.call("revert", {"n": 1})
    assert rev["ok"] and rev["verified"]
    assert rev["undone"] == [a3["edit_id"]], "n=1 reverts exactly the newest edit"
    assert rev["restored"] == [N2]
    assert rev["remaining"] == [a1["edit_id"], a2["edit_id"]]
    assert a3["edit_id"] not in [x["edit_id"] for x in rt.applied]
    assert {x["edit_id"] for x in rt.applied} == {a1["edit_id"], a2["edit_id"]}


def test_v3_n_counts_edits_not_transactions(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    ids = [rt.call("apply_edit", {"operation": _op(N1, [i], 0.1), "reason": str(i)})["edit_id"]
           for i in range(3)]                       # 3 edits, ONE transaction
    rev = rt.call("revert", {"n": 1})
    assert rev["ok"] and rev["undone"] == [ids[-1]], \
        "n=1 must revert one edit, not the whole transaction"
    assert rt.applied and len(rt.applied) == 2
    rev2 = rt.call("revert", {"n": 2})
    assert rev2["ok"] and rev2["undone"] == [ids[1], ids[0]]
    assert rt.applied == []


def test_v3_n_newest_first_order(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    ids = [rt.call("apply_edit", {"operation": _op(N1, [i], 0.1), "reason": "x"})["edit_id"]
           for i in range(4)]
    rev = rt.call("revert", {"n": 2})
    assert rev["undone"] == [ids[3], ids[2]], "reverts in reverse execution order"


def test_v3_revert_transaction_whole_txn(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    orig1, orig2 = backend.get_tensor(N1), backend.get_tensor(N2)
    rt.begin_transaction()
    a1 = rt.call("apply_edit", {"operation": _op(N1, [0], 0.1), "reason": "1"})
    a2 = rt.call("apply_edit", {"operation": _op(N2, [0], 0.1), "reason": "2"})
    rev = rt.call("revert_transaction", {})
    assert rev["ok"] and rev["verified"] is True
    assert sorted(rev["undone"]) == sorted([a1["edit_id"], a2["edit_id"]])
    assert np.array_equal(backend.get_tensor(N1), orig1)
    assert np.array_equal(backend.get_tensor(N2), orig2)
    assert rt.applied == [] and rev["remaining"] == []


def test_v3_partial_revert_rejects_txn_split(backend, cfg, tmp_project):
    """Out-of-order (non newest-first) reverts are rejected BEFORE any tensor
    write: reverting an older edit while a newer edit on the SAME tensor is
    still applied would make the newer edit's pre-image wrong."""
    rt = _runtime(backend, cfg, tmp_project)
    ids = [rt.call("apply_edit", {"operation": _op(N1, [i], 0.1), "reason": str(i)})["edit_id"]
           for i in range(3)]
    recs = rt._records()
    with pytest.raises(RevertConflict) as ei:
        rt._validate_partial_revert([recs[ids[0]]])          # oldest, out of order
    assert "newer edit" in str(ei.value)
    # the newest-first path used by t_revert never conflicts on one tensor
    rev = rt.call("revert", {"n": 3})
    assert rev["ok"] and rev["undone"] == [ids[2], ids[1], ids[0]]


def test_v3_repeated_revert_idempotent(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    a = rt.call("apply_edit", {"operation": _op(N1, [0], 0.1), "reason": "r"})
    r1 = rt.call("revert", {"n": 1})
    r2 = rt.call("revert", {"n": 1})
    r3 = rt.call("revert_transaction", {})
    assert r1["ok"] and r1["undone"] == [a["edit_id"]]
    assert r2["ok"] and r2["undone"] == []
    assert r3["ok"] and r3["undone"] == []
    assert rt.applied == []


def test_v3_n_too_large_is_rejected_cleanly(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    rt.call("apply_edit", {"operation": _op(N1, [0], 0.1), "reason": "r"})
    rev = rt.call("revert", {"n": 9})
    assert rev["ok"] is False and "n=9" in rev["error"]
    assert len(rt.applied) == 1, "oversized n must not silently revert everything"


def test_v3_sequential_edits_same_tensor_partial_revert_exact(backend, cfg, tmp_project):
    """3 edits on ONE tensor: reverting only the newest 2 restores the state
    after edit 1 exactly (not the pristine tensor, not a lie)."""
    rt = _runtime(backend, cfg, tmp_project)
    pristine = backend.get_tensor(N1)
    ids = [rt.call("apply_edit", {"operation": _op(N1, [i], 0.1), "reason": str(i)})["edit_id"]
           for i in range(3)]
    after_e1 = None
    # recompute the expected intermediate state by replaying edit 1 on a copy
    rev = rt.call("revert", {"n": 2})
    assert rev["ok"] and rev["undone"] == [ids[2], ids[1]]
    # after edit 1, column 0 was scaled; recompute expectation from pristine
    expected = pristine.copy()
    expected[:, 0] = expected[:, 0] * 1.1
    assert np.allclose(backend.get_tensor(N1), expected, rtol=0, atol=1e-6), \
        "partial revert must leave exactly edit 1 applied"


def test_v3_ledger_and_agent_state_synchronized(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    ids = [rt.call("apply_edit", {"operation": _op(N1, [i], 0.1), "reason": str(i)})["edit_id"]
           for i in range(2)]
    rt.call("revert", {"n": 1})
    # the durable txn ledger records the reverted status for that edit
    recs = {e["edit_id"]: e for e in rt.txn.edits}
    assert recs[ids[1]]["status"] == "reverted"
    assert recs[ids[0]]["status"] == "applied"
    assert len(rt.applied) == 1 and rt.applied[0]["edit_id"] == ids[0]


def test_v3_edit_ledger_durable_revert_status(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    ids = [rt.call("apply_edit", {"operation": _op(N1, [i], 0.1), "reason": str(i)})["edit_id"]
           for i in range(2)]
    rt.commit_transaction()
    rev = rt.call("revert", {"n": 1})
    assert rev["ok"] and rev["undone"] == [ids[1]]
    d = json.load(open(Path(rt.ledger.root) / rev["txn_id"] / "ledger.json"))
    st = {e["edit_id"]: e["status"] for e in d["edits"]}
    assert st[ids[1]] == "reverted" and st[ids[0]] == "applied"
    idx = json.load(open(Path(rt.ledger.root) / "ledger_index.json"))
    assert ids[1] in idx[rev["txn_id"]]["reverted_edit_ids"]


def test_v3_failed_rollback_never_touches_active_ids(backend, cfg, tmp_project):
    """rollback_transaction with a broken pristine journal raises, and the
    active tracking (rt.applied) survives unchanged."""
    rt = _runtime(backend, cfg, tmp_project)
    a = rt.call("apply_edit", {"operation": _op(N1, [0], 0.1), "reason": "r"})
    # corrupt the open transaction's journal
    for f in Path(rt.txn.root).glob("*.npy"):
        f.unlink()
    with pytest.raises(Exception):
        rt.rollback_transaction(error="boom")
    assert rt.applied and rt.applied[0]["edit_id"] == a["edit_id"]


def test_v3_loop_register_loop_edit_visible_to_revert(backend, cfg, tmp_project):
    """The experiment-loop path (register_loop_edit) feeds the same revert."""
    rt = _runtime(backend, cfg, tmp_project)
    rt.begin_transaction()
    rt.txn.capture_pristine(N1)
    store = DeltaStore(tmp_project["paths"]["deltas"])
    ed = TensorEditor(backend, cfg, store)
    applied = ed.apply(_op(N1, [0], 0.1), reason="loop", meta={"txn_id": rt.txn.txn_id})
    rt.register_loop_edit(applied, hypothesis="h", agent="test")
    rev = rt.call("revert", {"n": 1})
    assert rev["ok"] and rev["undone"] == [applied["edit_id"]]
    assert rt.applied == []


# ============================================================ B. loader selection
@pytest.fixture()
def qwen35_cfg_dir(tmp_path):
    """A synthetic config.json mirroring the REAL published Qwen3.5-0.8B config."""
    real = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "image_token_id": 248056, "video_token_id": 248057,
        "vision_start_token_id": 248053, "vision_end_token_id": 248054,
        "tie_word_embeddings": True,
        "transformers_version": "4.57.0.dev0",
        "text_config": {"model_type": "qwen3_5_text", "hidden_size": 1024,
                        "num_hidden_layers": 24, "vocab_size": 248320,
                        "intermediate_size": 3584, "num_attention_heads": 8,
                        "num_key_value_heads": 2, "tie_word_embeddings": True,
                        "dtype": "bfloat16"},
        "vision_config": {"depth": 12, "hidden_size": 768, "patch_size": 16},
    }
    d = tmp_path / "fake_qwen35"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(real), encoding="utf-8")
    return d


def test_v3_detect_qwen35_architecture(qwen35_cfg_dir):
    from aladdin.backends.qwen_loader import detect_architecture
    a = detect_architecture(qwen35_cfg_dir)
    assert a["is_qwen3_5"] is True
    assert a["is_multimodal"] is True
    assert a["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert a["text_model_type"] == "qwen3_5_text"
    assert a["text_hidden_size"] == 1024
    assert a["text_num_hidden_layers"] == 24
    assert a["text_vocab_size"] == 248320
    assert a["tie_word_embeddings"] is True


def test_v3_auto_class_selection(qwen35_cfg_dir):
    from aladdin.backends.qwen_loader import _auto_class_for, detect_architecture
    a = detect_architecture(qwen35_cfg_dir)
    cls, reason = _auto_class_for(a)
    assert cls == "AutoModelForImageTextToText"
    assert "AutoModelForCausalLM" in reason or "NOT" in reason


def test_v3_classic_causal_lm_stays_on_causallm(tmp_path):
    from aladdin.backends.qwen_loader import _auto_class_for
    classic = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
    cls, _ = _auto_class_for({"is_qwen3_5": False, "is_multimodal": False,
                              "architectures": classic["architectures"]})
    assert cls == "AutoModelForCausalLM"


def test_v3_missing_dependency_message(tmp_path, monkeypatch):
    """Missing torch produces an actionable BackendError, not a traceback."""
    import builtins
    from aladdin.backends import hf_torch as ht
    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    cfg = {"backend": {"kind": "hf_torch", "model_path": str(tmp_path)}}
    from aladdin.backends.base import BackendError
    with pytest.raises(BackendError, match="pip install torch"):
        ht.HFTorchBackend(cfg)


def test_v3_unsupported_transformers_version(tmp_path, monkeypatch):
    from aladdin.backends import qwen_loader as ql
    monkeypatch.setattr(ql, "transformers_version", lambda: "4.46.0")
    r = ql.check_transformers_version()
    assert r["ok"] is False and r["installed"] == "4.46.0"
    assert "pip install" in r["error"]
    with pytest.raises(ql.LoaderCompatibilityError, match="too old"):
        ql.load_qwen35(tmp_path)


def test_v3_model_path_validation(tmp_path):
    from aladdin.backends.qwen_loader import detect_architecture
    with pytest.raises(FileNotFoundError, match="config.json"):
        detect_architecture(tmp_path / "does_not_exist")


def test_v3_loader_rejects_silent_model_swap(qwen35_cfg_dir, monkeypatch):
    """A loaded object whose class contradicts config.json must be rejected."""
    from aladdin.backends import qwen_loader as ql
    arch = ql.detect_architecture(qwen35_cfg_dir)

    class FakeCompletelyDifferent:      # noqa: does not match any architecture
        config = type("C", (), {"architectures": ["SomeoneElsesModel"]})()

    chk = ql.verify_loaded_model(FakeCompletelyDifferent(), qwen35_cfg_dir, arch)
    assert chk["ok"] is False and chk["problems"]
