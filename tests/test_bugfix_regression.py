"""Regression tests for the three critical weight-space bugs.

BUG 1: CheckpointManager backend_factory injection + recoverable large-model checkpoints.
BUG 2: t_revert() restores the actual tensors (not just edit IDs).
BUG 3: Multi-edit transactions: every edit tracked, group rollback exact.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from aladdin.agent.tools import ToolRuntime
from aladdin.checkpoints import manager as ckpt_mod
from aladdin.checkpoints.manager import CheckpointManager
from aladdin.evaluation.datasets import Dataset
from aladdin.evaluation.runner import run_eval
from aladdin.experiment.state import ExperimentState
from aladdin.forward_pass import ForwardPassEngine
from aladdin.sparse_deltas.store import DeltaStore
from aladdin.tensor_editor import TensorEditor

N1 = "model.layers.0.mlp.down_proj.weight"
N2 = "model.layers.1.mlp.down_proj.weight"


def _op(name, cols, alpha, kind="scale_columns"):
    return {"kind": kind, "tensor": name, "columns": cols, "alpha": alpha}


def _setup(backend, cfg, tmp_project, **kw):
    store = DeltaStore(tmp_project["paths"]["deltas"])
    ed = TensorEditor(backend, cfg, store)
    ck = CheckpointManager(backend, tmp_project["paths"]["checkpoints"],
                           deltas_root=tmp_project["paths"]["deltas"],
                           backend_factory=lambda: type(backend)(cfg), **kw)
    ck.save_base()
    return ed, ck, store


# --------------------------------------------------------------------- BUG 1
def test_checkpoint_manager_receives_backend_factory(backend, cfg, tmp_project):
    ed, ck, _ = _setup(backend, cfg, tmp_project)
    assert callable(ck.backend_factory)
    fresh = ck.backend_factory()
    assert fresh.tensor_hash(N1) == ck._data["base"]["hashes"][N1]


def test_small_model_rollback_restores_exact_bits(backend, cfg, tmp_project):
    ed, ck, _ = _setup(backend, cfg, tmp_project)
    h0 = backend.tensor_hash(N1)
    ed.apply(_op(N1, [0, 2], 0.1), reason="x")
    ed.apply(_op(N1, [3], -0.05), reason="y")
    assert backend.tensor_hash(N1) != h0
    r = ck.rebuild_chain_after_reset(backend, [])
    assert r["ok"], r
    assert backend.tensor_hash(N1) == h0, "rollback must be byte-exact"


def test_large_model_modified_only_checkpoint_is_recoverable(backend, cfg, tmp_project, monkeypatch):
    """Force the large-model path with a lowered threshold: hashes alone must
    never be the only recovery data."""
    monkeypatch.setattr(ckpt_mod, "SNAPSHOT_PARAM_LIMIT", 1)
    ed, ck, _ = _setup(backend, cfg, tmp_project)
    h0 = backend.tensor_hash(N1)
    e1 = ed.apply(_op(N1, [0], 0.1), reason="a")
    e2 = ed.apply(_op(N2, [1], -0.08), reason="b")
    ck.save("large_ckpt")
    assert ck._data["large_ckpt"]["mode"] == "modified_only"
    # the checkpoint physically stores recoverable tensor data for changed tensors
    assert (Path(tmp_project["paths"]["checkpoints"]) / "large_ckpt" /
            (N1.replace(".", "__") + ".npy")).exists()
    assert (Path(tmp_project["paths"]["checkpoints"]) / "large_ckpt" /
            (N2.replace(".", "__") + ".npy")).exists()
    # restore and verify hashes
    r = ck.restore("large_ckpt")
    assert r["ok"] and r["verified"], r
    assert backend.tensor_hash(N1) == e1["edited_sha256"]
    assert backend.tensor_hash(N2) == e2["edited_sha256"]
    # rollback to base via pristine reload (backend_factory) also works
    r2 = ck.rebuild_chain_after_reset(backend, [])
    assert r2["ok"], r2
    assert backend.tensor_hash(N1) == h0
    assert backend.tensor_hash(N2) == ck._data["base"]["hashes"][N2]


def test_large_model_reset_to_base_verifies_fresh_backend(backend, cfg, tmp_project, monkeypatch):
    monkeypatch.setattr(ckpt_mod, "SNAPSHOT_PARAM_LIMIT", 1)
    ed, ck, _ = _setup(backend, cfg, tmp_project)
    ed.apply(_op(N1, [0], 0.1), reason="x")
    r = ck.reset_to_base()
    assert r["ok"] and backend.tensor_hash(N1) == ck._data["base"]["hashes"][N1]


def test_base_hash_mismatch_is_detected_not_silent(backend, cfg, tmp_project, monkeypatch):
    monkeypatch.setattr(ckpt_mod, "SNAPSHOT_PARAM_LIMIT", 1)
    ed, ck, _ = _setup(backend, cfg, tmp_project)
    ed.apply(_op(N1, [0], 0.1), reason="x")
    # corrupt the pristine source the factory would load: point the factory at a
    # fresh backend, then mutate its base to simulate a drifted disk
    def bad_factory():
        b = type(backend)(cfg)
        w = b.get_tensor(N1)
        w[0, 0] += 7.0
        b.set_tensor(N1, w)
        return b
    ck.backend_factory = bad_factory
    with pytest.raises(RuntimeError, match="do NOT match"):
        ck.reset_to_base()


def test_rejected_edit_rollback_restores_previous_state(backend, cfg, tmp_project):
    ed, ck, store = _setup(backend, cfg, tmp_project)
    kept = ed.apply(_op(N1, [0], 0.1), reason="keep")     # an accepted earlier edit
    target = {n: backend.tensor_hash(n) for n in backend.param_names()}
    rej = ed.apply(_op(N1, [1], 0.2), reason="reject me")
    store.update_status(rej["edit_id"], "rejected")
    r = ck.rebuild_chain_after_reset(backend, [kept["edit_id"]])
    assert r["ok"], r
    for n, h in target.items():
        assert backend.tensor_hash(n) == h, f"{n} not restored exactly"


# --------------------------------------------------------------------- BUG 2
def _runtime(backend, cfg, tmp_project):
    store = DeltaStore(tmp_project["paths"]["deltas"])
    ed = TensorEditor(backend, cfg, store)
    fwd = ForwardPassEngine(backend, budget=20, run_dir=tmp_project["paths"]["logs"])
    ds = Dataset([{"id": "a1", "category": "roman_hinglish", "prompt": "mujhe samaj nahi aya",
                   "expect_script": "roman", "must_include": [["samaj"]], "answer_tokens": []}], "dev")
    return ToolRuntime(backend, cfg, ed, fwd, ds,
                       Path(tmp_project["paths"]["experiments"]))


def test_t_revert_restores_actual_tensor_bytes(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    orig = backend.get_tensor(N1)
    a = rt.call("apply_edit", {"operation": _op(N1, [0, 1], 0.15), "reason": "r"})
    assert a["ok"]
    assert not np.array_equal(backend.get_tensor(N1), orig), "edit must change the tensor"
    rev = rt.call("revert", {"n": 1})
    assert rev["ok"], rev
    assert np.array_equal(backend.get_tensor(N1), orig), \
        "t_revert must restore the tensor byte-for-byte"
    assert backend.tensor_hash(N1) == hash_of(orig, backend, N1)


def hash_of(arr, backend, name):
    from aladdin.sparse_deltas.cast import cast_to_dtype
    from aladdin.sparse_deltas.store import sha256_arr
    return sha256_arr(cast_to_dtype(arr, backend.dtype_of(name)))


def test_t_revert_verifies_hashes(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    orig = backend.get_tensor(N1)
    rt.call("apply_edit", {"operation": _op(N1, [2], 0.1), "reason": "r"})
    rev = rt.call("revert", {"n": 1})
    assert rev["ok"] and rev.get("verified") is True
    assert backend.tensor_hash(N1) == hash_of(orig, backend, N1)


def test_t_revert_repeated_edits_same_tensor(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    orig = backend.get_tensor(N1)
    rt.call("apply_edit", {"operation": _op(N1, [0], 0.1), "reason": "1"})
    rt.call("apply_edit", {"operation": _op(N1, [1], 0.1), "reason": "2"})
    rt.call("apply_edit", {"operation": _op(N1, [2], 0.1), "reason": "3"})
    rev = rt.call("revert", {"n": 3})
    assert rev["ok"], rev
    assert np.array_equal(backend.get_tensor(N1), orig)
    assert rt.applied == []


def test_t_revert_idempotent(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    rt.call("apply_edit", {"operation": _op(N1, [0], 0.1), "reason": "1"})
    r1 = rt.call("revert", {"n": 1})
    r2 = rt.call("revert", {"n": 1})
    assert r1["ok"] and r2["ok"] and r2.get("undone") == []


# --------------------------------------------------------------------- BUG 3
def test_multi_edit_interaction_all_ids_tracked(backend, cfg, tmp_project):
    """3+ edits across 2 tensors in ONE interaction: all IDs returned, all
    registered, group rollback exact."""
    rt = _runtime(backend, cfg, tmp_project)
    orig1, orig2 = backend.get_tensor(N1), backend.get_tensor(N2)
    rt.begin_transaction()
    a1 = rt.call("apply_edit", {"operation": _op(N1, [0], 0.1), "reason": "e1"})
    a2 = rt.call("apply_edit", {"operation": _op(N2, [0], 0.1), "reason": "e2"})
    a3 = rt.call("apply_edit", {"operation": _op(N1, [1], 0.12), "reason": "e3"})
    ids = [a1["edit_id"], a2["edit_id"], a3["edit_id"]]
    assert len(set(ids)) == 3, "every successful edit gets its own unique edit ID"
    assert a1["txn_id"] == a2["txn_id"] == a3["txn_id"]
    assert len(rt.applied) == 3
    # the open transaction itself is the source of truth until it is closed
    assert len(rt.txn.edits) == 3
    recs = {e["edit_id"]: e for e in rt.txn.edits}
    for eid in ids:
        assert eid in recs
        assert recs[eid]["tensor_sha256_after"] and recs[eid]["shape"] and recs[eid]["dtype"]
    assert set(rt.txn.tensors_touched()) == {N1, N2}
    # reject the whole transaction -> exact restore of BOTH tensors
    rb = rt.rollback_transaction(error="rejected by test", status="rejected")
    assert rb["ok"], rb
    assert np.array_equal(backend.get_tensor(N1), orig1)
    assert np.array_equal(backend.get_tensor(N2), orig2)
    # after the rollback the ledger index carries the full rejected group
    assert len(rt.ledger.edits_in_order()) == 3
    assert all(e["status"] == "rejected" for e in rt.ledger.edits_in_order())
    assert rt.applied == []


def test_exception_mid_transaction_rolls_back_all(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    orig1, orig2 = backend.get_tensor(N1), backend.get_tensor(N2)
    rt.begin_transaction()
    rt.call("apply_edit", {"operation": _op(N1, [0], 0.1), "reason": "e1"})
    assert not np.array_equal(backend.get_tensor(N1), orig1)
    # simulate Edit 2 exploding after a partial write attempt
    class Boom(RuntimeError):
        pass
    orig_apply = rt.editor.apply
    def exploding(op, **kw):
        name = op.get("tensor") or op.get("name")
        if name == N2:
            w = backend.get_tensor(N2); w[0, 0] += 1.0; backend.set_tensor(N2, w)
            raise Boom("disk died mid-edit")
        return orig_apply(op, **kw)
    rt.editor.apply = exploding
    res = rt.call("apply_edit", {"operation": _op(N2, [0], 0.1), "reason": "e2"})
    assert res["ok"] is False and "disk died" in res.get("error", "")
    assert np.array_equal(backend.get_tensor(N1), orig1), "already-applied edit must be undone"
    assert np.array_equal(backend.get_tensor(N2), orig2), "partial write must be undone"
    # failed transaction recorded with the original error information
    txns = rt.ledger.transactions()
    assert txns and txns[-1]["status"] == "failed" and "disk died" in (txns[-1]["error"] or "")


def test_loop_records_all_edit_ids_in_history(project, cfg):
    """End-to-end: the experiment loop captures every edit ID of the candidate."""
    from aladdin.experiment.loop import run_experiment
    c = dict(project)
    c["acceptance"] = dict(cfg["acceptance"], min_composite_gain=99.0)  # reject all
    final = run_experiment(c, experiment="multi_edit", max_steps=1, agent_kind="scripted",
                           verbose=False)
    assert final["ok"], final
    exp = Path(final["dir"])
    hist = [json.loads(l) for l in (exp / "history.jsonl").read_text().splitlines() if l.strip()]
    assert hist
    for h in hist:
        assert h["edit_ids"], "every candidate records ALL its edit IDs"
        assert len(h["edit_ids"]) == h["n_edits"] >= 1
        assert h["txn_id"]
    st = json.load(open(exp / "state.json"))
    n_ids = sum(len(h["edit_ids"]) for h in hist)
    assert st["applied_total"] == n_ids
    assert len(st["rejected_edits"]) == n_ids, "no edit ID silently discarded"
    # ledger / checkpoint consistency with the state
    led = Path(tmp_ledger(c, "multi_edit")) / "edit_ledger"
    idx = json.load(open(led / "ledger_index.json"))
    all_ledger_ids = {e for t in idx.values() for e in t["edit_ids"]}
    assert all_ledger_ids == set(st["rejected_edits"]) | set(st["kept_edits"])
    assert final["verify"]["ok"] is True, "weights back to pristine after group rejection"


def tmp_ledger(cfg, experiment):
    return Path(cfg["paths"]["experiments"]) / experiment


def test_checkpoint_history_contains_group(project, cfg):
    from aladdin.experiment.loop import run_experiment
    c = dict(project)
    c["acceptance"] = dict(cfg["acceptance"], min_composite_gain=-1.0)  # accept all
    final = run_experiment(c, experiment="multi_accept", max_steps=1, agent_kind="scripted",
                           verbose=False)
    assert final["ok"], final
    exp = Path(final["dir"])
    hist = [json.loads(l) for l in (exp / "history.jsonl").read_text().splitlines() if l.strip()]
    man = json.load(open(Path(c["paths"]["checkpoints"]) / "checkpoints.json"))
    best = [v for k, v in man.items() if k.startswith("best_step")]
    assert best, "accepted candidate must have a checkpoint"
    assert all(b.get("txn") for b in best), "checkpoint linked to its transaction"
    assert all(h["txn_id"] == best[-1]["txn"] for h in hist if h["accepted"]), \
        "checkpoint history contains the complete edit group"


def test_state_multi_edit_accounting(tmp_path):
    p = tmp_path / "state.json"
    st = ExperimentState.create(p, experiment="e", backend="tiny_numpy",
                                model_desc={}, config_path="", max_steps=5)
    st.record_baseline(0.4)
    st.record_step({"step": 0, "edit_ids": ["e1", "e2", "e3"], "txn_id": "t1",
                    "accepted": True, "candidate_composite": 0.5})
    assert st.data["kept_edits"] == ["e1", "e2", "e3"]
    assert st.data["applied_total"] == 3
    assert st.data["transactions"][0]["txn_id"] == "t1"
    st.record_step({"step": 1, "edit_ids": ["e4", "e5"], "txn_id": "t2",
                    "accepted": False, "candidate_composite": 0.3})
    assert st.data["rejected_edits"] == ["e4", "e5"]
    assert st.data["applied_total"] == 5


# ------------------------------------------------- interrupted experiment
def test_rollback_after_interrupted_experiment(backend, cfg, tmp_project):
    ed, ck, _ = _setup(backend, cfg, tmp_project)
    kept = ed.apply(_op(N1, [0], 0.1), reason="kept")
    ed.apply(_op(N2, [0], 0.1), reason="uncommitted")
    # simulate a crash: state says interrupted; resume replays only kept deltas
    p = Path(tmp_project["paths"]["experiments"]) / "interrupted" / "state.json"
    st = ExperimentState.create(p, experiment="interrupted", backend="tiny_numpy",
                                model_desc={}, config_path="", max_steps=3)
    st.record_step({"step": 0, "edit_ids": [kept["edit_id"]], "txn_id": "t0",
                    "accepted": True, "candidate_composite": 0.5})
    st.data["status"] = "interrupted"
    st.save()
    st2 = ExperimentState.load(p)
    rep = ck.rebuild_chain_after_reset(backend, st2.data["kept_edits"])
    assert rep["ok"], rep
    assert backend.tensor_hash(N1) == kept["edited_sha256"]
    assert backend.tensor_hash(N2) == ck._data["base"]["hashes"][N2], \
        "interrupted (never-accepted) edit must not survive rollback"


def test_bf16_and_fp32_restoration_semantics(backend, cfg, tmp_project):
    """The restore pipeline is dtype-exact: hashes over raw storage bits."""
    from aladdin.sparse_deltas.cast import cast_to_dtype
    from aladdin.sparse_deltas.store import sha256_arr
    ed, ck, _ = _setup(backend, cfg, tmp_project)
    e = ed.apply(_op(N1, [0, 5], 0.1), reason="x")
    bits = cast_to_dtype(backend.get_tensor(N1), backend.dtype_of(N1))
    assert sha256_arr(bits) == e["edited_sha256"]
    r = ck.rebuild_chain_after_reset(backend, [])
    assert r["ok"] and backend.tensor_hash(N1) == e["base_sha256"]
    # bf16 bit handling itself
    from aladdin.sparse_deltas.cast import from_bf16_bits, to_bf16_bits
    x = np.array([1.0, -2.5, 3.14159], dtype=np.float32)
    back = from_bf16_bits(to_bf16_bits(x))
    assert np.allclose(back, x, atol=1e-2)
