"""Checkpoints, reconstruction from base+deltas, full rollback, resume state."""
import numpy as np
import pytest

from aladdin.checkpoints.manager import CheckpointManager
from aladdin.experiment.state import ExperimentState
from aladdin.sparse_deltas.store import DeltaStore
from aladdin.tensor_editor import TensorEditor

NAME = "model.layers.0.mlp.down_proj.weight"


def _setup(backend, cfg, tmp_project):
    store = DeltaStore(tmp_project["paths"]["deltas"])
    ed = TensorEditor(backend, cfg, store)
    ck = CheckpointManager(backend, tmp_project["paths"]["checkpoints"])
    ck.save_base()
    return ed, ck, store


def test_base_checkpoint_is_immutable(backend, cfg, tmp_project):
    ed, ck, _ = _setup(backend, cfg, tmp_project)
    h_base = dict(ck._data["base"]["hashes"])
    ed.apply({"kind": "scale_columns", "tensor": NAME, "columns": [0], "alpha": 0.1}, reason="x")
    ck.save_base()                               # must NOT overwrite
    assert ck._data["base"]["hashes"] == h_base


def test_restore_full_snapshot(backend, cfg, tmp_project):
    ed, ck, _ = _setup(backend, cfg, tmp_project)
    h0 = backend.tensor_hash(NAME)
    ed.apply({"kind": "scale_columns", "tensor": NAME, "columns": [0], "alpha": 0.1}, reason="x")
    assert backend.tensor_hash(NAME) != h0
    r = ck.restore("base")
    assert r["ok"] and backend.tensor_hash(NAME) == h0


def test_rebuild_chain_after_reset_reproduces_edited_state(backend, cfg, tmp_project):
    ed, ck, store = _setup(backend, cfg, tmp_project)
    e1 = ed.apply({"kind": "scale_columns", "tensor": NAME, "columns": [0], "alpha": 0.1}, reason="a")
    e2 = ed.apply({"kind": "scale_columns", "tensor": NAME, "columns": [3], "alpha": -0.08}, reason="b")
    target = backend.tensor_hash(NAME)
    r = ck.rebuild_chain_after_reset(backend, [e1["edit_id"], e2["edit_id"]])
    assert r["ok"], r
    assert backend.tensor_hash(NAME) == target, "base + deltas must reproduce the exact bits"


def test_reset_without_kept_deltas_returns_to_base(backend, cfg, tmp_project):
    ed, ck, _ = _setup(backend, cfg, tmp_project)
    h0 = backend.tensor_hash(NAME)
    ed.apply({"kind": "scale_columns", "tensor": NAME, "columns": [1], "alpha": 0.2}, reason="x")
    r = ck.rebuild_chain_after_reset(backend, [])
    assert r["ok"] and backend.tensor_hash(NAME) == h0


def test_chain_fails_loudly_on_wrong_order(backend, cfg, tmp_project):
    ed, ck, _ = _setup(backend, cfg, tmp_project)
    e1 = ed.apply({"kind": "scale_columns", "tensor": NAME, "columns": [0], "alpha": 0.1}, reason="a")
    # applying the same delta twice must fail the base-hash check, not silently stack
    r = ck.rebuild_chain([e1["edit_id"], e1["edit_id"]])
    assert r["ok"] is False and "sha256" in json_dump(r)


def json_dump(o):
    import json
    return json.dumps(o)


def test_checkpoint_diff(backend, cfg, tmp_project):
    ed, ck, _ = _setup(backend, cfg, tmp_project)
    ed.apply({"kind": "scale_columns", "tensor": NAME, "columns": [0], "alpha": 0.1}, reason="x")
    ck.save("step1", metrics={"dev": 0.5})
    d = ck.diff("base", "step1")
    assert NAME in d["changed"] and d["metrics_b"]["dev"] == 0.5


def test_state_roundtrip_and_resumability(tmp_path):
    p = tmp_path / "state.json"
    st = ExperimentState.create(p, experiment="e", backend="tiny_numpy",
                                model_desc={"x": 1}, config_path="c", max_steps=3)
    st.record_baseline(0.4)
    st.record_step({"step": 0, "edit_id": "e1", "accepted": True, "candidate_composite": 0.45})
    assert st.data["step"] == 1 and st.data["kept_edits"] == ["e1"]
    st2 = ExperimentState.load(p)
    assert st2.data["best_composite"] == 0.45
    assert st2.is_resumable()
    st2.finish("finished")
    assert not ExperimentState.load(p).is_resumable()


def test_state_atomic_write_leaves_no_tmp(tmp_path):
    p = tmp_path / "state.json"
    st = ExperimentState.create(p, experiment="e", backend="tiny_numpy",
                                model_desc={}, config_path="", max_steps=1)
    st.save()
    assert not p.with_suffix(".tmp").exists()
    import json
    json.load(open(p))                       # must always be valid JSON
