"""Bounded edit ops: validation, refusal rules, preview and direct application."""
import numpy as np
import pytest

from aladdin.sparse_deltas.store import DeltaStore
from aladdin.tensor_editor import EditRefused, TensorEditor, ops

NAME = "model.layers.0.mlp.down_proj.weight"


def test_validate_accepts_a_bounded_edit(backend, cfg):
    ed = TensorEditor(backend, cfg)
    v = ed.validate({"kind": "scale_columns", "tensor": NAME, "columns": [1, 2], "alpha": 0.05})
    assert v["ok"], v["errors"]
    assert v["info"]["entries_touched"] == 2 * backend.shape_of(NAME)[0]


def test_validate_rejects_bad_column(backend, cfg):
    ed = TensorEditor(backend, cfg)
    v = ed.validate({"kind": "scale_columns", "tensor": NAME, "columns": [99999], "alpha": 0.05})
    assert not v["ok"]


def test_validate_rejects_bad_alpha(backend, cfg):
    ed = TensorEditor(backend, cfg)
    assert not ed.validate({"kind": "scale_columns", "tensor": NAME, "columns": [1], "alpha": 5.0})["ok"]
    assert not ed.validate({"kind": "scale_columns", "tensor": NAME, "columns": [1], "alpha": "x"})["ok"]


def test_validate_rejects_unknown_tensor_and_kind(backend, cfg):
    ed = TensorEditor(backend, cfg)
    assert not ed.validate({"kind": "scale_columns", "tensor": "nope", "columns": [0], "alpha": 0.1})["ok"]
    assert not ed.validate({"kind": "multiply_all", "tensor": NAME, "columns": [0], "alpha": 0.1})["ok"]


def test_uniform_rescale_refused(backend, cfg):
    ed = TensorEditor(backend, cfg)
    n_cols = backend.shape_of(NAME)[1]
    v = ed.validate({"kind": "scale_columns", "tensor": NAME, "columns": list(range(n_cols)), "alpha": 0.05})
    assert not v["ok"]
    assert any("uniform rescale" in e for e in v["errors"])


def test_frozen_tensor_refused(backend, cfg):
    ed = TensorEditor(backend, cfg)
    v = ed.validate({"kind": "scale_columns", "tensor": "lm_head.weight", "columns": [0], "alpha": 0.05})
    assert not v["ok"]
    assert any("frozen" in e for e in v["errors"])


def test_max_entries_enforced(backend, cfg):
    c = dict(cfg)
    c["editing"] = dict(cfg["editing"], max_entries_per_tensor=10)
    ed = TensorEditor(backend, c)
    v = ed.validate({"kind": "scale_columns", "tensor": NAME, "columns": [1, 2, 3], "alpha": 0.05})
    assert not v["ok"] and any("max_entries" in e for e in v["errors"])


def test_ops_are_pure(backend):
    w = backend.get_tensor(NAME)
    before = w.copy()
    out = ops.apply(w, {"kind": "scale_columns", "tensor": NAME, "columns": [0], "alpha": 0.5})
    assert np.array_equal(w, before), "ops.apply must not mutate its input"
    assert not np.allclose(out[:, 0], before[:, 0])
    assert np.allclose(out[:, 1:], before[:, 1:])


def test_shift_columns_uses_std(backend):
    w = backend.get_tensor(NAME)
    out = ops.apply(w, {"kind": "shift_columns", "tensor": NAME, "columns": [0], "alpha": 1.0})
    d = (out[:, 0] - w[:, 0])
    assert np.allclose(d, np.std(w), atol=1e-4)


def test_scale_rows(backend):
    w = backend.get_tensor(NAME)
    out = ops.apply(w, {"kind": "scale_rows", "tensor": NAME, "rows": [0], "alpha": -0.5})
    assert np.allclose(out[0], w[0] * 0.5, atol=1e-6)
    assert np.allclose(out[1:], w[1:])


def test_scale_block_with_beta(backend):
    w = backend.get_tensor(NAME)
    out = ops.apply(w, {"kind": "scale_block", "tensor": NAME, "rows": [0, 1],
                        "columns": [0, 1], "alpha": 0.1, "beta": 0.0})
    assert out.shape == w.shape
    assert not np.allclose(out[:2, :2], w[:2, :2])


def test_preview_does_not_mutate(backend, cfg):
    ed = TensorEditor(backend, cfg)
    h0 = backend.snapshot_hashes()
    p = ed.preview({"kind": "scale_columns", "tensor": NAME, "columns": [0], "alpha": 0.1})
    assert p["ok"] and p["would_change_entries"] > 0
    assert backend.snapshot_hashes() == h0


def test_apply_writes_delta_and_verifies_hash(backend, cfg, tmp_project):
    store = DeltaStore(tmp_project["paths"]["deltas"])
    ed = TensorEditor(backend, cfg, store)
    before = backend.tensor_hash(NAME)
    res = ed.apply({"kind": "scale_columns", "tensor": NAME, "columns": [0, 5], "alpha": 0.1},
                  reason="unit test", meta={"experiment": "t"})
    assert res["verified"] is True
    assert backend.tensor_hash(NAME) == res["edited_sha256"] != before
    man, arrays = store.load(res["edit_id"])
    assert man["tensors"][0]["changed_entries"] == res["changed_entries"] > 0
    assert man["reason"] == "unit test"
    assert arrays["t0_idx"].dtype == np.int64


def test_apply_refuses_out_of_bounds_and_leaves_model_intact(backend, cfg, tmp_project):
    store = DeltaStore(tmp_project["paths"]["deltas"])
    ed = TensorEditor(backend, cfg, store)
    h0 = backend.snapshot_hashes()
    with pytest.raises(EditRefused):
        ed.apply({"kind": "scale_columns", "tensor": NAME, "columns": [0], "alpha": 9.0}, reason="bad")
    assert backend.snapshot_hashes() == h0
