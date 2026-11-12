"""Sparse delta format: bf16 bit exactness, hash checks, reconstruction."""
import numpy as np
import pytest

from aladdin.sparse_deltas import cast
from aladdin.sparse_deltas.store import DeltaStore, build_record, reconstruct


def test_bf16_roundtrip_is_lossy_but_stable():
    x = np.array([0.0, 1.0, -2.5, 1e-8, 123.456], dtype=np.float32)
    bits = cast.to_bf16_bits(x)
    back = cast.from_bf16_bits(bits)
    assert np.allclose(back, x, rtol=1e-2, atol=1e-6)
    assert cast.to_bf16_bits(back).tolist() == bits.tolist()


def test_bf16_round_to_nearest_even():
    # 1 + 2^-8 is exactly halfway between two bfloat16 values -> ties to even
    x = np.array([1.0 + 2 ** -8], dtype=np.float32)
    assert cast.to_bf16_bits(x)[0] == cast.to_bf16_bits(np.array([1.0], np.float32))[0]


def test_build_record_sparse_only_records_changed_entries():
    base = np.zeros((4, 4), np.float32)
    new = base.copy()
    new[1, 2] = 0.5
    new[3, 0] = -0.25
    rec, idx, val = build_record("w", "float32", base, new, reason="t")
    assert rec["changed_entries"] == 2
    assert idx.tolist() == [6, 12]
    assert rec["changed_rows"] == 2 and rec["changed_columns"] == 2
    assert rec["relative_change"] > 0


def test_build_record_bf16_ignores_sub_precision_noise():
    """A change far below bf16 resolution of the stored value is not recorded."""
    base = np.ones((2, 2), np.float32)
    new = base.copy()
    new[0, 0] = 1.0 + 1e-9              # rounds back to exactly 1.0 in bfloat16
    rec, idx, _ = build_record("w", "bfloat16", base, new)
    assert rec["changed_entries"] == 0
    assert idx.size == 0
    # ... but a change above bf16 resolution IS recorded
    new2 = base.copy()
    new2[0, 0] = 1.01
    rec2, idx2, _ = build_record("w", "bfloat16", base, new2)
    assert rec2["changed_entries"] == 1 and idx2.tolist() == [0]


def test_reconstruct_is_exact_and_hash_verified():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.1, (6, 5)).astype(np.float32)
    new = base.copy()
    new[2, 3] += 0.02
    new[0, 0] -= 0.01
    rec, idx, val = build_record("w", "bfloat16", base, new)
    out = reconstruct(rec, idx, val, base)
    assert cast.to_bf16_bits(out).tobytes() == cast.cast_to_dtype(new, "bfloat16").tobytes()


def test_reconstruct_detects_wrong_base():
    base = np.zeros((3, 3), np.float32)
    new = base.copy()
    new[0, 0] = 1.0
    rec, idx, val = build_record("w", "float32", base, new)
    with pytest.raises(ValueError):
        reconstruct(rec, idx, val, base + 1.0)


def test_store_save_load_update(backend, cfg, tmp_project):
    from aladdin.tensor_editor import TensorEditor

    store = DeltaStore(tmp_project["paths"]["deltas"])
    ed = TensorEditor(backend, cfg, store)
    name = "model.layers.0.mlp.down_proj.weight"
    res = ed.apply({"kind": "scale_columns", "tensor": name, "columns": [1], "alpha": 0.05},
                  reason="store test")
    eid = res["edit_id"]
    assert store.has(eid)
    man, arrays = store.load(eid)
    assert man["n_tensors"] == 1 and "t0_idx" in arrays
    assert man["status"] == "applied"
    store.update_status(eid, "accepted", {"note": "ok"})
    man2, _ = store.load(eid)
    assert man2["status"] == "accepted"
    assert [h["status"] for h in man2["status_history"]] == ["applied", "accepted"]
    rows = store.list_edits()
    assert rows and rows[0]["edit_id"] == eid
