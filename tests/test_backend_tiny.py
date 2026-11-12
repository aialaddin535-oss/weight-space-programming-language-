"""The tiny CPU backend must behave like a real transformer."""
import numpy as np

from aladdin.backends import build_backend


def test_shapes_and_names(backend):
    names = backend.param_names()
    assert "model.embed_tokens.weight" in names
    assert "model.layers.0.mlp.down_proj.weight" in names
    assert "lm_head.weight" in names
    assert backend.shape_of("model.embed_tokens.weight") == (300, 32)
    assert backend.shape_of("model.layers.0.attn.q_proj.weight" if "model.layers.0.attn.q_proj.weight" in names
                            else "model.layers.0.self_attn.q_proj.weight") == (32, 32)


def test_forward_logits_shape_and_finiteness(backend):
    ids = backend.encode("mujhe samaj nahi aya")
    logits = backend.forward_logits(ids)
    assert logits.shape == (len(ids), 300)
    assert np.isfinite(logits).all()


def test_causal_masking(backend):
    """Changing a later token must not change an earlier position's logits."""
    a = backend.encode("mujhe samaj")
    b = a + backend.encode(" nahi aya", add_special_tokens=False)
    la = backend.forward_logits(a)
    lb = backend.forward_logits(b)
    assert np.allclose(la, lb[:len(a)], atol=1e-5)


def test_generation_is_deterministic(backend):
    o1 = backend.generate("iska matlab kya hai", max_new_tokens=8, temperature=0.0)
    o2 = backend.generate("iska matlab kya hai", max_new_tokens=8, temperature=0.0)
    assert o1 == o2


def test_capture_produces_activations(backend):
    cap = {}
    backend.forward_logits(backend.encode("test"), capture=cap)
    assert "final_hidden" in cap
    assert "layer0.hidden" in cap
    assert cap["final_hidden"].ndim == 2


def test_set_tensor_roundtrip_and_hash_change(backend):
    n = "model.layers.0.mlp.down_proj.weight"
    h0 = backend.tensor_hash(n)
    w = backend.get_tensor(n)
    w[0, 0] += 0.5
    backend.set_tensor(n, w)
    assert backend.tensor_hash(n) != h0
    assert abs(backend.get_tensor(n)[0, 0] - w[0, 0]) < 1e-6


def test_build_backend_factory(backend, cfg):
    b2 = build_backend(cfg)
    assert b2.describe()["kind"] == "tiny_numpy"
    assert b2.n_params() == backend.n_params()
