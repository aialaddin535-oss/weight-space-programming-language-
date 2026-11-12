"""A real, small, numpy-only decoder used for CPU infrastructure tests.

It is *not* a pretrained language model: weights are randomly initialised from a
seed, so its text output is gibberish and its Hindi/Hinglish quality scores are
near zero. That is intentional and is reported as such - the point of this
backend is to prove the whole pipeline (inspection -> forward pass -> activation
capture -> bounded tensor edit -> sparse delta -> checkpoint -> evaluation ->
accept/reject -> rollback) works end to end without a GPU or an 8 GB download.

Architecture: decoder-only transformer, RMSNorm, causal multi-head attention with
GQA, SwiGLU MLP, untied LM head. Tensor names mirror the Qwen3.5 layout
(`model.layers.N.mlp.down_proj.weight`, ...) so tools/prompts transfer unchanged.
"""
from __future__ import annotations

import numpy as np

from .base import Backend


def _rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class TinyNumpyBackend(Backend):
    kind = "tiny_numpy"
    dtype_str = "float32"

    def __init__(self, cfg: dict):
        t = cfg["tiny"]
        self.cfg = cfg
        self.hidden_size = int(t["hidden_size"])
        self.num_layers = int(t["num_layers"])
        self.num_heads = int(t["num_attention_heads"])
        self.num_kv_heads = int(t["num_key_value_heads"])
        self.intermediate_size = int(t["intermediate_size"])
        self.vocab_size = int(t["vocab_size"])
        self.max_pos = int(t["max_position_embeddings"])
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.head_dim = self.hidden_size // self.num_heads
        self._seed = int(cfg["tiny"].get("seed", 0))
        self._shapes: dict[str, tuple] = {}
        self._tensors: dict[str, np.ndarray] = {}
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        rng = np.random.default_rng(self._seed)
        d, ffn, V = self.hidden_size, self.intermediate_size, self.vocab_size
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        def reg(name, shape, scale=0.02, ones=False):
            arr = np.ones(shape, np.float32) if ones else rng.normal(0.0, scale, shape).astype(np.float32)
            self._tensors[name] = arr
            self._shapes[name] = tuple(arr.shape)

        reg("model.embed_tokens.weight", (V, d))
        for i in range(self.num_layers):
            p = f"model.layers.{i}."
            reg(p + "input_layernorm.weight", (d,), ones=True)
            reg(p + "post_attention_layernorm.weight", (d,), ones=True)
            reg(p + "self_attn.q_proj.weight", (nh * hd, d))
            reg(p + "self_attn.k_proj.weight", (nkv * hd, d))
            reg(p + "self_attn.v_proj.weight", (nkv * hd, d))
            reg(p + "self_attn.o_proj.weight", (d, nh * hd))
            reg(p + "mlp.gate_proj.weight", (ffn, d))
            reg(p + "mlp.up_proj.weight", (ffn, d))
            reg(p + "mlp.down_proj.weight", (d, ffn))
        reg("model.norm.weight", (d,), ones=True)
        reg("lm_head.weight", (V, d))

    # --------------------------------------------------------------- identity
    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "model_path": "<in-memory tiny model, random init>",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_pos,
            "dtype": self.dtype_str,
            "n_params": self.n_params(),
            "note": "randomly initialised test model: proves infrastructure, not language quality",
        }

    def param_names(self):
        return sorted(self._tensors)

    def shape_of(self, name):
        return self._shapes[name]

    def dtype_of(self, name):
        return self.dtype_str

    # ---------------------------------------------------------------- tensors
    def get_tensor(self, name):
        return np.array(self._tensors[name], dtype=np.float32, copy=True)

    def raw_bytes(self, name):
        return np.ascontiguousarray(self._tensors[name]).tobytes()

    def set_tensor(self, name, value):
        arr = np.asarray(value)
        if tuple(arr.shape) != self._shapes[name]:
            raise ValueError(f"{name}: shape {arr.shape} != {self._shapes[name]}")
        self._tensors[name] = arr.astype(np.float32)

    # ----------------------------------------------------------------- text
    # byte-level tokenizer: token id == UTF-8 byte value (0..255) + 256 = EOS
    EOS = 256
    BOS = 257

    def encode(self, text, add_special_tokens=True):
        ids = [self.BOS] + [int(b) for b in text.encode("utf-8")][: self.max_pos - 2]
        return ids

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            i = int(i)
            if i in (self.EOS, self.BOS):
                continue
            if 0 <= i < 256:
                out.append(i)
        return out.decode("utf-8", errors="replace")

    # -------------------------------------------------------------- inference
    def forward_logits(self, input_ids, capture=None):
        ids = np.asarray(input_ids, dtype=np.int64)
        T = ids.shape[0]
        if T == 0:
            raise ValueError("empty input")
        if T > self.max_pos:
            ids = ids[-self.max_pos:]
            T = ids.shape[0]
        W = self._tensors
        d, nh, nkv, hd = self.hidden_size, self.num_heads, self.num_kv_heads, self.head_dim
        scale = 1.0 / np.sqrt(hd)

        x = W["model.embed_tokens.weight"][ids].astype(np.float32)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)

        for i in range(self.num_layers):
            p = f"model.layers.{i}."
            h = _rmsnorm(x, W[p + "input_layernorm.weight"])
            q = (h @ W[p + "self_attn.q_proj.weight"].T).reshape(T, nh, hd)
            k = (h @ W[p + "self_attn.k_proj.weight"].T).reshape(T, nkv, hd)
            v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(T, nkv, hd)
            rep = nh // nkv
            k = np.repeat(k, rep, axis=1)
            v = np.repeat(v, rep, axis=1)
            # [nh, T, hd]
            qh, kh, vh = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)
            scores = np.einsum("htd,hSd->htS", qh, kh) * scale
            scores = np.where(mask[None, :, :], -1e9, scores)
            scores = scores - scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs = probs / probs.sum(axis=-1, keepdims=True)
            ctx = np.einsum("htS,hSd->htd", probs, vh)
            attn = ctx.transpose(1, 0, 2).reshape(T, nh * hd)
            x = x + attn @ W[p + "self_attn.o_proj.weight"].T
            h2 = _rmsnorm(x, W[p + "post_attention_layernorm.weight"])
            gate = _silu(h2 @ W[p + "mlp.gate_proj.weight"].T)
            up = h2 @ W[p + "mlp.up_proj.weight"].T
            x = x + (gate * up) @ W[p + "mlp.down_proj.weight"].T
            if capture is not None:
                capture[f"layer{i}.attn_out"] = attn.astype(np.float32)
                capture[f"layer{i}.mlp_out"] = (gate * up).astype(np.float32)
                capture[f"layer{i}.hidden"] = x.astype(np.float32)

        x = _rmsnorm(x, W["model.norm.weight"])
        logits = x @ W["lm_head.weight"].T
        if capture is not None:
            capture["final_hidden"] = x.astype(np.float32)
        return logits.astype(np.float32)

    def generate(self, prompt, max_new_tokens=48, temperature=0.0, top_k=0, seed=0):
        rng = np.random.default_rng(seed)
        ids = self.encode(prompt)[-self.max_pos:]
        for _ in range(int(max_new_tokens)):
            logits = self.forward_logits(ids)[-1]
            if temperature and temperature > 0:
                z = logits.astype(np.float64) / float(temperature)
                if top_k and top_k > 0:
                    thr = np.sort(z)[-int(top_k)]
                    z = np.where(z < thr, -np.inf, z)
                z = z - z.max()
                p = np.exp(z)
                p = p / p.sum()
                nxt = int(rng.choice(len(p), p=p))
            else:
                nxt = int(np.argmax(logits))
            ids.append(nxt)
            if nxt == self.EOS:
                break
        return self.decode(ids[len(self.encode(prompt)):])
