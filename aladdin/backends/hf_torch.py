"""Target-model backend: the required edited WSP checkpoint through torch.

Loading is delegated to `qwen_loader`, which detects the architecture from the
checkpoint's own config.json, loads the TEXT tower with the official
``Qwen3_5ForCausalLM`` class and an explicit ``model.language_model.*`` ->
``model.*`` key remap, and verifies identity/geometry before returning. A
different repository id, a truncated checkpoint, or a key-layout mismatch is a
hard error, never a silent substitution.

Weights are read, edited with direct tensor writes (`param.data.copy_`), and
re-written to a separate checkpoint directory. There is no objective function,
no derivative pass and no solver - docs/METHOD.md and tests/test_no_training.py
enforce that. torch/transformers are imported lazily.
"""
from __future__ import annotations

import numpy as np

from ..model_identity import (REQUIRED_MODEL_ID, SNAPSHOT_METADATA_FILE)
from .base import Backend, BackendError
from .qwen_loader import (LoaderCompatibilityError, ModelIdentityError, _DT_MAP,
                          check_transformers_version, detect_architecture,
                          load_qwen35, read_model_config,
                          read_snapshot_metadata, verify_loaded_model)


class HFTorchBackend(Backend):
    kind = "hf_torch"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        b = cfg["backend"]
        self.model_path = b.get("model_path") or ""
        self.model_id = b.get("model_id") or REQUIRED_MODEL_ID
        if not self.model_path:
            raise BackendError(
                "backend.model_path is empty. Download the required checkpoint and "
                "set it:\n"
                f"  python -m aladdin.cli download --model {self.model_id}\n"
                "  export ALADDIN_MODEL_PATH=\"$PWD/models/"
                f"{self.model_id.split('/')[-1]}\"")
        try:
            import torch  # noqa: F401
        except Exception as e:  # pragma: no cover - depends on env
            raise BackendError(
                "hf_torch backend needs torch + transformers. Install: "
                "pip install torch 'transformers>=4.57' accelerate safetensors "
                f"huggingface_hub. Import error: {e}"
            ) from e
        self._torch = torch

        dtype = _DT_MAP.get(str(b.get("dtype", "float32")), "float32")
        self._dtype_name = dtype
        dev = str(b.get("device", "cpu"))
        if dev.startswith("cuda") and not torch.cuda.is_available():
            raise BackendError("backend.device=cuda but torch.cuda.is_available() is False")

        v = check_transformers_version()
        self.transformers_version = v["installed"]
        if not v["ok"]:
            raise BackendError(v["error"])

        # ---- architecture + required-checkpoint identity -------------------
        self.arch = detect_architecture(self.model_path,
                                        expected_model_id=self.model_id)
        self.load_mode = str(b.get("load_mode", "text")).lower()
        try:
            model, tokenizer, self.load_info = load_qwen35(
                self.model_path, dtype=dtype, device=dev,
                trust_remote_code=bool(b.get("trust_remote_code", False)),
                revision=b.get("revision") or "", mode=self.load_mode,
                expected_model_id=self.model_id,
                device_map=b.get("device_map"))
        except (LoaderCompatibilityError, ModelIdentityError) as e:
            raise BackendError(str(e)) from e
        self.model = model
        self.tokenizer = tokenizer
        self.device = dev
        tc = getattr(getattr(model, "config", None), "text_config", None)
        self.text_config = tc if tc is not None else getattr(model, "config", None)
        self._params = {n: p for n, p in self.model.named_parameters()}
        if not self._params:
            raise BackendError("loaded model exposes no parameters")

    # --------------------------------------------------------------- identity
    def describe(self) -> dict:
        c = self.text_config
        meta = read_snapshot_metadata(self.model_path) or {}
        return {
            "kind": self.kind,
            "model_id": self.model_id,
            "recorded_model_id": self.arch.get("recorded_model_id")
                                 or meta.get("repo_id"),
            "revision": self.arch.get("revision") or meta.get("sha"),
            "identity_verified": bool(self.load_info.get("identity_verified")),
            "model_path": self.model_path,
            "load_mode": self.load_mode,
            "auto_class": self.load_info["auto_class"],
            "class_selection_reason": self.load_info["selection_reason"],
            "architectures": getattr(getattr(self.model, "config", None),
                                     "architectures", None),
            "model_type": self.arch.get("model_type"),
            "text_model_type": self.arch.get("text_model_type"),
            "is_multimodal_checkpoint": self.arch.get("is_multimodal"),
            "loaded_class": self.load_info.get("loaded_class"),
            "hidden_size": getattr(c, "hidden_size", None),
            "num_hidden_layers": getattr(c, "num_hidden_layers", None),
            "num_attention_heads": getattr(c, "num_attention_heads", None),
            "num_key_value_heads": getattr(c, "num_key_value_heads", None),
            "intermediate_size": getattr(c, "intermediate_size", None),
            "vocab_size": getattr(c, "vocab_size", None),
            "embedding_shape": self.load_info.get("embedding_shape"),
            "lm_head_shape": self.load_info.get("lm_head_shape"),
            "tied_embeddings": self.load_info.get("tied_embeddings"),
            "dtype": self._dtype_name,
            "device": self.device,
            "n_params": self.n_params(),
            "n_tensors": len(self._params),
            "torch": self._torch.__version__,
            "transformers": self.transformers_version,
            "weights": self.load_info.get("weights"),
            "tokenizer": self.load_info.get("tokenizer"),
            "verification": self.load_info.get("verification"),
            "note": "required edited WSP checkpoint; edits are direct tensor writes",
        }

    def verify(self) -> dict:
        """Re-run identity/geometry verification against the live module tree."""
        return verify_loaded_model(
            self.model, self.model_path, self.arch,
            expected={"model_id": self.model_id,
                      "hidden_size": self.arch.get("text_hidden_size"),
                      "num_hidden_layers": self.arch.get("text_num_hidden_layers"),
                      "vocab_size": self.arch.get("text_vocab_size"),
                      "tie_word_embeddings": self.arch.get("tie_word_embeddings")},
            mode=self.load_mode)

    def param_names(self):
        return sorted(self._params)

    def shape_of(self, name):
        return tuple(int(x) for x in self._params[name].shape)

    def dtype_of(self, name):
        return str(self._params[name].dtype).replace("torch.", "")

    # ---------------------------------------------------------------- tensors
    def get_tensor(self, name):
        p = self._params[name]
        return p.detach().to("cpu", dtype=self._torch.float32).numpy().copy()

    def raw_bytes(self, name):
        p = self._params[name].detach()
        if p.dtype == self._torch.bfloat16:
            arr = p.contiguous().view(self._torch.int16).cpu().numpy()
        else:
            arr = p.contiguous().cpu().numpy()
        return np.ascontiguousarray(arr).tobytes()

    def set_tensor(self, name, value):
        p = self._params[name]
        arr = np.asarray(value)
        if tuple(arr.shape) != tuple(p.shape):
            raise ValueError(f"{name}: shape {arr.shape} != {tuple(p.shape)}")
        with self._torch.no_grad():
            p.data.copy_(self._torch.from_numpy(np.ascontiguousarray(arr))
                         .to(p.dtype).to(p.device))

    # ----------------------------------------------------------------- text
    def encode(self, text, add_special_tokens=True):
        return list(self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])

    def decode(self, ids):
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=True)

    def apply_chat_template(self, prompt: str) -> str:
        """Render one user turn with the checkpoint's own chat template."""
        msgs = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def forward_logits(self, input_ids, capture=None):
        torch = self._torch
        ids = torch.tensor([list(map(int, input_ids))], device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=ids,
                             output_hidden_states=capture is not None,
                             output_attentions=False, use_cache=False)
        logits = out.logits[0].to("cpu", dtype=torch.float32).numpy()
        if capture is not None:
            hs = getattr(out, "hidden_states", None) or ()
            for i, h in enumerate(hs):
                capture[f"hidden_states[{i}]"] = h[0].to("cpu", dtype=torch.float32).numpy()
            if hs:
                capture["final_hidden"] = capture[f"hidden_states[{len(hs) - 1}]"]
        return logits

    def generate(self, prompt, max_new_tokens=48, temperature=0.0, top_k=0, seed=0):
        torch = self._torch
        tok = self.tokenizer
        text = self.apply_chat_template(prompt)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        torch.manual_seed(int(seed))
        kw = dict(max_new_tokens=int(max_new_tokens),
                  do_sample=bool(temperature and temperature > 0),
                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if temperature and temperature > 0:
            kw.update(temperature=float(temperature),
                      top_k=int(top_k) if top_k else 0)
        with torch.no_grad():
            out = self.model.generate(ids, **kw)
        new = out[0][ids.shape[1]:]
        return tok.decode(new, skip_special_tokens=True).strip()
