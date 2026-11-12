"""Target-model backend: the genuine Qwen3.5-0.8B (or any HF model) via torch.

The loading path is Qwen3.5-aware: the architecture is detected from the model's
OWN config.json and loaded through the officially supported transformers class
(AutoModelForImageTextToText for the multimodal
Qwen3_5ForConditionalGeneration; AutoModelForCausalLM only for classic
`*ForCausalLM` configs). See backends/qwen_loader.py and docs/QWEN3_SETUP.md.

Nothing here fits anything to data: weights are read, edited in place with
direct tensor writes (`param.data.copy_`), and re-written to a checkpoint
directory. There is no objective function, no derivative pass and no solver -
see docs/METHOD.md and tests/test_no_training.py, which greps the whole tree for
any such call.

torch/transformers are imported lazily so the rest of the project runs on a CPU
box without them.
"""
from __future__ import annotations

import numpy as np

from .base import Backend, BackendError
from .qwen_loader import (LoaderCompatibilityError, _DT_MAP,
                          check_transformers_version, detect_architecture,
                          load_qwen35, transformers_version)


class HFTorchBackend(Backend):
    kind = "hf_torch"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        b = cfg["backend"]
        self.model_path = b.get("model_path") or ""
        if not self.model_path:
            raise BackendError(
                "backend.model_path is empty. Set ALADDIN_MODEL_PATH or run "
                "`python -m aladdin.cli download --model Qwen/Qwen3.5-0.8B`."
            )
        try:
            import torch  # noqa: F401
        except Exception as e:  # pragma: no cover - depends on env
            raise BackendError(
                "hf_torch backend needs torch + transformers. Install: "
                "pip install torch 'transformers>=4.57' accelerate safetensors. "
                f"Import error: {e}"
            ) from e
        self._torch = torch

        dtype = _DT_MAP.get(str(b.get("dtype", "float32")), "float32")
        self._dtype_name = dtype
        dev = str(b.get("device", "cpu"))
        if dev.startswith("cuda") and not torch.cuda.is_available():
            raise BackendError("backend.device=cuda but torch.cuda.is_available() is False")

        # ---- Qwen3.5-aware loading --------------------------------------
        self.arch = detect_architecture(self.model_path)   # from config.json
        v = check_transformers_version()
        self.transformers_version = v["installed"]
        if not v["ok"]:
            raise BackendError(v["error"])
        self._auto_class = ("AutoModelForImageTextToText" if self.arch["is_qwen3_5"]
                            or self.arch["is_multimodal"] else "AutoModelForCausalLM")
        try:
            model, tokenizer, self.load_info = load_qwen35(
                self.model_path, dtype=dtype, device=dev,
                trust_remote_code=bool(b.get("trust_remote_code", False)),
                revision=b.get("revision") or "")
        except LoaderCompatibilityError as e:
            raise BackendError(str(e)) from e
        self.model = model
        self.tokenizer = tokenizer
        self.device = dev
        # expose the TEXT tower's config as the primary config view
        tc = getattr(getattr(model, "config", None), "text_config", None)
        self.text_config = tc if tc is not None else getattr(model, "config", None)
        self._params = {n: p for n, p in self.model.named_parameters()}

    # --------------------------------------------------------------- identity
    def describe(self) -> dict:
        c = self.text_config
        return {
            "kind": self.kind,
            "model_path": self.model_path,
            "auto_class": self.load_info["auto_class"],
            "class_selection_reason": self.load_info["selection_reason"],
            "architectures": getattr(getattr(self.model, "config", None),
                                     "architectures", None),
            "model_type": getattr(getattr(self.model, "config", None),
                                  "model_type", None),
            "text_model_type": self.arch.get("text_model_type"),
            "is_multimodal": self.arch.get("is_multimodal"),
            "hidden_size": getattr(c, "hidden_size", None),
            "num_hidden_layers": getattr(c, "num_hidden_layers", None),
            "num_attention_heads": getattr(c, "num_attention_heads", None),
            "num_key_value_heads": getattr(c, "num_key_value_heads", None),
            "intermediate_size": getattr(c, "intermediate_size", None),
            "vocab_size": getattr(c, "vocab_size", None),
            "dtype": self._dtype_name,
            "device": self.device,
            "n_params": self.n_params(),
            "n_tensors": len(self._params),
            "torch": self._torch.__version__,
            "transformers": self.transformers_version,
            "note": "real pretrained weights; edits are direct tensor writes",
        }

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
            p.data.copy_(self._torch.from_numpy(np.ascontiguousarray(arr)).to(p.dtype).to(p.device))

    # ----------------------------------------------------------------- text
    def encode(self, text, add_special_tokens=True):
        return list(self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])

    def decode(self, ids):
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=True)

    def forward_logits(self, input_ids, capture=None):
        torch = self._torch
        ids = torch.tensor([list(map(int, input_ids))], device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=ids, output_hidden_states=capture is not None,
                             output_attentions=False, use_cache=False)
        logits = out.logits[0].to("cpu", dtype=torch.float32).numpy()
        if capture is not None:
            hs = out.hidden_states or ()
            for i, h in enumerate(hs):
                capture[f"hidden_states[{i}]"] = h[0].to("cpu", float).numpy()
            capture["final_hidden"] = capture.get(f"hidden_states[{len(hs) - 1}]")
        return logits

    def generate(self, prompt, max_new_tokens=48, temperature=0.0, top_k=0, seed=0):
        torch = self._torch
        tok = self.tokenizer
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        torch.manual_seed(int(seed))
        kw = dict(max_new_tokens=int(max_new_tokens), do_sample=bool(temperature and temperature > 0),
                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if temperature and temperature > 0:
            kw.update(temperature=float(temperature), top_k=int(top_k) if top_k else 0)
        with torch.no_grad():
            out = self.model.generate(ids, **kw)
        new = out[0][ids.shape[1]:]
        return tok.decode(new, skip_special_tokens=True).strip()
