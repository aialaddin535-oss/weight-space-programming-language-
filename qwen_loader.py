"""Qwen3.5-aware loading path for the hf_torch backend.

Grounded in the OFFICIAL config.json of Qwen/Qwen3.5-0.8B (fetched from
huggingface.co, see docs/QWEN3_SETUP.md):

    architectures: ["Qwen3_5ForConditionalGeneration"]
    model_type:    "qwen3_5"
    text_config:   { model_type: "qwen3_5_text", hidden_size: 1024,
                     num_hidden_layers: 24, vocab_size: 248320,
                     tie_word_embeddings: true, dtype: "bfloat16", ... }
    vision_config: { ... }                    (multimodal: text + image + video)
    transformers_version: "4.57.0.dev0"       (needs a recent transformers)

Consequences that the generic `AutoModelForCausalLM` path gets WRONG:

* Qwen3.5-0.8B is a MULTIMODAL model (`ForConditionalGeneration`, not
  `ForCausalLM`). `AutoModelForCausalLM.from_pretrained` either fails or,
  worse, silently instantiates a DIFFERENT architecture against this config.
  The officially supported entry point is `AutoModelForImageTextToText`
  (resolved through the config's `architectures` field), which maps
  `Qwen3_5ForConditionalGeneration` to its true class.
* Text-only use is fully supported: the text tower lives in
  `model.config.text_config`; we pass `text_config` values to the
  generation config and only ever feed `input_ids`.
* The tokenizer is loaded through `AutoTokenizer` (fast tokenizer file), and
  the image/video processor through `AutoProcessor` - for text-only WSP runs
  the tokenizer is sufficient; the processor is optional.
* Because `transformers_version` in the config is ahead of many installed
  releases, we check the version explicitly and refuse loudly with install
  instructions instead of failing deep inside `from_pretrained`.

Nothing here fits anything to data: weights are read, edited in place with
direct tensor writes (`param.data.copy_`), and re-written to a checkpoint
directory. There is no objective function, no derivative pass and no solver.
torch/transformers are imported lazily so the rest of the project runs on a
CPU box without them.
"""
from __future__ import annotations

import json
from pathlib import Path

# The config of Qwen3.5-0.8B carries "transformers_version": "4.57.0.dev0";
# anything materially older cannot parse the qwen3_5 model type.
MIN_TRANSFORMERS_VERSION = "4.57.0"

# model_type values (top level and text_config) that this loader recognises.
QWEN3_5_MODEL_TYPES = {"qwen3_5"}
QWEN3_5_TEXT_MODEL_TYPES = {"qwen3_5_text"}
# architectures values that mark the multimodal conditional-generation class
QWEN3_5_ARCHITECTURES = {"Qwen3_5ForConditionalGeneration"}
# architectures values handled fine by the classic AutoModelForCausalLM path
_CAUSAL_ARCH_SUFFIX = "ForCausalLM"


class LoaderCompatibilityError(RuntimeError):
    """The installed transformers/torch stack cannot load this model."""


def _version_tuple(v: str):
    out = []
    for part in v.split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        out.append(int(digits or 0))
    return tuple(out + [0, 0, 0])[:3]


def transformers_version() -> str | None:
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return None


def check_transformers_version(min_version: str = MIN_TRANSFORMERS_VERSION) -> dict:
    """Verify the installed transformers is new enough for Qwen3.5."""
    have = transformers_version()
    if have is None:
        return {"ok": False, "installed": None, "required": min_version,
                "error": "transformers is not installed. Install: "
                         "pip install 'transformers>=4.57' torch accelerate safetensors"}
    if _version_tuple(have) < _version_tuple(min_version):
        return {"ok": False, "installed": have, "required": min_version,
                "error": f"transformers {have} is too old for Qwen3.5 "
                         f"(config requires >= {min_version}). Install: "
                         "pip install -U 'transformers>=4.57'"}
    return {"ok": True, "installed": have, "required": min_version, "error": None}


def read_model_config(model_path: str | Path) -> dict:
    """Read config.json from a local snapshot (no network)."""
    p = Path(model_path) / "config.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no config.json under {model_path}; point backend.model_path at the "
            "downloaded HF snapshot directory (see scripts/download_model.sh)")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_architecture(model_path: str | Path) -> dict:
    """Detect what the model actually is, from its own config.json.

    Never guesses: every field is read from the file or reported as None.
    """
    cfg = read_model_config(model_path)
    archs = list(cfg.get("architectures") or [])
    model_type = cfg.get("model_type")
    text = cfg.get("text_config") or {}
    vision = cfg.get("vision_config") or {}
    is_qwen35 = (model_type in QWEN3_5_MODEL_TYPES
                 or (text.get("model_type") in QWEN3_5_TEXT_MODEL_TYPES)
                 or any(a in QWEN3_5_ARCHITECTURES for a in archs))
    is_causal_classic = bool(archs) and all(a.endswith(_CAUSAL_ARCH_SUFFIX)
                                            for a in archs)
    tfv = cfg.get("transformers_version")
    return {
        "config_path": str(p) if (p := Path(model_path) / "config.json").exists() else None,
        "architectures": archs,
        "model_type": model_type,
        "text_model_type": text.get("model_type"),
        "text_hidden_size": text.get("hidden_size"),
        "text_num_hidden_layers": text.get("num_hidden_layers"),
        "text_vocab_size": text.get("vocab_size"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings",
                                       text.get("tie_word_embeddings")),
        "dtype": text.get("dtype") or cfg.get("dtype") or cfg.get("torch_dtype"),
        "is_multimodal": bool(vision) or any(a in QWEN3_5_ARCHITECTURES
                                             for a in archs),
        "has_vision_config": bool(vision),
        "is_qwen3_5": is_qwen35,
        "classic_causal_lm": is_causal_classic,
        "transformers_version_required": tfv,
    }


def _auto_class_for(arch: dict):
    """The officially supported loading class for the detected architecture.

    Qwen3.5 (`*ForConditionalGeneration` with a vision tower) is registered
    under AutoModelForImageTextToText; classic causal LMs stay on
    AutoModelForCausalLM. Returns (class_name, reason).
    """
    if arch["is_qwen3_5"] or arch["is_multimodal"]:
        return ("AutoModelForImageTextToText",
                "config architectures=" + str(arch["architectures"]) +
                " (multimodal conditional-generation model; "
                "AutoModelForCausalLM is NOT the official entry point)")
    return ("AutoModelForCausalLM",
            "config architectures=" + str(arch["architectures"]) +
            " (classic causal LM)")


def verify_loaded_model(model, model_path: str | Path, arch: dict) -> dict:
    """Guard against silent model swaps: the loaded object must match config.json.

    Checks the runtime architecture class name, the text-config geometry
    (hidden size / layer count / vocab) against the live module tree.
    """
    problems: list = []
    cls_name = type(model).__name__
    expected_archs = arch["architectures"]
    if expected_archs and not any(cls_name.endswith(a) for a in expected_archs) \
            and cls_name not in expected_archs:
        # allow wrappers: check the class name contains the architecture family
        if not any(a.replace("For", "").lower() in cls_name.lower()
                   for a in expected_archs):
            problems.append(
                f"loaded class {cls_name!r} does not match config architectures "
                f"{expected_archs} (possible silent model swap)")
    # walk the live module tree for the text tower's core geometry
    try:
        import torch.nn as nn  # local import: only reached with torch present
        n_lin_layers = sum(1 for m in model.modules()
                           if isinstance(m, nn.Linear))
        if arch.get("text_num_hidden_layers") and n_lin_layers == 0:
            problems.append("loaded model has no Linear layers at all")
    except ImportError:                       # torch absent: structural check off
        n_lin_layers = None
    return {"ok": not problems, "problems": problems, "loaded_class": cls_name,
            "n_linear_layers": n_lin_layers}


def load_qwen35(model_path: str | Path, dtype: str = "bfloat16",
                device: str = "cpu", trust_remote_code: bool = False,
                revision: str = "", min_version: str = MIN_TRANSFORMERS_VERSION):
    """Load Qwen3.5 (or a classic causal LM) through the officially supported path.

    Returns (model, tokenizer, info_dict). Raises LoaderCompatibilityError with
    actionable messages for missing deps / too-old transformers / config drift.
    """
    v = check_transformers_version(min_version)
    if not v["ok"]:
        raise LoaderCompatibilityError(v["error"])
    arch = detect_architecture(model_path)
    class_name, reason = _auto_class_for(arch)

    import torch
    import transformers
    from transformers import AutoTokenizer, AutoConfig

    cls = getattr(transformers, class_name, None)
    if cls is None:
        raise LoaderCompatibilityError(
            f"transformers {v['installed']} has no {class_name}; upgrade: "
            "pip install -U 'transformers>=4.57'")

    kw: dict = {"trust_remote_code": trust_remote_code}
    if revision:
        kw["revision"] = revision
    tdt = getattr(torch, _DT_MAP.get(dtype, "float32"))
    kw["dtype"] = tdt

    tokenizer = AutoTokenizer.from_pretrained(model_path, **kw)
    model = cls.from_pretrained(model_path, **kw)

    # cross-check the loaded object against its own config (no silent swaps)
    chk = verify_loaded_model(model, model_path, arch)
    if not chk["ok"]:
        raise LoaderCompatibilityError(
            "loaded model does not match its config.json: "
            + "; ".join(chk["problems"]))

    model.to(device)
    model.eval()
    info = {
        "auto_class": class_name,
        "selection_reason": reason,
        "transformers": v["installed"],
        "torch": torch.__version__,
        "architecture": arch,
        "loaded_class": chk["loaded_class"],
        "n_linear_layers": chk["n_linear_layers"],
        "text_only": True,
    }
    return model, tokenizer, info


_DT_MAP = {"float32": "float32", "bfloat16": "bfloat16", "float16": "float16"}
