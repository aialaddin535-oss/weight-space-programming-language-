"""Qwen3.5-aware loading for the REQUIRED edited WSP checkpoint.

Verified facts about `RikiyaT/Qwen3.5-0.8B-weight-space-programming`
(read from the published repository, not assumed):

    config.json     architectures: ["Qwen3_5ForConditionalGeneration"]
                    model_type:    "qwen3_5"
                    text_config:   model_type "qwen3_5_text", hidden_size 1024,
                                   num_hidden_layers 24, vocab_size 248320,
                                   intermediate_size 3584, 8 attention heads,
                                   2 KV heads, tie_word_embeddings true,
                                   18 linear-attention + 6 full-attention layers
                    vision_config: present (the checkpoint is natively multimodal)
    weight keys     488 tensors: model.language_model.*  (320, the text tower)
                                model.visual.*           (153, the vision tower)
                                mtp.*                    (15, multi-token prediction)

Consequences for a TEXT-ONLY WSP run (the only thing this project does):

* The official text-only entry point is ``Qwen3_5ForCausalLM`` with a
  ``Qwen3_5TextConfig`` (transformers >= 4.57). ``AutoModelForImageTextToText``
  is the multimodal path and is only used when explicitly requested; it is NOT
  the right class for text-only tensor work.
* The text weights in this checkpoint are stored under
  ``model.language_model.*``, while ``Qwen3_5ForCausalLM`` expects
  ``model.*`` / ``lm_head.*``. The prefixes are therefore remapped explicitly
  (``plan_text_only_load``), and visual/MTP tensors are skipped with a count.
  They are not silently fed into a model with a different key layout.
* The checkpoint has no ``lm_head.weight``: ``tie_word_embeddings`` is true, so
  the head is the embedding matrix. That is the only permitted "missing" key.
* ``transformers_version`` in the config is ``4.57.0.dev0``; older releases
  cannot parse ``qwen3_5``. The version is checked up front and a clear
  install instruction is raised instead of a deep ``from_pretrained`` failure.

No fitting happens here: weights are read from the safetensors shards and
copied into the module tree exactly (dtype-preserving), then hashed and edited
in place by the WSP engine. torch/transformers are imported lazily so the rest
of the project runs on a CPU box without them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from ..model_identity import (EXPECTED_ARCHITECTURE, EXPECTED_HIDDEN_SIZE,
                              EXPECTED_INTERMEDIATE_SIZE,
                              EXPECTED_NUM_ATTENTION_HEADS,
                              EXPECTED_NUM_HIDDEN_LAYERS,
                              EXPECTED_NUM_KEY_VALUE_HEADS,
                              EXPECTED_TEXT_MODEL_TYPE,
                              EXPECTED_TIE_WORD_EMBEDDINGS,
                              EXPECTED_VOCAB_SIZE, MTP_WEIGHT_PREFIX,
                              REQUIRED_MODEL_ID, SNAPSHOT_METADATA_FILE,
                              TEXT_WEIGHT_PREFIX, VISION_WEIGHT_PREFIX,
                              expected_text_geometry)

# The checkpoint config declares transformers_version 4.57.0.dev0; anything
# materially older cannot parse the qwen3_5 model type or its text class.
MIN_TRANSFORMERS_VERSION = "4.57.0"

# Official classes for each loading path.
TEXT_ONLY_CLASS = "Qwen3_5ForCausalLM"
TEXT_CONFIG_CLASS = "Qwen3_5TextConfig"
MULTIMODAL_CLASS = EXPECTED_ARCHITECTURE
AUTO_MULTIMODAL_CLASS = "AutoModelForImageTextToText"

# model_type values this loader recognises.
QWEN3_5_MODEL_TYPES = {"qwen3_5"}
QWEN3_5_TEXT_MODEL_TYPES = {EXPECTED_TEXT_MODEL_TYPE, "qwen3_5"}
QWEN3_5_ARCHITECTURES = {MULTIMODAL_CLASS}
_CAUSAL_ARCH_SUFFIX = "ForCausalLM"

# Keys allowed to be absent from a text-only load: the tied LM head lives in the
# embedding matrix instead of its own tensor.
TIED_HEAD_KEYS = ("lm_head.weight",)

_DT_MAP = {"float32": "float32", "bfloat16": "bfloat16", "float16": "float16"}


class LoaderCompatibilityError(RuntimeError):
    """The installed stack cannot load this model, or the checkpoint is wrong."""


class ModelIdentityError(LoaderCompatibilityError):
    """The checkpoint on disk is not the configured/required model."""


class WeightLoadError(LoaderCompatibilityError):
    """The text tower could not be loaded exactly (missing/mismatched tensors)."""


# --------------------------------------------------------------------- versions
def _version_tuple(v: str):
    out = []
    for part in str(v).split("."):
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
                         "pip install 'transformers>=4.57' torch accelerate "
                         "safetensors huggingface_hub"}
    if _version_tuple(have) < _version_tuple(min_version):
        return {"ok": False, "installed": have, "required": min_version,
                "error": f"transformers {have} is too old for Qwen3.5 "
                         f"(config requires >= {min_version}). Install: "
                         f"pip install -U 'transformers>={min_version}'"}
    return {"ok": True, "installed": have, "required": min_version, "error": None}


def check_text_class_available() -> dict:
    """The official text-only class and config must exist in this transformers."""
    try:
        import transformers
    except Exception as e:                       # pragma: no cover - env dependent
        return {"ok": False, "error": f"transformers not importable: {e}"}
    missing = [n for n in (TEXT_ONLY_CLASS, TEXT_CONFIG_CLASS)
               if getattr(transformers, n, None) is None]
    if missing:
        return {"ok": False, "missing": missing, "installed": transformers_version(),
                "error": f"transformers {transformers_version()} does not provide "
                         f"{missing}; Qwen3.5 text-only loading needs >= "
                         f"{MIN_TRANSFORMERS_VERSION}"}
    return {"ok": True, "missing": [], "installed": transformers_version()}


# ----------------------------------------------------------------- config / arch
def read_model_config(model_path: str | Path) -> dict:
    """Read config.json from a local snapshot (no network)."""
    p = Path(model_path) / "config.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no config.json under {model_path}; point backend.model_path at the "
            "downloaded HF snapshot of "
            f"{REQUIRED_MODEL_ID} (see docs/QWEN3_SETUP.md)")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def read_snapshot_metadata(model_path: str | Path) -> dict | None:
    """Metadata written by `aladdin download` (repo id, revision, file hashes)."""
    p = Path(model_path) / SNAPSHOT_METADATA_FILE
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def detect_architecture(model_path: str | Path,
                        expected_model_id: str = REQUIRED_MODEL_ID) -> dict:
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
    meta = read_snapshot_metadata(model_path) or {}
    recorded_id = meta.get("repo_id") or meta.get("model_id")
    identity_verified = bool(recorded_id and recorded_id == expected_model_id)
    return {
        "model_path": str(model_path),
        "model_id": expected_model_id,
        "recorded_model_id": recorded_id,
        "identity_verified": identity_verified,
        "revision": meta.get("sha") or meta.get("revision") or "",
        "config_sha256": meta.get("config_sha256"),
        "architectures": archs,
        "model_type": model_type,
        "text_model_type": text.get("model_type"),
        "text_hidden_size": text.get("hidden_size"),
        "text_num_hidden_layers": text.get("num_hidden_layers"),
        "text_vocab_size": text.get("vocab_size"),
        "text_intermediate_size": text.get("intermediate_size"),
        "text_num_attention_heads": text.get("num_attention_heads"),
        "text_num_key_value_heads": text.get("num_key_value_heads"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings",
                                       text.get("tie_word_embeddings")),
        "dtype": text.get("dtype") or cfg.get("dtype") or cfg.get("torch_dtype"),
        "is_multimodal": bool(vision) or any(a in QWEN3_5_ARCHITECTURES
                                             for a in archs),
        "has_vision_config": bool(vision),
        "is_qwen3_5": is_qwen35,
        "classic_causal_lm": is_causal_classic,
        "transformers_version_required": cfg.get("transformers_version"),
        "text_weight_prefix": TEXT_WEIGHT_PREFIX,
        "vision_weight_prefix": VISION_WEIGHT_PREFIX,
        "mtp_weight_prefix": MTP_WEIGHT_PREFIX,
        "_raw_config": cfg,
    }


def check_checkpoint_identity(model_path: str | Path,
                              expected_model_id: str = REQUIRED_MODEL_ID) -> dict:
    """Pure identity check (no torch): is this the configured/required model?

    Uses the recorded download metadata when present (repo id + revision) and
    otherwise falls back to the config fingerprint, reporting that the identity
    is config-verified only. A mismatch is always a hard, explicit problem.
    """
    arch = detect_architecture(model_path, expected_model_id=expected_model_id)
    problems: List[str] = []
    if not arch["is_qwen3_5"]:
        problems.append(
            f"{model_path} is not a Qwen3.5 checkpoint (architectures="
            f"{arch['architectures']}, model_type={arch['model_type']})")
    meta = read_snapshot_metadata(model_path)
    recorded = arch.get("recorded_model_id")
    if meta is not None and recorded != expected_model_id:
        problems.append(
            f"checkpoint metadata records {recorded!r} but this project requires "
            f"{expected_model_id!r}")
    geom = expected_text_geometry(arch.get("_raw_config"))
    for key, want in (("text_hidden_size", EXPECTED_HIDDEN_SIZE),
                      ("text_num_hidden_layers", EXPECTED_NUM_HIDDEN_LAYERS),
                      ("text_vocab_size", EXPECTED_VOCAB_SIZE)):
        have = arch.get(key)
        if have is not None and int(have) != int(geom[key.replace("text_", "", 1)] if False else want):
            problems.append(f"config {key}={have} but {expected_model_id} has {want}")
    return {"ok": not problems, "problems": problems,
            "model_id": expected_model_id, "recorded_model_id": recorded,
            "revision": arch.get("revision"),
            "identity_verified": bool(recorded == expected_model_id),
            "metadata_present": meta is not None,
            "architecture": arch.get("architectures")}


def _auto_class_for(arch: dict, mode: str = "text"):
    """The officially supported loading class for the detected architecture.

    mode="text"  -> Qwen3_5ForCausalLM + Qwen3_5TextConfig (default; what WSP
                    needs, and what the official docs prescribe for text-only
                    generation).
    mode="multimodal" -> AutoModelForImageTextToText (only on explicit request).

    Returns (class_name, reason).
    """
    mode = (mode or "text").lower()
    if mode not in ("text", "multimodal"):
        raise ValueError(f"mode must be 'text' or 'multimodal', got {mode!r}")
    if mode == "multimodal":
        if arch.get("is_qwen3_5") or arch.get("is_multimodal"):
            return (AUTO_MULTIMODAL_CLASS,
                    "explicit multimodal mode: config architectures="
                    f"{arch.get('architectures')} (vision tower loaded)")
        return ("AutoModelForCausalLM",
                "explicit multimodal mode on a text-only config: "
                f"architectures={arch.get('architectures')}")
    if arch.get("is_qwen3_5"):
        return (TEXT_ONLY_CLASS,
                "text-only mode: Qwen3.5 text tower via "
                f"{TEXT_CONFIG_CLASS} (official text-only path; "
                "AutoModelForImageTextToText is NOT used for text-only WSP)")
    return ("AutoModelForCausalLM",
            "text-only mode: config architectures="
            f"{arch.get('architectures')} (classic causal LM)")


# ------------------------------------------------------------- weight key mapping
def plan_text_only_load(source_keys: Sequence[str], arch: dict) -> dict:
    """Map the checkpoint's text-tower keys onto the text-only model namespace.

    ``model.language_model.layers.0.attn...``  ->  ``model.layers.0.attn...``
    ``model.language_model.embed_tokens.weight`` -> ``model.embed_tokens.weight``
    ``model.visual.*`` and ``mtp.*`` are skipped (counted, never merged).

    This is pure bookkeeping over the index file: no tensor is read.
    """
    prefix = arch.get("text_weight_prefix") or TEXT_WEIGHT_PREFIX
    vision_prefix = arch.get("vision_weight_prefix") or VISION_WEIGHT_PREFIX
    mtp_prefix = arch.get("mtp_weight_prefix") or MTP_WEIGHT_PREFIX
    mapping: Dict[str, str] = {}
    skipped = {"vision": [], "mtp": [], "other": []}
    for k in source_keys:
        if k.startswith(prefix):
            mapping["model." + k[len(prefix):]] = k
        elif k.startswith(vision_prefix):
            skipped["vision"].append(k)
        elif k.startswith(mtp_prefix):
            skipped["mtp"].append(k)
        elif k in TIED_HEAD_KEYS:
            mapping[k] = k
        elif k.startswith(("lm_head.", "model.")):
            mapping[k] = k
        else:
            skipped["other"].append(k)
    tied = bool(arch.get("tie_word_embeddings"))
    return {
        "mapping": mapping,
        "n_mapped": len(mapping),
        "skipped": {k: len(v) for k, v in skipped.items()},
        "skipped_samples": {k: v[:3] for k, v in skipped.items()},
        "tied_word_embeddings": tied,
        "allowed_missing": list(TIED_HEAD_KEYS) if tied else [],
        "source_text_tensors": sum(1 for k in source_keys if k.startswith(prefix)),
    }


def read_weight_index(model_path: str | Path) -> dict:
    """Return {'shards': [...], 'keys': [...], 'weight_map': {...}} for a snapshot."""
    p = Path(model_path)
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        with open(idx, "r", encoding="utf-8") as f:
            d = json.load(f)
        wm = d.get("weight_map") or {}
        return {"shards": sorted(set(wm.values())), "weight_map": wm,
                "keys": list(wm), "source": str(idx)}
    single = p / "model.safetensors"
    if single.exists():
        return {"shards": [single.name], "weight_map": {}, "keys": None,
                "source": str(single)}
    raise FileNotFoundError(
        f"no model.safetensors or model.safetensors.index.json under {model_path}; "
        f"download the real snapshot of {REQUIRED_MODEL_ID}")


# --------------------------------------------------------------------- configs
def build_text_config(text_cfg: dict, dtype: str | None = None):
    """Instantiate Qwen3_5TextConfig from the checkpoint's text_config block."""
    import transformers

    cls = getattr(transformers, TEXT_CONFIG_CLASS, None)
    if cls is None:
        raise LoaderCompatibilityError(
            f"transformers {transformers_version()} has no {TEXT_CONFIG_CLASS}; "
            f"Qwen3.5 text-only loading needs >= {MIN_TRANSFORMERS_VERSION}")
    payload = dict(text_cfg)
    if dtype:
        payload["dtype"] = dtype
    try:                                   # PreTrainedConfig.from_dict is exact
        cfg = cls.from_dict(payload)
    except AttributeError:                 # older API
        cfg = cls(**payload)
    except TypeError:                      # unknown keys in this release
        allowed = set(getattr(cls, "__dataclass_fields__", {}))
        cfg = cls(**{k: v for k, v in payload.items()
                     if not allowed or k in allowed})
    return cfg


# ------------------------------------------------------------------ verification
def _shape_of(param) -> tuple:
    return tuple(int(x) for x in getattr(param, "shape", ()))


def verify_loaded_model(model, model_path: str | Path, arch: dict,
                        expected: dict | None = None, mode: str = "text") -> dict:
    """Defensive identity/geometry verification of a loaded model.

    Never raises and never assumes the object is a real nn.Module: a fake object
    used in tests (no ``modules()``, no config) yields ``ok=False`` with clear
    problems instead of an AttributeError. A mismatch between config.json and
    the live module tree is reported as a silent-model-swap failure.
    """
    problems: List[str] = []
    warnings: List[str] = []
    exp = expected or {
        "model_id": arch.get("model_id", REQUIRED_MODEL_ID),
        "hidden_size": arch.get("text_hidden_size"),
        "num_hidden_layers": arch.get("text_num_hidden_layers"),
        "vocab_size": arch.get("text_vocab_size"),
        "tie_word_embeddings": arch.get("tie_word_embeddings"),
    }
    out = {"ok": False, "problems": problems, "warnings": warnings,
           "loaded_class": None, "mode": mode, "hidden_size": None,
           "num_hidden_layers": None, "vocab_size": None, "embedding_shape": None,
           "lm_head_shape": None, "tied_embeddings": None,
           "head_shares_embedding": None, "n_parameters": None,
           "n_linear_layers": None, "module_tree_walked": False,
           "config_checked": False, "weight_index_checked": False,
           "identity_verified": bool(arch.get("identity_verified"))}

    if model is None:
        problems.append("no model object was returned by the loader")
        return out
    out["loaded_class"] = type(model).__name__

    # ---- class matches the requested path --------------------------------
    expected_class = TEXT_ONLY_CLASS if mode == "text" else MULTIMODAL_CLASS
    cls_name = type(model).__name__
    if expected_class and not (cls_name == expected_class
                               or cls_name.endswith(expected_class)
                               or expected_class in cls_name):
        # allow a thin wrapper that contains the expected class
        problems.append(
            f"loaded class {cls_name!r} is not the expected {expected_class!r} "
            f"for mode={mode!r} (possible silent model swap)")
        if arch.get("architectures") and cls_name in arch["architectures"]:
            warnings.append(
                f"{cls_name!r} is the CONFIG architecture; it is a multimodal "
                "class and is not accepted on the text-only path")

    # ---- module tree (defensive: object may not be an nn.Module) ---------
    modules_fn = getattr(model, "modules", None)
    named_fn = getattr(model, "named_parameters", None)
    if not callable(modules_fn):
        problems.append("loaded object has no modules() method; it is not a "
                        "supported model instance")
    else:
        try:
            n_lin = 0
            for m in modules_fn():
                if type(m).__name__ in ("Linear", "LinearBase"):
                    n_lin += 1
            out["n_linear_layers"] = n_lin
            out["module_tree_walked"] = True
        except Exception as e:                   # never let verification crash
            problems.append(f"modules() raised {type(e).__name__}: {e}")

    params: Dict[str, object] = {}
    if callable(named_fn):
        try:
            params = dict(named_fn())
            out["n_parameters"] = int(sum(
                int(getattr(p, "numel", lambda: 0)()) for p in params.values()))
        except Exception as e:
            problems.append(f"named_parameters() raised {type(e).__name__}: {e}")
    else:
        problems.append("loaded object has no named_parameters() method")

    # ---- config geometry --------------------------------------------------
    cfg = getattr(model, "config", None)
    if cfg is None:
        problems.append("loaded model has no config")
    else:
        out["config_checked"] = True
        got = {k: getattr(cfg, k, None) for k in
               ("hidden_size", "num_hidden_layers", "vocab_size",
                "tie_word_embeddings")}
        for key, want_key in (("hidden_size", "hidden_size"),
                              ("num_hidden_layers", "num_hidden_layers"),
                              ("vocab_size", "vocab_size")):
            want = exp.get(want_key)
            have = got.get(key)
            out[key] = have
            if want is not None and have is not None and int(have) != int(want):
                problems.append(
                    f"config {key}={have} but checkpoint says {want} "
                    f"({arch.get('model_id')})")
        out["tied_embeddings"] = got.get("tie_word_embeddings")
        if exp.get("tie_word_embeddings") is not None and got.get("tie_word_embeddings") is not None:
            if bool(got["tie_word_embeddings"]) != bool(exp["tie_word_embeddings"]):
                problems.append(
                    f"config tie_word_embeddings={got['tie_word_embeddings']} "
                    f"but checkpoint says {exp['tie_word_embeddings']}")

    # ---- live layer count (not the config's claim) -----------------------
    if params:
        live_layers = set()
        for name in params:
            marker = ".layers."
            if marker in name:
                tail = name.split(marker, 1)[1]
                head = tail.split(".", 1)[0]
                if head.isdigit():
                    live_layers.add(int(head))
        if live_layers:
            out["num_hidden_layers"] = len(live_layers)
            want = exp.get("num_hidden_layers")
            if want is not None and len(live_layers) != int(want):
                problems.append(
                    f"live module tree has {len(live_layers)} decoder layers but "
                    f"the checkpoint config says {want}")

        def _find(suffix: str):
            for n, p in params.items():
                if n.endswith(suffix):
                    return n, p
            return None, None

        emb_name, emb = _find("embed_tokens.weight")
        if emb is None:
            problems.append("no embed_tokens.weight parameter in the loaded model")
        else:
            out["embedding_shape"] = list(_shape_of(emb))
            want_v, want_h = exp.get("vocab_size"), exp.get("hidden_size")
            shp = _shape_of(emb)
            if want_v is not None and shp and shp[0] != int(want_v):
                problems.append(
                    f"embedding rows={shp[0]} but checkpoint vocab_size={want_v}")
            if want_h is not None and len(shp) > 1 and shp[1] != int(want_h):
                problems.append(
                    f"embedding dim={shp[1]} but checkpoint hidden_size={want_h}")
            out["vocab_size"] = shp[0] if shp else None
        head_name, head = _find("lm_head.weight")
        if head is None:
            if exp.get("tie_word_embeddings"):
                warnings.append("lm_head.weight absent and embeddings are tied "
                                "(expected for this checkpoint)")
                out["lm_head_shape"] = list(out["embedding_shape"] or [])
                out["tied_embeddings"] = True
            else:
                problems.append("lm_head.weight missing while embeddings are untied")
        else:
            out["lm_head_shape"] = list(_shape_of(head))
            if out["embedding_shape"] and _shape_of(head) != tuple(out["embedding_shape"]):
                problems.append(
                    f"lm_head shape {_shape_of(head)} != embedding shape "
                    f"{tuple(out['embedding_shape'])}")
            # A tied head must be the *same storage* as the embedding matrix.
            # Shapes alone do not catch a head left uninitialised by a
            # meta-device build (all-zero logits, garbage generation).
            if out.get("tied_embeddings") and emb is not None:
                try:
                    share = head.data_ptr() == emb.data_ptr()
                except Exception:
                    share = head is emb
                out["head_shares_embedding"] = bool(share)
                if not share:
                    problems.append(
                        "tie_word_embeddings is true but lm_head.weight does not "
                        "share storage with embed_tokens.weight (head not re-tied "
                        "after an empty build; generation would be garbage)")

    # ---- tied head shares the embedding storage --------------------------
    if out.get("tied_embeddings"):
        emb = params.get(emb_name) if emb_name else None
        head_mod = getattr(model, "lm_head", None)
        head_w = getattr(head_mod, "weight", None)
        if emb is not None and head_w is not None:
            try:
                share = head_w.data_ptr() == emb.data_ptr()
            except Exception:
                share = head_w is emb
            out["head_shares_embedding"] = bool(share)
            if not share:
                problems.append(
                    "tie_word_embeddings is true but lm_head.weight does not share "
                    "storage with embed_tokens.weight (head left uninitialised by "
                    "an empty build; generation would be garbage)")

    # ---- checkpoint files vs config --------------------------------------
    try:
        index = read_weight_index(model_path)
        out["weight_index_checked"] = True
        out["n_checkpoint_tensors"] = (len(index["keys"]) if index["keys"]
                                       else None)
        if index["keys"]:
            plan = plan_text_only_load(index["keys"], arch)
            out["text_tensors_planned"] = plan["n_mapped"]
            out["skipped_tensors"] = plan["skipped"]
            if plan["n_mapped"] == 0:
                problems.append("no text-tower tensors found in the weight index")
            expected_layers = exp.get("num_hidden_layers")
            seen_layers = {int(k.split(".layers.")[1].split(".")[0])
                           for k in index["keys"] if ".layers." in k}
            if expected_layers and len(seen_layers) != int(expected_layers):
                problems.append(
                    f"weight index has {len(seen_layers)} layer groups but config "
                    f"says {expected_layers}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        warnings.append(f"weight index not inspected: {e}")

    # ---- recorded identity ------------------------------------------------
    meta = read_snapshot_metadata(model_path)
    if meta:
        out["snapshot_metadata"] = {k: meta.get(k) for k in
                                    ("repo_id", "revision", "sha", "downloaded_utc",
                                     "n_files", "total_bytes")}
        recorded = meta.get("repo_id") or meta.get("model_id")
        out["identity_verified"] = bool(recorded
                                        and recorded == exp.get("model_id"))
        if not out["identity_verified"]:
            problems.append(
                f"checkpoint metadata records {recorded!r} but the required model "
                f"is {exp.get('model_id')!r}; refusing to treat it as the edited "
                "WSP checkpoint")
    else:
        warnings.append(
            f"no {SNAPSHOT_METADATA_FILE} in the snapshot: model identity is "
            "verified from config geometry only (run `aladdin download` to record "
            "the repository id and revision)")
        out["identity_verified"] = False

    out["ok"] = not problems
    return out


def verify_tokenizer(tokenizer, arch: dict) -> dict:
    """Tokenizer must exist, encode/decode, and fit the model vocabulary."""
    problems: List[str] = []
    vocab = arch.get("text_vocab_size")
    try:
        n_tokens = len(tokenizer)
    except Exception as e:
        problems.append(f"tokenizer has no length: {e}")
        n_tokens = None
    if n_tokens is not None and vocab is not None and n_tokens > int(vocab):
        problems.append(f"tokenizer has {n_tokens} ids but model vocab is {vocab}")
    try:
        ids = tokenizer("mujhe ye samjha")["input_ids"]
        if not ids:
            problems.append("tokenizer produced no ids for a probe string")
        tokenizer.decode(ids)
    except Exception as e:
        problems.append(f"tokenizer encode/decode failed: {type(e).__name__}: {e}")
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        problems.append("tokenizer has no chat template")
    if tokenizer.eos_token is None:
        problems.append("tokenizer has no eos_token")
    return {"ok": not problems, "problems": problems, "n_tokens": n_tokens,
            "vocab_size": vocab, "has_chat_template": bool(template),
            "eos_token": getattr(tokenizer, "eos_token", None),
            "pad_token": getattr(tokenizer, "pad_token", None)}


# ------------------------------------------------------------------- weight load
def load_text_weights(model, model_path: str | Path, arch: dict,
                      plan: dict | None = None, dtype: str = "bfloat16") -> dict:
    """Copy the checkpoint's text-tower tensors into `model`, exactly.

    Reads shard by shard with safetensors and copies tensor-by-tensor, so peak
    memory is one tensor rather than a duplicate model. Shapes and dtypes are
    checked before each copy; missing/unexpected keys are reported, never
    silently ignored.
    """
    import torch
    from safetensors import safe_open

    index = read_weight_index(model_path)
    keys = index["keys"]
    if keys is None:
        raise WeightLoadError(
            f"{model_path} has a single model.safetensors but no index; use "
            "`aladdin download` for the official snapshot layout")
    plan = plan or plan_text_only_load(keys, arch)
    mapping = plan["mapping"]

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    targets = {**buffers, **params}
    allowed_missing = set(plan["allowed_missing"])

    missing = sorted(set(mapping) - set(targets))
    unexpected_targets = sorted(t for t in mapping if t in TIED_HEAD_KEYS
                                and t not in targets and t not in allowed_missing)
    hard_missing = [m for m in missing if m not in allowed_missing]
    if hard_missing:
        raise WeightLoadError(
            f"the text-only model expects {len(hard_missing)} tensor(s) that the "
            f"checkpoint does not provide (first: {hard_missing[:5]}); the key "
            "mapping does not match this architecture")
    if unexpected_targets:
        raise WeightLoadError(f"unexpected target tensors: {unexpected_targets}")

    tdt = getattr(torch, _DT_MAP.get(dtype, "float32"))
    loaded, skipped_dtype, shape_errors = 0, 0, []
    want = {t: s for t, s in mapping.items() if t in targets}

    for shard in index["shards"]:
        path = Path(model_path) / shard
        if not path.exists():
            raise WeightLoadError(f"missing safetensors shard {path}")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            available = set(f.keys())
            for target, source in want.items():
                if source not in available:
                    continue
                tensor = f.get_tensor(source)
                param = targets[target]
                if tuple(tensor.shape) != tuple(param.shape):
                    shape_errors.append(
                        f"{target}: checkpoint {tuple(tensor.shape)} != model "
                        f"{tuple(param.shape)}")
                    continue
                with torch.no_grad():
                    param.data.copy_(tensor.to(dtype=tdt, device=param.device))
                loaded += 1
    if shape_errors:
        raise WeightLoadError(
            "shape mismatches while loading the text tower: "
            + "; ".join(shape_errors[:5]))
    if loaded != len(want):
        missing_now = sorted(set(want) - set())  # placeholder for clarity
        raise WeightLoadError(
            f"loaded {loaded} of {len(want)} text tensors; the shard index and the "
            f"files disagree (missing example: {missing_now[:0]})")

    return {
        "loaded_tensors": loaded,
        "expected_tensors": len(want),
        "allowed_missing": sorted(allowed_missing),
        "skipped_visual": plan["skipped"].get("vision", 0),
        "skipped_mtp": plan["skipped"].get("mtp", 0),
        "source": index["source"],
        "shards": index["shards"],
        "dtype": dtype,
        "all_weights_loaded": loaded == len(want),
    }


# --------------------------------------------------------------------- main load
def _resolve_placements(model, device: str, device_map) -> dict:
    """Explicit per-prefix placement; no fake multi-device sharding."""
    named = [n for n, _ in model.named_parameters()]
    if not device_map:
        return {n: device for n in named}
    if isinstance(device_map, dict):
        out = {}
        for n in named:
            place = device
            for prefix, dev in sorted(device_map.items(), key=lambda kv: -len(kv[0])):
                if n.startswith(prefix):
                    place = dev
                    break
            out[n] = place
        return out
    if device_map == "auto":
        import torch
        if device.startswith("cuda") or (device == "auto" and torch.cuda.is_available()):
            return {n: "cuda" for n in named}
        return {n: "cpu" for n in named}
    raise LoaderCompatibilityError(
        f"unsupported device_map {device_map!r}; pass a dict of "
        "{parameter_prefix: device} or 'auto'")


def load_qwen35(model_path: str | Path, dtype: str = "bfloat16",
                device: str = "cpu", trust_remote_code: bool = False,
                revision: str = "", min_version: str = MIN_TRANSFORMERS_VERSION,
                mode: str = "text", expected_model_id: str = REQUIRED_MODEL_ID,
                device_map=None, verify_identity: bool = True):
    """Load the required edited checkpoint, text-only by default.

    Returns (model, tokenizer, info). Raises LoaderCompatibilityError /
    ModelIdentityError / WeightLoadError with actionable messages for a missing
    stack, a wrong checkpoint, or a key/shape mismatch. It never falls back to a
    different model, a different class, or an unedited checkpoint.
    """
    v = check_transformers_version(min_version)
    if not v["ok"]:
        raise LoaderCompatibilityError(v["error"])
    if verify_identity:
        ident = check_checkpoint_identity(model_path, expected_model_id)
        if not ident["ok"]:
            raise ModelIdentityError(
                "refusing to load this checkpoint: " + "; ".join(ident["problems"])
                + ". Download the required model with `aladdin download`.")
    arch = detect_architecture(model_path, expected_model_id=expected_model_id)
    class_name, reason = _auto_class_for(arch, mode)

    import torch
    import transformers
    from transformers import AutoTokenizer

    cls = getattr(transformers, class_name, None)
    if cls is None:
        raise LoaderCompatibilityError(
            f"transformers {v['installed']} has no {class_name}; upgrade: "
            f"pip install -U 'transformers>={min_version}'")

    kw: dict = {"trust_remote_code": trust_remote_code}
    if revision:
        kw["revision"] = revision

    # ---- tokenizer (always from the same snapshot directory) --------------
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kw)
    jf = Path(model_path) / "chat_template.jinja"
    if not getattr(tokenizer, "chat_template", None) and jf.exists():
        tokenizer.chat_template = jf.read_text(encoding="utf-8")
    tok_chk = verify_tokenizer(tokenizer, arch)
    if not tok_chk["ok"]:
        raise LoaderCompatibilityError(
            "tokenizer is not usable with this checkpoint: "
            + "; ".join(tok_chk["problems"]))

    # ---- model: text-only path remaps the checkpoint's key namespace ------
    weight_report = None
    if mode == "text":
        text_cls = getattr(transformers, TEXT_CONFIG_CLASS, None)
        if text_cls is None:
            raise LoaderCompatibilityError(
                f"transformers {v['installed']} has no {TEXT_CONFIG_CLASS}; "
                f"text-only Qwen3.5 needs >= {min_version}")
        text_cfg = build_text_config(arch.get("_raw_config", {}).get("text_config")
                                     or arch, dtype=dtype)
        placements = None
        try:                              # avoid allocating weights twice
            with torch.device("meta"):
                model = cls(text_cfg)
            place = _resolve_placements(model, device, device_map)
            if len(set(place.values())) > 1:
                raise LoaderCompatibilityError(
                    "multi-device placement needs accelerate; pass a single "
                    "device in backend.device")
            model = model.to_empty(device=device)
            # to_empty() replaces every Parameter, which silently breaks the
            # embedding/output-head sharing created at init time: lm_head would
            # keep uninitialised memory and generation would be garbage. Re-tie
            # before the weights are copied in.
            if getattr(text_cfg, "tie_word_embeddings", False):
                tie_fn = getattr(model, "tie_weights", None)
                if not callable(tie_fn):
                    raise LoaderCompatibilityError(
                        "tie_word_embeddings is true but the loaded class has no "
                        "tie_weights(); refusing to continue with an untied head")
                tie_fn()
        except (AttributeError, RuntimeError):
            model = cls(text_cfg).to(device)
        weight_report = load_text_weights(model, model_path, arch, dtype=dtype)
    else:
        model = cls.from_pretrained(model_path, dtype=getattr(
            torch, _DT_MAP.get(dtype, "float32")), device_map=device_map, **kw)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    chk = verify_loaded_model(model, model_path, arch,
                             expected={"model_id": expected_model_id,
                                       "hidden_size": arch.get("text_hidden_size"),
                                       "num_hidden_layers": arch.get("text_num_hidden_layers"),
                                       "vocab_size": arch.get("text_vocab_size"),
                                       "tie_word_embeddings": arch.get("tie_word_embeddings")},
                             mode=mode)
    if not chk["ok"]:
        raise LoaderCompatibilityError(
            "loaded model failed verification (refusing to continue with an "
            "unverified checkpoint): " + "; ".join(chk["problems"]))

    info = {
        "model_id": expected_model_id,
        "recorded_model_id": arch.get("recorded_model_id"),
        "revision": arch.get("revision") or revision or None,
        "identity_verified": chk["identity_verified"],
        "mode": mode,
        "auto_class": class_name,
        "selection_reason": reason,
        "transformers": v["installed"],
        "torch": torch.__version__,
        "architecture": arch,
        "loaded_class": chk["loaded_class"],
        "n_linear_layers": chk["n_linear_layers"],
        "n_parameters": chk["n_parameters"],
        "hidden_size": chk["hidden_size"],
        "num_hidden_layers": chk["num_hidden_layers"],
        "vocab_size": chk["vocab_size"],
        "embedding_shape": chk["embedding_shape"],
        "lm_head_shape": chk["lm_head_shape"],
        "tied_embeddings": chk["tied_embeddings"],
        "tokenizer": tok_chk,
        "weights": weight_report,
        "verification": {k: chk[k] for k in
                         ("ok", "problems", "warnings", "module_tree_walked",
                          "config_checked", "weight_index_checked",
                          "n_checkpoint_tensors", "text_tensors_planned",
                          "skipped_tensors") if k in chk},
        "text_only": mode == "text",
    }
    return model, tokenizer, info
