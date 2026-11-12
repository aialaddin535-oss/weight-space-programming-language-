"""Configuration manager.

JSON configs (no YAML dependency), deep-merged over DEFAULTS, then overridden by
explicit overrides and by ALADDIN_* / NVIDIA_* environment variables. Secrets are
NEVER read from config files - only from the environment.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from .model_identity import REQUIRED_MODEL_ID

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "project": {"name": "Aladdin_WSP_Hindi_Hinglish", "seed": 0},
    "backend": {
        "kind": "tiny_numpy",          # tiny_numpy | hf_torch
        "model_id": REQUIRED_MODEL_ID,  # the edited WSP checkpoint (never the base)
        "model_path": "",              # local dir of that checkpoint
        "dtype": "float32",            # float32 | bfloat16 | float16 (hf_torch)
        "device": "cpu",               # cpu | cuda
        "device_map": None,            # None | "auto" | {prefix: device}
        "load_mode": "text",           # text (default) | multimodal
        "trust_remote_code": False,
        "revision": "",
    },
    "tiny": {
        "hidden_size": 64,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 512,
        "max_position_embeddings": 512,
        "seed": 0,
    },
    "paths": {
        "checkpoints": "checkpoints",
        "deltas": "checkpoints/deltas",
        "experiments": "experiments",
        "logs": "logs",
        "datasets": "datasets",
        "cache": ".cache",
        "models": "models",
    },
    "editing": {
        "max_tensors_per_edit": 2,
        "max_entries_per_tensor": 4096,
        "max_relative_change": 0.25,
        "min_relative_change": 1e-9,
        "frozen_suffixes": [
            "model.norm.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ],
        "refuse_uniform_rescale": True,
    },
    "forward_pass": {
        "max_new_tokens": 48,
        "budget": 64,
        "max_prompt_tokens": 256,
    },
    "evaluation": {
        "dev_file": "dev.jsonl",
        "heldout_file": "heldout.jsonl",
        "max_new_tokens": 48,
        "temperature": 0.0,
        "top_k": 0,
        "limit_per_category": 0,
        "seed": 0,
    },
    "acceptance": {
        "min_composite_gain": 0.01,
        "max_category_regression": 0.05,
        "min_abs_score": 0.0,
    },
    "agent": {
        "provider": "nvidia",          # nvidia | scripted
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "",                   # set NVIDIA_MODEL - never hardcoded
        "judge_model": "",
        "max_steps": 14,
        "temperature": 0.2,
        "max_tokens": 2048,
        "timeout": 180,
        "max_retries": 3,
    },
}


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _env_overrides(cfg: dict) -> dict:
    a = cfg["agent"]
    b = cfg["backend"]
    if os.environ.get("NVIDIA_BASE_URL"):
        a["base_url"] = os.environ["NVIDIA_BASE_URL"].strip()
    if os.environ.get("NVIDIA_MODEL"):
        a["model"] = os.environ["NVIDIA_MODEL"].strip()
    if os.environ.get("NVIDIA_JUDGE_MODEL"):
        a["judge_model"] = os.environ["NVIDIA_JUDGE_MODEL"].strip()
    if os.environ.get("ALADDIN_BACKEND"):
        b["kind"] = os.environ["ALADDIN_BACKEND"].strip()
    if os.environ.get("ALADDIN_MODEL_ID"):
        b["model_id"] = os.environ["ALADDIN_MODEL_ID"].strip()
    if os.environ.get("ALADDIN_MODEL_PATH"):
        b["model_path"] = os.environ["ALADDIN_MODEL_PATH"].strip()
    if os.environ.get("ALADDIN_DEVICE"):
        b["device"] = os.environ["ALADDIN_DEVICE"].strip()
    if os.environ.get("ALADDIN_LOAD_MODE"):
        b["load_mode"] = os.environ["ALADDIN_LOAD_MODE"].strip()
    return cfg


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    """Load config: DEFAULTS <- file <- overrides <- env."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            cfg = _deep_merge(cfg, json.load(f))
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg = _env_overrides(cfg)
    cfg["_project_root"] = str(PROJECT_ROOT)
    cfg["_config_path"] = str(path) if path else ""
    return cfg


def path_of(cfg: dict, key: str) -> Path:
    """Absolute path for a configured relative directory."""
    raw = cfg["paths"][key]
    p = Path(raw)
    if not p.is_absolute():
        p = Path(cfg["_project_root"]) / p
    return p


def ensure_dirs(cfg: dict) -> dict:
    out = {}
    for key in cfg["paths"]:
        p = path_of(cfg, key)
        p.mkdir(parents=True, exist_ok=True)
        out[key] = p
    return out


def redacted(cfg: dict) -> dict:
    """Config copy safe for logging / zipping (no secrets live in config anyway)."""
    c = copy.deepcopy(cfg)
    for k in list(c.get("agent", {})):
        if "key" in k.lower() or "token" in k.lower():
            c["agent"][k] = "<redacted>"
    return c
