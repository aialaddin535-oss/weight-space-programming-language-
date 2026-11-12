"""Guard rails: no training, no secrets, no forbidden training APIs anywhere.

This test is the executable version of the project's main promise.
"""
import hashlib
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "aladdin"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", "experiments",
             "checkpoints", "logs", ".cache", "models", ".venv", "venv",
             "site-packages"}

FORBIDDEN_CALLS = [
    r"\.backward\s*\(",
    r"\btorch\.optim\b",
    r"\boptimizer\s*\.\s*step\s*\(",
    r"\bAdamW?\s*\(",
    r"\bSGD\s*\(",
    r"\bTrainer\s*\(",
    r"\btrain\(\)",
    r"\bPeftModel\b",
    r"\bget_peft_model\b",
    r"\bLoraConfig\b",
    r"\bSFTTrainer\b",
    r"\.requires_grad_\s*\(\s*True",
    r"\bloss\s*\.\s*backward",
    r"\bno_grad\(\)\s*:\s*\n\s*[^\n]*backward",
]

ALLOWED_TRAIN_WORDS = ["train(", "training", "train_", "trainable"]


def _py_files(base: Path):
    self_path = Path(__file__).resolve()
    for p in base.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.resolve() == self_path:      # this file only holds the patterns
            continue
        yield p


def test_no_training_api_anywhere():
    hits = []
    for p in _py_files(ROOT):
        text = p.read_text(encoding="utf-8", errors="replace")
        for pat in FORBIDDEN_CALLS:
            for m in re.finditer(pat, text):
                hits.append(f"{p.relative_to(ROOT)}: {m.group(0)!r}")
    assert not hits, "forbidden training API found:\n" + "\n".join(hits)


def test_no_gradient_descent_semantics_in_editing_path():
    """The editor / sparse-delta modules must not import any fitting machinery."""
    import ast

    banned_modules = {"torch.optim", "torch.autograd", "peft", "accelerate.optimizer",
                      "transformers.Trainer", "trl"}
    banned_names = {"backward", "optimizer", "SGD", "Adam", "AdamW", "LoraConfig",
                    "get_peft_model", "SFTTrainer"}
    for name in ("aladdin/tensor_editor/engine.py", "aladdin/tensor_editor/ops.py",
                 "aladdin/sparse_deltas/store.py", "aladdin/sparse_deltas/cast.py",
                 "aladdin/checkpoints/manager.py"):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name not in banned_modules, f"{name} imports {a.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert mod not in banned_modules, f"{name} imports from {mod}"
                for a in node.names:
                    assert a.name not in banned_names, f"{name} imports {a.name}"
            elif isinstance(node, ast.FunctionDef):
                assert node.name not in banned_names, f"{name} defines {node.name}"
            elif isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute):
                    assert fn.attr not in ("backward", "step"), \
                        f"{name} calls .{fn.attr}()"


def test_no_secret_literals_in_tracked_files():
    """No live-looking API key anywhere in the shipped project."""
    pats = [re.compile(r"nvapi-[A-Za-z0-9]{12,}"),
            re.compile(r"\bhf_[A-Za-z0-9]{25,}"),
            re.compile(r"\bsk-[A-Za-z0-9]{25,}")]
    hits = []
    self_path = Path(__file__).resolve()
    for p in ROOT.rglob("*"):
        if p.is_dir() or any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.resolve() == self_path:          # this file only holds the patterns
            continue
        if p.suffix.lower() in {".zip", ".npz", ".npy", ".jsonl"} or p.stat().st_size > 2_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in pats:
            m = pat.search(text)
            if m:
                hits.append(f"{p.relative_to(ROOT)}: {m.group(0)[:8]}...")
    assert not hits, f"possible secret in: {hits}"


def test_api_key_is_read_only_from_env():
    src = (PKG / "agent" / "nvidia_client.py").read_text(encoding="utf-8")
    assert 'SECRET_ENV = "NVIDIA_API_KEY"' in src
    assert 'os.environ.get(SECRET_ENV' in src


def test_env_example_has_placeholders_only():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip() == "NVIDIA_API_KEY":
            assert val.strip() == "", f"example file must not carry a key: {line}"
        if key.strip() == "NVIDIA_MODEL":
            assert val.strip() == "", "model id must be chosen by the user, not baked in"
    assert "NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1" in text
    assert "ALADDIN_MODEL_PATH" in text


def test_gitignore_covers_secrets_and_weights():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for tok in (".env", "*.safetensors", "checkpoints/", "models/"):
        assert tok in text, tok


def test_no_model_weights_committed():
    weight_suffixes = {".safetensors", ".bin", ".gguf", ".pt", ".pth"}
    for p in ROOT.rglob("*"):
        # local virtualenvs / caches are not part of the repository
        if SKIP_DIRS & set(p.relative_to(ROOT).parts):
            continue
        if p.is_file() and p.suffix in weight_suffixes:
            pytest.fail(f"model weight file in the project: {p}")
