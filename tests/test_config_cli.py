"""Configuration + CLI surface (no GPU / no key required)."""
import json
from pathlib import Path

import pytest

from aladdin.config import DEFAULTS, load_config, path_of, redacted


def test_defaults_load_and_merge(tmp_path):
    c = load_config(Path(__file__).resolve().parent.parent / "config" / "default.json")
    assert c["backend"]["kind"] == "tiny_numpy"
    assert c["editing"]["max_relative_change"] > 0


def test_default_model_is_the_required_edited_checkpoint():
    from aladdin.model_identity import REQUIRED_MODEL_ID
    c = load_config(Path(__file__).resolve().parent.parent / "config" / "default.json")
    assert c["backend"]["model_id"] == REQUIRED_MODEL_ID
    assert REQUIRED_MODEL_ID == "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
    assert DEFAULTS["backend"]["model_id"] == REQUIRED_MODEL_ID


def test_env_overrides_model_path(monkeypatch):
    monkeypatch.setenv("ALADDIN_MODEL_PATH", "/tmp/fake-model")
    monkeypatch.setenv("NVIDIA_MODEL", "qwen/qwen3-next-80b-a3b-instruct")
    c = load_config()
    assert c["backend"]["model_path"] == "/tmp/fake-model"
    assert c["agent"]["model"] == "qwen/qwen3-next-80b-a3b-instruct"


def test_config_has_no_secret_fields():
    assert not any("api_key" in k.lower() for k in DEFAULTS["agent"])
    c = load_config()
    assert "api_key" not in c["agent"]


def test_redacted_removes_secret_like_keys():
    c = load_config()
    c["agent"]["api_key"] = "nvapi-should-not-survive"
    assert redacted(c)["agent"]["api_key"] == "<redacted>"


def test_all_config_files_are_valid_json():
    for p in (Path(__file__).resolve().parent.parent / "config").glob("*.json"):
        json.load(open(p)), p.name


def test_cli_parser_accepts_documented_commands():
    from aladdin.cli import build_parser
    p = build_parser()
    for argv in (["doctor"], ["inspect"], ["baseline", "--split", "dev"],
                 ["run", "--agent", "scripted", "--steps", "1"],
                 ["edits", "list"], ["rollback", "--keep"], ["verify"],
                 ["resume", "--experiment", "x"], ["report", "--experiment", "x"],
                 ["compare", "--a", "a.json", "--b", "b.json"],
                 ["download", "--repo-id",
                  "RikiyaT/Qwen3.5-0.8B-weight-space-programming"],
                 ["model-info"], ["verify-model"],
                 ["evaluate", "--split", "heldout"]):
        ns = p.parse_args(argv)
        assert hasattr(ns, "fn")


def test_paths_resolve_inside_project():
    c = load_config()
    assert str(path_of(c, "checkpoints")).startswith(c["_project_root"])
