import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aladdin.backends.tiny_numpy import TinyNumpyBackend  # noqa: E402
from aladdin.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    c = load_config(ROOT / "config" / "default.json")
    c["tiny"] = dict(c["tiny"], hidden_size=32, num_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, intermediate_size=64, vocab_size=300,
                     max_position_embeddings=128)
    c["editing"] = dict(c["editing"], max_entries_per_tensor=2048)
    c["evaluation"] = dict(c["evaluation"], max_new_tokens=6)
    c["_project_root"] = str(ROOT)
    return c


@pytest.fixture()
def backend(cfg):
    return TinyNumpyBackend(cfg)


@pytest.fixture()
def tmp_project(tmp_path, cfg):
    """An isolated project root with its own checkpoints/experiments dirs."""
    c = dict(cfg)
    c["paths"] = {k: str(tmp_path / v) for k, v in cfg["paths"].items()}
    c["_project_root"] = str(tmp_path)
    for sub in c["paths"].values():
        Path(sub).mkdir(parents=True, exist_ok=True)
    return c


def _write_datasets(root: Path):
    d = root / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (cat, prompt) in enumerate([
        ("roman_hinglish", "mujhe samaj nahi aya simple me batao"),
        ("devanagari_hindi", "मुझे समझ नहीं आया"),
        ("mixed_code_switch", "mera laptop slow chal raha hai kya karu"),
        ("instruction_following", "2 bullet me batao python kyu"),
    ]):
        rows.append({"id": f"d{i}", "category": cat, "prompt": prompt,
                     "expect_script": "either", "must_include": [["a", "e", "i", "o", "u"]],
                     "answer_tokens": [], "meaning": "test"})
    (d / "dev.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    held = [{"id": f"h{i}", "category": c, "prompt": p + " heldout",
             "expect_script": "either", "must_include": [], "answer_tokens": []}
            for i, (c, p) in enumerate([("factuality", "bharat ki rajdhani"),
                                        ("question_answering", "2+2 kitna")])]
    (d / "heldout.jsonl").write_text("\n".join(json.dumps(r) for r in held), encoding="utf-8")


@pytest.fixture()
def project(tmp_path, cfg):
    """Full isolated experiment project (shared by e2e + bugfix regression tests)."""
    c = dict(cfg)
    c["paths"] = {k: str(tmp_path / v) for k, v in cfg["paths"].items()}
    c["_project_root"] = str(tmp_path)
    for p in c["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
    c["forward_pass"] = dict(c["forward_pass"], budget=6)
    c["agent"] = dict(c["agent"], provider="scripted", max_steps=2)
    c["evaluation"] = dict(c["evaluation"], max_new_tokens=4)
    c["acceptance"] = dict(c["acceptance"], min_composite_gain=-1.0)   # accept everything
    _write_datasets(tmp_path)
    return c
