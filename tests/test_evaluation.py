"""Scoring, datasets, dev/held-out separation, accept/reject criteria."""
import json

import pytest

from aladdin.evaluation import datasets as D
from aladdin.evaluation.runner import aggregate, compare
from aladdin.evaluation.scorer import loop_penalty, score_item, script_ratio


def item(**kw):
    base = {"id": "x1", "category": "roman_hinglish", "prompt": "mujhe samaj nahi aya",
            "expect_script": "roman", "must_include": [["samaj", "simple"]],
            "answer_tokens": [], "meaning": ""}
    base.update(kw)
    return base


def test_good_answer_scores_high():
    s = score_item(item(), "Samajh gaya, simple words me batau: ye error ka matlab hai ki value galat hai.")
    assert s["composite"] > 0.6
    assert s["dims"]["intent"] == 1.0
    assert s["dims"]["script"] == 1.0


def test_translation_preamble_penalised():
    a = score_item(item(), "Here is the translation: mujhe samajh aya.")
    b = score_item(item(), "Samajh aya, simple me batau.")
    assert a["composite"] < b["composite"]


def test_dodge_and_empty_penalised():
    assert score_item(item(), "")["composite"] < 0.3
    assert score_item(item(), "As an AI, I cannot help with that.")["composite"] < 0.6


def test_devanagari_script_requirement():
    it = item(id="x2", category="devanagari_hindi", expect_script="devanagari",
              must_include=[["समझ"]], prompt="मुझे समझ नहीं आया")
    hi = score_item(it, "समझ गया, आसान भाषा में बताता हूँ।")
    roman = score_item(it, "samajh gaya, aasan bhasha me batata hu")
    assert hi["dims"]["script"] == 1.0
    assert roman["dims"]["script"] < 1.0
    assert hi["composite"] > roman["composite"]


def test_bullets_and_length_constraints():
    it = item(id="x3", category="instruction_following", expect_bullets=3, max_words=40,
              must_include=[["python"]], prompt="3 bullets")
    ok = score_item(it, "- Python easy hai\n- Python me jobs hain\n- Python fast hai")
    bad = score_item(it, "Python achi bhasha hai " * 30)
    assert ok["dims"]["compliance"] == 1.0
    assert bad["dims"]["compliance"] < 1.0


def test_fact_banned_detected():
    it = item(id="x4", category="factuality", fact_banned=["bill gates invented the internet"],
              must_include=[["internet"]], prompt="internet kisne banaya")
    bad = score_item(it, "Internet Bill Gates invented the internet.")
    assert bad["dims"]["factual"] == 0.0


def test_loop_penalty():
    text = " ".join(["the same phrase again and again"] * 12)
    assert loop_penalty(text) > 0.3
    assert loop_penalty("ek chhota sa saaf jawab hai ye") == 0.0


def test_script_ratio_mixed():
    r = script_ratio("Ye error API key की वजह से है")
    assert r["devanagari_chars"] > 0 and r["latin_chars"] > 0
    assert 0 < r["devanagari_ratio"] < 1


def test_shipped_datasets_are_valid_and_disjoint():
    dev, held = D.load_splits({"_project_root": __import__("pathlib").Path(__file__).resolve().parent.parent,
                               "paths": {"datasets": "datasets"},
                               "evaluation": {"dev_file": "dev.jsonl", "heldout_file": "heldout.jsonl"}})
    assert len(dev.rows) >= 40 and len(held.rows) >= 20
    assert D.validate(dev.rows)["ok"] and D.validate(held.rows)["ok"]
    assert not (dev.ids() & held.ids())
    assert len(dev.by_category()) == 10 and len(held.by_category()) == 10


def test_assert_no_heldout_leak_raises():
    from aladdin.evaluation.datasets import Dataset
    h = Dataset([{"id": "h1", "category": "factuality", "prompt": "p"}], "heldout")
    with pytest.raises(AssertionError):
        D.assert_no_heldout_leak({"h1", "d1"}, h)


def _res(comp, cats):
    """Build a fake result whose per-category means follow `cats` exactly."""
    scored = [{"composite": v, "category": c, "dims": {"intent": v}, "notes": [],
               "output": "ok"}
              for c, v in cats.items() for _ in range(2)]
    return {"aggregate": aggregate(scored), "items": scored, "label": "x", "split": "dev"}


def test_compare_accepts_clear_gain(cfg):
    b = _res(0.40, {"roman_hinglish": 0.40})
    c = _res(0.50, {"roman_hinglish": 0.50})
    d = compare(b, c, cfg)
    assert d["accept"] is True and d["delta"] == pytest.approx(0.1, abs=1e-6)


def test_compare_rejects_regression(cfg):
    b = _res(0.40, {"roman_hinglish": 0.40, "factuality": 0.40})
    c = _res(0.45, {"roman_hinglish": 0.60, "factuality": 0.30})
    d = compare(b, c, cfg)
    assert d["accept"] is False
    assert d["regressions"], d
    assert d["regressions"][0]["category"] == "factuality"


def test_compare_rejects_flat(cfg):
    b = _res(0.40, {"roman_hinglish": 0.40})
    d = compare(b, _res(0.401, {"roman_hinglish": 0.401}), cfg)
    assert d["accept"] is False
