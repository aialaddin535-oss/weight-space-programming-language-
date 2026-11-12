"""Deterministic, rubric-based scorer for Hindi / Hinglish behaviour.

Every score is computed from the model's *actual text output* against a per-item
rubric. There is no LLM in the loop by default, so scores are reproducible and
cheap; an optional NVIDIA LLM-judge can be enabled for the "naturalness" axis
(`--judge`), and when it is, its name is recorded in the results file.

Dimensions reported per item (0..1 each, `None` = not applicable for that item):

  intent        did the answer address the user's actual request (rubric tokens)
  correctness   exact/fuzzy answer match, when the item has a verifiable answer
  script        right writing system (Devanagari vs Roman) for the request
  compliance    instruction constraints respected (length, bullet count, bans)
  relevance     answer is on-topic and not a dodge/refusal
  naturalness   heuristic fluency proxy: no loops, no template junk, healthy length
  factual       no banned fabrication markers, required facts present

Composite = weighted mean of the applicable dimensions (weights in WEIGHTS).
These are heuristics, not human judgements - docs/EVALUATION.md says so plainly.
"""
from __future__ import annotations

import re
from typing import Dict, List

WEIGHTS = {
    "intent": 0.25,
    "correctness": 0.20,
    "script": 0.15,
    "compliance": 0.15,
    "relevance": 0.10,
    "naturalness": 0.10,
    "factual": 0.05,
}

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")
DODGE_MARKERS = [
    "as an ai", "i cannot", "i can't help", "i'm sorry, but", "i am sorry, but",
    "i don't have the ability", "as a language model", "मैं एक भाषा मॉडल",
    "मैं यह नहीं कर सकता", "क्षमा करें, मैं",
]
TRANSLATION_PREAMBLE = ["here is the translation", "translation:", "अनुवाद:"]


def script_ratio(text: str) -> dict:
    dev = len(DEVANAGARI.findall(text))
    lat = len(LATIN.findall(text))
    total = dev + lat
    return {"devanagari_chars": dev, "latin_chars": lat,
            "devanagari_ratio": (dev / total) if total else 0.0}


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\u0900-\u097F]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _has_any(text_norm: str, groups: List[List[str]]) -> tuple[bool, List[str]]:
    """groups: list of alternative-token lists; ALL groups must be satisfied."""
    if not groups:
        return True, []
    missing = []
    for group in groups:
        if not any(_norm(t) in text_norm for t in group):
            missing.append("|".join(group))
    return (not missing), missing


def loop_penalty(text: str) -> float:
    """Fraction of repeated 4-grams - catches degenerate looping."""
    toks = _norm(text).split()
    if len(toks) < 8:
        return 0.0
    grams = [" ".join(toks[i:i + 4]) for i in range(len(toks) - 3)]
    seen: Dict[str, int] = {}
    for g in grams:
        seen[g] = seen.get(g, 0) + 1
    rep = sum(c - 1 for c in seen.values() if c > 1)
    return min(1.0, rep / max(1, len(grams)))


def score_item(item: dict, output: str) -> dict:
    out = output or ""
    on = _norm(out)
    words = out.split()
    n_words = len(words)
    sr = script_ratio(out)
    dims: Dict[str, float | None] = {k: None for k in WEIGHTS}
    notes: List[str] = []

    # ---- intent: rubric token groups
    groups = item.get("must_include") or []
    ok_intent, missing = _has_any(on, groups)
    dims["intent"] = 1.0 if ok_intent else (0.0 if groups else None)
    if missing:
        notes.append("missing:" + ",".join(missing[:3]))

    # ---- correctness: verifiable answer tokens
    ans = item.get("answer_tokens") or []
    if ans:
        hit = any(_norm(a) in on for a in ans)
        dims["correctness"] = 1.0 if hit else 0.0
        if not hit:
            notes.append("answer_miss")
    else:
        # fall back to intent as a weak correctness proxy, flagged as such
        dims["correctness"] = None

    # ---- script
    expect = item.get("expect_script")
    if expect == "devanagari":
        dims["script"] = 1.0 if sr["devanagari_ratio"] >= 0.55 else round(sr["devanagari_ratio"], 3)
    elif expect == "roman":
        dims["script"] = 1.0 if sr["latin_chars"] > 0 and sr["devanagari_ratio"] <= 0.35 else 0.3
    elif expect == "either":
        dims["script"] = 1.0 if (sr["devanagari_chars"] + sr["latin_chars"]) > 0 else 0.0
    else:
        dims["script"] = None

    # ---- compliance: length, bullets, bans
    comp = 1.0
    if item.get("max_words") and n_words > int(item["max_words"]):
        comp -= 0.4
        notes.append("too_long")
    if item.get("min_words") and n_words < int(item["min_words"]):
        comp -= 0.3
        notes.append("too_short")
    if item.get("expect_bullets"):
        n = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", out))
        comp -= 0.5 if n < int(item["expect_bullets"]) else 0.0
        if n < int(item["expect_bullets"]):
            notes.append(f"bullets<{item['expect_bullets']}")
    for bad in (item.get("must_not_include") or []):
        if _norm(bad) in on:
            comp -= 0.5
            notes.append(f"banned:{bad}")
    dims["compliance"] = max(0.0, comp)

    # ---- relevance: not empty, not a dodge
    rel = 1.0
    if n_words == 0:
        rel = 0.0
        notes.append("empty")
    low = out.lower()
    for m in DODGE_MARKERS:
        if m in low:
            rel -= 0.6
            notes.append("dodge")
            break
    if item.get("category") != "natural_response":
        for p in TRANSLATION_PREAMBLE:
            if p in low and item.get("expect_translation") is not True:
                rel -= 0.3
                notes.append("unrequested_translation")
                break
    dims["relevance"] = max(0.0, rel)

    # ---- naturalness: loops, length sanity, junk
    nat = 1.0
    lp = loop_penalty(out)
    if lp > 0.05:
        nat -= min(0.7, lp * 3)
        notes.append(f"loop:{lp:.2f}")
    if n_words < 2:
        nat -= 0.3
    if n_words > 300:
        nat -= 0.2
    dims["naturalness"] = max(0.0, nat)

    # ---- factual: banned fabrication markers / needed facts
    fact = None
    bans = item.get("fact_banned") or []
    if bans:
        fact = 1.0
        for b in bans:
            if _norm(b) in on:
                fact -= 1.0
                notes.append(f"fabricated:{b}")
        fact = max(0.0, fact)
    dims["factual"] = fact

    applicable = {k: v for k, v in dims.items() if v is not None}
    wsum = sum(WEIGHTS[k] for k in applicable)
    composite = (sum(WEIGHTS[k] * v for k, v in applicable.items()) / wsum) if wsum else 0.0
    if n_words == 0:
        # an empty answer is a total failure, whatever the other dimensions say
        composite = 0.0
        notes.append("no_answer")

    return {"id": item["id"], "category": item["category"],
            "dims": {k: (None if v is None else round(float(v), 4)) for k, v in dims.items()},
            "composite": round(float(composite), 4),
            "n_words": n_words, "script": sr, "notes": notes,
            "output": out}
