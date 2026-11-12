"""Experiment reporting: before/after tables, kept-edit audit, Markdown output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .evaluation.runner import category_delta_table


def load_json(p) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def experiment_report(exp_dir: str | Path, heldout: dict | None = None) -> str:
    d = Path(exp_dir)
    state = load_json(d / "state.json") if (d / "state.json").exists() else {}
    base = load_json(d / "baseline_dev.json") if (d / "baseline_dev.json").exists() else None
    hist = []
    hp = d / "history.jsonl"
    if hp.exists():
        for line in hp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                hist.append(json.loads(line))

    L: List[str] = []
    L.append(f"# Experiment report — `{state.get('experiment', d.name)}`\n")
    L.append(f"- backend: `{state.get('backend')}`")
    L.append(f"- status: **{state.get('status')}**  |  steps: {state.get('step')}/{state.get('max_steps')}")
    L.append(f"- baseline dev composite: **{state.get('baseline_composite')}**")
    L.append(f"- best dev composite: **{state.get('best_composite')}**")
    L.append(f"- kept edits: {len(state.get('kept_edits', []))}  |  rejected: {len(state.get('rejected_edits', []))}\n")

    if base:
        L.append("## Baseline (dev split)\n")
        L.append("| category | n | mean | min | max |")
        L.append("|---|---|---|---|---|")
        for c, st in base["aggregate"]["categories"].items():
            L.append(f"| {c} | {st['n']} | {st['mean']} | {st['min']} | {st['max']} |")
        a = base["aggregate"]
        L.append(f"| **ALL** | {a['n']} | **{a['composite']}** | | |\n")

    if hist:
        L.append("## Candidates\n")
        L.append("| step | edit | tensor | op | Δ composite | verdict |")
        L.append("|---|---|---|---|---|---|")
        for h in hist:
            op = h.get("operation") or {}
            t = op.get("tensor", h.get("refused", "")[:30])
            kind = op.get("kind", "-")
            dl = h.get("delta")
            dl_s = f"{dl:+.4f}" if isinstance(dl, (int, float)) else "-"
            verdict = "ACCEPT" if h.get("accepted") else ("refused" if h.get("refused") else "reject")
            L.append(f"| {h.get('step')} | {h.get('edit_id', '-')} | `{t}` | {kind} | {dl_s} | {verdict} |")
        L.append("")

    if heldout:
        L.append("## Held-out split (touched only for this report)\n")
        L.append("| category | baseline | edited | Δ |")
        L.append("|---|---|---|---|")
        for r in category_delta_table(heldout["baseline"], heldout["edited"]) if "baseline" in heldout and "edited" in heldout else []:
            L.append(f"| {r['category']} | {r['baseline']} | {r['other']} | {r['delta']} |")
        L.append("")

    L.append("## How to read this\n")
    L.append("Scores come from `aladdin/evaluation/scorer.py`: a deterministic rubric over the")
    L.append("model's real output (intent tokens, verifiable answer, script, instruction")
    L.append("constraints, relevance, loop/naturalness, fabrication bans). They are heuristic,")
    L.append("not human ratings, and a `null` dimension means 'not applicable to that item'.\n")
    return "\n".join(L)


def write_markdown(text: str, out: str | Path) -> str:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)
