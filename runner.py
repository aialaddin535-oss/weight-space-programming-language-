"""Evaluation runner: one protocol, every checkpoint, machine-readable output.

`run_eval` generates an answer for every item of a split under a fixed sampling
protocol, scores it, and writes:
    <out>/results.json    full per-item records + aggregates
    <out>/summary.csv     category table (for spreadsheets)
    <out>/generations.jsonl  prompt -> output, for human reading

The same function is used for the baseline, every candidate and the final
held-out report, so numbers are comparable by construction.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .datasets import Dataset
from .scorer import WEIGHTS, score_item

DIMS = list(WEIGHTS)


def run_eval(backend, dataset: Dataset, cfg: dict, out_dir: str | Path | None = None,
             label: str = "baseline", limit_per_category: int = 0,
             judge=None) -> dict:
    ev = cfg["evaluation"]
    max_new = int(ev["max_new_tokens"])
    temp = float(ev["temperature"])
    top_k = int(ev["top_k"])
    seed = int(ev.get("seed", 0))

    rows = dataset.rows
    if limit_per_category:
        rows = []
        for cat, items in dataset.by_category().items():
            rows.extend(items[:limit_per_category])

    t0 = time.time()
    scored: List[dict] = []
    for item in rows:
        try:
            out = backend.generate(item["prompt"], max_new_tokens=max_new,
                                   temperature=temp, top_k=top_k, seed=seed)
        except Exception as e:                       # never crash a whole run
            out = ""
            item = dict(item)
            item["_error"] = repr(e)
        s = score_item(item, out)
        if item.get("_error"):
            s["notes"] = s.get("notes", []) + [f"error:{item['_error']}"]
        if judge is not None and judge.available():
            try:
                s["judge"] = judge.score(item, out)
            except Exception as e:
                s["judge"] = {"error": repr(e)}
        scored.append(s)

    agg = aggregate(scored)
    result = {
        "label": label,
        "split": dataset.split,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_items": len(scored),
        "elapsed_s": round(time.time() - t0, 3),
        "protocol": {"max_new_tokens": max_new, "temperature": temp, "top_k": top_k, "seed": seed},
        "backend": backend.describe(),
        "aggregate": agg,
        "items": scored,
    }
    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
        with open(d / "generations.jsonl", "w", encoding="utf-8") as f:
            for s in scored:
                f.write(json.dumps({"id": s["id"], "category": s["category"],
                                    "output": s["output"], "composite": s["composite"]},
                                   ensure_ascii=False) + "\n")
        write_summary_csv(result, d / "summary.csv")
    return result


def aggregate(scored: List[dict]) -> dict:
    if not scored:
        return {"composite": 0.0, "n": 0, "categories": {}, "dims": {}}
    comps = [s["composite"] for s in scored]
    cats: Dict[str, List[float]] = {}
    for s in scored:
        cats.setdefault(s["category"], []).append(s["composite"])
    dims: Dict[str, List[float]] = {d: [] for d in DIMS}
    for s in scored:
        for d in DIMS:
            v = s["dims"].get(d)
            if v is not None:
                dims[d].append(v)
    cat_stats = {}
    for c, vals in sorted(cats.items()):
        arr = np.array(vals, dtype=float)
        cat_stats[c] = {"n": len(vals), "mean": round(float(arr.mean()), 4),
                        "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4),
                        "stdev": round(float(arr.std()), 4)}
    return {
        "composite": round(float(np.mean(comps)), 4),
        "composite_stdev": round(float(np.std(comps)), 4),
        "n": len(scored),
        "categories": cat_stats,
        "dims": {k: round(float(np.mean(v)), 4) for k, v in dims.items() if v},
        "empty_outputs": sum(1 for s in scored if not str(s.get("output", "")).strip()),
    }


def write_summary_csv(result: dict, path: str | Path) -> None:
    p = Path(path)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "split", "category", "n", "mean", "min", "max", "stdev"])
        for c, st in result["aggregate"]["categories"].items():
            w.writerow([result["label"], result["split"], c, st["n"], st["mean"],
                        st["min"], st["max"], st["stdev"]])
        a = result["aggregate"]
        w.writerow([result["label"], result["split"], "ALL", a["n"], a["composite"],
                    "", "", a["composite_stdev"]])


def compare(baseline: dict, candidate: dict, cfg: dict) -> dict:
    """Accept/reject decision with the configured criteria (never on vibes)."""
    acc = cfg["acceptance"]
    b, c = baseline["aggregate"], candidate["aggregate"]
    delta = round(c["composite"] - b["composite"], 4)
    per_cat = {}
    regressions = []
    for cat, st in c["categories"].items():
        bst = b["categories"].get(cat)
        if not bst:
            continue
        d = round(st["mean"] - bst["mean"], 4)
        per_cat[cat] = {"baseline": bst["mean"], "candidate": st["mean"], "delta": d}
        if d < -float(acc["max_category_regression"]):
            regressions.append({"category": cat, "delta": d})
    improved_cats = [k for k, v in per_cat.items() if v["delta"] > 0]
    accept = (delta >= float(acc["min_composite_gain"]) and not regressions
              and c["composite"] >= float(acc["min_abs_score"]))
    return {
        "baseline_composite": b["composite"],
        "candidate_composite": c["composite"],
        "delta": delta,
        "per_category": per_cat,
        "improved_categories": improved_cats,
        "regressions": regressions,
        "accept": bool(accept),
        "criteria": dict(acc),
        "reason": ("composite gain %.4f >= %.4f with no category regression > %.2f"
                   % (delta, float(acc["min_composite_gain"]), float(acc["max_category_regression"])))
                  if accept else
                  ("rejected: gain %.4f (needed %.4f), regressions=%d"
                   % (delta, float(acc["min_composite_gain"]), len(regressions))),
    }


def category_delta_table(baseline: dict, other: dict) -> List[dict]:
    rows = []
    b = baseline["aggregate"]["categories"]
    o = other["aggregate"]["categories"]
    for cat in sorted(set(b) | set(o)):
        bv = b.get(cat, {}).get("mean")
        ov = o.get(cat, {}).get("mean")
        rows.append({"category": cat, "baseline": bv, "other": ov,
                     "delta": None if (bv is None or ov is None) else round(ov - bv, 4)})
    return rows
