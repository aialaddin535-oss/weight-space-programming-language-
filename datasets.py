"""Dataset loading, splitting and validation.

Two JSONL files ship with the project: `datasets/dev.jsonl` (used while searching
for edits) and `datasets/heldout.jsonl` (touched only for the final report, never
during the search). Splitting is enforced in code: the experiment loop is given a
Dataset object built with split="dev", and `assert_no_heldout_leak` fails the run
if a held-out id shows up in a dev evaluation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

CATEGORIES = (
    "roman_hinglish",       # 1. Roman Hinglish understanding
    "devanagari_hindi",     # 2. Devanagari Hindi understanding
    "mixed_code_switch",    # 3. Mixed Hindi-English
    "context_followup",     # 4. Context / follow-up / reference resolution
    "instruction_following",  # 5. Instruction following
    "question_answering",   # 6. QA
    "natural_response",     # 7. Natural response quality
    "spelling_robustness",  # 8. Spelling / typos / no punctuation
    "technical_hindi",      # 9. Technical Hindi/Hinglish
    "factuality",           # 10. Hallucination / factual accuracy
)

REQUIRED_FIELDS = ("id", "category", "prompt")


def load_jsonl(path: str | os.PathLike) -> List[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {p}")
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{i}: invalid JSON ({e})") from e
    return rows


def validate(rows: List[dict], name: str = "dataset") -> dict:
    seen, problems = set(), []
    for i, r in enumerate(rows):
        for fld in REQUIRED_FIELDS:
            if fld not in r or not str(r[fld]).strip():
                problems.append(f"{name}[{i}]: missing {fld}")
        if r.get("category") not in CATEGORIES:
            problems.append(f"{name}[{i}]: unknown category {r.get('category')!r}")
        if r.get("id") in seen:
            problems.append(f"{name}[{i}]: duplicate id {r.get('id')!r}")
        seen.add(r.get("id"))
    return {"name": name, "n": len(rows), "ok": not problems, "problems": problems,
            "categories": sorted({r.get("category") for r in rows})}


class Dataset:
    def __init__(self, rows: List[dict], split: str):
        self.rows = rows
        self.split = split
        self._validate()

    def _validate(self) -> None:
        v = validate(self.rows, self.split)
        if not v["ok"]:
            raise ValueError("invalid dataset:\n  " + "\n  ".join(v["problems"]))

    def ids(self) -> set:
        return {r["id"] for r in self.rows}

    def by_category(self) -> Dict[str, List[dict]]:
        out: Dict[str, List[dict]] = {}
        for r in self.rows:
            out.setdefault(r["category"], []).append(r)
        return out

    def head(self, n: int) -> "Dataset":
        return Dataset(self.rows[:n], self.split)

    @classmethod
    def load(cls, path, split: str) -> "Dataset":
        return cls(load_jsonl(path), split)


def load_splits(cfg: dict) -> tuple[Dataset, Dataset]:
    base = Path(cfg["_project_root"]) / cfg["paths"]["datasets"]
    dev = Dataset.load(base / cfg["evaluation"]["dev_file"], "dev")
    held = Dataset.load(base / cfg["evaluation"]["heldout_file"], "heldout")
    overlap = dev.ids() & held.ids()
    if overlap:
        raise ValueError(f"dev/heldout id overlap: {sorted(overlap)[:5]}")
    same_prompt = ({r["prompt"] for r in dev.rows} & {r["prompt"] for r in held.rows})
    if same_prompt:
        raise ValueError(f"{len(same_prompt)} identical prompts in dev and heldout "
                         "(held-out must be unseen)")
    return dev, held


def assert_no_heldout_leak(evaluated_ids: set, held: Dataset) -> None:
    leak = evaluated_ids & held.ids()
    if leak:
        raise AssertionError(
            f"held-out items were evaluated during the search: {sorted(leak)[:5]} - "
            "the search must only see the dev split")
