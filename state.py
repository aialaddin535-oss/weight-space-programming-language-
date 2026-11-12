"""Resumable experiment state.

Written after every step, atomic (tmp + rename), so Ctrl-C / OOM / spot-instance
loss never leaves a half-written state. `resume` re-reads it, reloads the base
model, replays the kept sparse deltas and continues at the recorded step.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List

STATE_VERSION = 2


class ExperimentState:
    def __init__(self, path: str | Path, data: dict):
        self.path = Path(path)
        self.data = data

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def create(cls, path: str | Path, *, experiment: str, backend: str,
               model_desc: dict, config_path: str, max_steps: int) -> "ExperimentState":
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": STATE_VERSION,
            "experiment": experiment,
            "backend": backend,
            "model": model_desc,
            "config_path": config_path,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_utc": None,
            "status": "running",
            "step": 0,
            "max_steps": int(max_steps),
            "baseline_composite": None,
            "best_composite": None,
            "kept_edits": [],          # edit_ids in application order
            "rejected_edits": [],
            "applied_total": 0,
            "history": [],
            "config_snapshot": {},
        }
        st = cls(p, data)
        st.save()
        return st

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentState":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"no state file at {p}")
        data = json.load(open(p, "r", encoding="utf-8"))
        if data.get("version") != STATE_VERSION:
            raise ValueError(f"state version {data.get('version')} != {STATE_VERSION}")
        return cls(p, data)

    def save(self) -> None:
        self.data["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ---------------------------------------------------------------- updates
    def record_baseline(self, composite: float) -> None:
        self.data["baseline_composite"] = round(float(composite), 5)
        if self.data["best_composite"] is None:
            self.data["best_composite"] = round(float(composite), 5)
        self.save()

    def record_step(self, entry: Dict) -> None:
        self.data["history"].append(entry)
        self.data["step"] = int(self.data.get("step", 0)) + 1
        # multi-edit tracking: every edit ID of the candidate is recorded (BUG 3)
        ids = list(entry.get("edit_ids") or ([entry["edit_id"]] if entry.get("edit_id") else []))
        if ids:
            self.data["applied_total"] = int(self.data.get("applied_total", 0)) + len(ids)
            if entry.get("accepted"):
                self.data["kept_edits"].extend(ids)
                self.data["best_composite"] = round(float(entry["candidate_composite"]), 5)
            else:
                self.data["rejected_edits"].extend(ids)
        if entry.get("txn_id"):
            self.data.setdefault("transactions", []).append(
                {"txn_id": entry["txn_id"], "step": entry.get("step"),
                 "edit_ids": ids, "accepted": bool(entry.get("accepted"))})
        self.save()

    def finish(self, status: str = "finished") -> None:
        self.data["status"] = status
        self.save()

    def is_resumable(self) -> bool:
        return (self.data.get("status") in ("running", "interrupted")
                and self.data.get("step", 0) < self.data.get("max_steps", 0))

    def summary(self) -> dict:
        d = self.data
        return {
            "experiment": d["experiment"], "status": d["status"], "step": d["step"],
            "max_steps": d["max_steps"], "baseline": d["baseline_composite"],
            "best": d["best_composite"], "kept": len(d["kept_edits"]),
            "rejected": len(d["rejected_edits"]), "kept_edits": d["kept_edits"],
        }
