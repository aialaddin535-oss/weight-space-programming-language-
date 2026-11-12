"""Budgeted forward-pass experiments.

Every call that runs the model costs one unit of the configured budget; the
budget exists so an agent cannot brute-force the evaluation set, and so a run's
cost is known in advance. Each completed pass is appended to
`<run_dir>/forward_passes.jsonl` with its prompts, outputs and timing, which
makes an experiment auditable after the fact.

Nothing here trains anything: passes are `torch.no_grad()`/numpy forward
evaluations that only read the current weights.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..inspection.activations import capture, logprob_of


class BudgetExhausted(RuntimeError):
    """The configured forward-pass budget has been used up."""


class ForwardPassEngine:
    def __init__(self, backend, budget: int = 50, run_dir: str | Path | None = None,
                 max_new_tokens: int = 32):
        self.backend = backend
        self.budget = max(0, int(budget))
        self.run_dir = Path(run_dir) if run_dir else None
        self.max_new_tokens = int(max_new_tokens)
        self.used = 0
        self.by_kind: Dict[str, int] = {}
        self.records: List[dict] = []
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = (self.run_dir / "forward_passes.jsonl") if self.run_dir else None

    # ------------------------------------------------------------------ budget
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    def _charge(self, kind: str, payload: dict) -> dict:
        """Consume one unit and persist the record. Callers charge only after a
        pass actually succeeded, so a failed request is free."""
        if self.remaining() <= 0:
            raise BudgetExhausted(
                f"forward-pass budget exhausted ({self.used}/{self.budget}); "
                "raise forward_pass.budget or continue with the evidence you have")
        self.used += 1
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
        rec = {"id": f"fp_{uuid.uuid4().hex[:8]}", "kind": kind,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "index": self.used, **payload}
        self.records.append(rec)
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False)[:20000] + "\n")
        return rec

    # ------------------------------------------------------------------ passes
    def generate(self, prompts: Sequence[str], max_new_tokens: int | None = None,
                 temperature: float = 0.0, seed: int = 0) -> dict:
        """Generate a continuation for each prompt under one fixed protocol."""
        prompts = [str(p) for p in prompts]
        if not prompts:
            raise ValueError("no prompts given")
        n = int(max_new_tokens if max_new_tokens is not None else self.max_new_tokens)
        t0 = time.time()
        outputs = [self.backend.generate(p, max_new_tokens=n, temperature=temperature,
                                         top_k=0, seed=seed) for p in prompts]
        rec = self._charge("generate", {
            "prompts": prompts, "outputs": outputs,
            "max_new_tokens": n, "temperature": float(temperature),
            "elapsed_s": round(time.time() - t0, 3)})
        return {**rec, "budget_left": self.remaining()}

    def teacher_forced(self, pairs: Sequence[Tuple[str, str]]) -> dict:
        """Mean log-probability of each continuation given its prompt.

        A deterministic, cheap signal for 'does the model consider this answer
        likely', not an accuracy score.
        """
        pairs = [(str(a), str(b)) for a, b in pairs]
        if not pairs:
            raise ValueError("no (prompt, continuation) pairs given")
        vals = [logprob_of(self.backend, p, c) for p, c in pairs]
        clean = [None if (v is None or not np.isfinite(v)) else round(float(v), 6)
                 for v in vals]
        rec = self._charge("teacher_forced", {
            "pairs": [{"prompt": p, "continuation": c} for p, c in pairs],
            "mean_logprob": clean})
        return {**rec, "budget_left": self.remaining()}

    def activations(self, prompts: Sequence[str]) -> dict:
        """Capture per-layer activations for each prompt (for contrast analysis)."""
        prompts = [str(p) for p in prompts]
        if not prompts:
            raise ValueError("no prompts given")
        snaps = [capture(self.backend, p) for p in prompts]
        rec = self._charge("activations", {
            "prompts": prompts,
            "snapshots": [{k: v for k, v in s.items() if k != "_raw"} for s in snaps]})
        return {**rec, "budget_left": self.remaining()}

    # ------------------------------------------------------------------ report
    def summary(self) -> dict:
        return {"budget": self.budget, "used": self.used,
                "remaining": self.remaining(), "by_kind": dict(self.by_kind),
                "log": str(self.log_path) if self.log_path else None}
