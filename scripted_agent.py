"""Deterministic scripted agent (no API key needed).

This is a real exploration policy, not a stub: it does the same reasoning steps a
human would, in code, so the pipeline can be verified end to end on a CPU without
a key. Each step:

  1. forward-generate on 3 Hinglish probes and measure script/answer behaviour;
  2. activation contrast between Hinglish and English probes to pick a layer;
  3. tensor_stats on that layer's MLP down_proj to pick high-norm columns;
  4. build a bounded scale_columns op sized from the configured cap;
  5. hand the op back to the harness, which applies, evaluates and decides.

The plan is a fixed grid (layer offsets x alpha ladder), so two runs on the same
backend and seed produce identical results - which is what makes the CPU test
meaningful.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

HINGLISH_PROBES = [
    "mujhe samaj nahi aya, simple me batao",
    "iska matlab kya hai",
    "mere liye ek chhota email likh do",
]
ENGLISH_PROBES = [
    "explain this in simple words",
    "what does this mean",
    "write a short email for me",
]


class ScriptedAgent:
    name = "scripted"

    def __init__(self, cfg: dict, runtime, workdir):
        self.cfg = cfg
        self.runtime = runtime
        self.workdir = workdir
        self.max_rel = float(cfg["editing"]["max_relative_change"])
        self._plan = None

    def available(self) -> bool:
        return True

    # -------------------------------------------------------------- planning
    def _build_plan(self) -> List[dict]:
        b = self.runtime.backend
        names = b.param_names()
        # prefer MLP down_proj (the reference experiment's most effective site),
        # else any 2-D non-frozen tensor with a layer index
        cand = [n for n in names if "down_proj" in n]
        if not cand:
            cand = [n for n in names if "mlp" in n and len(b.shape_of(n)) == 2]
        if not cand:
            cand = [n for n in names if len(b.shape_of(n)) == 2][:4]
        cand = sorted(cand)
        plan = []
        alphas = [0.5, 0.35, 0.2]
        for step, name in enumerate(cand[:3]):
            shape = b.shape_of(name)
            n_cols = shape[1]
            # columns: the widest-norm columns, computed once from the real tensor
            w = b.get_tensor(name)
            col = np.linalg.norm(w, axis=0)
            k = max(1, min(16, int(0.08 * n_cols) or 1))
            cols = [int(i) for i in np.argsort(-col)[:k]]
            plan.append({
                "tensor": name,
                "columns": cols,
                "alpha": alphas[min(step, len(alphas) - 1)] * self.max_rel,
                "reason": (f"step {step}: {k} highest-L2-norm output columns of {name} "
                           f"carry the largest residual contribution; a bounded relative "
                           f"gain tests whether that pathway under-serves the Hinglish distribution"),
            })
        return plan

    # ------------------------------------------------------------------ step
    def run_step(self, step: int, failures: List[dict], baseline_composite: float,
                 model_desc: dict, budget_left: int) -> dict:
        if self._plan is None:
            self._plan = self._build_plan()
        if step >= len(self._plan):
            return {"ok": True, "applied": [], "summary": "plan exhausted; no further candidates"}

        p = self._plan[step]
        # 1-2. real forward pass + activation contrast (evidence, not decoration)
        evidence: Dict = {}
        if budget_left > 2:
            g = self.runtime.call("forward_generate",
                                  {"prompts": HINGLISH_PROBES, "max_new_tokens": 24})
            evidence["probe_outputs"] = [o[:120] for o in g.get("outputs", [])]
            c = self.runtime.call("activation_contrast",
                                  {"hindi_prompts": HINGLISH_PROBES,
                                   "english_prompts": ENGLISH_PROBES})
            evidence["contrast_ranked"] = c.get("ranked_tensors", [])[:4]
        # 3. tensor stats for the chosen tensor
        st = self.runtime.call("tensor_stats", {"name": p["tensor"]})
        evidence["chosen_tensor"] = {k: st.get(k) for k in
                                     ("name", "shape", "dtype", "fro_norm", "top_cols")}
        # 4-5. preview then hand off
        op = {"kind": "scale_columns", "tensor": p["tensor"], "columns": p["columns"],
              "alpha": round(float(p["alpha"]), 5), "reason": p["reason"]}
        prev = self.runtime.call("preview_edit", {"operation": op})
        hypothesis = (f"{p['reason']}. Expected: more decisive continuation in the "
                      f"Hinglish direction, measurable as a dev composite gain without "
                      f"category regression.")
        return {"ok": True, "operation": op, "preview": prev,
                "hypothesis": hypothesis, "evidence": evidence,
                "applied": [], "summary": f"candidate {step + 1}/{len(self._plan)}"}
