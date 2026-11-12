"""The NVIDIA-backed reasoning/editing agent.

Loop: build system prompt from the live tool schema -> send the failure digest ->
the model answers with one JSON tool call -> the runtime executes it against the
real model -> the compact result goes back as the next user turn -> repeat until
`finish`, a step cap, or an unrecoverable API error.

Everything the agent does is recorded (agent_transcript.jsonl) with the API key
redacted, so an experiment can be audited or replayed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from .nvidia_client import NvidiaClient, NvidiaError, redact
from .prompt import first_user_message, render_system_prompt
from .tools import ToolRuntime, parse_tool_call


class NvidiaAgent:
    name = "nvidia"

    def __init__(self, cfg: dict, runtime: ToolRuntime, workdir: str | Path):
        self.cfg = cfg
        self.runtime = runtime
        self.workdir = Path(workdir)
        self.client = NvidiaClient(cfg)
        self.transcript = self.workdir / "agent_transcript.jsonl"
        self.max_steps = int(cfg["agent"].get("max_steps", 14))
        self.applied_before = 0

    def available(self) -> bool:
        return self.client.configured()

    def _log(self, rec: dict) -> None:
        rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.transcript, "a", encoding="utf-8") as f:
            f.write(redact(json.dumps(rec, ensure_ascii=False)) + "\n")

    # --------------------------------------------------------------- one step
    def run_step(self, step: int, failures: List[dict], baseline_composite: float,
                 model_desc: dict, budget_left: int) -> dict:
        sys_p = render_system_prompt(self.runtime.schema())
        messages: List[dict] = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": first_user_message({
                "baseline": {"composite": baseline_composite},
                "failures": failures[:6],
                "model": model_desc,
                "budget_left": budget_left,
                "step": step,
                "max_steps": self.max_steps,
            })},
        ]
        self._log({"event": "step_start", "step": step})
        applied_here: List[str] = []
        summary = ""
        for turn in range(int(self.cfg["agent"].get("max_steps", 14))):
            try:
                reply = self.client.chat(
                    messages,
                    max_tokens=int(self.cfg["agent"].get("max_tokens", 2048)),
                    temperature=float(self.cfg["agent"].get("temperature", 0.2)),
                )
            except NvidiaError as e:
                self._log({"event": "api_error", "error": str(e)})
                return {"ok": False, "error": str(e), "applied": applied_here,
                        "summary": "NVIDIA API unavailable"}
            self._log({"event": "assistant", "turn": turn, "content": reply})
            call = parse_tool_call(reply)
            if not call:
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user",
                                 "content": "No tool call found. Reply with exactly one fenced "
                                            "json block: {\"tool\": ..., \"args\": {...}}"})
                if turn >= 2:
                    break
                continue
            tool, args = call["tool"], call["args"]
            result = self.runtime.call(tool, args)
            if tool == "apply_edit" and result.get("ok"):
                applied_here.append(result["edit_id"])
            if tool in ("revert", "revert_transaction"):
                if result.get("ok"):
                    # 'undone' contains REAL edit IDs (set only after the tensors
                    # were restored and hash-verified by the runtime), so the
                    # tracking below can never claim a stale edit as applied.
                    undone = set(result.get("undone") or [])
                    removed = [e for e in applied_here if e in undone]
                    applied_here = [e for e in applied_here if e not in undone]
                    self._log({"event": "revert_applied", "undone": sorted(undone),
                               "removed_from_applied_here": removed,
                               "still_applied": list(applied_here),
                               "verified": result.get("verified")})
                else:
                    # failed revert: applied_here intentionally UNCHANGED
                    self._log({"event": "revert_failed", "result": result})
            if tool == "finish":
                summary = result.get("summary") or args.get("summary", "")
                self._log({"event": "finish", "summary": summary})
                break
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",
                             "content": "TOOL RESULT " + tool + ": " +
                                        redact(json.dumps(result, ensure_ascii=False))[:6000]})
            if self.runtime.fwd.remaining() <= 0 and tool in ("forward_generate", "activation_contrast"):
                messages.append({"role": "user", "content": "Budget exhausted; use preview_edit/"
                                                            "evaluate_candidate or finish."})
        return {"ok": True, "applied": applied_here, "summary": summary or "(no summary)",
                "turns": turn + 1}
