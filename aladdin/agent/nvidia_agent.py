"""The NVIDIA-backed reasoning/editing agent.

Loop: build the system prompt from the live tool schema -> send the failure
digest -> the model answers with one JSON tool call -> the runtime executes it
against the real model -> the compact result goes back as the next user turn ->
repeat until `finish`, a step cap, or an unrecoverable API error.

Synchronization rules (enforced here, tested in tests/test_v4_ledger_and_agent.py):
* the set of active edit IDs comes from the RUNTIME (`runtime.applied`), which
  changes only after a hash-verified tensor restore;
* a revert result's `undone` list removes those IDs; a failed revert removes
  nothing;
* `revert_transaction` is forwarded with its `txn_id`/`cascade` arguments, so a
  committed transaction can be reverted by ID;
* the ledger's own status is logged after every revert, so an audit can compare
  the agent's claims with the durable record.

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
        self.applied_here: List[str] = []

    def available(self) -> bool:
        return self.client.configured()

    def _log(self, rec: dict) -> None:
        rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.transcript, "a", encoding="utf-8") as f:
            f.write(redact(json.dumps(rec, ensure_ascii=False)) + "\n")

    def active_edit_ids(self) -> List[str]:
        """The runtime is authoritative: never report a reverted edit as active."""
        return [a.get("edit_id") for a in self.runtime.applied]

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
        self._log({"event": "step_start", "step": step,
                   "active_edit_ids": self.active_edit_ids()})
        applied_here: List[str] = []
        self.applied_here = applied_here
        summary = ""
        turn = 0
        for turn in range(int(self.cfg["agent"].get("max_turns", self.max_steps))):
            try:
                reply = self.client.chat(
                    messages,
                    max_tokens=int(self.cfg["agent"].get("max_tokens", 2048)),
                    temperature=float(self.cfg["agent"].get("temperature", 0.2)),
                )
            except NvidiaError as e:
                self._log({"event": "api_error", "error": str(e)})
                return {"ok": False, "error": str(e), "applied": list(applied_here),
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
                # 'undone' contains REAL edit IDs, set only after the tensors were
                # restored and hash-verified by the runtime, so this can never
                # claim a stale edit as applied.
                undone = set(result.get("undone") or [])
                if result.get("ok"):
                    removed = [e for e in applied_here if e in undone]
                    applied_here = [e for e in applied_here if e not in undone]
                    self.applied_here = applied_here
                    self._log({"event": "revert_applied", "tool": tool,
                               "txn_id": result.get("txn_id"),
                               "undone": sorted(undone),
                               "removed_from_applied_here": removed,
                               "still_applied": list(applied_here),
                               "runtime_active": self.active_edit_ids(),
                               "ledger": self.runtime.ledger.status_report(),
                               "verified": result.get("verified")})
                else:
                    # failed revert: applied_here intentionally UNCHANGED
                    self._log({"event": "revert_failed", "tool": tool,
                               "error": result.get("error"),
                               "still_applied": list(applied_here),
                               "runtime_active": self.active_edit_ids()})
            if tool == "finish":
                summary = result.get("summary") or args.get("summary", "")
                self._log({"event": "finish", "summary": summary,
                           "active_edit_ids": self.active_edit_ids()})
                break
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",
                             "content": "TOOL RESULT " + tool + ": " +
                                        redact(json.dumps(result, ensure_ascii=False))[:6000]})
            if self.runtime.fwd.remaining() <= 0 and tool in ("forward_generate", "activation_contrast"):
                messages.append({"role": "user", "content": "Budget exhausted; use preview_edit/"
                                                            "evaluate_candidate or finish."})
        return {"ok": True, "applied": list(applied_here),
                "summary": summary or "(no summary)", "turns": turn + 1}
