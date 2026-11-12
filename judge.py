"""Optional NVIDIA LLM judge for the 'naturalness' axis.

Disabled unless `--judge` is passed AND NVIDIA_API_KEY is set. Returns nothing but
a boolean+score; the deterministic rubric in scorer.py stays authoritative for
accept/reject so a flaky judge can never silently rewrite the decision.
"""
from __future__ import annotations

import json
import os
import re


class NvidiaJudge:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg["agent"].get("judge_model") or cfg["agent"].get("model") or ""
        self.api_key = os.environ.get("NVIDIA_API_KEY", "")

    def available(self) -> bool:
        return bool(self.api_key and self.model)

    def score(self, item: dict, output: str) -> dict:
        from ..agent.nvidia_client import NvidiaClient

        client = NvidiaClient(self.cfg)
        sys_p = ("You are a strict bilingual Hindi/Hinglish evaluator. Score the assistant reply "
                 "for naturalness and appropriateness for an Indian user. Reply ONLY with JSON: "
                 '{"score": 0.0-1.0, "why": "<12 words"}')
        usr = json.dumps({"user_message": item["prompt"],
                          "expected_intent": item.get("meaning", ""),
                          "assistant_reply": output}, ensure_ascii=False)
        txt = client.chat([{"role": "system", "content": sys_p},
                           {"role": "user", "content": usr}], max_tokens=120, temperature=0.0)
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return {"score": None, "why": "unparseable judge reply"}
        try:
            d = json.loads(m.group(0))
            return {"score": float(d.get("score")), "why": str(d.get("why", ""))[:80],
                    "model": self.model}
        except Exception:
            return {"score": None, "why": "invalid judge json"}
