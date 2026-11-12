"""Thin OpenAI-compatible client for NVIDIA's hosted endpoints.

Secrets: the key is read from NVIDIA_API_KEY only. It is never written to config,
logs, state files, checkpoints or the packaged ZIP - `redact()` scrubs it from any
string that is about to be persisted, and tests/test_secrets.py enforces that.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import requests

SECRET_ENV = "NVIDIA_API_KEY"


def redact(text: str, extra: List[str] | None = None) -> str:
    """Remove any live secret from a string before it is logged/stored."""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    secrets = [os.environ.get(SECRET_ENV, "")] + list(extra or [])
    for s in secrets:
        if s and len(s) > 8:
            text = text.replace(s, "<redacted>")
            # also catch the common "nvapi-" prefix fragment
            if s.startswith("nvapi-") and len(s) > 20:
                text = text.replace(s[6:], "<redacted>")
    return text


class NvidiaError(RuntimeError):
    pass


class NvidiaClient:
    def __init__(self, cfg: dict):
        a = cfg["agent"]
        self.base_url = (a.get("base_url") or "https://integrate.api.nvidia.com/v1").rstrip("/")
        self.model = a.get("model") or ""
        self.timeout = int(a.get("timeout", 180))
        self.max_retries = int(a.get("max_retries", 3))
        self.api_key = os.environ.get(SECRET_ENV, "")

    # ------------------------------------------------------------------ state
    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    def status(self) -> dict:
        return {
            "api_key_present": bool(self.api_key),
            "api_key_prefix_ok": self.api_key.startswith("nvapi-") if self.api_key else False,
            "model": self.model or "<unset>",
            "base_url": self.base_url,
            "configured": self.configured(),
        }

    # ------------------------------------------------------------------- calls
    def _post(self, path: str, payload: dict) -> dict:
        if not self.api_key:
            raise NvidiaError(
                f"{SECRET_ENV} is not set. Export it (see .env.example) or use "
                "`--agent scripted` to run the deterministic exploration policy.")
        if not self.model:
            raise NvidiaError(
                "agent.model is empty. Set NVIDIA_MODEL (see docs/NVIDIA_MODEL.md for "
                "the models that were available when this project was written).")
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json", "Accept": "application/json"}
        last = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code}: {redact(r.text[:400])}"
                if r.status_code in (400, 401, 403, 404):
                    break                        # retrying will not help
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(8, 1.5 * (attempt + 1)))
        raise NvidiaError(f"NVIDIA API call failed ({url}): {last}")

    def list_models(self) -> List[str]:
        """GET /models - verify the model id instead of trusting memory."""
        if not self.api_key:
            raise NvidiaError(f"{SECRET_ENV} is not set")
        r = requests.get(f"{self.base_url}/models",
                         headers={"Authorization": f"Bearer {self.api_key}"}, timeout=60)
        if r.status_code != 200:
            raise NvidiaError(f"HTTP {r.status_code}: {redact(r.text[:300])}")
        data = r.json()
        return sorted(m.get("id") for m in data.get("data", []) if m.get("id"))

    def chat(self, messages: List[Dict], max_tokens: int = 1024, temperature: float = 0.2,
             stop: List[str] | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": 0.9,
        }
        if stop:
            payload["stop"] = stop
        data = self._post("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise NvidiaError(f"unexpected response shape: {redact(json.dumps(data)[:300])}")
