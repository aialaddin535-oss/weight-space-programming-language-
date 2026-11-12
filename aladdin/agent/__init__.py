"""Agent factory: NVIDIA-hosted reasoning agent or the deterministic scripted one."""
from __future__ import annotations


def build_agent(cfg: dict, runtime, workdir):
    """Return the configured agent.

    `scripted` needs no credentials and is used by the test suite and by CPU
    runs. `nvidia` talks to an OpenAI-compatible endpoint and requires
    NVIDIA_API_KEY + NVIDIA_MODEL; it is never constructed with a hardcoded key.
    """
    provider = str(cfg["agent"].get("provider", "scripted")).strip().lower()
    if provider == "nvidia":
        from .nvidia_agent import NvidiaAgent
        return NvidiaAgent(cfg, runtime, workdir)
    if provider == "scripted":
        from .scripted_agent import ScriptedAgent
        return ScriptedAgent(cfg, runtime, workdir)
    raise ValueError(f"unknown agent provider {provider!r}; "
                     "choose 'nvidia' or 'scripted'")


__all__ = ["build_agent"]
