"""Backend factory: the tiny CPU fixture model or the real HF/WSP checkpoint."""
from __future__ import annotations

from .base import Backend, BackendError

SUPPORTED_KINDS = ("tiny_numpy", "hf_torch")


def build_backend(cfg: dict) -> Backend:
    """Instantiate the configured backend.

    `tiny_numpy` is a randomly-initialised infrastructure fixture (no language
    ability). `hf_torch` loads real weights through the Qwen3.5-aware loader,
    which refuses any checkpoint other than the configured model id.
    """
    kind = str(cfg["backend"].get("kind", "tiny_numpy")).strip()
    if kind == "tiny_numpy":
        from .tiny_numpy import TinyNumpyBackend
        return TinyNumpyBackend(cfg)
    if kind == "hf_torch":
        from .hf_torch import HFTorchBackend
        return HFTorchBackend(cfg)
    raise BackendError(
        f"unknown backend kind {kind!r}; choose one of {SUPPORTED_KINDS}")


__all__ = ["Backend", "BackendError", "build_backend", "SUPPORTED_KINDS"]
