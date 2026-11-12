"""Backend contract shared by the tiny CPU model and the real HF model.

Every backend exposes tensors as float32 numpy arrays for analysis plus the exact
stored bit pattern for integrity hashing, so a delta recorded against one dtype
can be verified byte-for-byte after it is written back.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Tuple

import numpy as np


class BackendError(RuntimeError):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Backend:
    """Interface. Subclasses implement the underscored primitives."""

    kind = "base"
    supports_generate = True

    # ---- identity -------------------------------------------------------
    def describe(self) -> dict:
        raise NotImplementedError

    def param_names(self) -> List[str]:
        raise NotImplementedError

    def shape_of(self, name: str) -> Tuple[int, ...]:
        raise NotImplementedError

    def dtype_of(self, name: str) -> str:
        raise NotImplementedError

    # ---- tensors --------------------------------------------------------
    def get_tensor(self, name: str) -> np.ndarray:
        """float32 view for analysis (a copy - never a live handle)."""
        raise NotImplementedError

    def raw_bytes(self, name: str) -> bytes:
        """Exact stored bytes (bf16 -> uint16 bits) for hashing."""
        raise NotImplementedError

    def set_tensor(self, name: str, value: np.ndarray) -> None:
        """Write float32 values, cast to the tensor's own dtype."""
        raise NotImplementedError

    def tensor_hash(self, name: str) -> str:
        return sha256_bytes(self.raw_bytes(name))

    # ---- inference ------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Iterable[int]) -> str:
        raise NotImplementedError

    def forward_logits(self, input_ids: List[int], capture: dict | None = None) -> np.ndarray:
        """[T, vocab] float32 logits. `capture` collects activations by layer name."""
        raise NotImplementedError

    def generate(self, prompt: str, max_new_tokens: int, temperature: float = 0.0,
                 top_k: int = 0, seed: int = 0) -> str:
        raise NotImplementedError

    # ---- convenience ----------------------------------------------------
    def n_params(self) -> int:
        return int(sum(int(np.prod(self.shape_of(n))) for n in self.param_names()))

    def snapshot_hashes(self) -> Dict[str, str]:
        return {n: self.tensor_hash(n) for n in self.param_names()}

    def frozen(self, name: str, suffixes: Iterable[str]) -> bool:
        return any(name.endswith(s) for s in suffixes)
