"""dtype-exact casting shared by the delta store and the editor.

bfloat16 has no numpy dtype, so it is handled as its raw uint16 bit pattern -
exactly the reference implementation's approach - which makes hash comparisons
byte-exact without requiring torch.
"""
from __future__ import annotations

import numpy as np

DT_BYTES = {
    "float32": (np.float32, 4),
    "float16": (np.float16, 2),
    "bfloat16": (np.uint16, 2),
}


def to_bf16_bits(x32: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit pattern, round-to-nearest-even."""
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def from_bf16_bits(bits: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def cast_to_dtype(x: np.ndarray, dtype: str) -> np.ndarray:
    """Cast float input to the storage representation of `dtype`."""
    dtype = dtype.lower()
    if dtype in ("float32", "fp32", "f32"):
        return np.ascontiguousarray(x, dtype=np.float32)
    if dtype in ("float16", "fp16", "f16", "half"):
        return np.ascontiguousarray(x, dtype=np.float16)
    if dtype in ("bfloat16", "bf16", "b16"):
        return to_bf16_bits(x)
    raise ValueError(f"unsupported dtype: {dtype}")


def to_dtype_bytes(x: np.ndarray, dtype: str) -> bytes:
    return np.ascontiguousarray(cast_to_dtype(x, dtype)).tobytes()


def bits_to_values(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Storage representation -> float32 values (for analysis)."""
    dtype = dtype.lower()
    if dtype in ("bfloat16", "bf16", "b16"):
        return from_bf16_bits(arr)
    return np.ascontiguousarray(arr, dtype=np.float32)
