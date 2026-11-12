"""Aladdin WSP - Weight-Space Programming for Hindi & Hinglish.

Runs on the MIT WSP researcher's edited checkpoint
`RikiyaT/Qwen3.5-0.8B-weight-space-programming` (see aladdin.model_identity).

Direct tensor editing only: inspect -> hypothesise -> bounded edit -> sparse
delta -> measure -> accept/reject -> exact rollback. No LoRA, no SFT, no
fine-tuning, no gradient descent, no optimizer. See docs/METHOD.md.
"""

__version__ = "4.0.0"

from .model_identity import (BASE_MODEL_ID, KNOWN_MODEL_REVISION,  # noqa: E402
                             REQUIRED_MODEL_ID)

DEFAULT_MODEL_ID = REQUIRED_MODEL_ID

__all__ = ["__version__", "REQUIRED_MODEL_ID", "DEFAULT_MODEL_ID",
           "KNOWN_MODEL_REVISION", "BASE_MODEL_ID"]
