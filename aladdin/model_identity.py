"""Single source of truth for the REQUIRED runtime checkpoint.

Aladdin WSP must run on the MIT WSP researcher's already-edited Qwen3.5-0.8B
checkpoint, never on the base Qwen/Qwen3.5-0.8B release. Every module that names
a default model (config, loader, CLI, docs, tests) imports from here so the
identity cannot drift between components.

The values below were read from the real repository on 2026-11-12, not guessed:

    GET https://huggingface.co/api/models/RikiyaT/Qwen3.5-0.8B-weight-space-programming
    sha            d1671a178705bcab7644806d23c535517f35723b
    architectures  ["Qwen3_5ForConditionalGeneration"]
    model_type     qwen3_5
    text_config    model_type qwen3_5_text, hidden_size 1024,
                   num_hidden_layers 24, vocab_size 248320,
                   intermediate_size 3584, num_attention_heads 8,
                   num_key_value_heads 2, tie_word_embeddings true,
                   layer_types 18x linear_attention + 6x full_attention
    weights        488 tensors: model.language_model.* (320),
                   model.visual.* (153), mtp.* (15); 1,746,882,752 bytes
"""
from __future__ import annotations

# The edited checkpoint. This is the default everywhere and is never replaced by
# the base release.
REQUIRED_MODEL_ID = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
# Revision published at the time this project was written. A snapshot whose
# metadata records a different commit is reported, not silently accepted.
KNOWN_MODEL_REVISION = "d1671a178705bcab7644806d23c535517f35723b"
BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"   # reference only - NOT a fallback target

# Geometry from the checkpoint's own text_config (used to spot a swapped or
# truncated checkpoint when download metadata is unavailable).
EXPECTED_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
EXPECTED_MODEL_TYPE = "qwen3_5"
EXPECTED_TEXT_MODEL_TYPE = "qwen3_5_text"
EXPECTED_HIDDEN_SIZE = 1024
EXPECTED_NUM_HIDDEN_LAYERS = 24
EXPECTED_VOCAB_SIZE = 248320
EXPECTED_INTERMEDIATE_SIZE = 3584
EXPECTED_NUM_ATTENTION_HEADS = 8
EXPECTED_NUM_KEY_VALUE_HEADS = 2
EXPECTED_TIE_WORD_EMBEDDINGS = True

# Weight-namespace layout of the published checkpoint (verified from
# model.safetensors.index.json).
TEXT_WEIGHT_PREFIX = "model.language_model."
VISION_WEIGHT_PREFIX = "model.visual."
MTP_WEIGHT_PREFIX = "mtp."
TOTAL_CHECKPOINT_TENSORS = 488
TEXT_CHECKPOINT_TENSORS = 320
VISION_CHECKPOINT_TENSORS = 153
MTP_CHECKPOINT_TENSORS = 15

# Metadata file written next to a downloaded snapshot by aladdin.model.loader.
SNAPSHOT_METADATA_FILE = "aladdin_checkpoint.json"


def expected_text_geometry(cfg: dict | None = None) -> dict:
    """Geometry a text tower must have to be this checkpoint's text tower.

    `cfg` (a parsed config.json) overrides the recorded constants so the check
    works for a different but explicitly requested model, while the default path
    always compares against the required checkpoint.
    """
    text = ((cfg or {}).get("text_config") or {})
    return {
        "model_id": REQUIRED_MODEL_ID,
        "architecture": EXPECTED_ARCHITECTURE,
        "text_model_type": text.get("model_type", EXPECTED_TEXT_MODEL_TYPE),
        "hidden_size": int(text.get("hidden_size", EXPECTED_HIDDEN_SIZE)),
        "num_hidden_layers": int(text.get("num_hidden_layers",
                                         EXPECTED_NUM_HIDDEN_LAYERS)),
        "vocab_size": int(text.get("vocab_size", EXPECTED_VOCAB_SIZE)),
        "intermediate_size": int(text.get("intermediate_size",
                                          EXPECTED_INTERMEDIATE_SIZE)),
        "num_hidden_layers_cfg": int(text.get("num_hidden_layers",
                                              EXPECTED_NUM_HIDDEN_LAYERS)),
    }
