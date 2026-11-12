# The required checkpoint: `RikiyaT/Qwen3.5-0.8B-weight-space-programming`

## 1. What it is

The MIT WSP researcher's already-edited Qwen3.5-0.8B checkpoint. It is the
project's default and only supported runtime model for real experiments:

* repository: <https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming>
* base model: `Qwen/Qwen3.5-0.8B` (reference only — **not** a default, **not** a
  fallback)
* revision at the time of writing: `d1671a178705bcab7644806d23c535517f35723b`
  (2026-11-02)
* weights: `model.safetensors-00001-of-00001.safetensors`, 1,746,882,752 bytes
* config: `architectures: ["Qwen3_5ForConditionalGeneration"]`,
  `model_type: "qwen3_5"`, `transformers_version: "4.57.0.dev0"`

Everything below was read from the published repository, not remembered.

## 2. Verified contents

```
config.json                                  text_config + vision_config + MTP
tokenizer.json, tokenizer_config.json,       fast tokenizer, chat template
merges.txt, vocab.json, chat_template.jinja
preprocessor_config.json,                    multimodal processor (unused in text mode)
video_preprocessor_config.json
model.safetensors-00001-of-00001.safetensors 1.75 GB
model.safetensors.index.json                 488 tensors
```

Tensor namespaces:

| prefix | tensors | role |
|---|---|---|
| `model.language_model.` | 320 | text tower (`embed_tokens`, 24 layers, `norm`) |
| `model.visual.` | 153 | vision tower |
| `mtp.` | 15 | multi-token-prediction head |

`text_config` geometry: hidden 1024, 24 layers (18 linear-attention + 6
full-attention), vocab 248320, intermediate 3584, 8 attention heads, 2 KV heads,
`tie_word_embeddings: true`.

There is no `lm_head.weight`: the head is tied to the embedding matrix.

## 3. Install the stack

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install torch 'transformers>=4.57' accelerate safetensors huggingface_hub
```

`transformers>=4.57` is required for the `qwen3_5` model type and the text-only
class. The loader checks the version first and raises an install instruction:

```
LoaderCompatibilityError: transformers 4.46.0 is too old for Qwen3.5
(config requires >= 4.57). Install: pip install -U 'transformers>=4.57'
```

It also checks that `Qwen3_5ForCausalLM` and `Qwen3_5TextConfig` exist.

## 4. Download

```bash
python -m aladdin.cli download
# or
bash scripts/download_model.sh
# or explicitly
python -m aladdin.cli download \
  --repo-id RikiyaT/Qwen3.5-0.8B-weight-space-programming \
  --dest models/Qwen3.5-0.8B-weight-space-programming

export ALADDIN_MODEL_PATH="$PWD/models/Qwen3.5-0.8B-weight-space-programming"
export ALADDIN_BACKEND=hf_torch
```

The download writes `aladdin_checkpoint.json`:

```json
{
 "repo_id": "RikiyaT/Qwen3.5-0.8B-weight-space-programming",
 "revision": "main",
 "sha": "d1671a178705bcab7644806d23c535517f35723b",
 "config_sha256": "...", "n_files": 15, "total_bytes": 1746882752,
 "is_required_wsp_checkpoint": true,
 "note": "pristine downloaded checkpoint; Aladdin WSP never writes here"
}
```

Downloading the base model is allowed but warns; loading it as the project model
is refused.

## 5. Verify before doing anything expensive

```bash
python -m aladdin.cli model-info     # identity + architecture + geometry + index
python -m aladdin.cli verify-model   # load and verify the live module tree
```

`model-info` exits non-zero when the snapshot does not record the required repo
id. `verify-model` loads the text tower, then checks class, config geometry, live
layer count, embedding/head shapes, parameter count, tokenizer, and the weight
index. A mismatch is an error, never a warning.

## 6. How the text tower is loaded

For text-only WSP (the default), the official class is `Qwen3_5ForCausalLM` with
`Qwen3_5TextConfig` — `AutoModelForImageTextToText` is only used when
`--load-mode multimodal` is requested explicitly.

Because the checkpoint stores the text tower under `model.language_model.*` and
`Qwen3_5ForCausalLM` expects `model.*`, the loader:

1. reads `model.safetensors.index.json` and plans the mapping:

   ```
   model.language_model.embed_tokens.weight -> model.embed_tokens.weight
   model.language_model.layers.N.*           -> model.layers.N.*
   model.language_model.norm.weight          -> model.norm.weight
   model.visual.*                            skipped (153)
   mtp.*                                     skipped (15)
   ```

2. builds the model on the `meta` device and `to_empty(device)`, so only one copy
   of the weights is ever allocated;
3. copies tensor-by-tensor from the safetensors shards, checking every shape and
   casting to the requested dtype;
4. treats an absent `lm_head.weight` as expected **only** because embeddings are
   tied;
5. verifies the result and raises if anything is missing, unexpected or
   geometrically wrong.

Nothing is trained or re-initialised: the published tensor values are copied
verbatim.

## 7. Run

GPU (bf16, text-only):

```bash
python -m aladdin.cli --config config/qwen35.json inspect --layers 6
python -m aladdin.cli --config config/qwen35.json baseline --split dev --limit-per-category 2
```

CPU smoke test (slow):

```bash
python -m aladdin.cli --config config/qwen35_cpu.json inspect
```

## 8. Troubleshooting

| symptom | cause / fix |
|---|---|
| `no config.json under ...` | `ALADDIN_MODEL_PATH` does not point at the snapshot |
| `transformers ... too old` | `pip install -U 'transformers>=4.57'` |
| `has no Qwen3_5ForCausalLM` | same upgrade |
| `refusing to load this checkpoint: ... records 'Qwen/...'` | you downloaded the base model; download the edited WSP checkpoint |
| `no model.safetensors...` | incomplete download; re-run `aladdin download` |
| `shape mismatches while loading the text tower` | the directory is not this checkpoint, or the key layout changed; do not force it |
| CPU killed during load | use bf16 (`config/qwen35_cpu.json` uses fp32 and needs ~2.5 GB) |

## 9. Never modify the snapshot

Aladdin WSP writes edits only in memory and stores deltas/checkpoints under
`checkpoints/` and `experiments/` (git-ignored). `tests/test_qwen35_real.py`
asserts the snapshot directory file list/sizes are unchanged after loading and
editing, and `tests/test_no_training.py` refuses committed weight files.
