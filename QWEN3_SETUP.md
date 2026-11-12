# Qwen3.5-0.8B — official loading, download and verification

All values below were verified against the **published Hugging Face files** of
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) (config.json and
model card, fetched 2026-11-11). Nothing is guessed.

## 1. What the official config actually says

| Field | Value (from `config.json`) |
|---|---|
| `architectures` | `["Qwen3_5ForConditionalGeneration"]` |
| `model_type` | `qwen3_5` |
| `text_config.model_type` | `qwen3_5_text` |
| `text_config.hidden_size` | `1024` |
| `text_config.num_hidden_layers` | `24` |
| `text_config.vocab_size` | `248320` |
| `text_config.tie_word_embeddings` | `true` |
| `text_config.dtype` | `bfloat16` |
| `vision_config` | present → the model is **multimodal** (text + image + video) |
| `transformers_version` | `4.57.0.dev0` → requires a **recent transformers** |

Consequences implemented in `aladdin/backends/qwen_loader.py`:

1. **`AutoModelForCausalLM` is NOT the official entry point** for this model:
   the architecture is `Qwen3_5ForConditionalGeneration` (a conditional-
   generation multimodal class). The loader resolves the class through
   `AutoModelForImageTextToText`, which maps the `architectures` field to the
   true class. Classic `*ForCausalLM` configs keep using `AutoModelForCausalLM`.
2. **Text-only use is fully supported**: the text tower lives in
   `model.config.text_config`; WSP only ever feeds `input_ids`. The image/video
   processor (`AutoProcessor`) is optional and not needed for WSP runs.
3. The loader **refuses silently-wrong loads**: after `from_pretrained` it
   checks the loaded class name against `architectures` and the module tree for
   the text-tower geometry, raising `LoaderCompatibilityError` on mismatch.
4. `transformers` older than **4.57** is rejected up front with an install
   command instead of a deep failure inside `from_pretrained`.

## 2. Download (Linux / macOS / cloud GPU, bash)

```bash
# needs: pip install huggingface_hub (opt-in, large download ~1.6 GB)
bash scripts/download_model.sh                          # Qwen/Qwen3.5-0.8B -> models/Qwen3.5-0.8B
# or explicitly:
bash scripts/download_model.sh Qwen/Qwen3.5-0.8B /data/models/Qwen3.5-0.8B
export ALADDIN_MODEL_PATH="$PWD/models/Qwen3.5-0.8B"
export ALADDIN_BACKEND=hf_torch
export ALADDIN_DEVICE=cuda        # or cpu (float32, slow smoke test)
python -m aladdin.cli inspect     # prints the REAL tensor table
```

## 2b. Download (Windows PowerShell)

```powershell
python -m aladdin.cli download --model Qwen/Qwen3.5-0.8B --dest models\Qwen3.5-0.8B
$env:ALADDIN_MODEL_PATH = "$PWD\models\Qwen3.5-0.8B"
$env:ALADDIN_BACKEND    = "hf_torch"
$env:ALADDIN_DEVICE     = "cuda"      # or "cpu"
python -m aladdin.cli inspect
```

Model ID and local path are always configurable (`backend.model_path`,
`ALADDIN_MODEL_PATH`); no weights are packaged inside the project ZIP.

## 3. Dependency install

```bash
pip install "transformers>=4.57" torch accelerate safetensors huggingface_hub
# CPU-only box: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 4. Real-model verification tests

```bash
export ALADDIN_MODEL_PATH="$PWD/models/Qwen3.5-0.8B"
python -m pytest tests/test_qwen35_real.py -v
#   without the weights these report SKIPPED ("not run"), never a fake pass
python -m pytest tests/test_v3_revert_and_loader.py -v   # mock/infrastructure level
```

`tests/test_qwen35_real.py` performs, on the real weights: architecture
detection, official-class loading, real tensor read + exact hash, reversible
edit, byte-exact revert, agent `undone`/`remaining` synchronization, and a
text-only forward pass (asserting the real 248320 vocab). GPU variants
(`test_real_gpu_roundtrip`) skip honestly when no CUDA device exists.
