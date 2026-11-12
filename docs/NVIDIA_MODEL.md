# NVIDIA model selection (checked 2026-11-11)

## Why nothing is hardcoded

Model ids on hosted endpoints get renamed, replaced and retired. This project
therefore ships **no default model id**: `config/*.json` has `"model": ""`, and the
CLI refuses to call the API until `NVIDIA_MODEL` is set. The value you choose is
recorded in every experiment's `state.json` and in the manifest of every delta, so
an experiment is reproducible.

## Verify for your own key

```bash
export NVIDIA_API_KEY=nvapi-...
python -m aladdin.cli nvidia-models
```

That prints `status` (key present, prefix sanity, base URL), then every id returned
by `GET {NVIDIA_BASE_URL}/models`, then whether your configured `NVIDIA_MODEL` is in
that list. If it is not, pick one from the list.

Base URL and endpoint (per NVIDIA's NIM LLM API reference):

* `NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1`
* `POST /chat/completions` — OpenAI-compatible; this client uses that path only.

## Catalog snapshot

The public catalogue at `build.nvidia.com/models` and the NIM LLM API reference
(`docs.api.nvidia.com/nim/reference/llm-apis`) listed the ids below on **2026-11-11**.
They are recorded here only as *candidate* ids — confirm against `/models`, because
availability is per-account and changes.

Instruction-tuned / chat-capable ids seen in the docs' model table:

| id | note |
|---|---|
| `qwen/qwen3-next-80b-a3b-instruct` | Qwen family, instruct; good default for reasoning+tools |
| `qwen/qwen3-next-80b-a3b-thinking` | same family, thinking variant |
| `qwen/qwq-32b` | reasoning model |
| `qwen/qwen2.5-coder-32b-instruct` | code-focused |
| `deepseek-ai/deepseek-v4-pro` | strong general reasoning |
| `deepseek-ai/deepseek-v4-flash` | cheaper/faster |
| `meta/llama-3.3-70b-instruct` | general instruct |
| `meta/llama-3.1-8b-instruct` | small/fast |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | NVIDIA-tuned instruct |
| `nvidia/nemotron-3-super-120b-a12b` | large Nemotron |
| `mistralai/mistral-nemotron` | Mistral-based |
| `moonshotai/kimi-k2-instruct` | long-context instruct |
| `z-ai/glm-5.3` | GLM family |
| `openai/gpt-oss-120b` | open-weight GPT-OSS |
| `sarvamai/sarvam-m` | Indic-language focused — worth trying first for Hindi/Hinglish wording |

Recommended starting point: an instruct model with reliable JSON output
(`qwen/qwen3-next-80b-a3b-instruct` or `deepseek-ai/deepseek-v4-pro`) for the edit
agent, and optionally `sarvamai/sarvam-m` as the `NVIDIA_JUDGE_MODEL` for
naturalness, since it is Indic-tuned.

## How the agent is wired

```
NVIDIA_API_KEY (env only)
        │
        ▼
agent/nvidia_client.py ── POST /chat/completions ──► integrate.api.nvidia.com
        │                                              │
        │  reply: one fenced json block                │
        ▼                                              │
agent/tools.py::parse_tool_call                        │
        │                                              │
        ▼                                              │
ToolRuntime.call(tool, args) ── real model ──► result ─┘  (tool result fed back)
```

* `max_steps` bounds the whole conversation; `forward_pass.budget` bounds expensive
  passes independently (a failed call does not consume budget).
* The transcript is written to `experiments/<id>/agent_transcript.jsonl` with the
  key redacted (`redact()` runs over every persisted string).
* API errors are retried with backoff on 5xx/timeout, but not on 400/401/403/404;
  after `max_retries` the step ends with an error recorded in `state.json` instead of
  hanging the experiment.

## Fallback: no key at all

```bash
python -m aladdin.cli run --agent scripted --steps 3
```

`agent/scripted_agent.py` is a deterministic policy (activation contrast → pick an
MLP `down_proj` → highest-L2-norm output columns → alpha ladder from
`editing.max_relative_change`). It exercises the identical apply/evaluate/rollback
path, which is how the CPU test mode stays meaningful.
