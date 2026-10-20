# Capability evaluation

`eval/` is a repository-local coordinator for capability evaluations (AIME, GPQA-Diamond, IFBench,
BFCL-v4, ERQA, RealWorldQA, Needle-in-a-Haystack). It can evaluate `ninferno-serve`, another local
OpenAI-compatible service or a remote online model; each target and job declares the concurrency
that server run admits. [EvalScope](https://github.com/modelscope/evalscope) is the first real
backend; the coordinator, configuration, logging, progress, resume and result contracts do not
depend on it, and a deterministic `mock` backend exercises them without a model or network. Use it
to check the quality of a converted model or quantization; throughput belongs to
[bench](../bench/README.md).

## Quick start

```bash
python3 -m venv eval/.venv
eval/.venv/bin/python -m pip install -r eval/requirements.txt
export PYTHONPATH="$PWD/eval"

# Network-free check of the coordinator
eval/.venv/bin/python -m ninferno_eval run --config eval/configs/mock-suite.yaml --suite all

# Real model: serve it where the config points (127.0.0.1:18080, model id qwen3.6-27b), in
# another terminal, then run the small matrix
ninferno-serve models/qwen3_6_27b.ninferno --port 18080 --model-id qwen3.6-27b
eval/.venv/bin/python -m ninferno_eval run --config eval/configs/capability-suite.yaml --suite smoke
```

Installing the dependencies downloads no model and creates no `.ninferno` file; convert the model
first (`ninferno-convert --fetch org/name`, see the [conversion guide](../docs/weight-conversion.md))
and serve it. `requirements.txt` pins PyYAML 6.0.3, rich 15.0.0, EvalScope 1.10.0 with the
`bfcl`, `ifbench` and `needle-haystack` extras, and `soundfile` 0.14.0; the BFCL job additionally
checks for `bfcl-eval==2025.10.27.1`. Dataset and model caches stay owned by their upstream
libraries.

## Commands

All commands are `python -m ninferno_eval <command>` with `PYTHONPATH=eval`.

| Command | Options | Purpose |
|---|---|---|
| `validate` | `--config FILE` (required), `--suite NAME` (default: all suites) | Validate configuration and installed backend dependencies. |
| `plan` | `--config FILE`, `--suite NAME` (both required), `--check-runtime` | Show expected work and requirements without model requests; `--check-runtime` also resolves secret environment variables and pinned packages. |
| `run` | `--config FILE`, `--suite NAME` (both required) | Run one suite. |
| `status` | `--run DIR` | Print `state.json` of a run. |
| `resume` | `--run DIR` | Continue an interrupted run; rejects a changed effective configuration or backend version. Completed jobs are skipped and an incomplete EvalScope job reuses its prediction cache. |
| `summarize` | `--run DIR` | Print the normalized summary. |
| `list-backends` | none | List registered backends (`evalscope`, `mock`). |

Ctrl-C asks the active backend to checkpoint; a second Ctrl-C aborts. Typical sequence for a formal
evaluation:

```bash
eval/.venv/bin/python -m ninferno_eval plan \
  --config eval/configs/capability-suite.yaml --suite reasoning
eval/.venv/bin/python -m ninferno_eval run \
  --config eval/configs/capability-suite.yaml --suite reasoning
SERPAPI_API_KEY=... eval/.venv/bin/python -m ninferno_eval run \
  --config eval/configs/capability-suite.yaml --suite bfcl_full
eval/.venv/bin/python -m ninferno_eval status    --run eval/runs/<run-id>
eval/.venv/bin/python -m ninferno_eval summarize --run eval/runs/<run-id>
```

### Exit status

| Code | Meaning |
|---:|---|
| 0 | Completed, or status query of a running run. |
| 2 | Invalid configuration or missing configured secret. |
| 3 | Missing or incompatible backend dependency. |
| 4 | Partial evaluation. |
| 5 | Failed evaluation, missing run artifact or other error. |
| 6 | Cancelled. |

## Configuration

A YAML file (`schema_version: 1`) with `targets`, `suites` and an optional `runtime` section. Unknown
fields are rejected everywhere. Shipped examples:

| File | Suites | Content |
|---|---|---|
| `configs/capability-suite.yaml` | `smoke`, `reasoning`, `bfcl_full` | AIME25, AIME26, GPQA-Diamond, BFCL-v4. |
| `configs/mock-suite.yaml` | `all` | Network-free coordinator check. |
| `configs/qwen3_6_27b_reasoning.yaml` | `reasoning_full` | Qwen3.6-27B AIME25/26 and GPQA-Diamond. |
| `configs/qwen3_6_35b_aime.yaml`, `qwen3_6_35b_gpqa.yaml` | `smoke`, `aime_full`; `gpqa_full` | Qwen3.6-35B-A3B. |
| `configs/qwen3_6_35b_needle_haystack.yaml` | `smoke`, `standard`, `native_long` | Needle-in-a-Haystack (below). |
| `configs/qwen3_8_27b_nvfp4_reasoning.yaml`, `qwen3_8_27b_groupwise_reasoning.yaml` | `text_full`, `multimodal_full` | IFBench, AIME25/26, GPQA-Diamond; ERQA, RealWorldQA. |
| `configs/qwen3_8_27b_nvfp4_gpqa_budgets.yaml` | `gpqa_8k`, `gpqa_16k`, `gpqa_32k` | GPQA-Diamond under thinking budgets. |

### Targets

```yaml
targets:
  model_api:
    protocol: openai_chat
    base_url: http://127.0.0.1:18080/v1
    model: qwen3.6-27b
    api_key_env: MODEL_API_KEY   # optional; omit for an unauthenticated endpoint
    max_concurrency: 1
    request: { timeout_seconds: 3600, retries: 2 }
```

| Field | Default | Meaning |
|---|---|---|
| `protocol` | required | Only `openai_chat` is accepted by the EvalScope backend. |
| `base_url`, `model` | required | `http(s)://` endpoint and served model id. |
| `api_key_env` | none | Environment variable holding the key. Literal `api_key`, `Authorization` and `x-api-key` entries are rejected so secrets never reach saved configurations. |
| `max_concurrency` | 1 | Cap on aggregate requests to this endpoint. |
| `request.timeout_seconds` | 600 | Per-request timeout. |
| `request.retries` | 5 | Retries. |
| `request.retry_interval_seconds` | 10 | Pause between retries. |
| `request.headers` | none | Extra string headers (no credentials). |
| `provenance` | none | Free-form mapping recorded in the manifest. |

### Jobs

```yaml
- id: new_dataset
  backend: evalscope          # or mock
  dataset: evalscope_dataset_name
  target: model_api
  limit: 2                    # optional: positive integer, or float in (0, 1]
  repeats: 1
  max_concurrency: 1          # optional per-job cap
  generation: { temperature: 0, max_tokens: 16384, seed: 42 }
  backend_args: { subset_list: [subset_name] }
```

| Field | Default | Meaning |
|---|---|---|
| `id`, `backend`, `dataset` | required | Unique job id per suite; backend name; dataset name. |
| `target` | none (required by EvalScope) | Target name. |
| `max_concurrency` | target's | Cap on target slots the job may reserve. |
| `limit` | all | Sample count or fraction. |
| `repeats` | 1 | Repeats per sample. |
| `generation` | `{}` | Portable settings: `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stream`, `frequency_penalty`, `presence_penalty`, `parallel_tool_calls`, `stop_seqs`, `extra_body`. EvalScope defaults `seed` to 42. |
| `backend_args` | `{}` | Backend-specific. EvalScope: `subset_list`, `dataset_args`, `few_shot_num`, `shuffle`, `judge_strategy`, `judge_model_args`, `dataset_hub`, `dataset_dir`, `collect_perf`, `ignore_errors`, `debug`, `is_fc_model`, `underscore_to_dot`, `serpapi_api_key_env`, `allow_network_downloads`. Mock: `items` (10), `sleep_seconds`, `wrong_every`, `unknown_total`, `fail_at`. |

The EvalScope backend has built-in sample counts for `aime25`, `aime26`, `gpqa_diamond`, `ifbench`,
`erqa`, `real_world_qa` and `bfcl_v4`; the `needle_haystack` total is computed per subset
(`english`, `chinese`) as length intervals times depth intervals. Other EvalScope datasets run with
an unknown total.

### Runtime

| Field | Default | Meaning |
|---|---|---|
| `runtime.max_parallel_jobs` | 1 | Concurrently active dataset jobs. |
| `runtime.runs_dir` | `eval/runs` | Parent of run directories. |
| `runtime.progress.enabled` / `refresh_seconds` / `heartbeat_seconds` | true / 1 / 30 | Progress rendering. |
| `runtime.samples.retention` | `all` | `all`, `errors` or `none`. |

### Concurrency

Three limits apply together: `runtime.max_parallel_jobs` (active jobs), the target
`max_concurrency` (aggregate requests to the endpoint) and an optional job `max_concurrency` (target
slots one job may reserve). For EvalScope the granted slots become `eval_batch_size`; jobs sharing a
target never reserve more than its capacity. For `ninferno-serve`, set the target capacity to the
server's `--max-concurrency`; a long-output job may use fewer slots when its KV entitlement requires.

## Datasets and scores

Each benchmark is reported independently; no cross-benchmark average is computed.

| Benchmark | Samples | Reported |
|---|---:|---|
| AIME25, AIME26 | 30 each | Rule-scored accuracy. |
| GPQA-Diamond | 198 | Accuracy. |
| IFBench | 300 | Prompt- and instruction-level strict and loose adherence; primary `prompt_level_strict`. |
| ERQA | 400 (eight reasoning subsets) | Accuracy (multimodal). |
| RealWorldQA | 765 | Accuracy (multimodal). |
| BFCL-v4 | 5106 | Official `agentic`, `multi_turn`, `live`, `non_live`, `hallucination`, `overall` once the full score-bearing suite completes. |
| Needle-in-a-Haystack | by profile | Rule-scored retrieval. |

A partial or failed job makes the run `partial` or `failed`; an incomplete BFCL run is never labeled
the official full score.

- **BFCL-v4.** Multi-turn samples may make several requests per sample. Web Search subsets need
  `SERPAPI_API_KEY` (or a variable named by `backend_args.serpapi_api_key_env`); `memory_vector` may
  download an upstream model and requires `allow_network_downloads: true`.
- **Needle-in-a-Haystack** (`qwen3_6_35b_needle_haystack.yaml`): `standard` keeps EvalScope's
  1K-32K, ten-length, ten-depth English/Chinese matrix (200 samples); `native_long` evaluates 64K,
  128K and a safe 260K prompt at eleven depths in both languages (66 samples), leaving more than 2K
  context for chat framing and the bounded 512-token answer. Scoring is rule-based with thinking
  disabled. Download the corpus and edit `local_path` and `tokenizer_path` in the config (they ship
  as `/path/to/...` placeholders), then check the plan:

  ```bash
  eval/.venv/bin/python - <<'PY'
  from modelscope import dataset_snapshot_download
  print(dataset_snapshot_download('AI-ModelScope/Needle-in-a-Haystack-Corpus',
        allow_file_pattern=['PaulGraham_Essays.txt', 'Journey_to_the_West.txt']))
  PY
  eval/.venv/bin/python -m ninferno_eval plan \
    --config eval/configs/qwen3_6_35b_needle_haystack.yaml --suite native_long --check-runtime
  ```

## Managed Qwen3.8-27B runs

Three scripts start a fresh `build/apps/ninferno-serve` on `127.0.0.1:18080` (they refuse to run if a
healthy service already answers there), wait for `/health`, run the suite and stop the server. They
need `curl`, `eval/.venv` and the artifact under `out/`. The multimodal datasets are fetched through
ModelScope first:

```bash
eval/.venv/bin/python - <<'PY'
from modelscope import dataset_snapshot_download
for dataset_id in ('evalscope/ERQA', 'allenai/IFBench_test', 'lmms-lab/RealWorldQA'):
    print(dataset_snapshot_download(dataset_id))
PY
```

| Script | Artifact | Config | Server settings |
|---|---|---|---|
| `eval/run_qwen3_8_27b_nvfp4_reasoning.sh [--plan] [text\|multimodal]` | `out/qwen3_8_27b_nvfp4.ninferno` | `qwen3_8_27b_nvfp4_reasoning.yaml` | Concurrency 2; `text`: context 252928; `multimodal`: context 81920 and `--vision`. |
| `eval/run_qwen3_8_27b_groupwise_reasoning.sh [--plan] [text\|multimodal]` | `out/qwen3_8_27b.ninferno` | `qwen3_8_27b_groupwise_reasoning.yaml` | Concurrency 4; `text`: context 262144; `multimodal`: 81920 and `--vision`. |
| `eval/run_qwen3_8_27b_nvfp4_gpqa_budgets.sh [8k\|16k\|32k ...]` | `out/qwen3_8_27b_nvfp4.ninferno` | `qwen3_8_27b_nvfp4_gpqa_budgets.yaml` | Per tier `--default-thinking-budget` 8192/16384/32768, context 12352/20544/36928, concurrency 8/8/6; outputs under `profiles/eval/`. |

All use `--kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft --prefill-chunk 1024
--model-id qwen3.8-27b` and write server and request logs to `eval/server-logs/`. With no step
argument the reasoning scripts run `text` (IFBench, AIME25, AIME26, GPQA-Diamond) and then
`multimodal` (ERQA, RealWorldQA), restarting the server between steps; `--plan` previews the
selected steps without starting a server. The NVFP4 text context is the largest that fits a 32 GB
card after weights; pick a smaller context if the server rejects the startup reservation on your
hardware. Output budgets: AIME 122880 tokens, IFBench and both multimodal sets 65536, GPQA-Diamond
245760 at concurrency one.

## Runs, progress and logs

TTY runs show a live display (dataset phase, completed/total, elapsed, rate, ETA); other runs print
periodic heartbeats. Unknown totals stay `?`; no percentage or ETA is invented. Every run is stored
under `eval/runs/<timestamp>-<config-hash>/`:

| Artifact | Purpose |
|---|---|
| `effective-config.yaml` | Validated, secret-free effective configuration. |
| `manifest.json` | Git state, environment, backend versions, target and concurrency provenance. |
| `state.json` | Atomically updated operational and resume state. |
| `events.jsonl` | Append-only structured progress and lifecycle events. |
| `run.log` | Human-readable timestamps, progress, retries and failures. |
| `backends/<job>/` | Unchanged backend-native predictions, logs, cache and reports. |
| `summary.json` | Versioned normalized result contract. |
| `summary.md` | Compact score table. |

The sample-retention policy is recorded in the manifest. API keys and known secret values are
redacted from events and task snapshots.

## Adding an evaluation

An ordinary EvalScope dataset needs only another job (see the job example above). An evaluator
without EvalScope implements the four-method backend protocol (`validate`, `plan`, `run`,
`normalize`) in `ninferno_eval/backends/base.py`, registers a stable name in
`backends/registry.py`, keeps its raw artifacts and returns the normalized `DatasetResult`; the
coordinator and summary writer need no benchmark-specific change.

## Verification

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover -s eval/tests -p 'test_*.py'
PYTHONPATH=eval eval/.venv/bin/python -m ninferno_eval run \
  --config eval/configs/mock-suite.yaml --suite all
```
