# Perplexity evaluation

`ninferno-perplexity` measures the causal perplexity of a converted `.ninferno` model
([weight conversion](weight-conversion.md)). It uses the model's tokenizer, Text model, selected
Main KV representation, final normalization and main output head. It is an offline evaluator, not a
serving endpoint or a logits-export API. Only Text weights and resources are loaded; Vision and
speculative components are not needed. It scores Qwen 3.5/3.6/3.8 models; the other families
refuse causal scoring here. It comes with a source build ([building](building.md)), not with the
release archive.

## Quick start

Quick pass over the fixed corpus:

```bash
./build/apps/ninferno-perplexity models/qwen3_8_27b_nvfp4.ninferno \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype fp8
```

One UTF-8 file with a longer window:

```bash
./build/apps/ninferno-perplexity models/qwen3_8_27b_nvfp4.ninferno \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

Recommended long-context profile for KV-format comparisons: the full corpus (no `--quick`) with
`--context 65536 --stride 32768`.

## Options

Run `./build/apps/ninferno-perplexity --help` for the complete command surface. Give exactly one of
`--corpus` and `--text`.

| Option | Default | Meaning |
|---|---|---|
| `--version` | | Print the version and exit. |
| `--corpus MANIFEST` | none | Corpus manifest. |
| `--quick` | off | One stream per domain. Needs `--corpus`. |
| `--text FILE` | none | Score one UTF-8 file. |
| `--context N` | `4096` | Window size in tokens. |
| `--stride N` | `2048` | Targets scored per window, `1..context-1` (`--context` is at least 2). |
| `--device N` | `0` | CUDA device index. |
| `--kv-dtype bf16\|int8\|fp8\|nvfp4\|k8v4` | `fp8` | Main KV representation. |
| `--activations auto\|16\|8\|4` | `auto` | Activation precision (as the server's `--activations`). |
| `--state-dtype f32\|bf16\|f16` | `f32` | Recurrent-state storage (the values of the server's `--state-dtype`; no automatic choice here). |
| `--step-tokens N` | `0` (prefill chunks) | Run the scored targets `N` tokens per call, as decode (`1`) or a speculative round (`N`) runs them. |
| `--rope-yarn-factor F` | `1` | Qwen 3.5/3.6: rotate with YaRN over the checkpoint's own context and admit `F` (`1..64`) times it. Other families refuse any other value. |
| `--output DIR` | `profiles/perplexity/` | Empty directory for `report.json`. |
| `--kld-out FILE` | none | Record this (reference) model's top-K tokens per position. |
| `--kld-top-k N` | `32` | Top-K to record, `1..64`. Only with `--kld-out`. |
| `--kld FILE` | none | KL divergence and top-1 agreement against such a reference, scored on the same corpus, context and stride. Exclusive with `--kld-out`. |
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | `info` | Log detail. |

## The fixed corpus

`ninferno-ppl-1m-v1` is a fixed set of 16 independent UTF-8 streams: English reference text, English
long-form text, Chinese reference text, and NInferno C++/CUDA code. The default selects all streams;
`--quick` selects one stream from each domain.

## Output

- The model loads once. Every selected stream is read and tokenized before scoring.
- Readable startup, corpus, scoring and per-stream summaries go to stderr. On a terminal, weight
  loading and scoring use one transient progress line; redirected scoring prints persistent progress
  every ten seconds. `--log-level debug` adds startup and stream-begin detail.
- The final domain/overall table is product output on stdout.
- The independent full-precision machine report is `report.json` under `profiles/perplexity/`,
  unless `--output` gives an empty directory.
- The schema-v2 report records the artifact's architecture, public name, actual weight formats and
  prefill signature, with the workload and numerical results.
- Reports contain unrounded NLL/PPL for every window, stream, domain and the token-weighted overall
  aggregate.

## Metric

For a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded. The
metric is fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets and
keeps up to `context-stride` preceding tokens as local context. Streams never share history.

## Comparing runs

- Keep the corpus, context, stride and execution settings fixed except the variable you measure.
- Compare KV formats with the same artifact. Compare weight formats with the same KV format.
- The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
  come from the artifact's tokenizer at run time and are recorded in each report.
