# How the performance numbers are measured

Every number in the [performance matrix](README.md) and on the GPU, model and per-pair pages comes from one
protocol, run the same way on every GPU by [`tools/bench/perf_matrix.py`](../../tools/bench/perf_matrix.py).
The pages are generated from [`data/bench.jsonl`](data/bench.jsonl) by
`tools/docs/gen_performance.py`; nothing in them is edited by hand.

## Setup

- **Model.** Converted by `ninferno-serve` from the Hugging Face repository named in the row
  (`recipe`: the repository and its `--quant` label), exactly as the
  [quick start](../../README.md#quick-start) does. The model page links the original checkpoint.
- **Server.** A Release build for the GPU's architecture, started fresh for each workload with the
  flags listed on the pair's page and `--max-concurrency` equal to the workload's concurrency (1 for
  the long prompts and C=1, 4 for C=4, 8 for C=8), so the server is sized for its load as a user
  would size it: more concurrent slots leave less memory for a drafter, and `--spec auto` then
  decodes without speculation. The startup log's `speculation |` line is kept with the row.
- **GPU.** One GPU at stock clocks and power limit; the driver, CUDA version and power limit are
  recorded with every row (`machine`).
- **Sampling.** Greedy (temperature 0) for every number, so repeated runs generate the same tokens.
  Each workload starts with one untimed warm-up request.

## Long prompt: prefill and single-stream decode

Six different English prompts of about 24,000 tokens: slices of Federal Register documents published
in September and October 2026 (public domain, new to every model measured, so no model has
memorized them; [`long_niah_256k.json`](../../examples/cli/messages/long_niah_256k.json)), each
followed by a request to summarize the text section by section.

- **Prefill** is prompt tokens divided by the wall time of a request with `max_tokens=1`.
- **Decode @24K** is completion tokens divided by the wall time of the same prompt with
  `max_tokens=512`; its prefix is then already cached, so the time is the decode of 512 tokens
  after a 24K-token context.
- Each value is the median over the six prompts; the row records min..max (`spread`).

## Short chat: aggregate decode at C=1, 4 and 8

Twenty fixed short prompts (code, SQL, writing, translation, reasoning; listed in the tool), each
answered with up to 512 tokens. C client workers send them concurrently, worker *k* taking prompts
*k*, *k+C*, ….

- **C=n** is all completion tokens of the wave divided by its wall time: the throughput a
  server with *n* concurrent users delivers. **One stream** is C=1; **all streams** is the
  fastest C above 1 (up to 8).

## Reading the numbers

- Prefill and decode are never mixed in one number.
- Decode includes speculative decoding when the setting uses it (MTP, DFlash, DFlash2): tokens
  counted are committed output tokens, so speculation shows up as more tokens per second.
- A matrix cell shows one stream / all streams, each at its own best setting. The GPU and model
  pages give one row per pair (weights, one stream, all streams with its concurrency, prefill and
  decode after 24K tokens), every column at its own best setting. The pair's page lists every
  setting measured (weights, speculation, KV format, every workload, the date), the commands of
  the best ones and the server flags.
- A blank matrix cell means the pair was not measured; a dash, that one workload was not, for
  example because the model does not fit.
- Rows of other engines (`engine` `llama.cpp`, measured with `perf_matrix.py --engine llama.cpp`
  on the same GGUF, GPU and prompts) and quality rows (`perplexity`, `acceptance`,
  `greedy_match`; see [perplexity](../perplexity.md)) are shown where the data has them and never
  enter the speed tables.

## Reproducing a row

```bash
./build/apps/ninferno-serve unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_XL --with-drafters \
  --max-context 32768 --max-concurrency 1 &
python3 tools/bench/perf_matrix.py --port 8080 --concurrency 1 \
  --jsonl docs/performance/data/bench.jsonl --model Qwen/Qwen3.6-35B-A3B \
  --recipe "unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_XL" --spec dflash --kv auto \
  --flags "--max-context 32768 --max-concurrency 1"
# then a fresh server with --max-concurrency 4 and --skip-long --concurrency 4, and with 8
python3 tools/docs/gen_performance.py          # rewrites docs/performance/** and the README blocks
python3 tools/docs/gen_performance.py --check  # fails when a page differs from the data
```
