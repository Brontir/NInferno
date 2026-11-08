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
- **Decode @24K** is the same prompt with `max_tokens=512`, streamed: the tokens after the first
  divided by the time from the first token to the last. The prompt's cost never enters it, so a
  server that recomputes the prompt and one that reuses it from a prefix cache measure the same
  decode after a 24K-token context. (Rows before 2026-11-05 divided the completion by the whole
  request with the prefix cached, which counted the cached request's small prefill too.)
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

## Speculation under load

`--spec auto` runs the measured load table, and for a pair that was not measured it stops drafting from two
requests at once. For C=4 and C=8 the speculative setting was therefore also run with the startup's own
choice forced (`--spec X --draft-tokens D`), and the page shows the faster of the two for each cell. Only
that faster row is kept in `bench.jsonl`; the slower runs are in the measurement logs, not in the data.
A cell with a footnote marker explains a result that needs one.

## llama.cpp reference rows

The reference rows run `llama-server` (release tag in the row's `version`) on the same GGUF file and GPU,
with the same prompts, repeats and metrics, and the command-line flags listed on each run page:

- `-ngl 99 -fa on -b 2048 -ub 2048`. 2048 was the best physical batch on the RTX 5090 (512: 15.7K, 1024:
  18.0K, 2048: 18.9K, 4096: 16.9K tok/s prefill on gpt-oss-20b).
- One slot per stream (`-np C`). The context is 32768 for one stream (the 24K-token prompts need it) and
  8192 tokens per slot for C=4 and C=8 (the short chats need under 1K). When a start ran out of memory, the
  batch was lowered (1024, 512, 256) and, for a few C=8 runs, the context to 2048 tokens per slot. A row
  that needed either is marked on its run page.
- Speculation: `--spec-type draft-mtp` only where an MTP GGUF of the model in the same quantization was
  cached on the stand, with the `--spec-draft-n-max` that was fastest in a short sweep. Everywhere else
  llama.cpp runs plain. The Speculation column of the index says which; the summary line keeps the two
  groups apart. DFlash drafters were not converted for llama.cpp.
- The ratio in the index divides our best setting by the best llama.cpp row of the same quantization. A
  dagger marks a ratio whose NInferno row is older than the llama.cpp row.

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
