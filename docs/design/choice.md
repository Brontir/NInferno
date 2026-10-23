# The configuration choice

One rule picks how a model is served when the user leaves an axis to the engine, and the same code
says what any other value would change. It is used everywhere such a pick is made: `ninferno-serve`
without the flags, `ninferno-serve --plan --no-load`, and ninferno-ui's `/api/plan`. Code:
`src/product/choice/` (`choice.h`, the constants in `choice_constants.h`), test `tests/test_choice.cpp`.

## Inputs

| Kind | What | Where from |
|---|---|---|
| Model facts | layers by kind, KV heads and widths, window, recurrent state, experts, active parameters, weight bytes per quantization | config.json / GGUF header / artifact (`product/plan/plan_facts.h`) |
| Card facts | compute capability, SMs, reference clock, DRAM rate, FP4 tensor cores, free memory | PCI-ID facts (`core/data/gpu_facts.json`), `core/generation.h` |
| Task | context each stream holds of its own, streams, a typical request (prompt, answer tokens) | the user (defaults 32K, 1, 2048 + 512) |
| Pins | `--quant`, `--kv-dtype`, `--activations`, `--state-dtype`, `--sliding-kv-dtype` | the user's flags |
| Constants | decode and prefill efficiencies per generation and kind, KV read and exact-route factors, KLD terms, the accuracy weight | measured once on the lab's stands (`choice_constants.h` names each source) |

Nothing is measured on the user's machine.

## The three estimates

- **Memory.** The plan without the weights (`runtime/engine/plan_dry_run.h`): the family planner's own
  bytes for the asked options, the start's ladder. A candidate fits when each stream holds the asked
  context of its own.
- **Speed.** A decode step of C rows at context n:
  `t0 + (W(C) + C n b_kv k_kv + 2 C s + C min(n, window) b_ring) / (eta_dram BW) + C 2 P / (eta_compute peak_s8) f_route`,
  prefill `eta_prefill peak_route / (2 P)`. A request of P prompt and G answer tokens takes
  `P / prefill + G / decode(P + G/2)`. Fitted on 49 decode and 15 prefill points over four cards and
  three models: 5.2 % rms decode, 2.0 % prefill (`tests/fixtures/choice/speed_measured.json`; the
  test holds the code to it).
- **Accuracy.** KLD against the exact reference (`--activations exact`, BF16 KV, FP32 state) per
  switch, measured with `ninferno-perplexity`; terms combine as the root of the sum of squares
  (the product default: measured 0.0040, root sum square 0.0044, plain sum 0.0067). A value with no
  measurement takes its axis's worst measured term, never zero.

## The rule

Among the candidates that fit, the largest worth `-ln(T) - lambda * KLD` wins (`lambda` 25: 0.001
KLD is worth 2.5 % of a request's time). Where two candidates cost about the same, the more accurate
one wins; where one is much faster, it wins. Ties: the longer own context, then fewer weight bytes,
then the listing's order.

Until a measured KLD curve over the quantizations exists, the quantization is the quant rule's
(`runtime/engine/quant_rule.h`: the largest that fits, not below 3.5 bits per weight unless nothing
above fits) and the worth picks the KV codec and the activation mode at it.

The ring and the recurrent state follow the KV codec and the activations (`coupled_storages`):
today the engine's own defaults (BF16 rings; FP16 state under fast, FP32 under exact).

## Explaining

`explain` lists, for each axis, every other value with the chosen candidate's other axes kept: its
own context, decode and prefill speed, KLD, worth, and why it lost (does not fit, lower worth, or the
quant rule). A caller re-runs `choose` with another task (context, streams, request shape) to show
what an input changes.
