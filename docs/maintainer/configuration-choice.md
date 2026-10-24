# The configuration choice

One rule picks how a model is served when the user leaves an axis to the engine, and the same code
says what any other value would change. It is used everywhere such a pick is made: `ninferno-serve`
without the flags, `ninferno-serve --plan --no-load`, and ninferno-ui's `/api/plan`. Code:
`src/product/choice/` (`choice.h`, the constants in `choice_constants.h`), test `tests/test_choice.cpp`.

## Inputs

| Kind | What | Where from |
|---|---|---|
| Model facts | layers by kind, KV heads and widths, window, recurrent state, experts, active parameters, weight bytes per quantization | config.json / GGUF header / artifact (`product/plan/plan_facts.h`) |
| Card facts | compute capability, SMs, reference clock, DRAM rate, FP4 tensor cores, free memory | PCI-ID facts (`core/data/gpu_facts.json`), `core/generation.h`; the clock and the DRAM rate feed only the estimates shown, never a decision (below) |
| Task | context each stream holds of its own, streams, a typical request (prompt, answer tokens) | the user (defaults 32K, 1, 2048 + 512) |
| Pins | `--quant`, `--kv-dtype`, `--activations`, `--state-dtype`, `--sliding-kv-dtype`, `--prefill-chunk` | the user's flags |
| Constants | decode and prefill efficiencies per generation and kind, KV read and exact-route factors, KLD terms, the accuracy weight | measured once on the lab's stands (`choice_constants.h` names each source) |

Nothing is measured on the user's machine.

## Where it runs

One function decides the automatic axes everywhere: `product::choose_at_weights` (#89's
`product/plan/plan_choice`) over `choice::choose`. `ninferno-serve`'s start (`choose_served_axes`), the
`ninferno` CLI's start and `--plan`, `--plan --no-load`, the quantization rule's worth ranking
(`plan_quants`) and ninferno-ui's `/api/plan` all call it; none keeps a rule of its own for the KV
codec, the activations, the state, the ring or the chunk. The quantization is most often the user's
(`--quant`, the UI's row); the other axes are almost never pinned, so their default is the one that
serves. Where the choice cannot run (a model whose shape the plan cannot read), the engine takes the
family's defaults through `runtime::plan_family_storage`, the same function the choice takes its
defaults from. The measuring tools (`ninferno-perplexity`, the spec tuner, the maintainer apps) take
their options as given.

## The three estimates

- **Memory.** The plan without the weights (`runtime/engine/plan_dry_run.h`): the family planner's own
  bytes for the asked options, the start's ladder. A candidate fits when each stream holds the asked
  context of its own.
- **Speed.** A decode step of C rows at context n:
  `t0 + (W(C) + C n b_kv k_kv + 2 C s + C min(n, window) b_ring) / (eta_dram BW) + C 2 P / (eta_compute peak_s8) f_route`,
  prefill `eta_prefill peak_route / (2 P)` (an MoE's times its chunk share, below). A request of P
  prompt and G answer tokens takes `P / prefill + G / decode(P + G/2)`, the prompt as whole chunks
  and a final partial one, each at its own rate. Fitted on 49 decode and 15 prefill points over four cards and
  three models: 5.2 % rms decode, 2.0 % prefill (`tests/fixtures/choice/speed_measured.json`; the
  test holds the code to it).
- **Accuracy.** KLD against the exact reference (`--activations exact`, BF16 KV, FP32 state) per
  switch, measured with `ninferno-perplexity`; the switches' terms combine as the root of the sum
  of squares (the product default: measured 0.0040, root sum square 0.0044, plain sum 0.0067), and
  the quantization's own term adds on top: how it combines with them is not measured, and under a
  root sum square a coarse quantization (0.2) made a KV codec's error look free. A value with no
  measurement takes its axis's worst measured term, never zero (an FP8 ring takes the FP8 KV
  codec's term as a stand-in).

## The rule

Among the candidates that fit, the largest worth `-ln(T) - lambda * KLD` wins (`lambda` 25: 0.001
KLD is worth 2.5 % of a request's time). Where two candidates cost about the same, the more accurate
one wins; where one is much faster, it wins. Ties: the longer own context, then fewer weight bytes,
then the smaller quantization label (the quant rule's last key), then the listing's order.

**Gates.** A constant decides only once it rests on two lab cards (principle 4); until then the
choice falls back and says why. `choice_constants.h` holds the gates:

| Gate | Holds | Closed: what stands |
|---|---|---|
| `kReferenceCards` row's `kv_two_cards`, `exact_two_cards` | the generation's KV read and exact-route factors | the family's defaults |
| each `kKldTerms` row's `cards`, `kQuantKldCards` (`accuracy_decides`) | the KV codecs' and the int8 activations' terms (model constants, or a generation's own row), the quantization curve | the quant rule's largest and the family's defaults (automatic KV codec, `--activations fast`) |
| `KldTerm::family`, `kQuantKldFamily` | the families the switches' terms (Qwen 3.5 dense and MoE) and the quantization curve (Qwen 3.5 dense) were measured on | another family keeps its defaults; a family with switch terms but no curve keeps the quant rule's largest |
| `kMoeChunkTwoCards` | the MoE chunk shares | the default chunk (1024) |

The switches' terms are per family: Qwen 3.5 MoE (Qwen3.6-35B-A3B) loses 3.4 times dense's KLD
to INT8 KV (0.0033 against 0.00095) and more to fast (0.0049 against 0.0039), so one family's terms
never decide for another (they are shown, marked as borrowed). The MoE terms rest on the RTX 4090
and 5070 Ti (within 6 %); it has no quantization curve, so its quantization stays the quant rule's.

The KV codecs' terms, the int8 activations' term and the quantization curve agree across the two
generations measured (RTX 5070 Ti and RTX 4070 Ti / 4090: KV within 13 %, the curve within 2 %,
activations 0.0039 / 0.0040) and are model constants on two cards. The activations' term became
one only with ggml's MMQ prefill summing its q8_1 blocks from the codes: before, the boards that
prefill on it (RTX 4090, 4070 Ti, 2080 Ti) lost 0.030 KLD (top-1 95.8 %) on the min-term formats of
Qwen3.5-9B Q4_K_M, 7.6 times the W4A8 tiles' 0.0039. A generation's own row (`KldTerm::generation`)
overrides the constant where a generation's routes differ. The state's and the ring's terms only
price the candidates they follow (they are not decided), so they do not gate.

Today Ada decides (its reference card's factors and every term above rest on two cards); Blackwell
and Turing keep the family's defaults (their factors rest on one card). On Qwen3.5-9B Q4_K_M fast's
0.004 KLD is worth less than exact's time (+34 % on the RTX 4090's mixed workload; `lambda` 25 sets
the break-even at 0.012): fast. The chunk shares rest on two Ada cards, so an MoE's chunk is decided
on Ada too.

**What a decision reads.** The KV codec and the activation precision change the output bytes (tier
B1 of [Deterministic decisions](deterministic-decisions.md)): they may not read a board's clock or
bandwidth. So T is priced on the generation's *reference card* (`kReferenceCards`: the mean of the
generation's lab cards), never on the board itself: every board of a generation decides the same,
and a board's own facts feed only the estimates a caller shows. The reference card decides only
where the KV read factors and the exact-route factor were measured on two lab cards of the
generation (principle 4); elsewhere the family's defaults stand (the automatic KV codec,
`--activations fast`), and the choice says so. Ada has them (RTX 4070 Ti and 4090: KV factors within
6 %, exact within 20 %); Blackwell and Turing have one lab card each.

**Auto's steps.** Each own-context step (`quant_own_targets`: min(native, 32K), 16K, 8K) is tried
above the quality floor before any quantization below it. Where the worth decides, a narrower KV
codec is a memory lever of its own: at a step where the largest quantization fits only with it,
that pair is taken before the next step; under the family's defaults only the automatic codec
counts.

The quantization's own term is the measured curve `kQuantKld` (Qwen3.5-9B GGUF against Q8_0:
Q6_K 0.0063, Q5_K_M 0.0106, Q4_K_M 0.0193, IQ4_XS 0.0285, Q3_K_M 0.068, UD-IQ3_XXS 0.107, UD-IQ2_M
0.161, UD-IQ2_XXS 0.334), interpolated in log KLD over bits per weight for other quantizations and
models (marked as not measured); above Q6_K it falls on a straight line to 0 at Q8_0's bits, and
any model's Q8_0 is its own reference (0). Where the worth decides, the plan weighs the quantizations
of the quant rule's tier (`runtime/engine/quant_rule.h`: the own context per stream it reached, the
same side of the 3.5-bit floor) by the worth of each one's own choice, and the rule line says which
was the largest (`rule_code` `best_worth`). With the gate open, on the RTX 4090 and 4070 Ti
Qwen3.5-9B takes UD-Q5_K_XL where the largest that fits is UD-Q8_K_XL or Q8_0: Q8_0 costs about
26 % more time per request than Q6_K for 0.003 KLD, and Q4_K_M saves 20 % for 0.012 more. Elsewhere
the rule's largest stands.

The ring and the recurrent state follow the KV codec and the activations (`coupled_storages`, the
owner's rule): the sliding rings in the KV cache's kind (FP8 under any narrower KV codec where the
family's rings take FP8: Gemma 4 and the dense decoders; else BF16), the recurrent state one step
finer than the KV cache (FP32 under BF16 KV, FP16 under a narrower one, FP32 under exact). A flag
(`--sliding-kv-dtype`, `--state-dtype`) pins either. Their other values are scored beside the
chosen candidate (never chosen) so that `explain` can show them.

**The prefill chunk.** An MoE's prefill grows with the tokens each expert takes from a chunk
(chunk x experts per token / experts), along one curve across models and cards: Gemma 4 26B-A4B on
the RTX 4070 Ti and 4090 and Qwen3.6-35B-A3B on the 4090 lie within 6 % of `kMoeChunkShare`
(0.635, 0.813, 0.952, 1.0 at 16, 32, 64, 128 tokens per expert); decode does not move. A dense
model's prefill moves under 5 % from 512 tokens on. So an MoE's candidates take the chunks 1024,
2048 and 4096 (the widest: the ladder may still narrow it); a prompt fills only its own tokens of
a chunk, so with the default 2048-token request the 2048 chunk is the widest that pays, and a
wider one only costs workspace. A chunk has no accuracy term (under `--activations fast` where a
prompt is cut may move the answer's bytes, not its error), so it is decided by speed on the
reference card alone, behind its own gate, not the accuracy one.

## Explaining

`explain` lists, for each axis, every other value with the chosen candidate's other axes kept: its
own context, decode and prefill speed, KLD, worth, and why it lost:

| `why_code` | Meaning |
|---|---|
| `no_fit` | it does not hold the asked context per stream |
| `lower_worth` | it fits and its worth is lower |
| `tie_context` | the same worth; the chosen keeps more context per stream (or fewer weight bytes) |
| `tie_order` | the same in every respect; the smaller quantization label, else the listing's order |
| `quant_rule` | the quant rule decided: no worth here, or the other side of the 3.5-bit floor |
| `default` | a gate is closed: the family's default (or the default chunk) stands |
| `coupled` | a ring or state value: it follows the KV codec and the activations |

A caller re-runs `choose` with another task (context, streams, request shape) to show what an input
changes.
