# Inkling 2 model reference

This reference describes the text mathematics of Thinking Machines' Inkling 2 Mini
(`Inkling2LoopForConditionalGeneration`, `model_type = inkling2_loop`) as NInferno's Inkling 2
family implements it, and maps every step onto an existing or a new Op. The family will live in
`src/models/inkling2/` beside Gemma 4 and Qwen3.5. The vision tower, the 3-layer MTP block and the
DSpark drafter are out of scope for the first stage.

The reference implementation is llama.cpp's `src/models/inkling2.cpp` and `inkling2-loop.cpp`
(ggml-org/llama.cpp 29b52d398 and 690b87d88, November 2026); the GGUF names and the conversion
rules are in its `conversion/inkling2.py`. Numerical qualification compares NInferno's logits with
that build's on the same GGUF ([Op development](op-development.md), [perplexity](../perplexity.md)).

## A family, not a variant

Inkling 2 shares no layer schedule with the other families. A layer visit has no RoPE (a learned
relative-position bias carries the positions), a per-query learned sink, an output gate, short
causal convolutions on K, V and both block outputs, and residual streams that restart every six
layers and are mixed back by attention over the depth. The middle twelve layers run three times
with shared weights; each visit keeps its own KV cache and convolution state, and passes two and
three read messages from the previous pass through rank-64 gates.

## Configuration (Inkling 2 Mini)

| Quantity | Value |
|---|---:|
| Hidden width H | 2304 |
| Physical layers / visits | 24 / 48 (layers 0-5 once, 6-17 three times, 18-23 once) |
| Query heads x D, KV heads | 24 x 128, 8 |
| Full-attention layers | 5, 11, 17, 23 (one in each block of six); the rest sliding, window 1024 |
| Relative bias | d_rel = 16 per head; extent 1024 (full), the window (sliding); clip 8 |
| Q/K clip | L2 norm per head at most 22.627 (= sqrt(512)) |
| Log scaling (full layers) | tau(p) = 1 + 0.1 ln(max(1, (p + 1) / 8192)) |
| Short convolution | 4 taps, on K, V, the attention output and the expert output |
| Experts / per token / width | 128 / 8 / 1152, no shared expert |
| Router | sigmoid scores, selection bias, renormalized, times 8 |
| Expert activation (SITU) | 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25) |
| Loop controllers | 4 per looped layer (q, k, v, ffn), rank 64 |
| Vocabulary | 201024 padded rows, 200058 tokens; soft cap 30 |

## Layer mathematics

`n(x) = x / sqrt(mean(x^2) + 1e-6)` (no weight), `norm(x, w) = w * n(x)` (eps 1e-6), and
`clip(x, c) = c x / max(|x|, c)` over each head's row. Per token column, with `blocks` the list of
finished residual blocks (empty at the first layer):

```text
mix(blocks, cur, Q)   # attention over the depth, one output per query row of Q [H, m]:
    S = [n(b_last), ..., n(b_0), n(cur)]                          (states, H wide)
    out_r = sum_s softmax_s(S_s . Q_r / sqrt(H)) S_s

# attention inputs
h_q = h_k = h_v = cur                       # layer 0
h   = mix(blocks, cur, attn_res_query)      # 1 query (sliding) or 3 (full: q, k, v)
pass >= 2 (looped layers): h_j = control_j(h_j, m_j) for j = q, k, v
sliding: h_j = norm(h_j, attn_norm)         # the same input for q, k, v unless messages differ
full:    h_q, h_k, h_v = norm(., q_inp_norm), norm(., k_inp_norm), norm(., v_inp_norm)

q = clip(q_clip * (Wq h_q), 22.6)                             [24 x 128]
k = clip(conv(Wk h_k, k_conv, full ? Wkc h_k : none), 22.6)   [8 x 128]
v = n(conv(Wv h_v, v_conv, full ? Wvc h_v : none))            [8 x 128]
r = clip(rel_clip * (Wr h_q), 8)                              [24 x 16]
full layers: q *= tau(p), r *= tau(p)
t = Ws h_q                                                    [24] sink logits

s_ij = q_i . k_j / sqrt(128) + r_i . B[:, min(p_i - p_j, extent)]   (B[:, extent] = 0; the bias is not scaled)
o_i  = sum_j exp(s_ij) v_j / (exp(t_i) + sum_j exp(s_ij))  (causal; sliding: p_i - p_j < 1024)
a    = conv(Wo (o * sigmoid(Wg h_q)), attn_post_conv, none)

if layer % 6 == 0: blocks.append(cur); x = a        # a block starts a new residual stream
else:              x = cur + a

f = mix(blocks, x, ffn_res_query); pass >= 2: f = control_ffn(f, m_ffn)
f = norm(f, ffn_norm)
MoE: l = R f (FP32); s = sigmoid(l); top 8 by s + bias; w = 8 s / sum_top s
     y = sum_e w_e Down_e situ(Gate_e f, Up_e f)
cur = x + conv(y, ffn_post_conv, none)
```

The short convolution with its state of the last three inputs `x_{-3..-1}` per channel is

```text
conv(x, w, dyn)_t = (x_t + sum_{k<4} w_k x_{t-3+k} + [dyn] sum_{lag<4} dyn_{t,head(c),lag} x_{t-lag}) / sqrt(4 (dyn ? 2 : 1) + 1)
```

where `dyn = Wkc h_k` (or `Wvc h_v`) gives four taps per KV head and token. The four convolutions
of a visit share one state slot of `3 x (2 x 1024 + 2 x 2304)` channels.

The loop: every pass starts from the prelude output (after layer 5) with the prelude's blocks.
Pass 2 and 3 of looped layer i first build its four messages from the previous pass:

```text
bank = that pass's blocks after its first core block, and its output
m    = mix(bank, out_prev, msg_query_i)                 (4 queries: q, k, v, ffn)
control(c, m) = c - tanh(Wf z) * c + tanh(Wg z) * m,    z = Wd [n(c); n(m)]   (Wd: 64 x 2H)
```

The output is `norm(mix(blocks, cur, output_res_query), output_norm)`, the head over all 201024
rows, `30 tanh(z / 30)`, and the first 200058 columns. Token embeddings are normalized by
`tok_norm` before the first layer (image embeddings are not).

## Per-visit state

48 visit slots (24 + 2 x 12). A slot keeps its KV planes (a ring of the window plus the chunk
for the 40 sliding visits, a full plane for the 8 full visits: layers 5, 23 once and 11, 17 three
times) and its convolution state. KV per token: 8 full visits x 8 x 128 x 2 x 2 B = 32 KiB, so
128 K tokens of context hold 4 GiB of full-layer KV in BF16 (half in the one-byte row codec).

## Program integration (serve)

The standalone `Target` (one sequence, plain launches) qualifies the mathematics. The serving
family (`models/inkling2/program_family.cpp`, loaded by `runtime/engine/inkling2_family.cpp` for
`inkling2_text` artifacts) runs the same `Pass` over the Program's state:

- visit slot i's keys and values: the paged pool for the 8 full slots (`full_layers`, head-major
  BF16 pages), a ring in each sequence's state image for the 40 sliding slots (`sliding_layers`,
  window + the widest call - 1 rows), both through `program::ring_decoder_bindings`;
- visit slot i's convolution history: layer i of the state image's linear planes
  (`recurrent_layers` 48, `conv_channels` 6656 = K + V + attention output + expert output,
  `conv_width` 3, FP32), the kernel's `Inkling2ConvState` with the slot stride 3 x 6656 and each
  batch row's state slot;
- a fork (a prefix checkpoint's first write) copies the source's rings and its linear slot; a
  fresh sequence's slot is zeroed by the Program.

`StateFamily::Inkling2` names that layout (`sequence_bytes.cpp`, `inkling2_layout`); the keys and
values stay BF16 (`validate_state_storage`). Prefill and ordinary decode (captured as a decode
graph) follow the Gemma 4 family without its MTP and vision paths. No speculative backend. The
plan without the weights (`product/plan/plan_facts.cpp`, `state_of`) does not size Inkling 2 yet:
the server's automatic configuration choice leaves its axes to the engine's defaults.

First serve measurement (RTX 4090, Inkling-2-Mini UD-Q4_K_M, 256 tokens a stream, greedy):
97.3 tokens/s at one stream and 260 at four, against llama.cpp's llama-server on the same GGUF
at 78.3 and 188.

## Decode cost (first form)

The analyzer's sealed prices for the decode projections (analyzer_r20/sealed_inkling2.json,
written before any measurement) put a visit's seven attention projections at ~44 us on the RTX
4090, i.e. ~2.1 ms a token over 48 visits before the experts and the head: the calls are
latency-bound (3-10 us each). q, gate, relative and sink read the same input, so one packed
projection (and k / v / taps another) is the first speed step after qualification.

## Ops: reuse, extend, new

| Step | Op | Status |
|---|---|---|
| Every projection (Wq, Wk, Wv, Wo, Wg, Wr, Ws, Wkc, Wvc, controllers, head) | generic linear (analyzer routes) | reuse; new shapes N/K in {2304, 3072, 1024, 384, 24, 32, 64, 4608} |
| Router + experts | `moe` (Sigmoid, selection bias, renormalize, routed_scale 8) | extend: `MoeActivation::Situ` (beta, linear beta) |
| RMS norms | `rmsnorm` | reuse |
| clip | `l2norm` with eps = threshold, times threshold | reuse (fold into the attention prepare) |
| Attention with relative bias and sink | `gemma4_attention` (D 128, ring and full planes, paged, row codec) | extend: optional `r [16, Hq, T]`, table `B [16, extent + 1]`, sink logits `[Hq, T]`; no RoPE |
| Output gate | `sigmoid_mul` | reuse |
| Short convolution with state | `causal_conv1d_silu` (width 4, state in/out) | extend: an identity epilogue with the residual and the scale, dynamic per-head taps |
| Depth mix | — | new: `depth_mix` (<= 9 states, <= 4 queries a column) |
| Loop controller | linear (64 x 4608, 2304 x 64) + a fused tanh-gate update | new elementwise epilogue |
| Head soft cap | `logit_softcap` | reuse |

Every extension keeps the existing call sites bit-equal (the new fields default to absent); the
goldens and the family tests of Gemma 4 and Qwen3.5 are the check.

## Stages

1. Converter recipe for the GGUF (tensor names of `conversion/inkling2.py`, F32 conv / clip /
   query / bias tables), the config, the loader.
2. Target forward in plain launches, prefill chunk and decode, against llama.cpp mainline: logits
   KLD and top-1 on a fixed prompt set, perplexity on wikitext-2 (docs/perplexity.md).
3. CUDA-graph decode, fused prepares (norm + projection inputs, attention prepare, controller
   epilogue), analyzer routes for the new shapes; e2e against llama.cpp mainline.
4. MTP block and DSpark drafter; vision.

## Unknown-kernel exam

The new and extended kernels (biased attention, depth mix, convolution epilogue, controller
epilogue) and the new projection shapes are never timed while the analyzer is developed. Their
counted facts come from the same tools as every kernel's; the analyzer's prices for them are
written to a sealed file before the first measurement; the exam reports their price error and
pick regret apart from the known kernels.
