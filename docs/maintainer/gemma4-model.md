# Gemma 4 model reference

This reference describes the Gemma 4 text mathematics implemented by NInferno's Gemma 4 family: the
26B-A4B MoE and 31B Dense targets, Google's MTP drafter and z-lab's DFlash drafter. The family lives
in [`src/models/gemma4/`](../../src/models/gemma4/) beside Qwen3.5 and runs through
`ninferno-gemma4`; it is not registered with the Engine and has no frontend or serving route.

[Qwen3.5 model](qwen3_5-model.md) describes the neighbouring family;
[artifact container](artifact-container.md) defines stored objects and bindings;
[Op development](op-development.md) owns numerical qualification.

## A family, not a variant

Gemma 4 differs from Qwen3.5 at every point of the layer schedule: sandwich norms around both
blocks, two attention geometries in one model, K reused as V on the full layers, a dense MLP beside
the expert bank, a per-layer output scalar, a soft-capped tied head, and no recurrent state. The
family therefore owns its config interpretation, binding, prepared parameters, KV pools and
execution explicitly, and shares only Ops, the artifact reader and core types. There is no family
base class and no shared model graph.

## Source units

| Unit | Responsibility |
|---|---|
| `config.h`, `config.cpp` | `Config`, `DrafterConfig`, `DFlashConfig`: the converter's text configs, validated |
| `weights.h` | Prepared parameters: every projection as its Op consumes it (`Projection`), norms as device vectors |
| `model.h` | Immutable `Model`, `Drafter`, `DFlashDrafter`; `ActivationProfile` |
| `load.h`, `load*.cpp` | One loading skeleton (`open_text_component`, `Demands`, `Prepare`, `ArtifactBacking`) and one loader per model |
| `target.h`, `target.cu` | The target forward: KV pools, every layer over T columns, soft-capped and uncapped heads, workspace sizing |
| `mtp_round.h`, `mtp_round.cu` | The draft phase of a round with the MTP drafter |
| `dflash_round.h`, `dflash_round.cu` | The draft phase of a round with the DFlash drafter, its context cache and prefill hook |
| `speculation.h`, `speculation.cu` | Greedy speculative decoding: draft, verify, accept, one CUDA graph per draft depth, adaptive depth |
| `session.h`, `session.cu` | `Session`: chunked prefill, teacher-forced scoring, graph-captured decode, the public speculation entry |
| `gemma4.h` | The applications' single include |

Each loader follows the same skeleton: read the text component's config and check its pairing,
declare every parameter demand while the Binder is alive (a Use's policy and activation divisor
cannot be read after `finish()`), materialize, then prepare. Rows of one parent become one packed
projection only when they cover the parent completely, because the NVFP4 native Weight addresses
whole parents.

## Configuration

[`config.cpp`](../../src/models/gemma4/config.cpp) defines the exact fields. The target requires
`model_type = gemma4_text`, a tied head, one `layer_types` entry per layer with a full-attention
last layer, and sliding layers that rotate every pair. Per-kind `attention_geometry` and
`rope_parameters` give each kind its own head dimension, KV heads, theta and rotary pairs.

| Quantity | 26B-A4B | 31B |
|---|---:|---:|
| Hidden width H | 2816 | 5376 |
| Layers (full at every sixth, from 5) | 30 | 60 |
| Query heads | 16 | 32 |
| Sliding: KV heads × D, theta, rotated pairs | 8 × 256, 1e4, 128 | 16 × 256, 1e4, 128 |
| Full: KV heads × D, theta, rotated pairs | 2 × 512, 1e6, 64 | 4 × 512, 1e6, 64 |
| Sliding window | 1024 | 1024 |
| Dense GeGLU width | 2112 | 21504 |
| Experts / per token / width | 128 / 8 / 704 | — |
| Vocabulary, soft cap | 262144, 30 | 262144, 30 |

Both use `attention_k_eq_v` and `attention_scale = 1`. The drafters' configs are checked against
the target they are loaded for; each check is a pairing the attention or head call would otherwise
get wrong in silence:

- MTP (`gemma4_assistant`): backbone width, vocabulary, sliding window, query heads and both
  attention geometries equal the target's. The 26B drafter has 4 layers of width 1024; an optional
  `head_rows` shortlist restricts its head, with `text/head_token_ids` naming each row's token.
- DFlash (`gemma4_dflash`): width, vocabulary and `num_target_layers` equal the target's, and every
  tapped layer lies inside it. Five D128 grouped-query layers, block 16, sliding window 2048.

## Layer mathematics

`norm(x, w) = w * x / sqrt(mean(x^2) + eps)`; `v_norm(x)` is the same without a weight. Every
target layer computes, per token column:

```text
h = norm(x, input_norm)
q, k, v = query(h), key(h), value(h)          # full layers under k_eq_v: v = k, before any norm
q = rope(norm(q, query_norm)); k = rope(norm(k, key_norm)); v = v_norm(v)
x = x + norm(output(attention(q, k, v)), post_attention_norm)

d = down(gelu_tanh(gate(f)) * up(f)),  f = norm(x, pre_ffn_norm)
MoE: d = norm(d, post_mlp_norm) + norm(experts(router(x), norm(x, pre_moe_norm)), post_moe_norm)
x = (x + norm(d, post_ffn_norm)) * output_scale
```

Sliding layers attend over the last `sliding_window` positions with RoPE over the whole head.
Full layers use proportional RoPE: 64 of 256 pairs rotate, with the frequency ladder still taken
over D = 512. The attention scale is folded into `query_norm`. The router reads the residual stream
after the attention update; the expert contract is in
[`gemma4_moe.h`](../../include/ninferno/ops/gemma4_moe.h).

The embedding table is stored pre-scaled by `sqrt(H)`. The head is tied to the embedding source
but stored unscaled (Q6); logits are `softcap * tanh(z / softcap)`. The cap is monotonic, so
callers that only take an argmax (verify, greedy acceptance) use the uncapped head.

## KV pools

The target keeps BF16 `[D, capacity, kv_heads]` K and V planes per layer, in two pools:

- a sliding layer keeps a ring of `min(window + prefill_chunk, max_context)` slots: the first query
  of a chunk reaches `window - 1` keys back, so window plus chunk slots hold every key any query of
  that chunk can see;
- a full layer keeps a plane of `max_context` slots.

Under `k_eq_v` the full layers still store both planes: V is the unweighted, unrotated norm of the
same projection. The MTP drafter owns no keys or values; it attends over the target's last sliding
and last full layer caches. The DFlash drafter owns one cache per drafter layer (rings for its
sliding layers). A `Session` owns every pool and one workspace sized for the target's chunk plus the
DFlash drafter's share.

## Fused operations

Norms are folded into the launches around them so that the decode step is not a chain of small
norm kernels:

- `gemma4_qkv_prepare` applies the Q/K norms, RoPE and `v_norm` in one launch and writes K and V
  straight into the layer's cache slots.
- `gemma4_residual_norm` performs a residual update and emits the normalized inputs of whatever reads
  the updated stream: after attention, the feed-forward (and expert) inputs; after the
  feed-forward, the next layer's `input_norm` (or `final_norm`). Outputs are normalized from the
  stored BF16 residual, so the fusion changes no rounding the unfused schedule had.
- The expert bank runs on a side stream, a parallel branch of the captured graph, beside the dense
  MLP. Its output and workspace are carved before the fork, so the branches share no arena bytes.

Their contracts and FP64 oracles are in [`gemma4_fused.h`](../../include/ninferno/ops/gemma4_fused.h).
Decode is one captured CUDA graph per step whose token and position live on the device.

## Activation profiles and the verify rule

| Profile | CLI | Dense projections | 26B-A4B expert bank, calls of more than 16 tokens |
|---|---|---|---|
| `Artifact` | none | Each Use's recorded policy (W4A4 where the NVFP4 checkpoint was calibrated for it) | W4A4, each expert's input encoded against its checkpoint divisor |
| `A16` | `--a16` | BF16 activations | W4A4, as `Artifact`: on 26B-A4B this profile is mixed |
| `A16Full` | `--a16-full` | BF16 activations | W4A16: BF16 activations against the decoded NVFP4 weights |

On 31B, which has no expert bank, `A16Full` equals `A16`. The expert bank's decode (T = 1) and
small-T (T ≤ 16) routes keep BF16 activations under every profile. `Layer::moe_activations`
carries the choice to `ops::gemma4_moe`. Under all three profiles a token's prefill bytes do not
depend on the chunk it shares for calls of 17 tokens or more. The Op and linear token-invariance
tests check both expert tiers and the A16 dense shapes.

**A speculative verify pass runs every projection and the head at A16, whatever the profile.** It
must compute what k+1 single-token decode steps would. Plain decode never takes a W4A4 route, but
from eight columns the 26B's packed Q/K/V projection would, and the round would accept drafts
against a coarser model than the one decode runs. The draft depth is bounded so that the verify
width stays on routes plain decode also runs, the expert bank's small-T route included: at most 15
MTP drafts, and at most `min(block_size, 16) - 1` DFlash drafts. The profile still reaches
speculation through the prompt's prefill, which writes the context that verify attends over and
that DFlash reads.

The measured best draft configuration per model and GPU is the model's profile in the
[drafter registry](../../src/spec/data/drafter_registry.json), which `--spec auto` runs. The executable defaults
(`--draft 4` for MTP, `block_size - 1` for DFlash) are not those recommendations.

Speculation is greedy: a round accepts the longest drafted prefix the target's argmax agrees with.
The verify pass runs every projection with BF16 activations (`Activations::A16Only`). Whether its
stream equals plain greedy decode depends on the profile:

- **exact (`--a16-full`)** is invariant: decode and verify run the same A16 routes, and speculation
  reproduces plain decode byte for byte (Gemma 4 31B UD-IQ3_XXS + MTP Q8_0, K = 4/6, four chat
  prompts, 256/256 tokens each).
- **fast (the artifact's recorded policies)** guarantees quality and run-to-run determinism, not
  spec = plain. Plain decode takes the int8 R8 route at T = 1 for the formats that record AllowA8
  (q8_g32 and the IQ2/IQ3 codebooks), while verify stays A16, so the greedy stream may leave plain
  decode at a near-tie in the tail (31B IQ3: after 61-215 tokens). Giving verify the R8 route made
  the two agree more often but cost 27-63 % of speculative throughput (R8's dp4a work grows with T),
  so fast keeps the A16 verify.

## MTP drafter

A round from the pending token at position P takes k single-token steps. Step i reads
`[target_embedding(token_i); carried]`, where `carried` is the target's final-normed state behind the
pending token for the first step and the drafter's post-projected state afterwards:

```text
x = pre_projection([embedding; carried])       # 2 * backbone -> H
x = layers(x)                                   # query-only; K/V from the target's caches at P
h = norm(x, final_norm)
token_(i+1) = argmax(head(h))                   # remapped through head_token_ids when shortlisted
carried = post_projection(h)                    # H -> backbone
```

The drafter's layers follow the target's sandwich schedule without an expert branch. The prefill
writes the first carried state; every accepted round writes the state behind its new pending token.

## DFlash drafter

The drafter conditions on the target's residual stream after the tapped layers (the layer output
with its scalar applied). Every forward that commits tokens, each prefill chunk and each verify
pass, captures those columns; the drafter then appends their keys and values to its own cache:

```text
c = norm(feature_projection(concat(taps)), context_norm)
k_l, v_l = rows of one packed context_kv projection of c, per drafter layer l
k_l = rope(norm(k_l, key_norm_l))                # appended at the tokens' positions
```

A round drafts the block `[pending, mask × 15]` at positions P..P+15 in one pass through the five
layers over the committed context, then scores columns 1.. with the target's head. The first k
drafts are verified. The block's own keys and values occupy the slots where the committed rows of
P.. will be written; the next context append overwrites them before any round reads them as
context.

Two readings of the published sources were ambiguous and are settled by an FP32 reference of the
drafter run on the target's own residual stream, whose acceptance was highest with these choices:

- The drafter embeds with the target's table as stored, pre-scaled by `sqrt(H)`.
- The sliding layers are causal inside the block; the full layer sees the whole block.

The engine's round-by-round acceptance length matches that reference.

## Programmatic dependent launch and `__restrict__`

Kernels chained with programmatic dependent launch (PDL) start while their producer runs and call
`griddepcontrol.wait` before reading its output. A `const __restrict__` pointer makes each load
through it an invariant load, which the compiler may schedule above the wait. The small-T expert
route did exactly that: it read the expert count before the grouping kernel had written it, got
whatever the arena held, and speculation diverged from plain greedy decode.

Any pointer through which a PDL consumer reads producer output must therefore be a plain pointer.
`__restrict__` remains on weights and on call inputs that are complete before the chain starts.
A test must replay a graph that clears the workspace before the call; a warm workspace returns the
previous call's identical values from an early read and hides the hazard.

## Not shared with Qwen3.5

These stay separate because their mathematics differs or sharing would cost more than it saves:
the DFlash context of Qwen (fixed geometry and FP16 V) against the Gemma one, the two MTP drafters,
attention and RoPE (two geometries, K = V, block mask, frequency ladder over D), sandwich norms and
the soft-capped head, and Qwen's Program machinery (paged KV, state images) for a single-sequence
session.
