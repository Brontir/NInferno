#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninferno::ops {

/**
 * One Gemma 4 text-attention layer's KV storage.
 *
 * k and v are contiguous BF16 tensors of shape [head_dim, capacity, kv_heads] in head-major
 * order: element (d, slot, h) lives at ((h * capacity) + slot) * head_dim + d. Absolute position
 * p lives in slot p % capacity. The Ops below read and write these planes but own no allocation,
 * frontier, or request identity.
 *
 * window == 0 is a full layer: capacity is at least the largest context the caller will
 * ever address, so every position keeps its own slot. window > 0 is a sliding layer backed by a
 * ring: the caller promises capacity >= window + (tokens of the largest call) - 1, which is
 * exactly what keeps every key a call can see alive while that call appends its own rows.
 *
 * A layer may also serve B independent sequences at once (a batched call), each through a
 * binding read on the device from rows[b], so a captured call follows whichever sequences a
 * replay binds:
 *
 *   Images: block_tables is empty and k, v are [head_dim, capacity, kv_heads, images], one slot
 *           storage as above per image; batch row b uses image rows[b].
 *   Paged:  block_tables is contiguous I32 [pages_per_row, table rows] and k, v are a head-major
 *           page pool [head_dim, 64, pages, kv_heads]. Position p of batch row b lives in row
 *           block_tables[p / 64, rows[b]] * 64 + p % 64 of the pool viewed as
 *           [head_dim, 64 * pages, kv_heads]; a row can hold pages_per_row * 64 positions.
 *
 * rows is contiguous device I32 [B]; empty means one batch row bound to image 0. Every binding
 * of one call is distinct, so no two batch rows write the same bytes.
 *
 * FP8 storage: k and v may instead be FP8_E4M3FN planes of the same shape, each row (one position
 * of one head) with its own scale in k_scale / v_scale, contiguous FP32 planes shaped like k and v
 * without the D axis ([capacity, Hkv(, images)] or [64, pages, Hkv]). The row represents
 * scale * code: the writers take scale = absmax(row) / 448 (0 for a zero row) and
 * code = E4M3(x / scale), saturating (src/ops/softmax_attention/gemma4/fp8_row.cuh). BF16 planes
 * leave the scale tensors empty.
 *
 * stored is empty, or contiguous device I32 [B]: batch row b writes the keys and values of its
 * first stored[b] tokens only, and its later tokens write nothing. A verify call pads a row with
 * fewer drafts than the launch's width by repeating the position of its last token; without it
 * those columns would race that token for its slot. Readers ignore it.
 */
struct Gemma4KvLayer {
    Tensor k;
    Tensor v;
    std::int32_t window = 0;
    Tensor block_tables;
    Tensor rows;
    Tensor k_scale;
    Tensor v_scale;
    Tensor stored;
};

/**
 * Host launch/workspace promise for one gemma4_attention call site. max_visible_keys bounds
 * p + 1 over every query position of every call made under it; max_tokens bounds T. It sizes
 * launch grids and workspace only. It is never a mask and never changes the admitted key set or
 * the numerical result; a captured call is valid for every device state inside the promise.
 */
struct Gemma4AttentionEnvelope {
    std::uint32_t max_visible_keys = 0;
    std::int32_t max_tokens        = 0;
};

/**
 * Visible key set of one gemma4_attention call; see the Math section of gemma4_attention.
 *
 *   Causal: each query column sees the keys at or before its own position (and inside the window
 *           on a sliding layer). This is the Gemma 4 target model's mask.
 *   Block:  every query column of the call sees every key at or before the call's last position,
 *           positions[T-1]: all earlier context plus the whole block, bidirectionally (on a
 *           sliding layer, from its own window's oldest key). This is the DFlash block-diffusion
 *           drafter's mask, and, on a target's D256 sliding layers, the mask of an image's soft
 *           tokens (one call per image). Only head_dim 128 and 256 layers admit it.
 */
enum class Gemma4AttentionMask : std::uint8_t {
    Causal,
    Block,
};

/**
 * Op: append Gemma 4 K/V rows to one layer's slot storage.
 *
 * Math / indexing:
 *   cache.k[d, positions[t] % C, h] = k[d, h, t]
 *   cache.v[d, positions[t] % C, h] = v[d, h, t]      for every d < D, h < Hkv, t < T,
 *   where C is the cache capacity.
 *
 * Logical shapes:
 *   k, v: contiguous BF16 [D, Hkv, T]; positions: contiguous device I32 [T] holding nonnegative
 *   sequential absolute positions; cache planes as documented on Gemma4KvLayer. A batched call
 *   carries k, v [D, Hkv, T, B] and positions [T, B], and batch row b's rows go to its binding.
 *   A batch row's positions may repeat its last real one after it (a padded verify) when the
 *   binding's stored count ends the row before the repeats, so no slot is written twice.
 *
 * Supported domain:
 *   The KV side of the gemma4_attention geometries below: D in {128, 256, 512}, any Hkv >= 1,
 *   any window >= 0. 1 <= T <= C;
 *   T is not bounded by any attention envelope, so a context chunk wider than the attention
 *   calls that later read it (the DFlash drafter's committed-token appends) is admitted.
 *
 * Numeric:
 *   A bit-exact copy; no conversion or rounding. FP8 planes store the row codec of Gemma4KvLayer
 *   (one rounding of each value, one of the row scale).
 *
 * Effects:
 *   Writes exactly the T addressed rows of each KV head in both planes (and their scales); every other cache byte
 *   is unchanged. Inputs are unchanged. k, v, positions and the cache planes are pairwise
 *   non-overlapping. Positions are read on the device only.
 *
 * Workspace: none.
 *
 * Execution:
 *   Stream-ordered and capture-safe; the launch depends only on shapes.
 */
void gemma4_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                      const Gemma4KvLayer& cache, cudaStream_t stream);

/**
 * Op: Gemma 4 grouped-query Softmax Attention over one layer's slot storage, causal or block-wide.
 *
 * Math / indexing:
 *   For query column t at absolute position p = positions[t] and query head h, the KV head is
 *   kvh = floor(h / (Hq / Hkv)). With P_last = positions[T-1], the visible key set is
 *
 *     Causal, full layer    (window == 0): J(p) = { j : 0 <= j <= p }
 *     Causal, sliding layer (window == W): J(p) = { j : p - W < j <= p }
 *     Block,  full layer    (window == 0): J    = { j : 0 <= j <= P_last }, the same for every t
 *     Block,  sliding layer (window == W): J(p) = { j : p - W < j <= P_last }
 *
 *   so on a sliding layer p - W + 1 is the oldest visible key and distance W is excluded, and
 *   under Block the keys of the whole call are visible to every column of it (bidirectional
 *   inside the block) while no key after P_last is. With K[:, j] and V[:, j] the rows of KV head
 *   kvh cached for absolute position j,
 *
 *     score[j]     = scale * dot(FP64(q[:, h, t]), FP64(K[:, j]))
 *     prob[j]      = exp(score[j] - m) / (sum_{x in J} exp(score[x] - m) + [sinks] exp(sinks[h] - m)),
                    m = max(max_J score, [sinks] sinks[h])
 *     out[:, h, t] = sum_{j in J} prob[j] * FP64(V[:, j]).
 *
 *   sinks (optional, FP32 [Hq]) are learned per-head logits in score units (gpt-oss's attention
 *   sinks): each joins its head's softmax denominator and contributes no value.
 *   scale is an explicit argument. Gemma 4 passes 1.0 because its learned q-norm already carries
 *   the head-width scaling; the DFlash drafter passes 1/sqrt(128). No 1/sqrt(D) is applied
 *   implicitly.
 *
 * Logical shapes:
 *   q, out: contiguous BF16 [D, Hq, T]; positions: contiguous device I32 [T], nonnegative and
 *   sequential (positions[t] = positions[0] + t), or sequential up to a last real column and
 *   repeating it after (positions[t] = min(positions[0] + t, positions[T-1])): the padding of a
 *   speculative verify row with fewer drafts than the call's width. A padding column is the last
 *   real column again: it sees that column's keys, and the call reads no key past positions[T-1].
 *   A batched call carries q, out [D, Hq, T, B] and positions [T, B]; batch row b is the
 *   single-sequence call above over its own binding, with the same result bits as that call.
 *
 * Supported domain:
 *   [D, Hq, Hkv, window] with D in {64, 128, 256, 512} (128-column warp slices; a 64-wide head is one
 *   slice of its own), Hq a
 *   multiple of Hkv with the GQA group Hq / Hkv in 1..16 (powers of two have their own instances,
 *   every other group shares one that reads it at run time), and any window >= 0 (0: full
 *   layer). The shipped profiles are points of this set: [256,16,8,1024] and [512,16,2,0]
 *   (26B-A4B), [256,32,16,1024] and [512,32,4,0] (31B), [256,16,8,1024] and [512,16,1,0] (12B);
 *   the DFlash drafters' [128,32,8,2048|0] (26B-A4B), [128,64,8,2048|0] (31B), [128,32,8,2048|0]
 *   (12B). mask == Block is admitted only on head_dim 128 and 256 layers. Any other
 *   geometry, a negative window, or Block elsewhere throws std::invalid_argument before a route
 *   is chosen. scale is finite and positive;
 *   1 <= T <= envelope.max_tokens. The drafter calls use T <= 16; wider T is served by the same
 *   routes as the target profiles and is not tuned for head_dim 128.
 *
 * Numeric:
 *   The oracle evaluates the formula above naively in FP64 from the represented BF16 inputs (for
 *   FP8 planes, K and V are BF16(FP32(code) * scale), the values the kernel stages).
 *   The BF16 output is promoted for comparison; its storage rounding belongs to the Op's
 *   criterion. Accumulation precision, the operand type of the probability-value product,
 *   reduction order and decomposition are private implementation choices.
 *
 * Effects:
 *   out is completely overwritten and is the only observable mutation. The caller guarantees
 *   that every key/value at a visible position has already been appended and, on a sliding ring,
 *   that no row visible to this call has been overwritten since. Rows outside the union of the
 *   call's visible sets are never read and may hold any bits, NaN included; in particular no row
 *   after P_last is read under either mask, so slots still holding a rejected draft's rows past
 *   the block are harmless. q, positions, the cache planes, out and live workspace allocations
 *   are pairwise non-overlapping.
 *
 * Workspace:
 *   Caller-owned transient capacity from gemma4_attention_workspace_bytes for the same geometry
 *   and envelope, suballocated for the duration of the call only.
 *
 * Execution:
 *   Positions are read on the device only. Grid sizes and workspace depend only on shapes,
 *   geometry and the envelope, so one captured call stays valid across replays with new positions
 *   inside the envelope. The mask is a host argument and part of the captured call. The caller
 *   guarantees p + 1 <= envelope.max_visible_keys for every query (under Block this is
 *   P_last + 1, as for Causal) and, on a full layer, envelope.max_visible_keys <= capacity (a paged
 *   binding: pages_per_row * 64).
 */
/**
 * Workspace for a decode call of `batch` rows that runs them in as few launches as fit: the
 * one-row bound times batch, capped by what `arena` has free (at least the one-row bound).
 */
[[nodiscard]] std::size_t gemma4_attention_batch_workspace_bytes(
    const WorkspaceArena& arena, std::int32_t batch, std::int32_t head_dim, std::int32_t q_heads,
    std::int32_t kv_heads, Gemma4AttentionEnvelope envelope, std::int32_t window);

void gemma4_attention(const Tensor& q, const Tensor& positions, const Gemma4KvLayer& cache,
                      float scale, Gemma4AttentionEnvelope envelope, WorkspaceArena& workspace,
                      Tensor& out, cudaStream_t stream,
                      Gemma4AttentionMask mask = Gemma4AttentionMask::Causal,
                      const Tensor* sinks = nullptr);

/**
 * Minimum workspace capacity for every T in [1, envelope.max_tokens] and every mask at one
 * [D, Hq, Hkv] geometry with the given window (0: full layer). An unsupported geometry, a negative
 * window or an invalid envelope throws.
 */
[[nodiscard]] std::size_t gemma4_attention_workspace_bytes(std::int32_t head_dim,
                                                           std::int32_t q_heads,
                                                           std::int32_t kv_heads,
                                                           Gemma4AttentionEnvelope envelope,
                                                           std::int32_t window);

/**
 * The same for every window at once: a bound that covers any window the geometry may run with.
 */
[[nodiscard]] std::size_t gemma4_attention_workspace_bytes(std::int32_t head_dim,
                                                           std::int32_t q_heads,
                                                           std::int32_t kv_heads,
                                                           Gemma4AttentionEnvelope envelope);

} // namespace ninferno::ops
