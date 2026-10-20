#pragma once

#include "core/tensor.h"
#include "ops/softmax_attention/gemma4/kv_address.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops::detail {

// Largest T served by the split-K decode route; wider calls take the prefill route.
inline constexpr std::int32_t kGemma4DecodeMaxTokens = 16;
// Largest GQA group (query heads per KV head): a prefill row block must hold at least two tokens.
inline constexpr std::int32_t kGemma4MaxGroup = 16;

// Launch facts of the decode route for one exact T. Everything here is a function of geometry,
// the window and T only: never of device positions, nor of the envelope's key capacity.
struct Gemma4DecodePlan {
    std::int32_t rows       = 0; // query rows of one KV head: T * (Hq / Hkv)
    std::int32_t warp_rows  = 0; // WR: 16-row warp rows per CTA
    std::int32_t row_groups = 0; // CTAs sharing one key chunk
    std::int32_t splits     = 0; // key chunks per KV head (grid capacity; live count is on device)
};

[[nodiscard]] Gemma4DecodePlan
gemma4_attention_decode_plan(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads,
                             std::int32_t window, std::int32_t tokens);

struct Gemma4AttentionLaunch {
    const Tensor& q;
    const Tensor& positions;
    const Tensor& cache_k;
    const Tensor& cache_v;
    Gemma4KvAddress address;
    std::int32_t window;
    float scale;
    bool block; // Gemma4AttentionMask::Block: every column sees keys up to positions[T-1]
    Tensor& out;
    const Tensor* sinks   = nullptr; // optional FP32 [Hq] sink logits
    const Tensor* k_scale = nullptr; // FP8 planes: FP32 row scales (Gemma4KvLayer)
    const Tensor* v_scale = nullptr;
};

// Which prefill tile a 512-wide, group-8 call takes: 32 rows on 2 x 4 warps of 128 columns
// (Narrow), 64 rows on 4 x 2 warps of 256 columns (Wide), or the device's choice (Auto). Both give
// the same bytes; the override is for tests.
enum class Gemma4PrefillSlices : std::uint8_t { Auto, Narrow, Wide };
void gemma4_attention_set_prefill_slices(Gemma4PrefillSlices slices) noexcept;

// Whether a prefill call of a full causal layer runs chained (spans of key tiles handed between
// CTAs, gemma4_attention_kernel) when its workspace fits: Auto and On do, Off never. Both give the
// same bytes; the override is for tests and benchmarks.
enum class Gemma4PrefillChain : std::uint8_t { Auto, Off, On };
void gemma4_attention_set_prefill_chain(Gemma4PrefillChain chain) noexcept;

// Workspace of a chained prefill call: 0 when the call runs plain (a sliding layer, the Block
// mask, a row block wider than a key tile, a decode-sized call, or the override is Off).
[[nodiscard]] std::size_t gemma4_attention_chain_bytes(std::int32_t head_dim, std::int32_t q_heads,
                                                       std::int32_t kv_heads, std::int32_t window,
                                                       bool block, bool fp8, std::int32_t tokens);
// The most any prefill call of up to max_tokens takes (either prefill tile, BF16 or FP8 planes).
[[nodiscard]] std::size_t gemma4_attention_chain_bound(std::int32_t head_dim, std::int32_t q_heads,
                                                       std::int32_t kv_heads, std::int32_t window,
                                                       std::int32_t max_tokens);

// A batched call (batch > 1) runs its rows one after another, each the single-row call over its
// own slice of q, positions and out and its own binding.
// chain: gemma4_attention_chain_bytes of workspace for a chained call, or null to run plain.
void gemma4_attention_prefill_launch(const Gemma4AttentionLaunch& launch, std::int32_t batch,
                                     void* chain, cudaStream_t stream);

// A batched decode call runs its rows in launches of up to rows_per_launch rows (partials hold
// that many rows' splits); every row does the single-row call's work, so its bytes do not depend
// on rows_per_launch.
void gemma4_attention_decode_launch(const Gemma4AttentionLaunch& launch,
                                    const Gemma4DecodePlan& plan, std::int32_t batch,
                                    std::int32_t rows_per_launch, Tensor& partial_acc,
                                    Tensor& partial_m, Tensor& partial_l, cudaStream_t stream);

void gemma4_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                             const Gemma4KvLayer& cache, const Gemma4KvBinding& binding,
                             cudaStream_t stream);

} // namespace ninferno::ops::detail
