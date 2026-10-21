#pragma once

// The predicted device cost of a prefill unit and of a decode round, from their shapes alone, so
// that --concurrent-prefill shares the GPU between them (Scheduler::fair_decode_credit) the same
// way on every machine of a chip type for the same requests. Both sides are priced with the
// context cache's machine cost model (context_cost.h: per-call, per-token and per-attention-pair
// prefill costs, fitted per card class), which the decode side extends by declared ratios:
//
// - A decode round reads, once, the weights its tokens touch (memory-bound), where a prompt token
//   costs its share of the matrix FLOPs (compute-bound). With 4-bit weights and the FLOP-to-byte
//   ratios of the supported cards (about 100-200 from V100 to RTX 5090), reading the active
//   weights once costs about kWeightReadPromptTokens prompt tokens. A MoE round of L lanes touches
//   the experts its L tokens route to: an expected fraction 1 - (1 - k/E)^L of each layer's
//   routed experts under uniform routing, so L = 8 lanes of a 256-expert top-8 model read about
//   seven times the routed bytes one lane reads, while a prefill chunk reads every expert once.
// - Each lane's attention reads its whole context once per round, without the reuse a prefill
//   chunk's queries share: one context token costs kDecodeAttentionPairs prefill attention pairs.
// - Each lane's token adds one prompt token of compute.
// - A step pays the larger of the call's fixed cost and its weight read, never both: the prefill
//   fit's per-call cost (chunk_ns) is mostly the one read of the weights a call makes, so adding
//   the read on top of it counted the weights twice.
// - A round runs steps_per_round steps (round_pacing.h: 16 under a CPU quota, else 1, or
//   --decode-steps), each reading the weights again; a speculative round runs one verify step of
//   verify_tokens_per_lane tokens per lane (the largest draft window plus the bonus token), which
//   reads the weights once and computes and attends for every token. The engine prices a round
//   before it runs and does not know whether a speculative lane will draft: a lane that stopped
//   drafting (load_plain, context_plain) or a shallower window verifies fewer tokens, so such a
//   round is priced above its cost and decode gets somewhat less than its equal share beside a
//   prefill. A step's growth of the contexts within a round is not priced.
//
// The ratios are estimates to refit on lab cards; a wrong ratio only moves the split of GPU time
// between prefill and decode, never a request's output. Arithmetic is integer and saturating.

#include "runtime/contract/resources.h"
#include "runtime/engine/context_cache/context_cost.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>

namespace ninferno::runtime {

// Facts of the served model's shape, fixed at load.
struct DecodeRoundShape {
    std::uint64_t weight_bytes        = 0; // every weight tensor of the artifact
    std::uint64_t routed_expert_bytes = 0; // of them, the routed-expert objects (0: dense)
    std::uint32_t experts             = 0; // routed experts of a MoE layer
    std::uint32_t experts_per_token   = 0; // routed experts each token selects
    // The round's shape (Program::decode_pacing): steps an ordinary round runs, and tokens each
    // lane verifies in a speculative round (1: not speculative).
    std::uint32_t steps_per_round        = 1;
    std::uint32_t verify_tokens_per_lane = 1;
};

struct InterleavePricing {
    static constexpr std::uint64_t kWeightReadPromptTokens = 32;
    static constexpr std::uint64_t kDecodeAttentionPairs   = 16;
};

namespace interleave_detail {

inline constexpr std::uint64_t kSaturated = std::numeric_limits<std::uint64_t>::max();

[[nodiscard]] constexpr std::uint64_t add(std::uint64_t a, std::uint64_t b) noexcept {
    return a > kSaturated - b ? kSaturated : a + b;
}

[[nodiscard]] constexpr std::uint64_t mul(std::uint64_t a, std::uint64_t b) noexcept {
    return b != 0 && a > kSaturated / b ? kSaturated : a * b;
}

// value_q32 * n / 2^32 without a 128-bit type: with value = hi * 2^32 + lo and
// n = n_hi * 2^32 + n_lo, it is hi * n + lo * n_hi + lo * n_lo / 2^32.
[[nodiscard]] constexpr std::uint64_t q32_times(std::uint64_t value_q32, std::uint64_t n) noexcept {
    const std::uint64_t hi   = value_q32 >> 32U;
    const std::uint64_t lo   = value_q32 & 0xffff'ffffULL;
    const std::uint64_t n_hi = n >> 32U;
    const std::uint64_t n_lo = n & 0xffff'ffffULL;
    return add(add(mul(hi, n), mul(lo, n_hi)), (lo * n_lo) >> 32U);
}

} // namespace interleave_detail

// The expected fraction (Q32) of a layer's routed experts that `lanes` tokens touch, each choosing
// `per_token` of `experts`: 1 - (1 - per_token/experts)^lanes, in fixed point.
[[nodiscard]] constexpr std::uint64_t touched_expert_fraction_q32(std::uint32_t experts,
                                                                  std::uint32_t per_token,
                                                                  std::uint32_t lanes) noexcept {
    constexpr std::uint64_t one = 1ULL << 32U;
    if (experts == 0 || per_token == 0 || lanes == 0) { return 0; }
    if (per_token >= experts) { return one; }
    const std::uint64_t keep = (static_cast<std::uint64_t>(experts - per_token) << 32U) / experts;
    std::uint64_t untouched  = one;
    for (std::uint32_t lane = 0; lane < lanes; ++lane) { untouched = (untouched * keep) >> 32U; }
    return one - untouched;
}

// Every active request gets an equal share of the GPU: a prefill unit predicted to take
// `prefill_ns`, serving one of `prefilling` requests, owes the `decoding` ones
// prefill_ns * decoding / prefilling of decode, and each decode round pays its own predicted cost
// (decode_round_ns). One decode round per chunk (the plain alternation) gave a decoding request a
// few tokens a second beside a long prompt's 2048-token chunks.
[[nodiscard]] constexpr std::uint64_t fair_decode_credit(std::uint64_t prefill_ns,
                                                         std::uint32_t decoding,
                                                         std::uint32_t prefilling) noexcept {
    if (prefilling == 0 || decoding == 0) { return 0; }
    return interleave_detail::mul(prefill_ns / prefilling, decoding);
}

// Predicted device nanoseconds of a prefill unit of this work.
[[nodiscard]] inline std::uint64_t prefill_unit_ns(const ContextMachineCostModel& model,
                                                   const PrefillWork& work) noexcept {
    return model.prefill_ns(work);
}

// Predicted device nanoseconds of a decode round whose lanes hold these contexts (prompt plus
// generated tokens).
[[nodiscard]] constexpr std::uint64_t
decode_round_ns(const ContextPrefillCost& prefill, const DecodeRoundShape& shape,
                std::span<const std::uint64_t> lane_contexts) noexcept {
    using namespace interleave_detail;
    if (lane_contexts.empty()) { return 0; }
    const std::uint64_t per_lane = std::max(shape.verify_tokens_per_lane, 1U);
    // Tokens one step computes; the expert fraction saturates long before 2^32 tokens.
    const auto lanes = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(mul(lane_contexts.size(), per_lane), 0xffff'ffffULL));
    const std::uint64_t routed = std::min(shape.routed_expert_bytes, shape.weight_bytes);
    const std::uint64_t dense  = shape.weight_bytes - routed;
    const bool moe             = routed != 0 && shape.experts != 0 && shape.experts_per_token != 0;
    // Bytes one token reads (the active weights) and bytes the round's tokens read.
    const std::uint64_t active =
        moe ? add(dense,
                  q32_times(touched_expert_fraction_q32(shape.experts, shape.experts_per_token, 1),
                            routed))
            : shape.weight_bytes;
    const std::uint64_t touched =
        moe ? add(dense, q32_times(touched_expert_fraction_q32(shape.experts,
                                                               shape.experts_per_token, lanes),
                                   routed))
            : shape.weight_bytes;
    const std::uint64_t one_read_ns =
        q32_times(prefill.token_ns_q32, InterleavePricing::kWeightReadPromptTokens);
    // touched / active in Q16 (touched >= active).
    const std::uint64_t ratio_q16 =
        active == 0 ? (1ULL << 16U) : mul(touched, 1ULL << 16U) / active;
    std::uint64_t step  = std::max(prefill.chunk_ns, mul(one_read_ns, ratio_q16) >> 16U);
    step                = add(step, q32_times(prefill.token_ns_q32, lanes));
    std::uint64_t pairs = 0;
    for (const std::uint64_t context : lane_contexts) {
        pairs = add(pairs, mul(mul(context, InterleavePricing::kDecodeAttentionPairs), per_lane));
    }
    step = add(step, q32_times(prefill.attention_pair_ns_q32, pairs));
    return mul(step, std::max(shape.steps_per_round, 1U));
}

} // namespace ninferno::runtime
