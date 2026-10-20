#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninferno::ops {

// Counter-based RNG subkey. Distinct purposes keep draws at the same logical position separate.
enum SamplePurpose : std::int32_t {
    kSamplePurposePrefill               = 0,
    kSamplePurposeDecode                = 1,
    kSamplePurposeSpeculativeAccept     = 2,
    kSamplePurposeSpeculativeCorrection = 3,
    kSamplePurposeSpeculativeBonus      = 4,
    kSamplePurposeDFlash2Proposal       = 5,
};

// Device-resident sampling parameters. token_counts is an optional device I32
// [token_domain] committed generated-token occurrence-count array used by both penalties.
//
// token_mask is an optional allowed-token bitmask (constrained decoding). When it is non-null,
// token v may be selected at sampled column c only if bit (v & 31) of
// token_mask[c * token_mask_words + (v >> 5)] is set; column c is 0 for sample() and the verify
// column for speculative acceptance. A disallowed token is treated as a -inf adjusted logit before
// penalties, top-k, min-p and top-p, so the support is renormalized over allowed tokens only.
// Every column a kernel reads must admit at least one token.
//
// The mask may be written by the host after the launch: a block reads it only after one of its
// threads observes (int32)(*token_mask_release - token_mask_epoch) >= 0 with a system-scope
// acquire, and mask words are read with uncached loads, so both pointers may address device
// memory written by a concurrent copy or device-mapped pinned host memory. Mapped memory is
// correct but slow: every sampler block polls and reads over PCIe (hundreds of microseconds per
// call on the multi-block route), so the Engine keeps both in device memory. Rows must start
// 16-byte aligned (token_mask_words is a multiple of 4). If the release is not observed within
// the Op's wait bound (renewed while *token_mask_progress advances, up to an overall bound), the
// block stores 1 to *token_mask_fault (when non-null) and proceeds; the result of such a call is
// unconstrained and the caller must discard it. A null token_mask keeps the unconstrained route
// unchanged.
struct SamplingConfig {
    float temperature          = 0.0f; // <= 0 => greedy argmax (bit-identical to argmax())
    std::int32_t top_k         = 20;   // runtime contract is [1, kMaximumTopK]; Op caps otherwise
    float top_p                = 1.0f; // >= 1 => disabled
    float min_p                = 0.0f; // <= 0 => disabled
    float presence_penalty     = 0.0f;
    float frequency_penalty    = 0.0f;
    unsigned long long seed    = 0;
    std::int32_t* token_counts = nullptr; // device [token_domain] i32, or null
    const std::uint32_t* token_mask         = nullptr; // [columns, token_mask_words], or null
    const std::uint32_t* token_mask_release = nullptr; // release counter, required with a mask
    std::uint32_t* token_mask_fault         = nullptr; // optional wait-bound fault flag
    // Optional counter the host advances while it is still computing masks (device-mapped host
    // memory, read only once a wait is already long). Each advance renews the wait bound.
    const std::uint32_t* token_mask_progress = nullptr;
    std::uint32_t token_mask_epoch           = 0;
    std::int32_t token_mask_words            = 0; // >= ceil(token_domain / 32) with a mask
};

// How long one block waits for the host to release its token mask without the host showing
// progress, and the bound on the whole wait however the host progresses.
inline constexpr unsigned long long kTokenMaskReleaseWaitNs    = 1'500'000'000ULL;
inline constexpr unsigned long long kTokenMaskReleaseMaxWaitNs = 30'000'000'000ULL;

// Caller-owned transient capacity for every parallel sampling-lane count in the inclusive
// interval. For sample(), one lane is one batch row (a call of more than sixteen runs in slices of
// sixteen, so the interval's capacity is at most one slice's); speculative acceptance uses the same
// workspace substrate for its verification columns. token_domain is the fixed route profile.
// Invalid profiles or intervals throw; a legal single-block route returns zero.
[[nodiscard]] std::size_t sampling_workspace_capacity_bytes(std::int32_t token_domain,
                                                            std::int32_t min_lanes,
                                                            std::int32_t max_lanes);

/**
 * Produces one token id per independent request row. `logits` is contiguous BF16
 * [physical_rows,B], `out` and `logical_positions` are contiguous I32 [B], and only vocabulary
 * rows v in [0,token_domain) participate. `configs` is a device-resident contiguous
 * SamplingConfig[B] array. Greedy and stochastic rows may coexist in one invocation.
 *
 * For row b with configs[b].temperature<=0:
 *
 * With either greedy or positive-temperature sampling, let
 * c_v=configs[b].token_counts[v] (or zero when the pointer is null):
 *
 *   adjusted_v = float(logits[v,b])
 *                - configs[b].presence_penalty * (c_v > 0)
 *                - configs[b].frequency_penalty * c_v.
 *
 * A greedy row selects min argmax_v adjusted_v and skips filters and RNG. Candidates for a
 * positive-temperature row are sorted by adjusted_v descending with lower token id breaking
 * ties. Per-row top_k in [1, kMaximumTopK - 1] keeps that many candidates; top_k <= 0 or
 * top_k >= kMaximumTopK keeps min(kMaximumTopK, token_domain) (20 on the single-block fallback,
 * which only a vocabulary within one sampler tile takes).
 * Candidate weights are exp(adjusted_v/temperature-max). min_p removes the suffix below
 * min_p*max_weight; top_p keeps the shortest remaining prefix whose cumulative weight reaches
 * top_p times the pre-truncation candidate weight. At least the best candidate remains, the
 * support is renormalized, and one id is drawn for that row.
 *
 * Row b uses counter-based RNG key
 * (configs[b].seed,logical_positions[b],purpose), without mutable RNG state or dependence on the
 * compact row index. In every mode the selected token atomically increments
 * configs[b].token_counts when it is non-null. Non-null token-count arrays belonging to distinct
 * active requests must not alias. `out` must not overlap logits, configs, logical_positions, or any
 * token-count array. The Op writes all of out, uses caller-owned transient storage reported by
 * sampling_workspace_capacity_bytes(), and has no other persistent-state side effect.
 */
void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, std::int32_t purpose,
            WorkspaceArena& workspace, cudaStream_t stream);

// Adds every id in the contiguous non-empty I32 token_ids vector to the contiguous I32
// [token_domain] committed count array. IDs must be in [0,token_domain).
void increment_token_counts(const Tensor& token_ids, Tensor& token_counts, cudaStream_t stream);

} // namespace ninferno::ops
