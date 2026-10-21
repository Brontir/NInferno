#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninferno/ops/gemma4_moe.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail::gemma4 {

inline constexpr std::int32_t kHidden      = 2816;
inline constexpr std::int32_t kExperts     = 128;
inline constexpr std::int32_t kTopK        = 8;
inline constexpr std::int32_t kInter       = 704;
inline constexpr std::int32_t kGateUpRows  = 2 * kInter;
inline constexpr std::int32_t kGateUpTotal = kExperts * kGateUpRows;
inline constexpr std::int32_t kDownTotal   = kExperts * kHidden;

// Route frontiers. Decode is T=1 only; small-T shares each selected expert's weight stream across
// the tokens routed to it; prefill is the grouped tensor-core route. kSmallTCapacity is how far
// the small-T route can be taken at all; kSmallTMax is where the public dispatch leaves it.
//
// kSmallTMax covers the verify pass of speculative decoding (T=16), which must compute what T
// single-token steps would: small-T keeps the activation represented, and each of its columns is
// the decode route's result bit for bit. What it costs against A4 prefill depends on how many
// tokens share an expert (cold L2, graph replay, local 5090, null arm within 0.9 %): with random
// routing (82 experts at T=16) small-T leads from T=5 on - 245.0 us against 261.3 at T=16, 320.8
// against 328.7 at T=28 - and trails from T=32 (343.3 / 337.3); with every token repeated once
// (bench --duplicate 2, 49 experts) A4 leads at T=16 by 3 %, repeated three times by 1.5x. So the
// frontier is set by exactness, not by speed, and stays at the verify width.
inline constexpr std::int32_t kSmallTMin      = 2;
inline constexpr std::int32_t kSmallTMax      = 16;
inline constexpr std::int32_t kSmallTCapacity = 128;
inline constexpr std::int32_t kPrefillMin     = kSmallTMax + 1;
static_assert(kSmallTMax <= kSmallTCapacity);
// The prefill route works on slices of at most this many tokens; the workspace
// is sized by it.
inline constexpr std::int32_t kSliceMax = 4096;
// Router selection tile of the prefill route: one warp per token, eight per
// block.
inline constexpr std::int32_t kRouteTile = 8;
// Column tile of a grouped job; every grouped schedule stages exactly this many
// tokens.
inline constexpr std::int32_t kJobColumns = 64;

enum class Route : std::uint8_t { Decode, SmallT, PrefillA4, PrefillA16 };

// Compile-time half of the Ada carve-out (the runtime half is DeviceFeatures::mma_mxf4nvf4 in
// core/device_class.h): whether this build's device images can issue the block-scaled NVFP4 MMA.
// Keyed on the build flag, not __CUDA_ARCH__, so the host and device passes of one TU agree —
// a launch arm must not disappear only in the host pass. The A16 prefill tier below is the Ada
// answer: same router, same grouping, BF16 MMA over weights decoded from NVFP4 codes.
#ifdef NINFERNO_WITHOUT_NVFP4
inline constexpr bool kHasNvfp4Mma = false;
#else
inline constexpr bool kHasNvfp4Mma = true;
#endif

// How the four-bit prefill route encodes the gate/up input. PerExpert honours
// every expert's checkpoint divisor and therefore encodes the hidden state once
// per assignment; PerToken encodes it once per token against a divisor taken
// from that token's own maximum.
enum class A4Input : std::uint8_t { PerExpert, PerToken };

// Decode-route variants, each timed against the others (see gemma4_moe_simt.cu): top-8 by rank
// counting over the whole block, gate/up weight loads issued before the down prefetch, and a down
// kernel that stages what does not depend on the expert ids.
struct DecodeOptions {
    bool rank_select = false;
    bool loads_first = false;
    bool staged_down = false;
};

// What the public entry runs: rank selection alone. At T=1 (cold L2, graph replay, null arm
// +0.00 %) it took 31.97 us against 34.02; loads-first changed nothing, staged-down cost
// 2.04 us, and all three together came back to 34.05.
inline constexpr DecodeOptions kDefaultDecode{true, false, false};

// narrow picks the A4 row tile: the wide one (128 rows a CTA) or the narrow one (64), see
// gemma4_moe_prefill.cu.
struct PrefillOptions {
    Route route   = Route::PrefillA4;
    A4Input input = A4Input::PerExpert;
    bool narrow   = false;
};

// The one place the public entry picks a route. Every predicate is a closed
// interval of T, and they are tested in the order they are written: [1,1]
// decode, [2,kSmallTMax] small-T, the rest prefill.
//
// The prefill tier follows the caller's activation choice: A4 by default, A16 on request. Both
// tiers share the workspace layout, the router and the selection, and neither mixes columns: a
// grouped GEMM accumulates each packed column over the whole K in a fixed order, so a column's
// bytes do not depend on T or on where it lands in its expert's group.
[[nodiscard]] constexpr Route
default_route(std::int32_t tokens,
              Gemma4MoeActivations activations = Gemma4MoeActivations::Recorded) noexcept {
    if (tokens == 1) { return Route::Decode; }
    if (tokens <= kSmallTMax) { return Route::SmallT; }
    // Ada (and any device without kind::mxf4nvf4) takes the A16 prefill tier: it decodes the
    // NVFP4 expert weights to BF16 and runs the same sm_80-class MMA as the rest of the engine,
    // exact for NVFP4 weights. The A4 tier would issue a Blackwell-only instruction.
    if (!kHasNvfp4Mma) { return Route::PrefillA16; }
    return activations == Gemma4MoeActivations::A16Only ? Route::PrefillA16 : Route::PrefillA4;
}

// The router every route uses. Decode and small-T: one block per (expert row, four tokens) (SIMT)
// and a single K slice, which gives a token the same logits whichever of the two routes runs it.
// Prefill uses the tensor-core router, split over K into a fixed number of slices. The split must
// not follow T: the slices' partial sums are added one by one, so another split count moves every
// logit by an ulp, every route weight with it and now and then a near-tie selection, and a token's
// output would depend on how many tokens share its chunk (a prompt cut elsewhere, or a different
// prefill chunk, then decodes differently).
enum class RouterKind : std::uint8_t { Simt, Mma };

[[nodiscard]] constexpr RouterKind router_kind(Route route) noexcept {
    return route == Route::Decode || route == Route::SmallT ? RouterKind::Simt : RouterKind::Mma;
}

inline constexpr std::int32_t kRouterTileTokens  = 16;
inline constexpr std::int32_t kRouterTileExperts = 64;
inline constexpr std::int32_t kRouterKTiles      = kHidden / 64; // 44
// The most K slices a router splits into; a consumer reads that many partials per logit at most.
inline constexpr std::int32_t kMaxRouterSplits = 11;
// The prefill router's split at every T. Two is what a 2048-token chunk used when the split still
// followed T, so such chunks keep their bytes; against four (and against the old T-dependent
// split) prefill measured within the noise at 512, 2048 and 8192 tokens.
inline constexpr std::int32_t kPrefillRouterSplits = 2;
static_assert(kRouterKTiles % kPrefillRouterSplits == 0 && kPrefillRouterSplits <= kMaxRouterSplits);

[[nodiscard]] constexpr std::int32_t router_splits(RouterKind kind, std::int32_t) noexcept {
    return kind == RouterKind::Simt ? 1 : kPrefillRouterSplits;
}

// Rows of router partials a slice of up to `capacity` tokens can need: the
// product of splits and tokens is not monotone in the token count.
[[nodiscard]] inline std::int32_t router_partial_rows(RouterKind kind,
                                                      std::int32_t capacity) noexcept {
    std::int32_t rows = 0;
    for (std::int32_t tokens = 1; tokens <= capacity; ++tokens) {
        rows = std::max(rows, router_splits(kind, tokens) * tokens);
    }
    return rows;
}

// Every route allocates the selection first, so a test can read which experts a
// call picked from the start of the workspace without knowing the rest of the
// layout.
struct Selection {
    Tensor ids;     // I32 [8*capacity], token-major, rank order
    Tensor weights; // FP32 [8*capacity], w_k including expert_scale
};

template <class Arena>
Selection allocate_selection(Arena& arena, std::int32_t capacity) {
    Selection out;
    out.ids     = arena.alloc(DType::I32, {kTopK * capacity}, 256);
    out.weights = arena.alloc(DType::FP32, {kTopK * capacity}, 256);
    return out;
}

struct DecodeWorkspace {
    Selection selection;
    Tensor router_dot;    // FP32 [128]
    Tensor router_square; // FP32 [1]
    Tensor activation;    // FP32 [8*704]
};

template <class Arena>
DecodeWorkspace allocate_decode_workspace(Arena& arena) {
    DecodeWorkspace out;
    out.selection     = allocate_selection(arena, 1);
    out.router_dot    = arena.alloc(DType::FP32, {kExperts}, 256);
    out.router_square = arena.alloc(DType::FP32, {1}, 256);
    out.activation    = arena.alloc(DType::FP32, {kTopK * kInter}, 256);
    return out;
}

struct SmallTWorkspace {
    Selection selection;
    Tensor router_dot;      // FP32 [rows*128]
    Tensor router_square;   // FP32 [rows]
    Tensor meta;            // I32: offsets[129], unique[128], unique_count[1]
    Tensor slot_assignment; // I32 [8T]: packed slot -> assignment
    Tensor packed_index;    // I32 [8T]: assignment -> packed slot
    Tensor activation;      // FP32 [8T*704], packed order
    Tensor expert_out;      // FP32 [8T*2816], packed order
};

inline constexpr std::int32_t kMetaOffsets = 0;
inline constexpr std::int32_t kMetaUnique  = kExperts + 1;
inline constexpr std::int32_t kMetaCount   = kMetaUnique + kExperts;
inline constexpr std::int32_t kMetaWords   = kMetaCount + 1;

template <class Arena>
SmallTWorkspace allocate_small_t_workspace(Arena& arena, std::int32_t tokens) {
    SmallTWorkspace out;
    const std::int32_t assignments = kTopK * tokens;
    const std::int32_t rows        = router_partial_rows(router_kind(Route::SmallT), tokens);
    out.selection                  = allocate_selection(arena, tokens);
    out.router_dot                 = arena.alloc(DType::FP32, {kExperts * rows}, 256);
    out.router_square              = arena.alloc(DType::FP32, {rows}, 256);
    out.meta                       = arena.alloc(DType::I32, {kMetaWords}, 256);
    out.slot_assignment            = arena.alloc(DType::I32, {assignments}, 256);
    out.packed_index               = arena.alloc(DType::I32, {assignments}, 256);
    out.activation                 = arena.alloc(DType::FP32, {kInter, assignments}, 256);
    out.expert_out                 = arena.alloc(DType::FP32, {kHidden, assignments}, 256);
    return out;
}

struct PrefillWorkspace {
    Selection selection;
    Tensor router_dot;    // FP32 [rows*128]
    Tensor router_square; // FP32 [rows]
    Tensor local_rank;    // I32 [8C]
    Tensor tile_counts;   // I32 [tiles*128]
    Tensor tile_bases;    // I32 [tiles*128]
    Tensor offsets;       // I32 [129]
    Tensor job_experts;   // I32 [max jobs]
    Tensor job_columns;   // I32 [max jobs]
    Tensor job_count;     // I32 [2]
    Tensor packed_index;  // I32 [8C]
    Tensor packed_token;  // I32 [8C]
    Tensor packed_expert; // I32 [8C]
    Tensor token_divisor; // FP32 [C] (A4 per-token input)
    // Lifetime union: the encoded gate/up input dies with gate/up, the expert
    // outputs are born in down, so the second overwrites the first.
    Tensor expert_out;   // BF16 [8C*2816]
    Tensor input_codes;  // U8 inside expert_out: [8C or C][1408]
    Tensor input_scales; // U8 inside expert_out: [8C or C][176]
    // The intermediate: NVFP4 for A4, BF16 for A16.
    Tensor middle;        // A16: BF16 [8C*704]; A4: U8 codes [8C*352]
    Tensor middle_scales; // A4: U8 [8C*44]
};

[[nodiscard]] inline std::int32_t max_route_jobs(std::int32_t assignments) noexcept {
    return assignments / kJobColumns + std::min(kExperts, assignments);
}

template <class Arena>
PrefillWorkspace allocate_prefill_workspace(Arena& arena, std::int32_t capacity) {
    PrefillWorkspace out;
    const std::int32_t assignments = kTopK * capacity;
    const std::int32_t tiles       = (capacity + kRouteTile - 1) / kRouteTile;
    const std::int32_t jobs        = max_route_jobs(assignments);
    const std::int32_t rows        = router_partial_rows(router_kind(Route::PrefillA4), capacity);
    out.selection                  = allocate_selection(arena, capacity);
    out.router_dot                 = arena.alloc(DType::FP32, {kExperts * rows}, 256);
    out.router_square              = arena.alloc(DType::FP32, {rows}, 256);
    out.local_rank                 = arena.alloc(DType::I32, {assignments}, 256);
    out.tile_counts                = arena.alloc(DType::I32, {kExperts * tiles}, 256);
    out.tile_bases                 = arena.alloc(DType::I32, {kExperts * tiles}, 256);
    out.offsets                    = arena.alloc(DType::I32, {kExperts + 1}, 256);
    out.job_experts                = arena.alloc(DType::I32, {jobs}, 256);
    out.job_columns                = arena.alloc(DType::I32, {jobs}, 256);
    out.job_count                  = arena.alloc(DType::I32, {2}, 256);
    out.packed_index               = arena.alloc(DType::I32, {assignments}, 256);
    out.packed_token               = arena.alloc(DType::I32, {assignments}, 256);
    out.packed_expert              = arena.alloc(DType::I32, {assignments}, 256);
    out.token_divisor              = arena.alloc(DType::FP32, {capacity}, 256);
    out.expert_out                 = arena.alloc(DType::BF16, {kHidden, assignments}, 256);
    auto* const base               = static_cast<std::uint8_t*>(out.expert_out.data);
    const std::int64_t code_bytes  = static_cast<std::int64_t>(assignments) * (kHidden / 2);
    out.input_codes                = Tensor(base, DType::U8, {kHidden / 2, assignments});
    out.input_scales = Tensor(base != nullptr ? base + code_bytes : nullptr, DType::U8,
                              {kHidden / 16, assignments});
    static_assert(kHidden / 2 + kHidden / 16 <= kHidden * 2,
                  "the encoded gate/up input must fit the expert outputs it aliases");
    out.middle        = arena.alloc(DType::BF16, {kInter, assignments}, 256);
    out.middle_scales = arena.alloc(DType::U8, {kInter / 16, assignments}, 256);
    return out;
}

// The tensor-core router over `tokens` columns of x, split over K `splits`
// ways, writing the partials RouterPartials adds up.
void launch_router_mma(const void* x, std::int32_t tokens, const Gemma4MoeWeights& weights,
                       float* dot, float* square, std::int32_t splits, cudaStream_t stream);

// Launches. Each takes views already carved out of the caller's arena.
void launch_decode(const Tensor& router_input, const Tensor& expert_input,
                   const Gemma4MoeWeights& weights, float eps, Tensor& out,
                   const DecodeWorkspace& workspace, DecodeOptions options, cudaStream_t stream);

void launch_small_t(const Tensor& router_input, const Tensor& expert_input,
                    const Gemma4MoeWeights& weights, float eps, Tensor& out,
                    const SmallTWorkspace& workspace, cudaStream_t stream);

void launch_prefill(const Tensor& router_input, const Tensor& expert_input,
                    const Gemma4MoeWeights& weights, float eps, Tensor& out,
                    const PrefillWorkspace& workspace, std::int32_t capacity,
                    PrefillOptions options, cudaStream_t stream);

// Route-level entry for tests and benchmarks: the same validation as
// gemma4_moe, but the route is named instead of chosen from T. A decode route
// needs T=1; a small-T route needs T<=kSmallTMax.
void gemma4_moe_with_route(const Tensor& router_input, const Tensor& expert_input,
                           const Gemma4MoeWeights& weights, float eps, Tensor& out,
                           WorkspaceArena& workspace, cudaStream_t stream, Route route,
                           A4Input input = A4Input::PerExpert, bool narrow = false,
                           DecodeOptions decode = kDefaultDecode);

[[nodiscard]] std::size_t route_workspace_bytes(Route route, std::int32_t tokens);

} // namespace ninferno::ops::detail::gemma4
