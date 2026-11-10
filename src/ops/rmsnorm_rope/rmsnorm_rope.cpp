#include "ninferno/ops/rmsnorm_rope.h"

#include "core/cost_constants.h"
#include "core/generation.h"
#include "ops/common/sm_wave_rules.h"
#include "ops/generic/route_cost_constants.h"
#include "ops/rmsnorm_rope/launch.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::ops {
namespace {

constexpr std::int32_t kHeadDim       = 128;
constexpr std::int32_t kQueryHeads    = 32;
constexpr std::int32_t kKeyHeads      = 8;
constexpr std::int32_t kMaximumBatch  = 8;
constexpr std::int32_t kMaximumSingle = 2048;
constexpr std::int32_t kTextHeadDim   = 256;
// The text form has no width of its own to cap: one warp owns one head, so the only ceiling is the
// launch grid, and even the largest supported context stays four orders of magnitude below it.
constexpr std::int64_t kMaximumTextGrid       = 2147483647;
constexpr std::int32_t kMaximumTextHeadGroups = 12; // (32 + 2) heads, three per CTA

void require_tensor(const Tensor& tensor, DType dtype, const std::array<std::int32_t, 4>& shape,
                    const char* label) {
    bool shape_matches = true;
    for (std::size_t dim = 0; dim < shape.size(); ++dim) {
        shape_matches = shape_matches && tensor.ne[dim] == shape[dim];
    }
    if (tensor.dtype != dtype || !shape_matches || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 4)) {
        throw std::invalid_argument(std::string("rmsnorm_rope: invalid ") + label);
    }
}

void require_pair_nonoverlap(const Tensor& positions, const Tensor& q_norm_weight,
                             const Tensor& k_norm_weight, const Tensor& q, const Tensor& k) {
    if (overlaps(q, k) || overlaps(q, positions) || overlaps(q, q_norm_weight) ||
        overlaps(q, k_norm_weight) || overlaps(k, positions) || overlaps(k, q_norm_weight) ||
        overlaps(k, k_norm_weight)) {
        throw std::invalid_argument("rmsnorm_rope: pair mutable tensors overlap another operand");
    }
}

void require_single_nonoverlap(const Tensor& positions, const Tensor& norm_weight,
                               const Tensor& x) {
    if (overlaps(x, positions) || overlaps(x, norm_weight)) {
        throw std::invalid_argument(
            "rmsnorm_rope: single mutable tensor overlaps a read-only input");
    }
}

void require_text_nonoverlap(const Tensor& positions, const Tensor& q_norm_weight,
                             const Tensor& k_norm_weight, const Tensor& q_in, const Tensor& k_in,
                             const Tensor& q_out, const Tensor& k_out) {
    for (const Tensor* mutable_tensor : {&q_out, &k_out}) {
        for (const Tensor* other : {&q_in, &k_in, &positions, &q_norm_weight, &k_norm_weight}) {
            if (overlaps(*mutable_tensor, *other)) {
                throw std::invalid_argument("rmsnorm_rope: text output overlaps an input");
            }
        }
    }
    if (overlaps(q_out, k_out)) {
        throw std::invalid_argument("rmsnorm_rope: text outputs overlap each other");
    }
}

} // namespace

void rmsnorm_rope(const Tensor& positions, const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                  Tensor& q, Tensor& k, cudaStream_t stream) {
    const std::int32_t batch = q.ne[3];
    const std::int32_t width = q.ne[2];
    if (width < 2 || width > 16) throw std::invalid_argument("rmsnorm_rope: pair W must be 2..16");
    if (batch < 1 || batch > kMaximumBatch) {
        throw std::invalid_argument("rmsnorm_rope: pair B must be 1..8");
    }
    require_tensor(q, DType::BF16, {kHeadDim, kQueryHeads, width, batch}, "q");
    require_tensor(k, DType::BF16, {kHeadDim, kKeyHeads, width, batch}, "k");
    require_tensor(q_norm_weight, DType::BF16, {kHeadDim, 1, 1, 1}, "q norm weight");
    require_tensor(k_norm_weight, DType::BF16, {kHeadDim, 1, 1, 1}, "k norm weight");
    require_tensor(positions, DType::I32, {width, batch, 1, 1}, "positions");
    require_pair_nonoverlap(positions, q_norm_weight, k_norm_weight, q, k);
    detail::rmsnorm_rope_pair_launch(positions, q_norm_weight, k_norm_weight, q, k, width * batch,
                                     stream);
}

void rmsnorm_rope(const Tensor& positions, const Tensor& norm_weight, Tensor& x,
                  cudaStream_t stream) {
    const std::int32_t tokens = x.ne[2];
    if (tokens < 1 || tokens > kMaximumSingle) {
        throw std::invalid_argument("rmsnorm_rope: single T must be 1..2048");
    }
    require_tensor(x, DType::BF16, {kHeadDim, kKeyHeads, tokens, 1}, "x");
    require_tensor(norm_weight, DType::BF16, {kHeadDim, 1, 1, 1}, "norm weight");
    require_tensor(positions, DType::I32, {tokens, 1, 1, 1}, "positions");
    require_single_nonoverlap(positions, norm_weight, x);
    detail::rmsnorm_rope_single_launch(positions, norm_weight, x, tokens, stream);
}

namespace {

// The (query, key) geometries of rmsnorm_rope_text_heads, as the launch dispatches them.
constexpr std::array<std::array<std::int32_t, 2>, 5> kTextGeometries{
    {{8, 2}, {16, 2}, {16, 4}, {24, 4}, {32, 2}}};

// The width measured on the RTX 5090, where the law has no calibration yet (see the header).
constexpr std::int32_t kFusedTextDefaultMaximumTokens = 256;

} // namespace

namespace detail {

std::int32_t text_fused_max_tokens_with_excess(const DeviceClass& device, std::int32_t query_heads,
                                               std::int32_t key_heads, double excess_cycles) {
    if (!rmsnorm_rope_text_heads(query_heads, key_heads)) return 0;
    const std::string_view generation = generation_known(device.major, device.minor)
                                            ? generation_row(device.major, device.minor).name
                                            : std::string_view{};
    const double launch               = launch_node_seconds(generation);
    if (!(launch > 0.0)) return 0;
    const int warps_per_sm =
        ninferno::ops::detail::sm_rules::resident_ctas_per_sm(
            device.max_blocks_per_sm, device.max_threads_per_sm, kTextHeadsPerBlock * 32) *
        kTextHeadsPerBlock;
    return ninferno::ops::detail::sm_rules::launch_amortized_tokens(
        2, launch, device.sms, warps_per_sm, machine_facts_of(device).reference_clock_khz * 1.0e3,
        query_heads + key_heads, excess_cycles);
}

} // namespace detail

bool rmsnorm_rope_text_heads(std::int32_t query_heads, std::int32_t key_heads) {
    for (const auto& g : kTextGeometries) {
        if (g[0] == query_heads && g[1] == key_heads) return true;
    }
    return false;
}

std::int32_t rmsnorm_rope_text_fused_max_tokens(const DeviceClass& device, std::int32_t query_heads,
                                                std::int32_t key_heads) {
    // Not calibrated: nothing to compute (this runs for every layer's call).
    static constexpr const auto& fit =
        detail::generic::route_cost::pending_fit("fused_text_excess_cycles"); // compile-time lookup
    const auto& excess = fit.value;
    if (!excess) return kFusedTextDefaultMaximumTokens;
    const std::int32_t bound =
        detail::text_fused_max_tokens_with_excess(device, query_heads, key_heads, *excess);
    return bound > 0 ? bound : kFusedTextDefaultMaximumTokens;
}

bool rmsnorm_rope_text_takes_width(const DeviceClass& device, std::int32_t query_heads,
                                   std::int32_t key_heads, std::int32_t tokens) {
    return rmsnorm_rope_text_heads(query_heads, key_heads) &&
           tokens <= rmsnorm_rope_text_fused_max_tokens(device, query_heads, key_heads);
}

void rmsnorm_rope(const Tensor& positions, const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                  const Tensor& q_in, const Tensor& k_in, Tensor& q_out, Tensor& k_out,
                  cudaStream_t stream) {
    const std::int32_t tokens      = q_in.ne[2];
    const std::int32_t query_heads = q_in.ne[1];
    const std::int32_t key_heads   = k_in.ne[1];
    if (tokens < 1 ||
        static_cast<std::int64_t>(tokens) * kMaximumTextHeadGroups > kMaximumTextGrid) {
        throw std::invalid_argument(
            "rmsnorm_rope: text T must be positive and fit the launch grid");
    }
    if (!rmsnorm_rope_text_heads(query_heads, key_heads)) {
        throw std::invalid_argument("rmsnorm_rope: text (Q,K) not instantiated");
    }
    require_tensor(q_in, DType::BF16, {kTextHeadDim, query_heads, tokens, 1}, "text q in");
    require_tensor(k_in, DType::BF16, {kTextHeadDim, key_heads, tokens, 1}, "text k in");
    require_tensor(q_out, DType::BF16, {kTextHeadDim, query_heads, tokens, 1}, "text q out");
    require_tensor(k_out, DType::BF16, {kTextHeadDim, key_heads, tokens, 1}, "text k out");
    require_tensor(q_norm_weight, DType::BF16, {kTextHeadDim, 1, 1, 1}, "text q norm weight");
    require_tensor(k_norm_weight, DType::BF16, {kTextHeadDim, 1, 1, 1}, "text k norm weight");
    require_tensor(positions, DType::I32, {tokens, 1, 1, 1}, "text positions");
    require_text_nonoverlap(positions, q_norm_weight, k_norm_weight, q_in, k_in, q_out, k_out);
    detail::rmsnorm_rope_text_launch(positions, q_norm_weight, k_norm_weight, q_in, k_in, q_out,
                                     k_out, tokens, stream);
}

} // namespace ninferno::ops
