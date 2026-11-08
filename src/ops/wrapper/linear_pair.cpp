#include "core/weight.h"
#include "ninferno/ops/linear_pair.h"

#include "ninferno/ops/linear.h"
#include "ops/common/tensor_checks.h"
#include "ops/linear/linear_registry.h"
#include "ops/linear_pair/q8/q8_pair_plan.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::ops {
namespace {

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("linear_pair: invalid ") + label);
    }
}

std::uint64_t required_payload_bytes(std::int32_t rows, std::int32_t k) {
    const std::uint64_t r = static_cast<std::uint64_t>(rows);
    const std::uint64_t h = static_cast<std::uint64_t>(k);
    if (h > std::numeric_limits<std::uint64_t>::max() / r) {
        throw std::overflow_error("linear_pair: weight payload size overflows");
    }
    const std::uint64_t code_bytes = r * h;
    const std::uint64_t groups     = h / 32;
    if (groups > std::numeric_limits<std::uint64_t>::max() / (r * 2u)) {
        throw std::overflow_error("linear_pair: scale payload size overflows");
    }
    return code_bytes + r * groups * 2u;
}

void require_weight(const Weight& weight, std::int32_t k, const char* label) {
    const std::uint64_t payload_bytes = required_payload_bytes(1024, k);
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group != 32 || weight.group_size != 32 ||
        weight.ndim != 2 || weight.n != 1024 || weight.k != k || weight.shape[0] != 1024 ||
        weight.shape[1] != k || weight.padded_shape[0] != 1024 || weight.padded_shape[1] != k ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < payload_bytes || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 4)) {
        throw std::invalid_argument(std::string("linear_pair: invalid ") + label);
    }
}

void require_nonoverlap(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                        const Tensor& first_out, const Tensor& second_out) {
    const std::uint64_t cols        = static_cast<std::uint64_t>(x.ne[1]);
    const std::uint64_t k           = static_cast<std::uint64_t>(x.ne[0]);
    const std::uint64_t code_bytes  = 1024u * k;
    const std::uint64_t scale_bytes = 1024u * (k / 32u) * 2u;
    const std::array<OperandRange, 7> ranges{{
        {x.data, k * cols * 2u, "x"},
        {first_out.data, 1024u * cols * 2u, "first output"},
        {second_out.data, 1024u * cols * 2u, "second output"},
        {first_weight.qdata, code_bytes, "first codes"},
        {first_weight.scales, scale_bytes, "first scales"},
        {second_weight.qdata, code_bytes, "second codes"},
        {second_weight.scales, scale_bytes, "second scales"},
    }};
    require_disjoint(ranges, "linear_pair");
}

// Whether the weights are the paired kernel's row-split Q8 G32 layout with FP16 scales.
bool fused_layout(const Weight& first_weight, const Weight& second_weight) {
    try {
        require_weight(first_weight, first_weight.k, "first weight");
        require_weight(second_weight, first_weight.k, "second weight");
    } catch (const std::invalid_argument&) {
        return false;
    }
    return true;
}

// The paired Q8 kernel's domain: the Q8 G32 row-split problems its plan registers (the two Qwen
// drafter geometries), the K = 2048 one only on the drafter's adjacent K/V row views of one parent
// (q8_pair_row_views, which its dispatch requires). Every other pair runs as two linear() calls
// with the same numerics contract.
bool fused_domain(const Weight& first_weight, const Weight& second_weight) {
    return detail::q8_pair_registered(first_weight.n, first_weight.k) &&
           fused_layout(first_weight, second_weight) &&
           (first_weight.k != 2048 || detail::q8_pair_row_views(first_weight, second_weight));
}

// The chain of each alternative for a call of `tokens` columns, by the bytes its CTAs request (the
// law of generic::route_traffic: every column tile requests the weights again, every row tile the
// activation, the output is written once), both weights counted. Launch counts are the chain's
// kernels. Both alternatives read the same weight bytes from DRAM and write the same output, so
// what tells them apart is how often the tiles re-request them and how many launches carry that.
// No device enters: the comparison is of two tilings of one problem.
struct ChainPrice {
    double requested_bytes;
    int launches;
};

std::optional<ChainPrice> fused_chain(std::int32_t n_rows, std::int32_t k_in, std::int32_t tokens) {
    const detail::Q8PairProblem problem{n_rows, k_in, k_in, tokens};
    const auto geometry = detail::q8_pair_geometry(problem);
    if (!geometry) { return std::nullopt; }
    const double n = n_rows, k = k_in, t = tokens;
    const double weight_bytes = 2.0 * (n * k + n * (k / 32.0) * 2.0);
    const double passes       = std::ceil(t / geometry->tile_cols);
    return ChainPrice{weight_bytes * passes +
                          k * t * 2.0 * geometry->x_row_blocks * geometry->launches +
                          2.0 * n * t * 2.0,
                      geometry->launches};
}

// The two calls' chain: the weights linear() sees in the pair's domain are Q8_G32 row-split
// [n,k] ones, so the shape alone describes them (the selection and the warm-up share this).
std::optional<ChainPrice> separate_chain(std::int32_t n, std::int32_t k, std::int32_t tokens) {
    Weight w;
    w.qtype           = QType::Q8_G32_FP16;
    w.n               = n;
    w.k               = k;
    w.ndim            = 2;
    w.shape[0]        = n;
    w.shape[1]        = k;
    w.padded_shape[0] = n;
    w.padded_shape[1] = k;
    w.group           = 32;
    w.group_size      = 32;
    w.layout          = QuantLayout::RowSplit;
    w.scale_dtype     = DType::FP16;
    const auto one    = detail::linear_q8_requested_bytes(w, tokens);
    if (!one) { return std::nullopt; }
    return ChainPrice{2.0 * *one, 2};
}

// Whether a call of the pair's domain runs the fused kernel: the cheaper chain where both are
// priced and the fusing saves nothing. A fused schedule that is one launch saves a launch (and its
// drain and ramp) against the two calls, which a bytes law does not price, so it keeps the call as
// before. One that is two launches (K = 5120 up to 85 columns: the single-weight SIMT kernel twice,
// 8 rows by 4 columns a CTA) saves none, and the chains differ only in how often the tiles
// re-request the weights, so the one requesting fewer bytes runs; a tie keeps the fused kernel.
bool chain_prefers_fused(std::int32_t n, std::int32_t k, std::int32_t tokens) {
    const auto fused    = fused_chain(n, k, tokens);
    const auto separate = separate_chain(n, k, tokens);
    if (!fused || !separate || fused->launches < separate->launches) { return true; }
    return fused->requested_bytes <= separate->requested_bytes;
}

bool fused_pair(const Weight& first_weight, const Weight& second_weight, std::int32_t tokens) {
    return fused_domain(first_weight, second_weight) &&
           chain_prefers_fused(first_weight.n, first_weight.k, tokens);
}

void require_same_logical_shape(const Weight& first_weight, const Weight& second_weight) {
    if (first_weight.n <= 0 || first_weight.k <= 0 || first_weight.n != second_weight.n ||
        first_weight.k != second_weight.k) {
        throw std::invalid_argument("linear_pair: the two weights must share one [N,K] shape");
    }
}

} // namespace

namespace detail {

bool linear_pair_runs_fused(std::int32_t n, std::int32_t k, std::int32_t tokens) {
    return q8_pair_registered(n, k) && chain_prefers_fused(n, k, tokens);
}

LinearPairRequested linear_pair_requested_bytes(const Weight& first, const Weight& second,
                                                std::int32_t tokens) {
    LinearPairRequested out;
    if (tokens <= 0 || !fused_domain(first, second)) { return out; }
    if (const auto c = fused_chain(first.n, first.k, tokens)) { out.fused = c->requested_bytes; }
    if (const auto c = separate_chain(first.n, first.k, tokens)) {
        out.separate = c->requested_bytes;
    }
    return out;
}

std::vector<LinearPairCandidate> linear_pair_candidates(const Weight& first, const Weight& second,
                                                        std::int32_t tokens) {
    using Status = LinearCandidate::Status;
    if (tokens <= 0) { throw std::invalid_argument("linear_pair candidates: T must be positive"); }
    // As linear_pair runs the call: the fused kernel where it applies, else the two calls, which
    // need one shape (linear_pair refuses another).
    const bool domain = fused_domain(first, second);
    const bool fused  = domain && fused_pair(first, second, tokens);
    const bool same  = first.n > 0 && first.k > 0 && first.n == second.n && first.k == second.k;
    LinearPairCandidate kernel{"fused:q8_pair", fused ? Status::Selected : Status::Unavailable, "", {}};
    if (!fused && domain) {
        // In the kernel's domain, and its chain is dearer than the shape's own launches.
        kernel.status = Status::Dominated;
        kernel.why    = "the fused chain requests more bytes than the two linear() calls";
    } else if (!fused) {
        kernel.why = !same ? "the two weights must share one [N,K] shape"
                     : !q8_pair_registered(first.n, first.k) ? "no fused kernel for this [N,K]"
                     : !fused_layout(first, second)
                         ? "the weights are not the fused kernel's RowSplit Q8_G32 layout with FP16 "
                           "scales"
                         : "the weights are not adjacent K/V row views of one 6144-row parent";
    }
    LinearPairCandidate separate{"separate", Status::Unavailable, "", {}};
    if (!same) {
        separate.why = "the two weights must share one [N,K] shape";
    } else {
        // Each call runs only where linear() selects an implementation for it (a ggml_frag_v1
        // weight, or the invariant profile on a format only the shape's kernels read, has none).
        bool both_run = true;
        for (const Weight* w : {&first, &second}) {
            bool runs = false;
            // The two implementations linear() chooses between (the shape's kernel and the
            // generic route), not the generic families listed after them.
            auto parts = linear_candidates(*w, LinearPolicy::A16Only, tokens);
            parts.resize(std::min<std::size_t>(parts.size(), 2));
            for (LinearCandidate& c : parts) {
                runs = runs || c.status == Status::Selected;
                separate.parts.push_back(std::move(c));
            }
            both_run = both_run && runs;
        }
        if (!both_run) {
            separate.why = "linear() runs no implementation for a weight (its parts say why)";
        } else {
            separate.status = fused ? Status::Inadmissible : Status::Selected;
            separate.why    = fused ? "fused" : "";
        }
    }
    std::vector<LinearPairCandidate> out;
    out.reserve(2);
    out.push_back(std::move(kernel));
    out.push_back(std::move(separate));
    return out;
}

} // namespace detail

std::size_t linear_pair_workspace_capacity_bytes(const Weight& first_weight,
                                                 const Weight& second_weight,
                                                 std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear_pair: unsupported weight/column geometry");
    }
    // Both alternatives take no caller workspace, so the width that decides between them does not
    // matter here.
    if (fused_domain(first_weight, second_weight)) { return 0; }
    require_same_logical_shape(first_weight, second_weight);
    // Two A16 linear() calls; the A16 form of linear() takes no caller workspace.
    return 0;
}

void linear_pair(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                 Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (!fused_pair(first_weight, second_weight, cols)) {
        require_same_logical_shape(first_weight, second_weight);
        if (first_weight.k != x.ne[0]) {
            throw std::invalid_argument("linear_pair: weight K differs from input");
        }
        require_matrix(x, x.ne[0], cols, "x");
        require_matrix(first_out, first_weight.n, cols, "first output");
        require_matrix(second_out, second_weight.n, cols, "second output");
        linear(x, first_weight, first_out, stream);
        linear(x, second_weight, second_out, stream);
        return;
    }
    if (x.ne[0] != 5120 && x.ne[0] != 2048) {
        throw std::invalid_argument("linear_pair: x K must be 5120 or 2048");
    }
    require_matrix(x, x.ne[0], cols, "x");
    require_matrix(first_out, 1024, cols, "first output");
    require_matrix(second_out, 1024, cols, "second output");
    if (first_weight.k != x.ne[0]) {
        throw std::invalid_argument("linear_pair: weight K differs from input");
    }
    (void)linear_pair_workspace_capacity_bytes(first_weight, second_weight, cols, cols);
    require_nonoverlap(x, first_weight, second_weight, first_out, second_out);

    detail::q8_pair_dispatch(x, first_weight, second_weight, first_out, second_out, stream);
}

} // namespace ninferno::ops
