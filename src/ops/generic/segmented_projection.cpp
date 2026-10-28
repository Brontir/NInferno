#include "ninferno/ops/segmented_projection.h"

#include "core/layout.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/gemma4_fused.h"
#include "ninferno/ops/rmsnorm.h"
#include "ninferno/ops/silu_mul.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/generic/generic_gemm.h"
#include "ops/linear/linear_registry.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninferno::ops {
namespace detail::generic {
void launch_gdn_gating_runtime(const Tensor& ab, const Tensor& A_log, const Tensor& dt_bias,
                               Tensor& g, Tensor& beta, cudaStream_t stream);
void launch_gdn_norm_gating_simt(const Tensor& x, const Tensor& norm_weight, float eps,
                                 const void* a_rows, const void* b_rows, const Tensor& A_log,
                                 const Tensor& dt_bias, Tensor& h, Tensor& g, Tensor& beta,
                                 cudaStream_t stream);
} // namespace detail::generic

namespace {

namespace generic = detail::generic;

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

// A contiguous tensor read as [first dimension, all remaining extents].
Tensor as_matrix(const Tensor& t) {
    return t.view({t.ne[0], t.ne[1] * t.ne[2] * t.ne[3]});
}

bool same_parent(const ProjectionSlice& a, const ProjectionSlice& b) {
    return a.parent.qdata == b.parent.qdata && a.parent.qtype == b.parent.qtype &&
           a.policy == b.policy && a.row_begin + a.rows == b.row_begin &&
           a.parent.input_scale_divisor == b.parent.input_scale_divisor;
}

// Runs the projection: consecutive slices of one parent become one call; each call's rows are
// spread over the outputs they intersect.
void project(const Tensor& x_matrix, const SegmentedProjectionWeights& w,
             std::span<const ProjectionOutput> outputs, WorkspaceArena& workspace,
             cudaStream_t stream) {
    require(!w.slices.empty(), "segmented projection: no slices");
    std::int32_t output_rows = 0;
    for (const auto& o : outputs) {
        require(o.tensor != nullptr && o.rows > 0 && o.row_offset >= 0,
                "segmented projection: invalid output region");
        output_rows += o.rows;
    }
    require(output_rows == w.rows(), "segmented projection: outputs do not cover the rows");
    // Everything below is scratch: the outputs are the caller's.
    auto scope = workspace.scope();

    // ggml's vector kernel quantizes its x to q8_1 in every call: where two or more slice groups
    // take it (a GGUF layer's q/k/v/z or gate/up of mixed formats), x is quantized once for all.
    // A call whose route turns out otherwise ignores the activation; one that takes the kernel
    // without it quantizes x itself.
    const void* vec_activation = nullptr;
    std::size_t parent_groups  = 1;
    for (std::size_t g = 1; g < w.slices.size(); ++g) {
        if (!same_parent(w.slices[g - 1], w.slices[g])) ++parent_groups;
    }
    if (parent_groups >= 2) {
        const std::int32_t tokens = x_matrix.ne[1];
        int vec_groups            = 0;
        for (std::size_t g = 0; g < w.slices.size();) {
            std::size_t h       = g + 1;
            std::int32_t g_rows = w.slices[g].rows;
            while (h < w.slices.size() && same_parent(w.slices[h - 1], w.slices[h])) {
                g_rows += w.slices[h].rows;
                ++h;
            }
            const ProjectionSlice& s = w.slices[g];
            if (generic::ggml_vec_takes_tuned(x_matrix, s.parent, s.policy, s.row_begin, g_rows,
                                              stream)) {
                ++vec_groups;
            }
            g = h;
        }
        if (vec_groups >= 2) {
            void* q8 =
                workspace.alloc_bytes(generic::ggml_vec_activation_bytes(x_matrix.ne[0], tokens))
                    .data;
            generic::ggml_vec_quantize(x_matrix, q8, stream);
            vec_activation = q8;
        }
    }

    std::int32_t logical = 0; // logical row of the current slice group
    std::size_t i        = 0;
    // Every call reads x and writes its own output rows: from the second on, a call may run
    // alongside the one before it (generic::Call::join_previous), which releases it
    // (release_next). A call runs once the next one is known, so the last one releases nothing.
    std::optional<generic::Call> pending;
    LinearPolicy pending_policy = LinearPolicy::A16Only;
    const auto launch           = [&](bool next) {
        pending->release_next = next;
        generic::run(*pending, pending_policy, &workspace, stream);
    };
    while (i < w.slices.size()) {
        std::size_t j        = i + 1;
        std::int32_t rows    = w.slices[i].rows;
        while (j < w.slices.size() && same_parent(w.slices[j - 1], w.slices[j])) {
            rows += w.slices[j].rows;
            ++j;
        }
        // Logical rows [logical, logical + rows) of parent rows [row_begin, row_begin + rows).
        const ProjectionSlice& first = w.slices[i];
        std::int32_t done            = 0;
        while (done < rows) {
            generic::Call call;
            call.x         = &x_matrix;
            call.weight    = &first.parent;
            call.row_begin = first.row_begin + done;
            call.epilogue  = generic::Epilogue::Store;
            call.segments  = 0;
            call.vec_activation  = vec_activation;
            std::int32_t o_begin = 0;
            std::int32_t taken   = 0;
            for (const auto& o : outputs) {
                const std::int32_t lo = std::max(o_begin, logical + done + taken);
                const std::int32_t hi = std::min(o_begin + o.rows, logical + rows);
                if (lo < hi && lo == logical + done + taken &&
                    call.segments < generic::kMaxSegments) {
                    call.out[call.segments]            = o.tensor;
                    call.out_row_offset[call.segments] = o.row_offset + (lo - o_begin);
                    call.segment_rows[call.segments]   = hi - lo;
                    ++call.segments;
                    taken += hi - lo;
                }
                o_begin += o.rows;
            }
            require(taken > 0, "segmented projection: output mapping failed");
            call.rows          = taken;
            call.join_previous = pending.has_value();
            if (pending) launch(true);
            pending        = call;
            pending_policy = first.policy;
            done += taken;
        }
        logical += rows;
        i = j;
    }
    launch(false);
}

std::size_t projection_capacity(const SegmentedProjectionWeights& w, std::int32_t min_tokens,
                                std::int32_t max_tokens) {
    std::size_t bytes  = 0;
    std::size_t shared = 0; // project()'s q8_1 activation for the vector kernel's calls
    for (const auto& s : w.slices) {
        if (generic::ggml_vec_format(s.parent.qtype) && allows_a8(s.policy) &&
            (s.parent.k % static_cast<std::int32_t>(ggml_block_values(s.parent.qtype))) == 0 &&
            min_tokens <= generic::kGgmlVecMaxTokens) {
            shared =
                std::max(shared, generic::ggml_vec_activation_bytes(
                                     s.parent.k, std::min(max_tokens, generic::kGgmlVecMaxTokens)) +
                                     256);
        }
        // project() joins consecutive slices of one parent into one call: up to its rows.
        bytes = std::max(bytes, generic::workspace_capacity_bytes(
                                    {s.parent.qtype, s.policy, std::max(s.rows, s.parent.n),
                                     s.parent.k, generic::Epilogue::Store},
                                    &s.parent, min_tokens, max_tokens));
    }
    return bytes + shared;
}

// The two halves of a [gate; up] projection as one weight's consecutive rows each, or nullopt.
std::optional<std::pair<generic::PairHalf, generic::PairHalf>>
gated_halves(const SegmentedProjectionWeights& w) {
    const std::int32_t half = w.rows() / 2;
    if (w.rows() != 2 * half || half <= 0) return std::nullopt;
    std::optional<generic::PairHalf> gate, up;
    std::int32_t logical = 0;
    for (std::size_t i = 0; i < w.slices.size();) {
        std::size_t j     = i + 1;
        std::int32_t rows = w.slices[i].rows;
        while (j < w.slices.size() && same_parent(w.slices[j - 1], w.slices[j])) {
            rows += w.slices[j].rows;
            ++j;
        }
        const generic::PairHalf h{&w.slices[i].parent, w.slices[i].policy, w.slices[i].row_begin};
        if (logical == 0 && rows == half) {
            gate = h;
        } else if (logical == half && rows == half) {
            up = h;
        } else {
            return std::nullopt;
        }
        logical += rows;
        i = j;
    }
    if (!gate || !up) return std::nullopt;
    return std::pair{*gate, *up};
}

} // namespace

std::size_t segmented_swiglu_workspace_capacity_bytes(const SegmentedProjectionWeights& weights,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    require(min_tokens > 0 && max_tokens >= min_tokens,
            "segmented swiglu workspace: invalid token interval");
    // The separate path: the [gate; up] rows in BF16, then the projection's own workspace.
    const std::size_t separate =
        static_cast<std::size_t>(weights.rows()) * max_tokens * sizeof(std::uint16_t) + 256 +
        projection_capacity(weights, min_tokens, max_tokens);
    return std::max(
        separate, segmented_swiglu_pair_workspace_capacity_bytes(weights, min_tokens, max_tokens));
}

std::size_t
segmented_swiglu_pair_workspace_capacity_bytes(const SegmentedProjectionWeights& weights,
                                               std::int32_t min_tokens, std::int32_t max_tokens) {
    require(min_tokens > 0 && max_tokens >= min_tokens,
            "segmented swiglu workspace: invalid token interval");
    if (!gated_halves(weights)) return 0;
    const std::int32_t half = weights.rows() / 2;
    std::size_t bytes       = 0;
    if (min_tokens <= generic::kGgmlVecMaxTokens) {
        bytes = generic::ggml_vec_gated_pair_bytes(
            weights.columns(), half, std::min(max_tokens, generic::kGgmlVecMaxTokens));
    }
    // The matrix kernel's band, or its prefill slice where that route holds (its route decides
    // per call; sized for its widest width).
    if (max_tokens > generic::kGgmlVecMaxTokens) {
        const auto halves = *gated_halves(weights);
        const std::int32_t slice =
            std::min(generic::ggml_mmq_slice_tokens(halves.first.weight->qtype),
                     generic::ggml_mmq_slice_tokens(halves.second.weight->qtype));
        bytes = std::max(bytes, generic::ggml_mmq_gated_pair_bytes(weights.columns(), half,
                                                                   std::min(max_tokens, slice)));
    }
    return bytes;
}

bool segmented_swiglu_pair(const Tensor& x, const SegmentedProjectionWeights& weights, bool gelu,
                           Tensor& out, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t half = weights.rows() / 2;
    const Tensor xm         = as_matrix(x);
    if (weights.rows() != 2 * half || out.dtype != DType::BF16 || out.ne[0] != half ||
        out.ne[1] != xm.ne[1] || !out.is_contiguous()) {
        return false;
    }
    const auto halves = gated_halves(weights);
    if (!halves) return false;
    if (xm.ne[1] <= generic::kGgmlVecMaxTokens) {
        return generic::ggml_vec_gated_pair(xm, halves->first, halves->second, half, gelu, out,
                                            workspace, nullptr, stream);
    }
    return generic::ggml_mmq_gated_pair(xm, halves->first, halves->second, half, gelu, out,
                                        workspace, stream);
}

void segmented_swiglu(const Tensor& x, const SegmentedProjectionWeights& weights, bool gelu,
                      Tensor& out, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t half = weights.rows() / 2;
    const Tensor xm         = as_matrix(x);
    require(weights.rows() == 2 * half && out.dtype == DType::BF16 && out.ne[0] == half &&
                out.ne[1] == xm.ne[1] && out.is_contiguous(),
            "segmented swiglu: out must be BF16 [rows / 2, T]");
    if (segmented_swiglu_pair(x, weights, gelu, out, workspace, stream)) return;
    auto scope     = workspace.scope();
    Tensor gate_up = workspace.alloc(DType::BF16, {weights.rows(), xm.ne[1]});
    {
        auto call = workspace.scope();
        const std::array outputs{ProjectionOutput{&gate_up, 0, weights.rows()}};
        project(xm, weights, outputs, workspace, stream);
    }
    if (gelu) {
        gelu_tanh_mul(gate_up, out, stream);
    } else {
        silu_mul(gate_up.slice(0, 0, half), gate_up.slice(0, half, half), out, stream);
    }
}

std::size_t segmented_projection_workspace_capacity_bytes(const SegmentedProjectionWeights& weights,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) {
    require(min_tokens > 0 && max_tokens >= min_tokens,
            "segmented projection workspace: invalid token interval");
    return projection_capacity(weights, min_tokens, max_tokens);
}

void segmented_projection(const Tensor& x, const SegmentedProjectionWeights& weights,
                          std::span<const ProjectionOutput> outputs, WorkspaceArena& workspace,
                          cudaStream_t stream) {
    require(x.ne[0] == weights.columns(), "segmented projection: x rows differ from K");
    const Tensor x_matrix = as_matrix(x);
    project(x_matrix, weights, outputs, workspace, stream);
}

std::size_t segmented_gdn_conv_workspace_capacity_bytes(const SegmentedProjectionWeights& weights,
                                                        std::int32_t channels,
                                                        std::int32_t batch_size,
                                                        std::int32_t min_width,
                                                        std::int32_t max_width) {
    require(channels > 0 && batch_size > 0 && min_width > 0 && max_width >= min_width,
            "segmented GDN conv workspace: invalid domain");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {channels, max_width * batch_size});
    const std::size_t projection =
        projection_capacity(weights, min_width * batch_size, max_width * batch_size);
    if (projection != 0) (void)layout.alloc_bytes(projection);
    return layout.peak_bytes(1);
}

void segmented_gdn_input_proj_conv_snapshot(
    const Tensor& x, const SegmentedProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_state_slots,
    const Tensor& snapshot_base_slots, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
    WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t channels = query.ne[0] + key.ne[0] + value.ne[0];
    const std::int32_t width    = x.ne[1];
    const std::int32_t batch    = x.ne[2];
    require(weights.rows() == channels + z.ne[0], "segmented GDN snapshot: row profile differs");
    auto scope       = workspace.scope();
    Tensor projected = workspace.alloc(DType::BF16, {channels, width * batch});
    Tensor z_matrix  = as_matrix(z);
    const ProjectionOutput outputs[] = {{&projected, 0, channels}, {&z_matrix, 0, z.ne[0]}};
    project(as_matrix(x), weights, outputs, workspace, stream);
    Tensor projected_block(projected.data, DType::BF16, {channels, width, batch});
    detail::gdn_projected_conv_snapshot_launch(projected_block, conv_weight, conv_states,
                                               valid_columns, initial_state_slots,
                                               snapshot_base_slots, query, key, value, stream);
}

void segmented_gdn_input_proj_conv_record(const Tensor& x,
                                          const SegmentedProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns,
                                          const Tensor& initial_state_slots, Tensor& conv_record,
                                          Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t channels = query.ne[0] + key.ne[0] + value.ne[0];
    require(weights.rows() == channels + z.ne[0] && conv_record.ne[0] == channels,
            "segmented GDN record: row profile differs");
    auto scope             = workspace.scope();
    Tensor record_matrix   = as_matrix(conv_record);
    Tensor z_matrix        = as_matrix(z);
    const ProjectionOutput outputs[] = {{&record_matrix, 0, channels}, {&z_matrix, 0, z.ne[0]}};
    project(as_matrix(x), weights, outputs, workspace, stream);
    detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states, valid_columns,
                                             initial_state_slots, query, key, value, stream);
}

std::size_t segmented_gdn_norm_gating_workspace_capacity_bytes(
    const SegmentedProjectionWeights& weights, std::int32_t min_tokens, std::int32_t max_tokens) {
    require(min_tokens > 0 && max_tokens >= min_tokens,
            "segmented GDN control workspace: invalid token interval");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {weights.rows(), max_tokens});
    const std::size_t projection = projection_capacity(weights, min_tokens, max_tokens);
    if (projection != 0) (void)layout.alloc_bytes(projection);
    return layout.peak_bytes(1);
}

namespace {

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0; }

// Rows [row, row + rows) of the bank as one BF16 row-major pointer: they lie in one slice of a
// contiguous BF16 parent. nullptr otherwise.
const void* bf16_bank_rows(const SegmentedProjectionWeights& w, std::int32_t row,
                           std::int32_t rows) {
    std::int32_t logical = 0;
    for (const auto& s : w.slices) {
        if (row >= logical && row + rows <= logical + s.rows) {
            const Weight& p = s.parent;
            if (p.qtype != QType::BF16 || p.layout != QuantLayout::Contiguous ||
                p.qdata == nullptr || p.scales != nullptr || p.qhigh != nullptr) {
                return nullptr;
            }
            const auto* data = static_cast<const std::uint8_t*>(p.qdata) +
                               (static_cast<std::size_t>(s.row_begin) + (row - logical)) *
                                   static_cast<std::size_t>(p.k) * sizeof(std::uint16_t);
            return aligned16(data) ? data : nullptr;
        }
        logical += s.rows;
    }
    return nullptr;
}

} // namespace

void segmented_gdn_norm_gating_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                                    const SegmentedProjectionWeights& weights, const Tensor& A_log,
                                    const Tensor& dt_bias, WorkspaceArena& workspace, Tensor& h,
                                    Tensor& g, Tensor& beta, cudaStream_t stream) {
    const std::int32_t heads  = weights.rows() / 2;
    const std::int32_t tokens = x.ne[1] * x.ne[2] * x.ne[3];
    require(weights.rows() == 2 * heads && heads > 0, "segmented GDN control: A/B rows differ");
    require(g.ne[0] == heads && beta.ne[0] == heads && A_log.ne[0] == heads &&
                dt_bias.ne[0] == heads,
            "segmented GDN control: head count differs");
    // Decode and verify widths with BF16 a/b: the norm, both projections and the gates in one
    // kernel (the 27B's fused form at this geometry), whose CTAs each read one head's two rows.
    // The composed form below costs four dependent launches and runs each projection on
    // heads / 16 CTAs; this one runs heads CTAs and one launch. It sums K in its own order, so it
    // stays out of the invariant route profile, and keeps to the decode/verify widths, where no
    // profile asks a token's bytes to follow a wider call's.
    const std::int32_t d = x.ne[0];
    if (tokens <= kSegmentedFusedControlMaxTokens && !invariant_routes_active() && d % 8 == 0 &&
        weights.columns() == d && aligned16(x.data) && aligned16(norm_weight.data) &&
        aligned16(h.data)) {
        const void* a_rows = bf16_bank_rows(weights, 0, heads);
        const void* b_rows = bf16_bank_rows(weights, heads, heads);
        if (a_rows != nullptr && b_rows != nullptr) {
            Tensor x_matrix    = as_matrix(x);
            Tensor h_matrix    = as_matrix(h);
            Tensor g_matrix    = as_matrix(g);
            Tensor beta_matrix = as_matrix(beta);
            detail::generic::launch_gdn_norm_gating_simt(x_matrix, norm_weight, eps, a_rows, b_rows,
                                                         A_log, dt_bias, h_matrix, g_matrix,
                                                         beta_matrix, stream);
            return;
        }
    }
    rmsnorm(x, norm_weight, eps, true, h, stream);
    auto scope = workspace.scope();
    Tensor ab  = workspace.alloc(DType::BF16, {2 * heads, tokens});
    const ProjectionOutput outputs[] = {{&ab, 0, 2 * heads}};
    project(as_matrix(h), weights, outputs, workspace, stream);
    Tensor g_matrix    = as_matrix(g);
    Tensor beta_matrix = as_matrix(beta);
    detail::generic::launch_gdn_gating_runtime(ab, A_log, dt_bias, g_matrix, beta_matrix, stream);
}

bool project_side_by_side(const Tensor& x, std::span<const SideBySideProjection> projections,
                          DeviceArena& work, cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    if (t > kSideBySideMaxTokens || projections.size() < 2 ||
        projections.size() > kSideBySideMaxProjections) {
        return false;
    }
    for (const auto& p : projections) {
        if (p.weight == nullptr || p.output == nullptr || p.output->ne[0] != p.weight->n ||
            p.output->ne[1] != t || !detail::side_by_side_admits(*p.weight, p.policy, t)) {
            return false;
        }
    }
    // Reused across layers and steps: one pass runs at a time per thread.
    thread_local SegmentedProjectionWeights weights;
    weights.slices.clear();
    ProjectionOutput regions[kSideBySideMaxProjections];
    for (std::size_t i = 0; i < projections.size(); ++i) {
        const Weight& wt = *projections[i].weight;
        weights.slices.push_back({wt, 0, wt.n, projections[i].policy});
        regions[i] = {projections[i].output, 0, wt.n};
    }
    auto scope = work.scope();
    WorkspaceArena scratch(work.alloc_bytes(
        std::max<std::size_t>(segmented_projection_workspace_capacity_bytes(weights, t, t), 256)));
    segmented_projection(x, weights, std::span(regions, projections.size()), scratch, stream);
    return true;
}

} // namespace ninferno::ops
