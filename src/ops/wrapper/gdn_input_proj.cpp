#include "core/weight.h"
#include "ops/generic/generic_gemm.h"
#include "ops/common/native_nvfp4.h"
#include "ninferno/ops/gdn_input_proj.h"
#include "ninferno/ops/prefill_routes.h"

#include "core/layout.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "ops/gdn_input_proj/q8/q8_gdn_input_kernels.h"
#include "ops/gdn_input_proj/q8/q8_gdn_input_plan.h"
#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_swiglu/q4a8/q4a8_linear_swiglu.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void require_conv_tensor(const Tensor& tensor, std::int32_t rows, std::int32_t width,
                         std::int32_t batch, const char* op, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != width ||
        tensor.ne[2] != batch || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }
}

void require_single_parent_nonoverlap(const Tensor& x, const Tensor& qkv, const Tensor& z) {
    if (overlaps(x, qkv) || overlaps(x, z) || overlaps(qkv, z)) {
        throw std::invalid_argument("gdn_input_proj: x, qkv, and z must not overlap");
    }
}

// Int8 prefill activations (prefill_a8_policy, generic_gemm.h): inside a prefill a call of 17 columns or more of the
// Q4/Q5 pair takes the generic path's int8 tiers, one call per parent written segment by segment,
// as linear does for these formats. The grouped kernels read A16 at every width; ninfer-all
// computes this projection on int8 activations (Qwen3.8-27B 8K prefill on an RTX 5090: 483 + 840 us
// against its 260 + 149). Off unless the option asks for it: it moves the numerics as linear's
// does.
constexpr std::int32_t kPairHidden     = 5120;
constexpr std::int32_t kPairQkRows     = 4096;
constexpr std::int32_t kPairValueRows  = 6144;
constexpr std::int32_t kPairZRows      = 6144;
constexpr std::int32_t kPairParentRows = kPairValueRows + kPairZRows;

LinearPolicy pair_prefill_a8_policy(QType qtype) {
    return detail::generic::fixed_parent_prefill_a8_policy(qtype);
}

std::size_t pair_prefill_a8_workspace_bytes(std::int32_t min_tokens, std::int32_t max_tokens) {
    // The dedicated int8 route of the listed cards (a8_input_pays) quantises x once for both.
    const std::size_t dedicated =
        max_tokens >= 128 && detail::a8_input_pays()
            ? detail::a8_input_workspace_capacity_bytes(std::max(min_tokens, 128), max_tokens)
            : 0;
    return std::max(dedicated, detail::generic::fixed_parents_prefill_a8_workspace_bytes(
                                   {{QType::Q4_G64_FP16, kPairQkRows},
                                    {QType::Q5_G64_FP16, kPairParentRows}},
                                   kPairHidden, min_tokens, max_tokens));
}

// qkv [10240, T] and z [6144, T] as the pair writes them; false leaves the call to the caller.
// The invariant route profile takes the same two generic calls at every width: the Q4/Q5 pair
// kernels pick their tile by width (direct to 12 columns, grouped tiles above), so a verify
// launch of several rows summed a row's columns in another order than the row alone.
bool pair_prefill_a8(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                     Tensor& qkv, Tensor& z, WorkspaceArena* workspace, cudaStream_t stream) {
    const bool invariant = invariant_routes_active();
    if (!invariant && (workspace == nullptr ||
                       !detail::generic::fixed_pair_prefill_a8(qk_weight, value_z_weight, x.ne[1]))) {
        return false;
    }
    if (!invariant && detail::a8_input_pays() &&
        detail::a8_gdn_input_supported(qk_weight, value_z_weight, x.ne[1]) && x.is_contiguous() &&
        aligned_to(x.data, 16) && qkv.is_contiguous() && z.is_contiguous()) {
        detail::a8_gdn_input_launch(x, qk_weight, value_z_weight, qkv, z, *workspace, stream);
        return true;
    }
    const LinearPolicy qk_policy =
        invariant ? LinearPolicy::A16Only : pair_prefill_a8_policy(qk_weight.qtype);
    const LinearPolicy value_policy =
        invariant ? LinearPolicy::A16Only : pair_prefill_a8_policy(value_z_weight.qtype);
    detail::generic::Call qk;
    qk.x               = &x;
    qk.weight          = &qk_weight;
    qk.rows            = kPairQkRows;
    qk.out[0]          = &qkv;
    qk.segment_rows[0] = kPairQkRows;
    qk.segments        = 1;
    detail::generic::run(qk, qk_policy, workspace, stream);
    detail::generic::Call value;
    value.x                 = &x;
    value.weight            = &value_z_weight;
    value.rows              = kPairParentRows;
    value.out[0]            = &qkv;
    value.out_row_offset[0] = kPairQkRows;
    value.segment_rows[0]   = kPairValueRows;
    value.out[1]            = &z;
    value.segment_rows[1]   = kPairZRows;
    value.segments          = 2;
    detail::generic::run(value, value_policy, workspace, stream);
    return true;
}

struct ConvGeometry {
    std::int32_t width;
    std::int32_t batch;
    std::int32_t aggregate_columns;
};

ConvGeometry require_snapshot_input(const Tensor& x, std::int32_t hidden) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMaximumWidth = 16;
    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    if (width <= 0 || batch <= 0 || batch > kMaximumBatch || (batch > 1 && width > kMaximumWidth)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: unsupported B/W domain");
    }
    require_conv_tensor(x, hidden, width, batch, "gdn_input_proj_conv_snapshot", "x");
    return {width, batch, width * batch};
}

ConvGeometry require_record_input(const Tensor& x, std::int32_t hidden) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMinimumWidth = 2;
    constexpr std::int32_t kMaximumWidth = 16;
    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    if (width < kMinimumWidth || width > kMaximumWidth || batch <= 0 || batch > kMaximumBatch) {
        throw std::invalid_argument("gdn_input_proj_conv_record: unsupported B/T domain");
    }
    require_conv_tensor(x, hidden, width, batch, "gdn_input_proj_conv_record", "x");
    return {width, batch, width * batch};
}

void require_snapshot_operands(const Tensor& conv_weight, const Tensor& conv_states,
                               const Tensor& valid_columns, const Tensor& initial_state_slots,
                               const Tensor& snapshot_base_slots, std::int32_t channels,
                               ConvGeometry geometry) {
    require_matrix(conv_weight, channels, 4, "conv weight");
    if (conv_states.dtype != DType::BF16 || conv_states.ne[0] != channels ||
        conv_states.ne[1] != 3 || conv_states.ne[2] < geometry.aggregate_columns ||
        conv_states.ne[3] != 1 || !conv_states.is_contiguous() ||
        !aligned_to(conv_states.data, 16)) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_snapshot: invalid convolution snapshot state");
    }
    const auto valid_selector = [batch = geometry.batch](const Tensor& selector) {
        return selector.dtype == DType::I32 && selector.ne[0] == batch && selector.ne[1] == 1 &&
               selector.ne[2] == 1 && selector.ne[3] == 1 && selector.is_contiguous() &&
               selector.data != nullptr;
    };
    if (!valid_selector(initial_state_slots) || !valid_selector(snapshot_base_slots)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: invalid state selector");
    }
    if (valid_columns.data != nullptr) {
        if (!valid_selector(valid_columns)) {
            throw std::invalid_argument("gdn_input_proj_conv_snapshot: invalid valid columns");
        }
    }
}

Tensor flatten_columns(const Tensor& tensor, std::int32_t rows, ConvGeometry geometry) {
    return Tensor(tensor.data, tensor.dtype, {rows, geometry.aggregate_columns});
}

void require_record_operands(const Tensor& conv_weight, const Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             std::int32_t channels, ConvGeometry geometry) {
    require_matrix(conv_weight, channels, 4, "conv weight");
    if (conv_states.dtype != DType::BF16 || conv_states.ne[0] != channels ||
        conv_states.ne[1] != 3 || conv_states.ne[2] <= 0 || conv_states.ne[3] != 1 ||
        !conv_states.is_contiguous() || !aligned_to(conv_states.data, 16)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid convolution state");
    }
    const auto valid_selector = [batch = geometry.batch](const Tensor& selector) {
        return selector.dtype == DType::I32 && selector.ne[0] == batch && selector.ne[1] == 1 &&
               selector.ne[2] == 1 && selector.ne[3] == 1 && selector.is_contiguous() &&
               selector.data != nullptr;
    };
    if (!valid_selector(initial_state_slots)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid initial state selector");
    }
    if (valid_columns.data != nullptr && !valid_selector(valid_columns)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid valid columns");
    }
}

bool overlaps_range(const Tensor& tensor, const void* base, std::size_t bytes) {
    if (tensor.data == nullptr || base == nullptr || bytes == 0) { return false; }
    const auto tensor_begin = reinterpret_cast<std::uintptr_t>(tensor.data);
    const auto range_begin  = reinterpret_cast<std::uintptr_t>(base);
    return tensor_begin < range_begin + bytes && range_begin < tensor_begin + tensor.bytes();
}

void require_record_nonoverlap(const Tensor& x, const Tensor& conv_weight,
                               const Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_state_slots, const Tensor& conv_record,
                               const Tensor& query, const Tensor& key, const Tensor& value,
                               const Tensor& z, const WorkspaceArena& workspace) {
    const std::array<const Tensor*, 10> tensors{
        &x,           &conv_weight, &conv_states, &valid_columns, &initial_state_slots,
        &conv_record, &query,       &key,         &value,         &z};
    for (std::size_t lhs = 0; lhs < tensors.size(); ++lhs) {
        if (tensors[lhs]->data == nullptr) { continue; }
        for (std::size_t rhs = lhs + 1; rhs < tensors.size(); ++rhs) {
            if (tensors[rhs]->data != nullptr && overlaps(*tensors[lhs], *tensors[rhs])) {
                throw std::invalid_argument(
                    "gdn_input_proj_conv_record: tensor operands must not overlap");
            }
        }
        if (overlaps_range(*tensors[lhs], workspace.base(), workspace.capacity())) {
            throw std::invalid_argument(
                "gdn_input_proj_conv_record: tensor operand overlaps live workspace");
        }
    }
}

void require_snapshot_nonoverlap(const Tensor& x, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_state_slots,
                                 const Tensor& snapshot_base_slots, const Tensor& query,
                                 const Tensor& key, const Tensor& value, const Tensor& z,
                                 const WorkspaceArena& workspace) {
    const std::array<const Tensor*, 10> tensors{&x,
                                                &conv_weight,
                                                &conv_states,
                                                &valid_columns,
                                                &initial_state_slots,
                                                &snapshot_base_slots,
                                                &query,
                                                &key,
                                                &value,
                                                &z};
    for (std::size_t lhs = 0; lhs < tensors.size(); ++lhs) {
        if (tensors[lhs]->data == nullptr) { continue; }
        for (std::size_t rhs = lhs + 1; rhs < tensors.size(); ++rhs) {
            const bool shared_state_selectors =
                tensors[lhs] == &initial_state_slots && tensors[rhs] == &snapshot_base_slots;
            if (!shared_state_selectors && tensors[rhs]->data != nullptr &&
                overlaps(*tensors[lhs], *tensors[rhs])) {
                throw std::invalid_argument(
                    "gdn_input_proj_conv_snapshot: tensor operands must not overlap");
            }
        }
        if (overlaps_range(*tensors[lhs], workspace.base(), workspace.capacity())) {
            throw std::invalid_argument(
                "gdn_input_proj_conv_snapshot: tensor operand overlaps live workspace");
        }
    }
}

template <std::size_t Count>
void require_parent_nonoverlap(const Weight& weight,
                               const std::array<const Tensor*, Count>& tensors,
                               const WorkspaceArena& workspace, const char* operation) {
    for (const Tensor* tensor : tensors) {
        if (tensor->data != nullptr &&
            overlaps_range(*tensor, weight.payload,
                           static_cast<std::size_t>(weight.payload_bytes))) {
            throw std::invalid_argument(std::string(operation) +
                                        ": tensor operand overlaps parent weight");
        }
    }
    if (weight.payload != nullptr && workspace.base() != nullptr && workspace.capacity() != 0) {
        const auto weight_begin    = reinterpret_cast<std::uintptr_t>(weight.payload);
        const auto workspace_begin = reinterpret_cast<std::uintptr_t>(workspace.base());
        if (weight_begin < workspace_begin + workspace.capacity() &&
            workspace_begin < weight_begin + weight.payload_bytes) {
            throw std::invalid_argument(std::string(operation) +
                                        ": parent weight overlaps live workspace");
        }
    }
}

void require_snapshot_capacity_domain(std::int32_t batch_size, std::int32_t min_width,
                                      std::int32_t max_width) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMaximumWidth = 16;
    if (batch_size <= 0 || batch_size > kMaximumBatch || min_width <= 0 || max_width < min_width ||
        (batch_size > 1 && max_width > kMaximumWidth)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot workspace: invalid B/W domain");
    }
}

void require_record_capacity_domain(std::int32_t batch_size, std::int32_t min_width,
                                    std::int32_t max_width) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMinimumWidth = 2;
    constexpr std::int32_t kMaximumWidth = 16;
    if (batch_size <= 0 || batch_size > kMaximumBatch || min_width < kMinimumWidth ||
        max_width < min_width || max_width > kMaximumWidth) {
        throw std::invalid_argument("gdn_input_proj_conv_record workspace: invalid B/T domain");
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t rows, const char* label) {
    const bool q4_planes =
        qtype != QType::Q4_G64_FP16 || (weight.qhigh == nullptr && weight.high_plane_bytes == 0);
    const bool q5_planes =
        qtype != QType::Q5_G64_FP16 || (weight.qhigh != nullptr && weight.high_plane_bytes != 0);
    if (weight.qtype != qtype || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 64 || weight.group != 64 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 5120 || weight.shape[0] != rows ||
        weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || !q4_planes || !q5_planes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4) ||
        (qtype == QType::Q5_G64_FP16 && !aligned_to(weight.qhigh, 16))) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void require_q8_rowsplit(const Weight& weight, std::int32_t rows, const char* label) {
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 2048 || weight.shape[0] != rows ||
        weight.shape[1] != 2048 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 2048 || weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("gdn_input_proj: invalid compute policy");
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_policy(policy);
    const std::int32_t cols = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }

    if (weight.qtype == QType::NVFP4) {
#ifdef NINFERNO_WITHOUT_NVFP4
        detail::no_native_nvfp4("gdn_input_proj");
#else
        constexpr std::int32_t kHidden  = 5120;
        constexpr std::int32_t kQkvRows = 10240;
        constexpr std::int32_t kZRows   = 6144;
        constexpr std::int32_t kRows    = kQkvRows + kZRows;
        require_matrix(x, kHidden, cols, "x");
        require_matrix(qkv, kQkvRows, cols, "qkv");
        require_matrix(z, kZRows, cols, "z");
        require_single_parent_nonoverlap(x, qkv, z);
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_gdn_input_dispatch(x, weight, qkv, z, policy, workspace, stream);
        return;
#endif
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        constexpr std::int32_t kHidden  = 5120;
        constexpr std::int32_t kQkvRows = 10240;
        constexpr std::int32_t kZRows   = 6144;
        constexpr std::int32_t kRows    = kQkvRows + kZRows;
        require_matrix(x, kHidden, cols, "x");
        require_matrix(qkv, kQkvRows, cols, "qkv");
        require_matrix(z, kZRows, cols, "z");
        require_single_parent_nonoverlap(x, qkv, z);
        detail::validate_fp8_weight(weight, "fp8 gdn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 gdn_input_proj: unsupported weight shape");
        }
        detail::fp8_gdn_input_dispatch(x, weight, qkv, z, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden  = 2048;
    constexpr std::int32_t kQkvRows = 8192;
    constexpr std::int32_t kZRows   = 4096;
    constexpr std::int32_t kRows    = kQkvRows + kZRows;
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_single_parent_nonoverlap(x, qkv, z);
    require_q8_rowsplit(weight, kRows, "query/key/value/z weight");
    // owned_decode_r8: at decode the generic R8 writes both parts in one call. The invariant route
    // profile takes the generic path at every width too: the Q8 kernels pick their K split by
    // width (sliced-K tiles of 8..32 columns, grouped split-K to 96), so a verify launch of
    // several rows summed a row's columns in another order than the row alone.
    if (invariant_routes_active() || detail::generic::owned_decode_r8(weight, policy, cols) ||
        detail::generic::q8_prefill_a8(weight, policy, cols) ||
        detail::generic::volta_tiled_route(weight, policy, cols)) {
        detail::generic::Call call;
        call.x               = &x;
        call.weight          = &weight;
        call.rows            = kRows;
        call.out[0]          = &qkv;
        call.segment_rows[0] = kQkvRows;
        call.out[1]          = &z;
        call.segment_rows[1] = kZRows;
        call.segments        = 2;
        detail::generic::run(call, policy, workspace, stream);
        return;
    }
    detail::q8_gdn_input_dispatch(x, weight, qkv, z, stream);
}

// Without tensor mma.sync (Volta) a call of 17 tokens or more that the tiled tensor-core route
// takes (generic::volta_tiled_route) runs there: the shape plans' MMA tiles run on FP32 FMAs on
// that GPU. One generic call per parent weight, its rows written segment by segment, each at its
// row offset into its output (a row slice of a [rows, T] matrix is not contiguous).
bool volta_tiled_parent(const Tensor& x, const Weight& weight, std::initializer_list<Tensor*> outs,
                        std::initializer_list<std::int32_t> rows,
                        std::initializer_list<std::int32_t> offsets, WorkspaceArena* workspace,
                        cudaStream_t stream) {
    if (!detail::generic::volta_tiled_route(weight, LinearPolicy::A16Only, x.ne[1])) return false;
    detail::generic::Call call;
    call.x      = &x;
    call.weight = &weight;
    int s       = 0;
    for (Tensor* out : outs) call.out[s++] = out;
    s = 0;
    for (const std::int32_t offset : offsets) call.out_row_offset[s++] = offset;
    s = 0;
    for (const std::int32_t r : rows) {
        call.segment_rows[s++] = r;
        call.rows += r;
    }
    call.segments = s;
    detail::generic::run(call, LinearPolicy::A16Only, workspace, stream);
    return true;
}

detail::Q4Q5GdnInputConvPlan resolve_q4_q5_conv_plan(std::int32_t tokens, std::int32_t batch_size) {
    return detail::q4_q5_gdn_input_conv_resolve_plan({5120, 4096, 12288, 10240, 6144, 5120, tokens},
                                                     batch_size);
}

detail::Q8GdnInputConvPlan resolve_q8_conv_plan(std::int32_t tokens, std::int32_t batch_size) {
    return detail::q8_gdn_input_conv_resolve_plan({2048, 8192, 4096, 12288, 2048, tokens},
                                                  batch_size);
}

struct ProjectedWorkspace {
    Tensor projected;
};

template <class Allocator>
ProjectedWorkspace allocate_projected_workspace(Allocator& allocator, std::int32_t channels,
                                                std::int32_t tokens) {
    ProjectedWorkspace out;
    out.projected = allocator.alloc(DType::BF16, {channels, tokens});
    return out;
}

std::size_t composed_snapshot_capacity(std::int32_t channels, std::int32_t aggregate_columns,
                                       std::size_t projection_workspace_bytes) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected_workspace(layout, channels, aggregate_columns);
    if (projection_workspace_bytes != 0) { (void)layout.alloc_bytes(projection_workspace_bytes); }
    return layout.peak_bytes(1);
}

template <class Project>
void compose_batched_snapshot(const Tensor& x, const Tensor& conv_weight, Tensor& conv_states,
                              const Tensor& valid_columns, const Tensor& initial_state_slots,
                              const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                              Tensor& value, Tensor& z, std::int32_t query_rows,
                              std::int32_t key_rows, std::int32_t value_rows, ConvGeometry geometry,
                              WorkspaceArena& workspace, cudaStream_t stream, Project&& project) {
    const std::int32_t channels = query_rows + key_rows + value_rows;
    auto scope                  = workspace.scope();
    ProjectedWorkspace scratch =
        allocate_projected_workspace(workspace, channels, geometry.aggregate_columns);

    Tensor x_flat = flatten_columns(x, x.ne[0], geometry);
    Tensor z_flat = flatten_columns(z, z.ne[0], geometry);
    project(x_flat, scratch.projected, z_flat);

    Tensor projected(scratch.projected.data, DType::BF16,
                     {channels, geometry.width, geometry.batch});
    detail::gdn_projected_conv_snapshot_launch(projected, conv_weight, conv_states, valid_columns,
                                               initial_state_slots, snapshot_base_slots, query, key,
                                               value, stream);
}

template <class Project>
void compose_record(const Tensor& x, const Tensor& conv_weight, const Tensor& conv_states,
                    const Tensor& valid_columns, const Tensor& initial_state_slots,
                    Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                    ConvGeometry geometry, WorkspaceArena& workspace, cudaStream_t stream,
                    Project&& project) {
    auto scope         = workspace.scope();
    Tensor x_flat      = flatten_columns(x, x.ne[0], geometry);
    Tensor record_flat = flatten_columns(conv_record, conv_record.ne[0], geometry);
    Tensor z_flat      = flatten_columns(z, z.ne[0], geometry);
    project(x_flat, record_flat, z_flat);
    detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states, valid_columns,
                                             initial_state_slots, query, key, value, stream);
}

void dispatch_single_parent_snapshot(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight, Tensor& conv_states,
                                     const Tensor& valid_columns, const Tensor& initial_state_slots,
                                     const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                     Tensor& value, Tensor& z, LinearPolicy policy,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    validate_policy(policy);

    if (weight.qtype == QType::NVFP4) {
#ifdef NINFERNO_WITHOUT_NVFP4
        detail::no_native_nvfp4("gdn_input_proj");
#else
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_snapshot_input(x, kHidden);
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj_conv_snapshot");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "nvfp4 gdn_input_proj_conv_snapshot: unsupported weight shape");
        }
        require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                  snapshot_base_slots, kChannels, geometry);
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "z");
        const detail::Nvfp4GdnConvPlan plan =
            detail::nvfp4_gdn_conv_resolve_plan(policy, geometry.width, geometry.batch);
        if (geometry.batch > 1) {
            if (plan.schedule != detail::Nvfp4GdnConvScheduleId::Materialized) {
                throw std::logic_error("batched NVFP4 GDN conv selected a fused schedule");
            }
            compose_batched_snapshot(x, conv_weight, conv_states, valid_columns,
                                     initial_state_slots, snapshot_base_slots, query, key, value, z,
                                     kQueryRows, kKeyRows, kValueRows, geometry, workspace, stream,
                                     [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                                         gdn_input_proj(x_flat, weight, projected, z_flat, policy,
                                                        workspace, stream);
                                     });
            return;
        }
        detail::nvfp4_gdn_snapshot_dispatch(x, weight, conv_weight, conv_states, valid_columns,
                                            initial_state_slots, snapshot_base_slots, query, key,
                                            value, z, policy, workspace, stream);
        return;
#endif
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_snapshot_input(x, kHidden);
        detail::validate_fp8_weight(weight, "fp8 gdn_input_proj_conv_snapshot");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "fp8 gdn_input_proj_conv_snapshot: unsupported weight shape");
        }
        require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                  snapshot_base_slots, kChannels, geometry);
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "z");
        require_snapshot_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                    snapshot_base_slots, query, key, value, z, workspace);
        const std::array<const Tensor*, 10> tensors{&x,
                                                    &conv_weight,
                                                    &conv_states,
                                                    &valid_columns,
                                                    &initial_state_slots,
                                                    &snapshot_base_slots,
                                                    &query,
                                                    &key,
                                                    &value,
                                                    &z};
        require_parent_nonoverlap(weight, tensors, workspace, "fp8 gdn_input_proj_conv_snapshot");
        detail::fp8_gdn_snapshot_dispatch(x, weight, conv_weight, conv_states, valid_columns,
                                          initial_state_slots, snapshot_base_slots, query, key,
                                          value, z, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kQueryRows = 2048;
    constexpr std::int32_t kKeyRows   = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    const ConvGeometry geometry       = require_snapshot_input(x, kHidden);
    require_q8_rowsplit(weight, kChannels + kZRows, "query/key/value/z weight");
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, kChannels, geometry);
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_snapshot",
                        "z");
    if (geometry.batch > 1) {
        compose_batched_snapshot(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                 snapshot_base_slots, query, key, value, z, kQueryRows, kKeyRows,
                                 kValueRows, geometry, workspace, stream,
                                 [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                                     gdn_input_proj(x_flat, weight, projected, z_flat, stream);
                                 });
        return;
    }

    // The fused routes sum the projection in their own split of K; the invariant profile
    // materializes it through the generic path (dispatch_single_parent) at every width.
    if (!invariant_routes_active()) {
        const detail::Q8GdnInputConvPlan plan =
            resolve_q8_conv_plan(geometry.width, geometry.batch);
        if (plan.schedule == detail::Q8GdnInputConvScheduleId::DecodeFused) {
            detail::q8_gdn_input_decode_conv_snapshot_launch(
                x, weight, conv_weight, conv_states, valid_columns, initial_state_slots,
                snapshot_base_slots, query, key, value, z, stream);
            return;
        }
        if (plan.schedule == detail::Q8GdnInputConvScheduleId::SplitKMmaFused) {
            detail::q8_gdn_input_splitk_conv_snapshot_launch(
                x, weight, conv_weight, conv_states, valid_columns, initial_state_slots,
                snapshot_base_slots, query, key, value, z, stream);
            return;
        }
    }

    auto scope                 = workspace.scope();
    ProjectedWorkspace scratch = allocate_projected_workspace(workspace, kChannels, geometry.width);
    gdn_input_proj(x, weight, scratch.projected, z, stream);
    detail::gdn_projected_conv_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                               valid_columns, initial_state_slots,
                                               snapshot_base_slots, query, key, value, stream);
}

void dispatch_single_parent_record(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   const Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& conv_record,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   LinearPolicy policy, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    validate_policy(policy);

    if (weight.qtype == QType::NVFP4) {
#ifdef NINFERNO_WITHOUT_NVFP4
        detail::no_native_nvfp4("gdn_input_proj");
#else
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_record_input(x, kHidden);
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj_conv_record");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "nvfp4 gdn_input_proj_conv_record: unsupported weight shape");
        }
        require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                kChannels, geometry);
        require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "conv record");
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                            "z");
        require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                  conv_record, query, key, value, z, workspace);

        const detail::Nvfp4GdnConvPlan plan =
            detail::nvfp4_gdn_conv_resolve_plan(policy, geometry.width, geometry.batch);
        if (plan.schedule == detail::Nvfp4GdnConvScheduleId::Materialized && geometry.batch > 1) {
            compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                           conv_record, query, key, value, z, geometry, workspace, stream,
                           [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                               gdn_input_proj(x_flat, weight, record_flat, z_flat, policy,
                                              workspace, stream);
                           });
            return;
        }
        if (plan.schedule == detail::Nvfp4GdnConvScheduleId::SmallTFusedA16) {
            detail::nvfp4_gdn_record_small_t_launch(x, weight, conv_weight, conv_states,
                                                    valid_columns, initial_state_slots, conv_record,
                                                    query, key, value, z, stream);
            return;
        }

        auto scope = workspace.scope();
        gdn_input_proj(x, weight, conv_record, z, policy, workspace, stream);
        detail::nvfp4_gdn_record_post_launch(conv_record, conv_weight, conv_states, valid_columns,
                                             initial_state_slots, query, key, value, stream);
        return;
#endif
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_record_input(x, kHidden);
        detail::validate_fp8_weight(weight, "fp8 gdn_input_proj_conv_record");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 gdn_input_proj_conv_record: unsupported weight shape");
        }
        require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                kChannels, geometry);
        require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "conv record");
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                            "z");
        require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                  conv_record, query, key, value, z, workspace);
        const std::array<const Tensor*, 10> tensors{
            &x,           &conv_weight, &conv_states, &valid_columns, &initial_state_slots,
            &conv_record, &query,       &key,         &value,         &z};
        require_parent_nonoverlap(weight, tensors, workspace, "fp8 gdn_input_proj_conv_record");
        detail::fp8_gdn_record_dispatch(x, weight, conv_weight, conv_states, valid_columns,
                                        initial_state_slots, conv_record, query, key, value, z,
                                        policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kQueryRows = 2048;
    constexpr std::int32_t kKeyRows   = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    const ConvGeometry geometry       = require_record_input(x, kHidden);
    require_q8_rowsplit(weight, kChannels + kZRows, "query/key/value/z weight");
    require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots, kChannels,
                            geometry);
    require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "conv record");
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "z");
    require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                              conv_record, query, key, value, z, workspace);

    if (geometry.batch > 1 || invariant_routes_active()) {
        compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots, conv_record,
                       query, key, value, z, geometry, workspace, stream,
                       [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                           gdn_input_proj(x_flat, weight, record_flat, z_flat, stream);
                       });
        return;
    }
    const detail::Q8GdnInputConvPlan plan = resolve_q8_conv_plan(geometry.width, geometry.batch);
    if (plan.schedule != detail::Q8GdnInputConvScheduleId::SplitKMmaFused) {
        throw std::logic_error("Q8 ReplaySSM record domain selected a non-record schedule");
    }
    detail::q8_gdn_input_splitk_conv_record_launch(x, weight, conv_weight, conv_states,
                                                   valid_columns, initial_state_slots, conv_record,
                                                   query, key, value, z, stream);
}

} // namespace

void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, cudaStream_t stream) {
    gdn_input_proj(x, qk_weight, value_z_weight, qkv, z, nullptr, stream);
}

void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    gdn_input_proj(x, qk_weight, value_z_weight, qkv, z, &workspace, stream);
}

std::size_t gdn_input_proj_pair_workspace_capacity_bytes(std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gdn_input_proj workspace: invalid token interval");
    }
    // The Volta tiled route takes each parent through CUTLASS, which decodes it here.
    const std::size_t volta =
        std::max(detail::generic::volta_prefill_workspace_bytes(QType::Q4_G64_FP16, kPairQkRows,
                                                                kPairHidden, max_tokens, false),
                 detail::generic::volta_prefill_workspace_bytes(QType::Q5_G64_FP16, kPairParentRows,
                                                                kPairHidden, max_tokens, false));
    return std::max(volta, pair_prefill_a8_workspace_bytes(min_tokens, max_tokens));
}

void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, WorkspaceArena* workspace, cudaStream_t stream) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQkRows     = 4096;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kQkvRows    = kQkRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const std::int32_t cols            = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_rowsplit(qk_weight, QType::Q4_G64_FP16, kQkRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5_G64_FP16, kParentRows, "value/z weight");

    if (pair_prefill_a8(x, qk_weight, value_z_weight, qkv, z, workspace, stream)) return;
    if (detail::generic::volta_tiled_route(qk_weight, LinearPolicy::A16Only, cols) &&
        detail::generic::volta_tiled_route(value_z_weight, LinearPolicy::A16Only, cols)) {
        volta_tiled_parent(x, qk_weight, {&qkv}, {kQkRows}, {0}, workspace, stream);
        volta_tiled_parent(x, value_z_weight, {&qkv, &z}, {kValueRows, kZRows}, {kQkRows, 0},
                           workspace, stream);
        return;
    }
    detail::q4_q5_gdn_input_dispatch(x, qk_weight, value_z_weight, qkv, z, stream);
}

std::size_t gdn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gdn_input_proj workspace: invalid token interval");
    }
    if (parent_qtype == QType::NVFP4) {
#ifdef NINFERNO_WITHOUT_NVFP4
        detail::no_native_nvfp4("gdn_input_proj");
#else
        if (parent_rows != detail::Nvfp4N16384K5120::kOutputRows ||
            input_rows != detail::Nvfp4N16384K5120::kInputRows) {
            throw std::invalid_argument("gdn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_gdn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
#endif
    }
    if (parent_qtype == QType::FP8_E4M3FN_ROW_BF16) {
        if (parent_rows != detail::Fp8N16384K5120::kOutputRows ||
            input_rows != detail::Fp8N16384K5120::kInputRows) {
            throw std::invalid_argument("gdn_input_proj workspace: unsupported FP8 profile");
        }
        return detail::fp8_gdn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (parent_qtype == QType::Q8_G32_FP16 && parent_rows == 12288 && input_rows == 2048) {
        (void)detail::q8_gdn_input_resolve_plan(
            {input_rows, 8192, 4096, parent_rows, input_rows, min_tokens});
        (void)detail::q8_gdn_input_resolve_plan(
            {input_rows, 8192, 4096, parent_rows, input_rows, max_tokens});
        if (allows_a8(policy) && max_tokens >= detail::generic::kQ8PrefillA8Tokens) {
            return detail::generic::workspace_capacity_bytes(
                {parent_qtype, policy, parent_rows, input_rows, detail::generic::Epilogue::Store},
                nullptr, min_tokens, max_tokens);
        }
        // A decode call may take the generic R8 (owned_decode_r8).
        if (allows_a8(policy) &&
            min_tokens <= detail::generic::kR8DecodeTokens) {
            return detail::generic::workspace_capacity_bytes(
                {parent_qtype, policy, parent_rows, input_rows, detail::generic::Epilogue::Store},
                nullptr, min_tokens, detail::generic::kR8DecodeTokens);
        }
        return 0;
    }
    throw std::invalid_argument("gdn_input_proj workspace: unsupported parent profile");
}

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_value_z_weight, qkv, z, policy, &workspace, stream);
}

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream) {
    dispatch_single_parent(x, query_key_value_z_weight, qkv, z, LinearPolicy::A16Only, nullptr,
                           stream);
}

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    const bool q4_q5 = query_rows == 2048 && key_rows == 2048 && value_rows == 6144;
    const bool q8    = query_rows == 2048 && key_rows == 2048 && value_rows == 4096;
    if (!q4_q5 && !q8) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot workspace: unregistered shape");
    }
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    const std::int32_t channels = query_rows + key_rows + value_rows;
    if (batch_size > 1) {
        if (q4_q5) {
            (void)resolve_q4_q5_conv_plan(min_width, batch_size);
            (void)resolve_q4_q5_conv_plan(max_width, batch_size);
        } else {
            (void)resolve_q8_conv_plan(min_width, batch_size);
            (void)resolve_q8_conv_plan(max_width, batch_size);
        }
        return composed_snapshot_capacity(
            channels, batch_size * max_width,
            q4_q5 ? pair_prefill_a8_workspace_bytes(batch_size * min_width, batch_size * max_width)
                  : 0);
    }

    std::int32_t largest_materialized_width = 0;
    if (q4_q5) {
        // The widest width the plan materializes (the projection-epilogue fused route covers
        // only some narrow widths, and which ones follows the projection route).
        for (std::int32_t width = min_width; width <= max_width; ++width) {
            if (resolve_q4_q5_conv_plan(width, 1).schedule ==
                detail::Q4Q5GdnInputConvScheduleId::Materialized) {
                largest_materialized_width = width;
            }
        }
    } else {
        (void)resolve_q8_conv_plan(min_width, 1);
        (void)resolve_q8_conv_plan(max_width, 1);
        // The invariant profile materializes every width (dispatch_single_parent_snapshot).
        if (max_width >= 17 || invariant_routes_active()) {
            largest_materialized_width = max_width;
        }
    }
    if (largest_materialized_width == 0) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected_workspace(layout, channels, largest_materialized_width);
    if (q4_q5) {
        if (const std::size_t a8 = pair_prefill_a8_workspace_bytes(min_width, max_width); a8 != 0) {
            (void)layout.alloc_bytes(a8);
        }
    }
    return layout.peak_bytes(1);
}

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(policy);
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    if (parent_qtype == QType::FP8_E4M3FN_ROW_BF16 &&
        parent_rows == detail::Fp8N16384K5120::kOutputRows &&
        input_rows == detail::Fp8N16384K5120::kInputRows) {
        return detail::fp8_gdn_snapshot_workspace_capacity_bytes(policy, batch_size, min_width,
                                                                 max_width);
    }
#ifdef NINFERNO_WITHOUT_NVFP4
    if (parent_qtype == QType::NVFP4) { detail::no_native_nvfp4("gdn_input_proj"); }
#endif
    if (parent_qtype != QType::NVFP4 || parent_rows != detail::Nvfp4N16384K5120::kOutputRows ||
        input_rows != detail::Nvfp4N16384K5120::kInputRows) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_snapshot workspace: unsupported single-parent profile");
    }
#ifdef NINFERNO_WITHOUT_NVFP4
    detail::no_native_nvfp4("gdn_input_proj");
#else
    if (batch_size == 1) {
        return detail::nvfp4_gdn_snapshot_workspace_capacity_bytes(policy, min_width, max_width);
    }

    (void)detail::nvfp4_gdn_conv_resolve_plan(policy, min_width, batch_size);
    (void)detail::nvfp4_gdn_conv_resolve_plan(policy, max_width, batch_size);

    constexpr std::int32_t kChannels       = 10240;
    const std::int32_t aggregate_columns   = batch_size * max_width;
    const std::size_t projection_workspace = gdn_input_proj_workspace_capacity_bytes(
        parent_qtype, parent_rows, input_rows, policy, batch_size * min_width, aggregate_columns);
    return composed_snapshot_capacity(kChannels, aggregate_columns, projection_workspace);
#endif
}

std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    const bool q4_q5 = query_rows == 2048 && key_rows == 2048 && value_rows == 6144;
    const bool q8    = query_rows == 2048 && key_rows == 2048 && value_rows == 4096;
    if (!q4_q5 && !q8) {
        throw std::invalid_argument("gdn_input_proj_conv_record workspace: unregistered shape");
    }
    require_record_capacity_domain(batch_size, min_width, max_width);
    if (q4_q5) {
        (void)resolve_q4_q5_conv_plan(min_width, batch_size);
        (void)resolve_q4_q5_conv_plan(max_width, batch_size);
        return pair_prefill_a8_workspace_bytes(batch_size * min_width, batch_size * max_width);
    }
    (void)resolve_q8_conv_plan(min_width, batch_size);
    (void)resolve_q8_conv_plan(max_width, batch_size);
    return 0;
}

std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(policy);
    require_record_capacity_domain(batch_size, min_width, max_width);
    if (parent_qtype == QType::FP8_E4M3FN_ROW_BF16 &&
        parent_rows == detail::Fp8N16384K5120::kOutputRows &&
        input_rows == detail::Fp8N16384K5120::kInputRows) {
        return detail::fp8_gdn_record_workspace_capacity_bytes(policy, batch_size, min_width,
                                                               max_width);
    }
#ifdef NINFERNO_WITHOUT_NVFP4
    if (parent_qtype == QType::NVFP4) { detail::no_native_nvfp4("gdn_input_proj"); }
#endif
    if (parent_qtype != QType::NVFP4 || parent_rows != detail::Nvfp4N16384K5120::kOutputRows ||
        input_rows != detail::Nvfp4N16384K5120::kInputRows) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_record workspace: unsupported single-parent profile");
    }
#ifdef NINFERNO_WITHOUT_NVFP4
    detail::no_native_nvfp4("gdn_input_proj");
#else
    const detail::Nvfp4GdnConvPlan minimum_plan =
        detail::nvfp4_gdn_conv_resolve_plan(policy, min_width, batch_size);
    const detail::Nvfp4GdnConvPlan maximum_plan =
        detail::nvfp4_gdn_conv_resolve_plan(policy, max_width, batch_size);
    if (batch_size == 1) {
        if (minimum_plan.schedule == detail::Nvfp4GdnConvScheduleId::DecodeFusedA16) {
            throw std::logic_error("ReplaySSM record planner admitted NVFP4 decode");
        }
        if (maximum_plan.schedule == detail::Nvfp4GdnConvScheduleId::SmallTFusedA16) { return 0; }
        // The record materializes from width 4 under A4 routes and at every width under
        // A16Only (whose projection needs no workspace).
        return detail::nvfp4_gdn_input_workspace_capacity_bytes(
            policy, std::min(std::max(min_width, 4), max_width), max_width);
    }
    return detail::nvfp4_gdn_input_workspace_capacity_bytes(policy, batch_size * min_width,
                                                            batch_size * max_width);
#endif
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_z_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQueryRows  = 2048;
    constexpr std::int32_t kKeyRows    = 2048;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const ConvGeometry geometry        = require_snapshot_input(x, kHidden);
    require_rowsplit(qk_weight, QType::Q4_G64_FP16, kQueryRows + kKeyRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5_G64_FP16, kParentRows, "value/z weight");
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, kChannels, geometry);
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_snapshot",
                        "z");

    if (geometry.batch > 1) {
        compose_batched_snapshot(
            x, conv_weight, conv_states, valid_columns, initial_state_slots, snapshot_base_slots,
            query, key, value, z, kQueryRows, kKeyRows, kValueRows, geometry, ws, stream,
            [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                gdn_input_proj(x_flat, qk_weight, value_z_weight, projected, z_flat, &ws, stream);
            });
        return;
    }

    const detail::Q4Q5GdnInputConvPlan plan =
        resolve_q4_q5_conv_plan(geometry.width, geometry.batch);
    const bool prefill_a8 = detail::generic::q8_prefill_a8(
        qk_weight, pair_prefill_a8_policy(qk_weight.qtype), geometry.width);
    if (!prefill_a8 &&
        plan.schedule == detail::Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused) {
        detail::q4_q5_gdn_input_conv_snapshot_launch(
            x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns,
            initial_state_slots, snapshot_base_slots, query, key, value, z, stream);
        return;
    }

    auto scope                 = ws.scope();
    ProjectedWorkspace scratch = allocate_projected_workspace(ws, kChannels, geometry.width);
    Tensor x_matrix            = flatten_columns(x, kHidden, geometry);
    Tensor z_matrix            = flatten_columns(z, kZRows, geometry);
    gdn_input_proj(x_matrix, qk_weight, value_z_weight, scratch.projected, z_matrix, &ws, stream);
    detail::gdn_projected_conv_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                               valid_columns, initial_state_slots,
                                               snapshot_base_slots, query, key, value, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& qk_weight,
                                const Weight& value_z_weight, const Tensor& conv_weight,
                                const Tensor& conv_states, const Tensor& valid_columns,
                                const Tensor& initial_state_slots, Tensor& conv_record,
                                Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQueryRows  = 2048;
    constexpr std::int32_t kKeyRows    = 2048;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const ConvGeometry geometry        = require_record_input(x, kHidden);
    require_rowsplit(qk_weight, QType::Q4_G64_FP16, kQueryRows + kKeyRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5_G64_FP16, kParentRows, "value/z weight");
    require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots, kChannels,
                            geometry);
    require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "conv record");
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "z");
    require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                              conv_record, query, key, value, z, workspace);

    const detail::Q4Q5GdnInputConvPlan plan =
        resolve_q4_q5_conv_plan(geometry.width, geometry.batch);
    const bool prefill_a8 = detail::generic::q8_prefill_a8(
        qk_weight, pair_prefill_a8_policy(qk_weight.qtype), geometry.aggregate_columns);
    if (!prefill_a8 &&
        plan.schedule == detail::Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused) {
        detail::q4_q5_gdn_input_conv_record_launch(x, qk_weight, value_z_weight, conv_weight,
                                                   conv_states, valid_columns, initial_state_slots,
                                                   conv_record, query, key, value, z, stream);
        return;
    }
    compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots, conv_record,
                   query, key, value, z, geometry, workspace, stream,
                   [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                       gdn_input_proj(x_flat, qk_weight, value_z_weight, record_flat, z_flat,
                                      &workspace, stream);
                   });
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    dispatch_single_parent_snapshot(x, query_key_value_z_weight, conv_weight, conv_states,
                                    valid_columns, initial_state_slots, snapshot_base_slots, query,
                                    key, value, z, policy, ws, stream);
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    dispatch_single_parent_snapshot(x, query_key_value_z_weight, conv_weight, conv_states,
                                    valid_columns, initial_state_slots, snapshot_base_slots, query,
                                    key, value, z, LinearPolicy::A16Only, ws, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    dispatch_single_parent_record(x, query_key_value_z_weight, conv_weight, conv_states,
                                  valid_columns, initial_state_slots, conv_record, query, key,
                                  value, z, policy, workspace, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent_record(x, query_key_value_z_weight, conv_weight, conv_states,
                                  valid_columns, initial_state_slots, conv_record, query, key,
                                  value, z, LinearPolicy::A16Only, workspace, stream);
}

} // namespace ninferno::ops
