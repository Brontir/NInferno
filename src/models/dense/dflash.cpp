// The dense family's side of the Program's DFlash rounds (the drafter's state, its context append,
// its block pass and the round around them). The math is DFlashRound's (models/speculation/
// dflash_round.cu, one sequence under ninferno-dense) over the Program's batched state: the
// drafter's sliding layers keep their keys and values in fixed rings of each sequence's state
// image, its full layers in a paged pool of their own, and the committed target features of each
// lane wait in the pending buffer until the next round appends them.

#include "models/dense/program_execution.h"

#include "core/nvtx.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "ninferno/ops/argmax.h"
#include "ninferno/ops/dense_fused.h"
#include "ninferno/ops/embedding.h"
#include "ninferno/ops/kv_cache_append.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/linear_add.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/prepare_masked_block.h"
#include "ninferno/ops/prepare_ragged_prefix.h"
#include "ninferno/ops/rmsnorm.h"
#include "ninferno/ops/scalar.h"
#include "ninferno/ops/scatter.h"
#include "ninferno/ops/silu_mul.h"
#include "ninferno/ops/sliding_window_attention.h"
#include "ninferno/ops/softmax_attention.h"
#include "ninferno/ops/speculative_round.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::models::dense::detail {
namespace {

using speculation::DFlashConfig;
using speculation::DFlashWeights;
using speculation::Projection;

constexpr std::size_t kAlign = 256;

// An arena allocation of `elements` values of `bytes` each, with its alignment.
std::size_t carved(std::int64_t elements, std::size_t bytes = 2) {
    return static_cast<std::size_t>(elements) * bytes + kAlign;
}

// One drafter projection under the policy its loader prepared (every activation BF16).
void project(WorkspaceArena& work, const Tensor& x, const Projection& p, Tensor& out,
             cudaStream_t stream) {
    auto scope = work.scope();
    ops::linear(x, p.weight, out, p.policy, work, stream);
}

// sum += the projection of x, rounded to BF16 once (the accumulator is the GEMM's).
void project_add(WorkspaceArena& work, const Tensor& x, const Projection& p, Tensor& sum,
                 cudaStream_t stream) {
    auto scope = work.scope();
    ops::linear_add(x, p.weight, sum, p.policy, work, stream);
}

// A split projection's slice, stored (the first tap) or added (the others).
std::size_t slice_bytes(const Projection& p, std::int32_t tokens) {
    return std::max(ops::linear_workspace_capacity_bytes(p, 1, tokens),
                    ops::linear_add_workspace_capacity_bytes(p, 1, tokens));
}

// Rows [first, first + rows) of every column of a packed projection's output BF16 [N, T], into the
// contiguous BF16 [rows, T] storage of `out`.
void copy_rows(const Tensor& packed, std::int32_t first, std::int32_t rows, Tensor& out,
               cudaStream_t stream) {
    const std::size_t row_bytes = static_cast<std::size_t>(rows) * dtype_size(DType::BF16);
    const auto* source =
        static_cast<const std::byte*>(packed.data) + static_cast<std::size_t>(first) * 2U;
    CUDA_CHECK(cudaMemcpy2DAsync(out.data, row_bytes, source,
                                 static_cast<std::size_t>(packed.nb[1]), row_bytes,
                                 static_cast<std::size_t>(packed.ne[1]), cudaMemcpyDeviceToDevice,
                                 stream));
}

} // namespace

ops::DenseQkvPrepare DenseExecution::qkv_prepare(const speculation::DFlashLayer& weights,
                                                const Tensor& positions) const {
    const DFlashConfig& dc = drafter_->config();
    ops::DenseQkvPrepare prep;
    prep.k_bias       = weights.key_bias;
    prep.v_bias       = weights.value_bias;
    prep.qk_norm      = ops::QkNorm::Head;
    prep.k_norm       = weights.key_norm;
    prep.eps          = dc.rms_norm_eps;
    prep.inv_freq     = drafter_->weights().rope_inv_freq;
    prep.rotary_dim   = dc.head_dim;
    prep.rotary_scale = dc.rope_attention_scaling;
    prep.positions    = positions;
    return prep;
}

void DenseExecution::prepare_keys_values(const speculation::DFlashLayer& weights,
                                         const Tensor& key_src, const Tensor& value_src,
                                         const Tensor& positions, Tensor key, Tensor value,
                                         cudaStream_t stream) const {
    ops::DenseQkvPrepare prep = qkv_prepare(weights, positions);
    prep.k_src                = key_src;
    prep.v_src                = value_src;
    prep.k_out                = key;
    prep.v_out                = value;
    ops::dense_qkv_prepare(prep, stream);
}

void DenseExecution::check_drafter() const {
    const DFlashConfig& dc = drafter_->config();
    if (dc.hidden_size != c_.hidden_size || dc.vocab_size != c_.vocab_size ||
        dc.target_layers != c_.layers ||
        drafter_->weights().feature_projection.weight.k !=
            static_cast<std::int64_t>(dc.taps.size()) * c_.hidden_size) {
        throw std::invalid_argument("dflash: the drafter does not pair with this target");
    }
    std::vector<bool> tapped(static_cast<std::size_t>(c_.layers), false);
    for (const std::int32_t tap : dc.taps) {
        if (tap < 0 || tap >= c_.layers || tapped[static_cast<std::size_t>(tap)]) {
            throw std::invalid_argument("dflash: the drafter's tapped layers are not distinct "
                                        "layers of this target");
        }
        tapped[static_cast<std::size_t>(tap)] = true;
    }
    if (dc.block_size < 2 ||
        dc.block_size > static_cast<std::int32_t>(qwen3_5::kDFlashDecodeMaximumWidth)) {
        throw std::invalid_argument("dflash: the Program's round holds a block of 2 to " +
                                    std::to_string(qwen3_5::kDFlashDecodeMaximumWidth) +
                                    " columns; this drafter's is " +
                                    std::to_string(dc.block_size));
    }
    const bool sliding =
        std::any_of(dc.windows.begin(), dc.windows.end(), [](std::int32_t w) { return w != 0; });
    if (sliding && !dc.sliding_block) {
        throw std::invalid_argument("dflash: the Program's sliding drafter attention is the block "
                                    "mask; this drafter's sliding layers are causal in the block");
    }
    // The drafter's attention runs on the profiles the Program's Ops register: D=128 with 32
    // query and 8 key/value heads (a sliding layer's window 2048 or 4096), or D=64 with 64 and 8
    // on full layers only (gpt-oss's drafter), and a batch of at most 8 rows.
    static_assert(kMaximumConcurrency <= 8, "the drafter's attention Ops take at most 8 rows");
    const bool d128 = dc.head_dim == 128 && dc.query_heads == 32 && dc.kv_heads == 8 &&
                      (!sliding || dc.sliding_window == 2048 || dc.sliding_window == 4096);
    const bool d64  = dc.head_dim == 64 && dc.query_heads == 64 && dc.kv_heads == 8 && !sliding;
    if (!d128 && !d64) {
        throw std::invalid_argument(
            "dflash: the Program's drafter attention runs D=128 with 32 query and 8 key/value "
            "heads (sliding windows 2048 or 4096), or D=64 with 64 and 8 on full layers; this "
            "drafter has D=" +
            std::to_string(dc.head_dim) + ", " + std::to_string(dc.query_heads) + " query and " +
            std::to_string(dc.kv_heads) + " key/value heads, window " +
            std::to_string(dc.sliding_window));
    }
}

void DenseExecution::dflash_workspace(const SequencePlanImpl& plan, WorkspacePlan& out) const {
    const auto chunk   = static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const auto context = static_cast<std::int32_t>(plan.capacity);
    const auto window  = static_cast<std::int32_t>(plan.draft_window);
    const std::int32_t width = window + 1;
    // A prefill chunk's append follows its forward, beside the chunk's ids, positions and rows.
    out.dflash_context =
        dflash_context_bytes(chunk, 1, false) + 2 * carved(chunk, 4) + carved(2, 4);
    out.dflash_round   = 0;
    for (std::int32_t batch = 1;
         batch <= static_cast<std::int32_t>(qwen3_5::detail::speculative_round_batches(plan));
         ++batch) {
        // The round's phases each start from an empty arena: the append of the compacted
        // features, the block pass, the verify (its tapped features and the target's call), and
        // the accept.
        const std::int32_t columns = width * batch;
        const std::size_t verify =
            carved(std::int64_t(traits_.draft_feature_rows) * columns) +
            DensePass::workspace_bytes(model_, context, columns, columns);
        const std::size_t accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            traits_.public_tokens, 1, window, batch, batch);
        const std::size_t append   = dflash_context_bytes(width, batch, true);
        const std::size_t proposal = dflash_proposal_bytes(plan.capacity, window, batch);
        out.dflash_round_append    = std::max(out.dflash_round_append, append);
        out.dflash_round           = std::max({out.dflash_round, append, proposal, verify, accept});
    }
}

std::size_t DenseExecution::dflash_context_bytes(std::int32_t width, std::int32_t batch,
                                                 bool compact_input) const {
    const DFlashConfig& dc     = drafter_->config();
    const DFlashWeights& dw    = drafter_->weights();
    const std::int32_t columns = width * batch;
    const std::int64_t h       = dc.hidden_size;
    const std::int64_t kv_rows = std::int64_t(dc.kv_heads) * dc.head_dim;
    std::size_t bytes =
        compact_input ? carved(std::int64_t(traits_.draft_feature_rows) * columns) : 0;
    bytes += 2 * carved(h * columns);       // the projected and the normed context
    bytes += 4 * carved(kv_rows * columns); // one layer's projected key and value, key and value
    std::size_t op = 0;
    if (!split_features_) {
        op = ops::linear_workspace_capacity_bytes(dw.feature_projection, 1, columns);
    } else if (compact_input) {
        // A round's features through the split projection: a tap's rows at a time.
        bytes += carved(h * columns);
        op = slice_bytes(dw.feature_slices.front(), columns);
    }
    if (dw.context_kv) {
        bytes += carved(2 * kv_rows * dc.layers * columns);
        op = std::max(op, ops::linear_workspace_capacity_bytes(*dw.context_kv, 1, columns));
    } else {
        for (const auto& layer : dw.layers) {
            op = std::max({op, ops::linear_workspace_capacity_bytes(layer.context_key, 1, columns),
                           ops::linear_workspace_capacity_bytes(layer.context_value, 1, columns)});
        }
    }
    return bytes + op;
}

std::size_t DenseExecution::dflash_proposal_bytes(std::uint32_t capacity, std::int32_t drafts,
                                                  std::int32_t batch) const {
    const DFlashConfig& dc     = drafter_->config();
    const DFlashWeights& dw    = drafter_->weights();
    const std::int32_t block   = dc.block_size;
    const std::int32_t columns = block * batch;
    const std::int32_t drafted = drafts * batch;
    const std::int64_t h       = dc.hidden_size;
    const std::int64_t q_rows  = std::int64_t(dc.query_heads) * dc.head_dim;
    const std::int64_t kv_rows = std::int64_t(dc.kv_heads) * dc.head_dim;
    const std::int64_t inter   = dc.intermediate_size;
    // The block's valid columns, ids and positions, its residual stream and its normed input.
    const std::size_t base =
        carved(batch, 4) + 2 * carved(columns, 4) + 2 * carved(h * columns);
    // Projected q/k/v, prepared q/k/v, the attention context and the branch.
    std::size_t attention = 3 * carved(q_rows * columns) + 4 * carved(kv_rows * columns) +
                            carved(h * columns);
    std::size_t mlp = carved(h * columns) + 3 * carved(inter * columns);
    const std::size_t tail =
        2 * carved(h * drafted) + carved(std::int64_t(c_.vocab_size) * drafted);
    std::size_t op = 0;
    const ops::AttentionHeadGeometry geometry{dc.head_dim, dc.query_heads, dc.kv_heads};
    if (drafter_local_layers_ != 0) {
        op = ops::sliding_window_attention_workspace_capacity_bytes(
            geometry, static_cast<std::uint32_t>(dc.sliding_window), {0, capacity}, block, block,
            batch);
    }
    if (drafter_full_layers_ != 0) {
        op = std::max(op, ops::context_softmax_attention_workspace_capacity_bytes(
                              geometry, {0, capacity}, block, block, batch));
    }
    for (const auto& layer : dw.layers) {
        if (layer.qkv) {
            op = std::max(op, ops::linear_workspace_capacity_bytes(*layer.qkv, 1, columns));
        } else {
            op = std::max({op, ops::linear_workspace_capacity_bytes(layer.query, 1, columns),
                           ops::linear_workspace_capacity_bytes(layer.key, 1, columns),
                           ops::linear_workspace_capacity_bytes(layer.value, 1, columns)});
        }
        if (layer.gate_up) {
            op = std::max(op, ops::linear_workspace_capacity_bytes(*layer.gate_up, 1, columns));
        } else {
            op = std::max({op, ops::linear_workspace_capacity_bytes(layer.gate, 1, columns),
                           ops::linear_workspace_capacity_bytes(layer.up, 1, columns)});
        }
        op = std::max({op, ops::linear_workspace_capacity_bytes(layer.output, 1, columns),
                       ops::linear_workspace_capacity_bytes(layer.down, 1, columns)});
    }
    const Projection& head = model_.weights().output_head;
    op                     = std::max(op, ops::linear_workspace_capacity_bytes(head, 1, drafted));
    return base + std::max({attention, mlp, tail}) + op;
}

FeatureTap DenseExecution::feature_tap(const Tensor& sink) const {
    return FeatureTap{.slots = tap_slots_, .sink = sink};
}

FeatureTap DenseExecution::prefill_tap(const Tensor& sink, WorkspaceArena& work,
                                       cudaStream_t stream) const {
    FeatureTap tap = feature_tap(sink);
    if (!split_features_) { return tap; }
    const std::int32_t first = tap_order_.front();
    tap.project = [this, sink, &work, stream, first](std::int32_t slot, const Tensor& x) {
        const auto& slices      = drafter_->weights().feature_slices;
        const Projection& slice = slices[static_cast<std::size_t>(slot)];
        Tensor sum              = sink;
        if (slot == first) {
            project(work, x, slice, sum, stream);
        } else {
            project_add(work, x, slice, sum, stream);
        }
    };
    return tap;
}

std::int32_t DenseExecution::prefill_feature_rows() const {
    return split_features_ ? c_.hidden_size : traits_.draft_feature_rows;
}

std::size_t DenseExecution::prefill_tap_bytes(std::int32_t chunk) const {
    if (!split_features_) { return 0; }
    return slice_bytes(drafter_->weights().feature_slices.front(), chunk);
}

void DenseExecution::project_features(WorkspaceArena& work, const Tensor& features, Tensor& out,
                                      cudaStream_t stream) const {
    const auto& slices         = drafter_->weights().feature_slices;
    const std::int32_t h       = c_.hidden_size;
    const std::int32_t columns = features.ne[1];
    auto scope                 = work.scope();
    Tensor rows                = work.alloc(DType::BF16, {h, columns});
    for (const std::int32_t slot : tap_order_) {
        const Projection& slice = slices[static_cast<std::size_t>(slot)];
        copy_rows(features, slot * h, h, rows, stream);
        if (slot == tap_order_.front()) {
            project(work, rows, slice, out, stream);
        } else {
            project_add(work, rows, slice, out, stream);
        }
    }
}

void DenseExecution::append_context(program::ExecutionCore& core, DFlashPersistentState& dflash,
                                    const Tensor& features, const Tensor& positions,
                                    const Tensor& commit_counts, const Tensor& lanes,
                                    const Tensor& table_rows,
                                    ops::KVCacheAppendPrefixExecutionEnvelope envelope,
                                    bool projected) const {
    const DFlashConfig& dc     = drafter_->config();
    const DFlashWeights& dw    = drafter_->weights();
    WorkspaceArena& work       = core.work;
    const cudaStream_t s       = core.device.stream;
    const std::int32_t width   = features.ne[1];
    const std::int32_t batch   = features.ne[2];
    const std::int32_t kv_rows = dc.kv_heads * dc.head_dim;
    nvtx::ScopedRange range(nvtx::Name::DFlashContextAppend, nvtx::Category::DFlash,
                            static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(batch));
    if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
        features.ne[0] != (projected ? c_.hidden_size : traits_.draft_feature_rows) ||
        (projected && batch != 1) || features.ne[3] != 1 ||
        !features.is_contiguous() || positions.dtype != DType::I32 || positions.ne[0] != width ||
        positions.ne[1] != batch || !positions.is_contiguous() ||
        commit_counts.dtype != DType::I32 || commit_counts.ne[0] != batch ||
        lanes.dtype != DType::I32 || lanes.ne[0] != batch || table_rows.dtype != DType::I32 ||
        table_rows.ne[0] != batch) {
        throw std::invalid_argument("DFlash context append inputs are invalid");
    }
    // A single sequence's append wider than the rings (a long prefill chunk) commits only its last
    // window columns to them, at their own positions; the full layers take every column.
    const std::int32_t window = dc.sliding_window;
    const bool oversized      = drafter_local_layers_ != 0 && batch == 1 && width > window;
    if (oversized && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                      envelope.max_count != static_cast<std::uint32_t>(width))) {
        throw std::invalid_argument(
            "DFlash oversized local append requires an exact full-prefix commit");
    }
    const std::int32_t skipped = oversized ? width - window : 0;
    Tensor local_counts        = commit_counts;
    auto local_envelope        = envelope;
    if (oversized) {
        if (!core.io.dflash_prefill) {
            throw std::logic_error("DFlash prefill count storage is unavailable");
        }
        local_counts = core.io.dflash_prefill->local_append_count;
        ops::set_i32_scalar(local_counts, window, s);
        local_envelope = {static_cast<std::uint32_t>(window), static_cast<std::uint32_t>(window)};
    }

    // The context of every column some layer keeps: context_norm(feature_projection(features)).
    const std::int32_t first   = drafter_full_layers_ == 0 ? skipped : 0;
    const std::int32_t kept    = width - first;
    const std::int32_t columns = kept * batch;
    const Tensor input         = features.slice(1, first, kept).view({features.ne[0], columns});
    Tensor projection;
    if (projected) {
        projection = input;
    } else {
        projection = work.alloc(DType::BF16, {dc.hidden_size, columns});
        if (split_features_) {
            project_features(work, input, projection, s);
        } else {
            project(work, input, dw.feature_projection, projection, s);
        }
    }
    Tensor context = work.alloc(DType::BF16, {dc.hidden_size, columns});
    ops::rmsnorm(projection, dw.context_norm, dc.rms_norm_eps, false, context, s);
    // Every layer's keys and values in one projection when the artifact packed them, rows
    // [k0; v0; k1; v1; ...].
    Tensor packed;
    if (dw.context_kv) {
        packed = work.alloc(DType::BF16, {2 * kv_rows * dc.layers, columns});
        project(work, context, *dw.context_kv, packed, s);
    }
    for (std::int32_t layer = 0; layer < dc.layers; ++layer) {
        auto scope          = work.scope();
        const auto& weights = dw.layers[static_cast<std::size_t>(layer)];
        const bool local    = dc.windows[static_cast<std::size_t>(layer)] != 0;
        // A local layer of an oversized append reads the last window columns (batch 1).
        const std::int32_t offset        = local ? skipped : 0;
        const std::int32_t layer_width   = width - offset;
        const std::int32_t layer_columns = layer_width * batch;
        Tensor value = work.alloc(DType::BF16, {dc.head_dim, dc.kv_heads, layer_width, batch});
        Tensor key   = work.alloc(DType::BF16, {dc.head_dim, dc.kv_heads, layer_width, batch});
        // The projected keys and values: row slices of the packed projection, or their own.
        Tensor key_src, value_src;
        if (dw.context_kv) {
            const Tensor source = packed.slice(1, offset - first, layer_columns);
            key_src             = source.slice(0, 2 * layer * kv_rows, kv_rows);
            value_src           = source.slice(0, (2 * layer + 1) * kv_rows, kv_rows);
        } else {
            const Tensor source = context.slice(1, offset - first, layer_columns);
            key_src             = work.alloc(DType::BF16, {kv_rows, layer_columns});
            value_src           = work.alloc(DType::BF16, {kv_rows, layer_columns});
            project(work, source, weights.context_key, key_src, s);
            project(work, source, weights.context_value, value_src, s);
        }
        // Bias, the per-head key norm and the rotation, in one pass (no queries).
        const Tensor layer_positions = positions.slice(0, offset, layer_width);
        prepare_keys_values(weights, key_src, value_src, layer_positions.view({layer_columns}),
                            key.view({dc.head_dim, dc.kv_heads, layer_columns}),
                            value.view({dc.head_dim, dc.kv_heads, layer_columns}), s);
        const std::uint32_t index = drafter_index_[static_cast<std::size_t>(layer)];
        if (local) {
            ops::kv_cache_append_prefix(key, value, layer_positions, local_counts, lanes,
                                        local_envelope, dflash.local_layer(index), s);
        } else {
            ops::kv_cache_append_prefix(key, value, layer_positions, commit_counts, table_rows,
                                        envelope, dflash.full_batch_layer(index), s);
        }
    }
}

void DenseExecution::propose(program::DFlashBatchContext& state, std::int32_t batch_size,
                             std::uint32_t k, program::DFlashEnvelopes envelopes) const {
    const DFlashConfig& dc       = drafter_->config();
    const DFlashWeights& dw      = drafter_->weights();
    program::ExecutionCore& core = state.execution;
    WorkspaceArena& work         = core.work;
    const cudaStream_t s         = core.device.stream;
    qwen3_5::DFlashDecodeState& frame = state.frame;
    // The block the drafter was trained on, or only the anchor and the k drafts the round verifies
    // (program::dflash_draft_width: by default the block for one row, k + 1 columns from two).
    const std::int32_t block =
        program::dflash_draft_width(core, batch_size) == DFlashDraftWidth::Block
            ? dc.block_size
            : std::min(dc.block_size, static_cast<std::int32_t>(k) + 1);
    const std::int32_t columns   = block * batch_size;
    const std::int32_t drafted   = static_cast<std::int32_t>(k) * batch_size;
    const std::int32_t h         = dc.hidden_size;
    const std::int32_t d         = dc.head_dim;
    const std::int32_t q_rows    = dc.query_heads * d;
    const std::int32_t kv_rows   = dc.kv_heads * d;
    const float eps              = dc.rms_norm_eps;
    nvtx::ScopedRange range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                            static_cast<std::uint64_t>(columns));
    const Tensor anchors   = frame.anchors.slice(0, 0, batch_size);
    const Tensor frontiers = frame.execution_frontiers.slice(0, 0, batch_size);
    const Tensor lanes     = frame.state_destination_slots.slice(0, 0, batch_size);
    const Tensor full_rows = frame.dflash_kv_table_rows.slice(0, 0, batch_size);

    work.reset();
    // The anchor, then block - 1 mask tokens, at the row's frontier and on; the round verifies the
    // first k drafts.
    Tensor valid = work.alloc(DType::I32, {batch_size});
    ops::fill_i32(valid, block, s);
    Tensor ids       = work.alloc(DType::I32, {block, batch_size});
    Tensor positions = work.alloc(DType::I32, {block, batch_size});
    ops::prepare_masked_block(anchors, frontiers, valid, dc.mask_token, ids, positions, s);
    Tensor residual = work.alloc(DType::BF16, {h, columns});
    // The target's table as the drafter reads it: its own rows, before any embedding norm the
    // target applies after it (Target::draft_embedding).
    ops::embedding(ids.view({columns}), model_.weights().token_embedding, residual, s);
    // The normed input of the next sublayer: each residual update writes it (residual_norm).
    Tensor hn = work.alloc(DType::BF16, {h, columns});
    ops::rmsnorm(residual, dw.layers.front().input_norm, eps, false, hn, s);

    const ops::AttentionHeadGeometry geometry{d, dc.query_heads, dc.kv_heads};
    const float scale = 1.0F / std::sqrt(static_cast<float>(d));
    for (std::int32_t layer = 0; layer < dc.layers; ++layer) {
        nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                      static_cast<std::uint64_t>(layer));
        const auto& weights = dw.layers[static_cast<std::size_t>(layer)];
        {
            nvtx::ScopedRange attention_range(nvtx::Name::DFlashAttention,
                                              nvtx::Category::Attention,
                                              static_cast<std::uint64_t>(layer));
            auto scope = work.scope();
            // The projected q/k/v: row slices of the packed projection, or each its own.
            Tensor query_src, key_src, value_src;
            if (weights.qkv) {
                Tensor packed = work.alloc(DType::BF16, {q_rows + 2 * kv_rows, columns});
                project(work, hn, *weights.qkv, packed, s);
                query_src = packed.slice(0, 0, q_rows);
                key_src   = packed.slice(0, q_rows, kv_rows);
                value_src = packed.slice(0, q_rows + kv_rows, kv_rows);
            } else {
                query_src = work.alloc(DType::BF16, {q_rows, columns});
                key_src   = work.alloc(DType::BF16, {kv_rows, columns});
                value_src = work.alloc(DType::BF16, {kv_rows, columns});
                project(work, hn, weights.query, query_src, s);
                project(work, hn, weights.key, key_src, s);
                project(work, hn, weights.value, value_src, s);
            }
            Tensor query = work.alloc(DType::BF16, {d, dc.query_heads, block, batch_size});
            Tensor key   = work.alloc(DType::BF16, {d, dc.kv_heads, block, batch_size});
            Tensor value = work.alloc(DType::BF16, {d, dc.kv_heads, block, batch_size});
            // Biases, the per-head q/k norms and the rotation, in one pass.
            ops::DenseQkvPrepare prep = qkv_prepare(weights, positions.view({columns}));
            prep.q_src                = query_src;
            prep.q_bias               = weights.query_bias;
            prep.q_norm               = weights.query_norm;
            prep.k_src                = key_src;
            prep.v_src                = value_src;
            prep.q_out                = query.view({d, dc.query_heads, columns});
            prep.k_out                = key.view({d, dc.kv_heads, columns});
            prep.v_out                = value.view({d, dc.kv_heads, columns});
            ops::dense_qkv_prepare(prep, s);
            // The block's own keys and values stay beside the committed context (the context
            // append of a later round writes the verified ones).
            Tensor attention = work.alloc(DType::BF16, {d, dc.query_heads, block, batch_size});
            const std::uint32_t index = drafter_index_[static_cast<std::size_t>(layer)];
            if (dc.windows[static_cast<std::size_t>(layer)] != 0) {
                ops::sliding_window_attention(query, key, value, positions, valid, lanes, geometry,
                                              static_cast<std::uint32_t>(dc.sliding_window), scale,
                                              state.dflash.local_layer(index), envelopes.local,
                                              work, attention, s);
            } else {
                ops::context_softmax_attention(query, key, value, frontiers, valid, full_rows,
                                               geometry, scale,
                                               state.dflash.full_batch_layer(index),
                                               envelopes.full, work, attention, s);
            }
            Tensor branch = work.alloc(DType::BF16, {h, columns});
            project(work, attention.view({q_rows, columns}), weights.output, branch, s);
            // residual += branch (+ the output bias); hn = post_attention_norm(residual).
            ops::residual_norm({.x          = residual,
                                .branch     = branch,
                                .bias       = weights.output_bias,
                                .eps        = eps,
                                .out_weight = weights.post_attention_norm,
                                .out        = hn},
                               s);
        }
        {
            nvtx::ScopedRange mlp_range(nvtx::Name::DFlashMlp, nvtx::Category::PostMixer,
                                        static_cast<std::uint64_t>(layer));
            auto scope               = work.scope();
            const std::int32_t inter = dc.intermediate_size;
            Tensor product = work.alloc(DType::BF16, {inter, columns});
            if (weights.gate_up) {
                Tensor gate_up = work.alloc(DType::BF16, {2 * inter, columns});
                project(work, hn, *weights.gate_up, gate_up, s);
                ops::silu_mul(gate_up.slice(0, 0, inter), gate_up.slice(0, inter, inter), product,
                              s);
            } else {
                Tensor gate = work.alloc(DType::BF16, {inter, columns});
                Tensor up   = work.alloc(DType::BF16, {inter, columns});
                project(work, hn, weights.gate, gate, s);
                project(work, hn, weights.up, up, s);
                ops::silu_mul(gate, up, product, s);
            }
            Tensor branch = work.alloc(DType::BF16, {h, columns});
            project(work, product, weights.down, branch, s);
            // residual += branch; hn = the next layer's input_norm(residual).
            const bool last = layer + 1 == dc.layers;
            ops::residual_norm(
                {.x      = residual,
                 .branch = branch,
                 .eps    = eps,
                 .out_weight =
                     last ? Tensor{} : dw.layers[static_cast<std::size_t>(layer) + 1].input_norm,
                 .out = last ? Tensor{} : hn},
                s);
        }
    }

    // Columns 1..k of each row's block are its drafts (column 0 re-reads the anchor): the drafter's
    // final norm, then the target's head, whose argmax is the draft (its logit multiplier and soft
    // cap are monotonic).
    Tensor tail                    = work.alloc(DType::BF16, {h, drafted});
    const std::size_t column_bytes = static_cast<std::size_t>(h) * dtype_size(DType::BF16);
    CUDA_CHECK(cudaMemcpy2DAsync(tail.data, column_bytes * k,
                                 static_cast<const std::byte*>(residual.data) + column_bytes,
                                 column_bytes * static_cast<std::size_t>(block), column_bytes * k,
                                 static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                 s));
    Tensor normed = work.alloc(DType::BF16, {h, drafted});
    ops::rmsnorm(tail, dw.final_norm, eps, false, normed, s);
    Tensor logits = work.alloc(DType::BF16, {c_.vocab_size, drafted});
    pass_.uncapped_head(work, normed, logits, Activations::Recorded);
    Tensor drafts = frame.draft_tokens.slice(1, 0, batch_size).view({drafted});
    ops::argmax(logits, drafts, traits_.public_tokens, s);
}

std::function<void()>
DenseExecution::dflash_body(program::DFlashBatchContext& state, std::int32_t batch_size,
                            std::uint32_t k, program::DFlashEnvelopes envelopes,
                            ops::CausalAttentionExecutionEnvelope target_envelope) const {
    return [this, &state, batch_size, k, envelopes, target_envelope] {
        if (drafter_ == nullptr || batch_size <= 0 ||
            batch_size > static_cast<std::int32_t>(kMaximumConcurrency) || k == 0 ||
            k + 1U > static_cast<std::uint32_t>(drafter_->config().block_size)) {
            throw std::logic_error("dense DFlash decode batch is out of range");
        }
        program::ExecutionCore& core      = state.execution;
        WorkspaceArena& work              = core.work;
        const cudaStream_t s              = core.device.stream;
        qwen3_5::DFlashDecodeState& frame = state.frame;
        const std::int32_t b              = batch_size;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns        = width * b;
        const std::int32_t features       = traits_.draft_feature_rows;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_5::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   s));
        Tensor anchors          = frame.anchors.slice(0, 0, b);
        Tensor frontiers        = frame.execution_frontiers.slice(0, 0, b);
        Tensor context_starts   = frame.context_frontiers.slice(0, 0, b);
        Tensor extents          = frame.proposal_extents.slice(0, 0, b);
        Tensor valid_columns    = frame.target_valid_columns.slice(0, 0, b);
        Tensor text_rows        = frame.text_kv_table_rows.slice(0, 0, b);
        Tensor dflash_rows      = frame.dflash_kv_table_rows.slice(0, 0, b);
        Tensor active_lanes     = frame.active_lanes.slice(0, 0, b);
        Tensor destinations     = frame.state_destination_slots.slice(0, 0, b);
        Tensor append_positions = frame.append_positions.slice(1, 0, b);
        Tensor append_counts    = frame.append_counts.slice(0, 0, b);
        Tensor drafts           = frame.draft_tokens.slice(1, 0, b);
        Tensor verify_ids       = frame.verify_ids.slice(1, 0, b);
        Tensor verify_positions = frame.verify_positions.slice(1, 0, b);
        Tensor target_tokens    = frame.target_argmax.slice(1, 0, b);
        Tensor target_logits    = frame.target_logits.slice(2, 0, b);
        Tensor target_hidden    = frame.target_hidden.slice(2, 0, b);
        Tensor selected_hidden  = frame.target_continuation_hidden.slice(1, 0, b);
        Tensor licensed_tokens  = frame.licensed_tokens.slice(1, 0, b);
        Tensor licensed_counts  = frame.licensed_counts.slice(0, 0, b);
        Tensor accepted         = frame.accepted_drafts.slice(0, 0, b);
        // Every lane's pending target features, at this launch window's width.
        Tensor& pending = frame.pending_features.data != nullptr ? frame.pending_features
                                                                 : state.dflash.pending_features;

        // The features the last round's verify left for its accepted columns (none after a
        // prefill, which appended the whole prompt) join the drafter's context.
        work.reset();
        {
            Tensor compact = work.alloc(DType::BF16, {features, width, b});
            ops::prepare_ragged_prefix(pending, active_lanes, context_starts, frontiers, compact,
                                       append_positions, append_counts, s);
            append_context(core, state.dflash, compact, append_positions, append_counts,
                           destinations, dflash_rows, envelopes.append);
        }

        propose(state, b, k, envelopes);
        ops::speculative_prepare_verify_inputs(anchors, drafts, frontiers, extents, verify_ids,
                                               verify_positions, s);
        // The drafts reach the host now, stamped with the round's epoch, so the host computes a
        // constrained row's verify masks along them while the target verifies
        // (fill_dflash_draft_masks waits for the stamp).
        ops::copy_to_pinned_host(state.host_egress.drafts.data(), drafts.data, drafts.bytes(), s);
        ops::copy_to_pinned_host(&state.host_egress.drafts_epoch,
                                 static_cast<const unsigned char*>(frame.ingress.data) +
                                     offsetof(qwen3_5::DFlashDecodeIngress, round_epoch),
                                 sizeof(std::uint32_t), s);

        // The target's verify under the policy plain decode runs: each row's anchor and first
        // extent drafts (the padding columns repeat its last real position and write no keys or
        // values), tapping the drafter's feature layers. Every verified column's features then
        // wait in its lane; the next round appends the accepted ones.
        work.reset();
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeDFlashTarget, nvtx::Category::DFlash,
                                           static_cast<std::uint64_t>(columns));
            const auto layers =
                bindings(state.text_cache, core.state_images,
                         state.text_cache.batch_layer_view(0).block_tables, text_rows,
                         destinations, valid_columns);
            auto scope            = work.scope();
            Tensor tapped         = work.alloc(DType::BF16, {features, columns});
            const FeatureTap tap  = feature_tap(tapped);
            const Tensor normed   = pass_.forward(
                work, verify_ids.view({columns}), verify_positions.view({columns}),
                {.layers = layers, .batch = b}, session_context(core),
                {.max_visible_keys = target_envelope.max_visible_keys, .max_tokens = width},
                Activations::Recorded, &tap);
            CUDA_CHECK(cudaMemcpyAsync(target_hidden.data, normed.data, target_hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, s));
            ops::scatter_bf16_batch(tapped.view({features, width, b}), active_lanes, valid_columns,
                                    pending, s);
        }
        {
            auto scope    = work.scope();
            Tensor logits = target_logits.view({target_logits.ne[0], columns});
            pass_.head(work, target_hidden.view({c_.hidden_size, columns}), logits);
            Tensor argmax = target_tokens.view({columns});
            ops::argmax(logits, argmax, traits_.public_tokens, s);
        }
        {
            auto scope = work.scope();
            ops::speculative_accept_greedy_drafts(target_tokens, target_logits, drafts, extents,
                                                  frontiers, anchors, licensed_tokens,
                                                  licensed_counts, accepted, traits_.public_tokens,
                                                  frame.sampling, work, s);
        }
        ops::speculative_select_accepted_hidden(target_hidden, accepted, selected_hidden, s);
        ops::scatter(selected_hidden, destinations, state.continuation_hidden_store, s);
        ops::copy_to_pinned_host(&state.host_egress, frame.egress.data,
                                 qwen3_5::kDFlashDecodeRoundEgressBytes, s);
    };
}

void DenseExecution::dflash_append_context(
    program::DFlashAppendContext& state, const Tensor& features, const Tensor& positions,
    const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
    ops::KVCacheAppendPrefixExecutionEnvelope envelope) const {
    if (drafter_ == nullptr) { throw std::logic_error("dense DFlash append without a drafter"); }
    const ops::ScopedInvariantRoutes routes(invariant_);
    append_context(state.execution, state.dflash, features, positions, commit_counts, lanes,
                   table_rows, envelope);
}

void DenseExecution::capture_dflash_decode_batch(
    program::DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
    program::DFlashEnvelopes envelopes, ops::CausalAttentionExecutionEnvelope target_envelope,
    DecodeGraphDefinition& definition) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    const ops::ScopedRoundLanes lanes(batch_size);
    program::capture_graph(state, definition,
                           dflash_body(state, batch_size, k, envelopes, target_envelope));
}

void DenseExecution::dflash_decode_batch(program::DFlashBatchContext& state,
                                         std::int32_t batch_size, std::uint32_t k,
                                         program::DFlashEnvelopes envelopes,
                                         ops::CausalAttentionExecutionEnvelope target_envelope,
                                         DecodeGraphExecutable* executable) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    const ops::ScopedRoundLanes lanes(batch_size);
    // A row that forks from a checkpoint writes its first token in this launch: its destination
    // takes the source's target rings first (the Program forked the drafter's rings at admission),
    // and the launch then runs in place.
    for (std::int32_t row = 0; row < batch_size; ++row) {
        state.execution.state_images.copy_rings(state.host_ingress.state_source_slots[row],
                                                state.host_ingress.state_destination_slots[row],
                                                state.execution.device.stream);
    }
    program::run_prepared(state, executable,
                          dflash_body(state, batch_size, k, envelopes, target_envelope));
}

} // namespace ninferno::models::dense::detail
