#include "models/qwen3_5/program/planning/workspace_plan.h"

#include "core/device.h"
#include "core/layout.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/execution/workspace.h"
#include "models/qwen3_5/program/program.h"
#include "ninferno/ops/candidate_selector.h"
#include "ninferno/ops/context_kv_materialize.h"
#include "ninferno/ops/dynamic_grouped_conv.h"
#include "ninferno/ops/gated_delta_net.h"
#include "ninferno/ops/linear_add.h"
#include "ninferno/ops/linear_swiglu.h"
#include "ninferno/ops/linear_topk.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/segmented_projection.h"
#include "ninferno/ops/sliding_window_attention.h"
#include "ninferno/ops/softmax_attention.h"
#include "ninferno/ops/speculative_round.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <variant>

namespace ninferno::models::qwen3_5::detail {
namespace {

using execution::dimension;
namespace workspace = execution::workspace;

enum class GdnWorkspacePath : std::uint8_t {
    Prefill,
    Snapshot,
    ReplayRecord,
};

} // namespace

WorkspacePlan build_workspace_plan(const WorkspaceInputs& inputs, const SequencePlanImpl& plan) {
    const DeviceExecutionView device_execution{nullptr, plan.multiprocessor_count};
    const auto& config = inputs.config;
    // The parts a plan's features read must be described (a start's load binds them).
    const bool optimized_head = plan.proposal_head == ProposalHead::Optimized;
    if (!config.attention || (!config.gdn && config.linear_attention_layers != 0) ||
        inputs.public_tokens <= 0) {
        throw std::invalid_argument("workspace inputs: the text geometry is incomplete");
    }
    if (plan.features.mtp() && inputs.mtp == nullptr) {
        throw std::invalid_argument("workspace inputs: an MTP plan without the MTP layer");
    }
    if (plan.features.masked_draft() &&
        (inputs.draft == nullptr || inputs.draft_config == nullptr)) {
        throw std::invalid_argument("workspace inputs: a masked-draft plan without the drafter");
    }
    if ((plan.features.mtp() || plan.features.masked_draft()) && optimized_head &&
        inputs.proposal == nullptr) {
        throw std::invalid_argument("workspace inputs: an optimized proposal head not described");
    }

    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    const ops::CausalAttentionExecutionEnvelope text_envelope{1, plan.capacity};

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace::text_prefill_roots(layout, config, tokens, plan.features.vision ? 3 : 0,
                                            plan.features.vision ? tokens : 0);
    };
    const auto linear_scratch = [&](WorkspaceLayoutBuilder& layout,
                                    const execution::LinearParameters& p, int first, int last) {
        scratch(layout, ops::linear_workspace_capacity_bytes(p, first, last));
    };
    const auto add_scratch = [&](WorkspaceLayoutBuilder& layout,
                                 const execution::LinearParameters& p, int first, int last) {
        scratch(layout, ops::linear_add_workspace_capacity_bytes(
                            p.weight.qtype, p.weight.n, p.weight.k, p.policy, first, last));
    };
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, TextPhase phase, GdnWorkspacePath path,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width,
                                 ops::CausalAttentionExecutionEnvelope envelope) {
        // The layers run a prefill chunk under the prefill route profile (TextContext::run_layers)
        // and everything else under the speed profile; the capacity queries answer for the active
        // one.
        const ops::ScopedPrefillRoutes route_profile(phase == qwen3_5::TextPhase::Prefill);
        for (const auto& block : inputs.text.layers) {
            {
                auto stage = layout.scope();
                if (const auto* attention =
                        std::get_if<execution::AttentionParameters>(&block.mixer)) {
                    (void)workspace::text_attention_projection(layout, config, last);
                    scratch(layout, execution::attention_projection_workspace_bytes(*attention,
                                                                                    first, last));
                    (void)workspace::text_attention_results(layout, config, last);
                    scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                                        {dimension(config.attention->head_dim),
                                         dimension(config.attention->num_attention_heads),
                                         dimension(config.attention->num_key_value_heads)},
                                        plan.kv_storage, envelope, batch_size, min_width, max_width,
                                        device_execution));
                    add_scratch(layout, attention->output, first, last);
                } else {
                    const auto& gdn = std::get<execution::GdnParameters>(block.mixer);
                    (void)workspace::gdn_control(layout, config, last);
                    scratch(layout,
                            execution::gdn_control_workspace_bytes(
                                gdn, *config.gdn, dimension(config.hidden_size), first, last));
                    (void)workspace::gdn_projection(layout, config, last);
                    if (path == GdnWorkspacePath::Snapshot) {
                        scratch(layout, execution::gdn_snapshot_workspace_bytes(
                                            gdn, *config.gdn, batch_size, min_width, max_width));
                    } else if (path == GdnWorkspacePath::ReplayRecord) {
                        scratch(layout, execution::gdn_record_workspace_bytes(
                                            gdn, *config.gdn, batch_size, min_width, max_width));
                    } else {
                        (void)workspace::gdn_prefill_conv(layout, config, last);
                        scratch(layout,
                                execution::gdn_projection_workspace_bytes(gdn, first, last));
                    }
                    (void)workspace::gdn_recurrent_output(layout, config, last);
                    if (path == GdnWorkspacePath::Prefill) {
                        scratch(layout,
                                ops::gated_delta_net_workspace_capacity_bytes(
                                    dimension(config.gdn->linear_num_key_heads),
                                    dimension(config.gdn->linear_num_value_heads), first, last));
                    }
                    (void)workspace::gdn_normalized_output(layout, config, last);
                    add_scratch(layout, gdn.output, first, last);
                }
            }
            auto stage = layout.scope();
            (void)workspace::post_mixer_hidden(layout, config, last);
            scratch(layout, execution::ffn_workspace_bytes(block.ffn, first, last));
        }
        if (!plan.causal_scoring) {
            // A prefill chunk projects only its last column through the head
            // (TextContext::prefill).
            const std::int32_t head_last = phase == qwen3_5::TextPhase::Prefill ? 1 : last;
            linear_scratch(layout, inputs.text.output_head, std::min(first, head_last), head_last);
        }
    };
    const auto mtp_post_mixer = [&](WorkspaceLayoutBuilder& layout, int first, int last) {
        linear_scratch(layout, inputs.mtp->output, first, last);
        scratch(layout, execution::ffn_workspace_bytes(inputs.mtp->ffn, first, last, true));
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, dimension(inputs.proposal->rows), columns);
            linear_scratch(layout, inputs.proposal->head, columns, columns);
        } else {
            linear_scratch(layout, inputs.mtp->output_head, columns, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace::mtp_stem(layout, config, tokens, !preembedded);
        linear_scratch(layout, inputs.mtp->input_projection, 1, tokens);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::CausalAttentionExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace::mtp_attention_projection(layout, config, tokens);
        scratch(layout, execution::mtp_projection_workspace_bytes(
                            inputs.mtp->projection, *config.attention, tokens, tokens));
        (void)workspace::mtp_attention_results(layout, config, tokens);
        scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(config.attention->head_dim),
                             dimension(config.attention->num_attention_heads),
                             dimension(config.attention->num_key_value_heads)},
                            plan.kv_storage, envelope, 1, tokens, tokens, device_execution));
        (void)workspace::mtp_post_attention(layout, config, tokens);
        mtp_post_mixer(layout, tokens, tokens);
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, bool preembedded) {
        auto call = layout.scope();
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, preembedded);
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
            scratch(layout, execution::mtp_kv_workspace_bytes(inputs.mtp->projection,
                                                              *config.attention, first, last));
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
        }
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        scratch(layout, execution::mtp_query_gate_workspace_bytes(inputs.mtp->projection,
                                                                  *config.attention, 1, 1));
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(config.attention->head_dim),
                             dimension(config.attention->num_attention_heads),
                             dimension(config.attention->num_key_value_heads)},
                            plan.kv_storage, text_envelope, 1, 1, 1, device_execution));
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        mtp_post_mixer(layout, 1, 1);
        proposal_scratch(layout, 1);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    target_body(text_prefill, 1, chunk, qwen3_5::TextPhase::Prefill, GdnWorkspacePath::Prefill, 1,
                1, chunk, text_envelope);
    if (!plan.causal_scoring) {
        scratch(text_prefill, ops::sampling_workspace_capacity_bytes(inputs.public_tokens, 1, 1));
    }
    out.text_prefill = finish(text_prefill);

    if (plan.causal_scoring) {
        WorkspaceLayoutBuilder causal_score;
        matrix(causal_score, DType::BF16, dimension(config.vocab_size),
               static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::I32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::FP32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        // The distribution detail (ScoreDetailRequest): top-K ids and log-probabilities, given ids
        // and theirs, at most ops::kTokenTopkMax per column.
        for (int plane = 0; plane < 2; ++plane) {
            matrix(causal_score, DType::I32, ops::kTokenTopkMax,
                   static_cast<std::int32_t>(kCausalScoreTile));
            matrix(causal_score, DType::FP32, ops::kTokenTopkMax,
                   static_cast<std::int32_t>(kCausalScoreTile));
        }
        linear_scratch(causal_score, inputs.text.output_head, 1, kCausalScoreTile);
        out.causal_score = finish(causal_score);
    }

    if (!plan.causal_scoring) {
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            WorkspaceLayoutBuilder ordinary;
            matrix(ordinary, DType::BF16, dimension(config.hidden_size), batch);
            target_body(ordinary, batch, batch, qwen3_5::TextPhase::Verify,
                        GdnWorkspacePath::Snapshot, batch, 1, 1, text_envelope);
            scratch(ordinary,
                    ops::sampling_workspace_capacity_bytes(inputs.public_tokens, batch, batch));
            out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
        }
    }

    if (plan.features.mtp()) {
        WorkspaceLayoutBuilder mtp_prefill;
        text_common_root(mtp_prefill, chunk);
        target_body(mtp_prefill, 1, chunk, qwen3_5::TextPhase::Prefill, GdnWorkspacePath::Prefill,
                    1, 1, chunk, text_envelope);
        matrix(mtp_prefill, DType::I32, 1, chunk);
        if (plan.features.vision) {
            matrix(mtp_prefill, DType::BF16, dimension(config.hidden_size), chunk);
            (void)workspace::visual_scatter_indices(mtp_prefill, chunk);
        }
        mtp_prefill_chunk(mtp_prefill, 1, chunk, plan.features.vision);
        for (std::int32_t i = 1; i < drafts; ++i) {
            matrix(mtp_prefill, DType::BF16, dimension(config.hidden_size), 1);
            mtp_full_call(mtp_prefill, 1, text_envelope, true);
        }
        out.mtp_prefill = finish(mtp_prefill);

        WorkspaceLayoutBuilder mtp_batch;
        mtp_full_call(mtp_batch, verify, text_envelope, false);
        WorkspaceLayoutBuilder mtp_ar;
        mtp_full_call(mtp_ar, 1, text_envelope, true);
        WorkspaceLayoutBuilder mtp_align;
        mtp_full_call(mtp_align, 1, text_envelope, false);
        WorkspaceLayoutBuilder mtp_proposal;
        proposal_scratch(mtp_proposal, 1);
        const std::size_t accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            inputs.public_tokens, drafts, drafts, 1, 1);
        out.mtp_round = std::max({accept, finish(mtp_batch), finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));

        for (std::int32_t batch = 1;
             batch <= static_cast<std::int32_t>(speculative_round_batches(plan)); ++batch) {
            // The widest window a round of this many rows runs (launch_window_batches): a load
            // bound caps every row of a step at least its lanes wide.
            const auto round_drafts = static_cast<std::int32_t>(
                speculative_round_window(plan, static_cast<std::uint32_t>(batch)));
            if (round_drafts == 0) { continue; }
            const std::int32_t round_verify = round_drafts + 1;
            const std::int32_t aggregate    = batch * round_verify;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, dimension(config.hidden_size), aggregate);
            target_body(target, aggregate, aggregate, qwen3_5::TextPhase::Verify,
                        GdnWorkspacePath::ReplayRecord, batch, round_verify, round_verify,
                        text_envelope);

            const auto mtp_decode_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t width) {
                const std::int32_t tokens = batch * width;
                auto core                 = layout.scope();
                mtp_stem(layout, tokens, false);
                (void)workspace::mtp_attention_projection(layout, config, tokens);
                scratch(layout, execution::mtp_projection_workspace_bytes(
                                    inputs.mtp->projection, *config.attention, tokens, tokens));
                (void)workspace::mtp_attention_results(layout, config, tokens);
                scratch(layout,
                        ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(config.attention->head_dim),
                             dimension(config.attention->num_attention_heads),
                             dimension(config.attention->num_key_value_heads)},
                            plan.kv_storage, text_envelope, batch, width, width, device_execution));
                (void)workspace::mtp_post_attention(layout, config, tokens);
                mtp_post_mixer(layout, tokens, tokens);
            };

            WorkspaceLayoutBuilder alignment;
            mtp_decode_core(alignment, round_verify);
            WorkspaceLayoutBuilder ar;
            mtp_decode_core(ar, 1);
            WorkspaceLayoutBuilder proposal;
            proposal_scratch(proposal, batch);
            const std::size_t batch_accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    inputs.public_tokens, round_drafts, round_drafts, batch, batch);
            out.mtp_round = std::max({out.mtp_round, finish(target), finish(alignment), finish(ar),
                                      finish(proposal), batch_accept});
        }
    }

    if (plan.features.masked_draft()) {
        {
            const auto* draft                  = inputs.draft_config;
            const auto dflash_context_capacity = [&](std::int32_t width, std::int32_t batch,
                                                     bool compact_input) {
                const auto tokens = width * batch;
                WorkspaceLayoutBuilder layout;
                if (compact_input) {
                    matrix(layout, DType::BF16,
                           dimension(config.hidden_size *
                                     std::uint64_t(draft->target_layer_ids.size())),
                           tokens);
                }
                if (draft->dflash2.has_value()) {
                    const auto local_width =
                        std::min(width, dimension(draft->sliding_window.value_or(0)));
                    (void)workspace::dflash_context(layout, config, *draft, local_width * batch);
                    linear_scratch(layout, inputs.draft->feature_projection, local_width * batch,
                                   local_width * batch);
                    scratch(layout, ops::context_kv_materialize_workspace_capacity_bytes(
                                        batch, local_width, local_width));
                    {
                        // The generic per-layer route of a drafter outside the fused Q8 profile.
                        auto layer = layout.scope();
                        (void)workspace::dflash_context_layer(layout, config, *draft,
                                                              local_width * batch);
                    }
                    return finish(layout);
                }
                (void)workspace::dflash_context(layout, config, *draft, tokens);
                linear_scratch(layout, inputs.draft->feature_projection, tokens, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace::dflash_context_layer(layout, config, *draft, tokens);
                }
                return finish(layout);
            };
            const auto dflash_proposal_capacity = [&](std::int32_t width, std::int32_t batch) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t tokens = width * batch;
                matrix(layout, DType::BF16, dimension(config.hidden_size), tokens);
                if (draft->dflash2.has_value()) {
                    // Every layer's convolution projections in their stored formats (the
                    // decomposed routes of a format without fused kernels take more scratch).
                    std::size_t prepare_bytes = 0, attention_add = 0, mlp_add = 0;
                    for (const auto& block : inputs.draft->layers) {
                        for (const auto* conv : {&block.attention_conv, &block.mlp_conv}) {
                            prepare_bytes = std::max(
                                prepare_bytes,
                                ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                                    (*conv)->kernel_projection.weight.qtype, width, width, batch,
                                    batch));
                        }
                        attention_add =
                            std::max(attention_add,
                                     ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                         block.output.weight.qtype,
                                         dimension(draft->attention.query_width()), width, width,
                                         batch, batch));
                        mlp_add = std::max(
                            mlp_add,
                            ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                block.mlp.down.weight.qtype, dimension(draft->intermediate_size),
                                width, width, batch, batch));
                    }
                    const auto prepare = [&] {
                        (void)workspace::dflash2_branch(layout, config, *draft, width, batch);
                        scratch(layout, prepare_bytes);
                    };
                    {
                        auto attention = layout.scope();
                        prepare();
                        matrix(layout, DType::BF16, dimension(draft->attention.query_width()),
                               tokens);
                        matrix(layout, DType::BF16, dimension(draft->attention.key_width()),
                               tokens);
                        matrix(layout, DType::BF16, dimension(draft->attention.key_width()),
                               tokens);
                        matrix(layout, DType::BF16, dimension(draft->attention.query_width()),
                               tokens);
                        for (const auto& block : inputs.draft->layers) {
                            scratch(layout, ops::segmented_projection_workspace_capacity_bytes(
                                                block.query_key_value_bank, tokens, tokens));
                        }
                        scratch(layout, ops::sliding_window_attention_workspace_capacity_bytes(
                                            {dimension(draft->attention.head_dim),
                                             dimension(draft->attention.num_attention_heads),
                                             dimension(draft->attention.num_key_value_heads)},
                                            dimension(draft->sliding_window.value_or(0)),
                                            {0, plan.capacity}, width, width, batch));
                        scratch(layout, attention_add);
                    }
                    {
                        auto mlp = layout.scope();
                        prepare();
                        matrix(layout, DType::BF16, dimension(draft->intermediate_size), tokens);
                        for (const auto& block : inputs.draft->layers) {
                            const auto& p = block.mlp.gate_up;
                            scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                                p.weight.qtype, p.weight.n, p.weight.k, p.policy,
                                                tokens, tokens));
                        }
                        scratch(layout, mlp_add);
                    }
                    const auto mask_columns = drafts * batch;
                    matrix(layout, DType::BF16, dimension(config.hidden_size), mask_columns);
                    matrix(layout, DType::FP32, dimension(draft->dflash2->selector_top_k),
                           mask_columns);
                    const auto& head = plan.proposal_head == ProposalHead::Optimized
                                           ? inputs.proposal->head
                                           : inputs.draft->output_head;
                    scratch(layout, ops::linear_topk_workspace_capacity_bytes(
                                        head.weight.qtype, head.weight.n, head.weight.k,
                                        mask_columns, mask_columns));
                    matrix(layout, DType::BF16, dimension(draft->dflash2->selector_rank),
                           mask_columns);
                    linear_scratch(layout, inputs.draft->selector->hidden_projection, mask_columns,
                                   mask_columns);
                    scratch(layout, ops::candidate_selector_path_workspace_capacity_bytes(
                                        drafts, drafts, batch, batch));
                    return finish(layout);
                }
                {
                    auto attention = layout.scope();
                    (void)workspace::dflash_attention(layout, config, *draft, tokens);
                    scratch(layout,
                            std::max(ops::sliding_window_attention_workspace_capacity_bytes(
                                         {dimension(draft->attention.head_dim),
                                          dimension(draft->attention.num_attention_heads),
                                          dimension(draft->attention.num_key_value_heads)},
                                         dimension(draft->sliding_window.value_or(0)),
                                         {0, plan.capacity}, width, width, batch),
                                     draft->fold_window != 0
                                         ? ops::context_fold_attention_workspace_capacity_bytes(
                                               {dimension(draft->attention.head_dim),
                                                dimension(draft->attention.num_attention_heads),
                                                dimension(draft->attention.num_key_value_heads)},
                                               {0, plan.capacity}, dimension(draft->fold_window),
                                               width, width, batch)
                                         : ops::context_softmax_attention_workspace_capacity_bytes(
                                               {dimension(draft->attention.head_dim),
                                                dimension(draft->attention.num_attention_heads),
                                                dimension(draft->attention.num_key_value_heads)},
                                               {0, plan.capacity}, width, width, batch)));
                    for (const auto& block : inputs.draft->layers) {
                        add_scratch(layout, block.output, tokens, tokens);
                        // Q/K/V not one RowSplit Q8 parent: the generic bank (draft.cpp).
                        if (!execution::fused_q8_weight(block.query_key_value.weight)) {
                            scratch(layout, ops::segmented_projection_workspace_capacity_bytes(
                                                block.query_key_value_bank, tokens, tokens));
                        }
                    }
                }
                {
                    auto mlp = layout.scope();
                    (void)workspace::dflash_mlp(layout, config, *draft, tokens);
                    for (const auto& block : inputs.draft->layers) {
                        const auto& p = block.mlp.gate_up;
                        scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                            p.weight.qtype, p.weight.n, p.weight.k, p.policy,
                                            tokens, tokens));
                    }
                    for (const auto& block : inputs.draft->layers) {
                        add_scratch(layout, block.mlp.down, tokens, tokens);
                    }
                }
                matrix(layout, DType::BF16, dimension(config.hidden_size), drafts * batch);
                matrix(layout, DType::BF16, dimension(config.hidden_size), drafts * batch);
                if (plan.proposal_head == ProposalHead::Optimized) {
                    matrix(layout, DType::BF16, dimension(inputs.proposal->rows), drafts * batch);
                } else {
                    matrix(layout, DType::BF16, dimension(config.vocab_size), drafts * batch);
                }
                const auto& head = plan.proposal_head == ProposalHead::Optimized
                                       ? inputs.proposal->head
                                       : inputs.draft->output_head;
                linear_scratch(layout, head, drafts * batch, drafts * batch);
                return finish(layout);
            };

            out.dflash_context = dflash_context_capacity(chunk, 1, false);
            for (std::int32_t batch = 1;
                 batch <= static_cast<std::int32_t>(speculative_round_batches(plan)); ++batch) {
                const std::int32_t aggregate = verify * batch;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, dimension(config.hidden_size), aggregate);
                target_body(target, aggregate, aggregate, qwen3_5::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);
                const std::size_t accept =
                    draft->dflash2.has_value()
                        ? ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                              inputs.public_tokens, {false}, drafts, drafts, batch, batch)
                        : ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                              inputs.public_tokens, drafts, drafts, batch, batch);
                const std::size_t proposal = dflash_proposal_capacity(verify, batch);
                const std::size_t append   = dflash_context_capacity(verify, batch, true);
                out.dflash_round_append    = std::max(out.dflash_round_append, append);
                out.dflash_round =
                    std::max({out.dflash_round, finish(target), accept, append, proposal});
            }
        }
    }

    out.general_capacity =
        std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                  out.dflash_context, out.dflash_round, out.causal_score});
    // A Vision start adds its encoder's region above this (qwen3_5_workspace_plan).
    out.capacity = out.general_capacity;
    return out;
}

} // namespace ninferno::models::qwen3_5::detail
