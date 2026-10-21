#include "models/qwen3_5/program/planning/startup.h"

#include <cstdlib>
#include <string_view>
#include "core/checked.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/program/qwen3_5_execution.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/execution/workspace.h"
#include "core/device.h"
#include "core/units.h"
#include "spec/engine_support.h"
#include "ninferno/ops/gated_delta_net.h"
#include "ninferno/ops/candidate_selector.h"
#include "ninferno/ops/context_kv_materialize.h"
#include "ninferno/ops/dynamic_grouped_conv.h"
#include "ninferno/ops/linear_topk.h"
#include "ninferno/ops/token_distribution.h"
#include "ninferno/ops/segmented_projection.h"
#include "ninferno/ops/gdn_gating_proj.h"
#include "ninferno/ops/gdn_input_proj.h"
#include "ninferno/ops/linear_add.h"
#include "ninferno/ops/linear_swiglu.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/sliding_window_attention.h"
#include "ninferno/ops/softmax_attention.h"
#include "ninferno/ops/speculative_round.h"
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninferno::models::qwen3_5::detail {
namespace {

using execution::dimension;
namespace workspace = execution::workspace;

constexpr std::size_t kArenaAlign = 256ULL;

// A DFlash drafter's sliding-layer ring keeps Int8Group64 K/V at D128 (about half the bytes of
// every StateImage slot's ring): on an RTX 5090, Qwen3.6-35B-A3B with DFlash decodes 0.7-2.4 %
// faster at 1-8 concurrent requests with 0.6 GiB less runtime. DFlash2 keeps BF16: an int8 ring
// takes it off the fused materialization onto the per-layer path (Qwen3.8-27B: +2.8 % at one
// request, -2.7 % at four, -0.6 % at eight with 31K contexts).
bool dflash_local_int8(std::int64_t head_dim, bool dflash2) {
    return head_dim == 128 && !dflash2;
}

enum class GdnWorkspacePath : std::uint8_t {
    Prefill,
    Snapshot,
    ReplayRecord,
};

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

template <class ProfileAllowance>
std::size_t graph_topology_allowance(const std::vector<GraphExecutionProfile>& profiles,
                                     ProfileAllowance&& profile_allowance, const char* label) {
    std::vector<std::pair<std::uint32_t, std::size_t>> classes;
    for (const GraphExecutionProfile profile : profiles) {
        const std::size_t allowance = profile_allowance(profile);
        const auto existing = std::find_if(classes.begin(), classes.end(), [&](const auto& entry) {
            return entry.first == profile.topology_class;
        });
        if (existing == classes.end()) {
            classes.emplace_back(profile.topology_class, allowance);
        } else {
            existing->second = std::max(existing->second, allowance);
        }
    }

    std::size_t total = 0;
    for (const auto& [topology_class, allowance] : classes) {
        (void)topology_class;
        total = checked_add(total, allowance, label);
    }
    return total;
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

PersistentLayout persistent_layout(const execution::Parameters& parameters,
                                   const SequencePlanImpl& plan) {
    const auto& config     = parameters.model.config().text;

    if (!plan.context_cache.device_state_slots) {
        throw std::logic_error("Qwen3.5 context cache options are not normalized");
    }
    const std::int32_t state_image_slots = checked_i32(
        static_cast<std::uint64_t>(plan.max_concurrency) + *plan.context_cache.device_state_slots,
        "Qwen3.5 StateImage slot count exceeds int32");
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const std::uint32_t logical_pages      = paged_kv_capacity_pages(plan.capacity);
    const std::uint32_t physical_pages = plan.main_page_groups;
    const std::uint64_t mtp_extra_pages    = plan.features.mtp()
                                                 ? static_cast<std::uint64_t>(plan.max_concurrency) *
                                                    paged_kv_pages(plan.draft_window - 1U)
                                                 : 0ULL;
    const std::uint32_t mtp_physical_pages = static_cast<std::uint32_t>(
        checked_i32(static_cast<std::uint64_t>(physical_pages) + mtp_extra_pages,
                    "MTP Paged KV physical pages exceed int32"));
    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen3_5::plan_decoder_state(
        builder, qwen3_5::DecoderStateSpec{
                     .full_attention_layers     = config.full_attention_layers,
                     .mtp_layers                = 1,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = dimension(config.attention->num_key_value_heads),
                     .attention_head_dim        = dimension(config.attention->head_dim),
                     .kv_storage                = plan.kv_storage,
                     .enable_mtp                = plan.features.mtp(),
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
                     .text_physical_page_groups = physical_pages,
                     .mtp_physical_page_groups  = mtp_physical_pages,
                 });
    qwen3_5::StateImageSpec state_image_spec{
        .linear =
            {
                .layers        = config.linear_attention_layers,
                .conv_channels = (config.gdn ? dimension(config.gdn->conv_channels()) : 0),
                .conv_width  = (config.gdn ? dimension(config.gdn->linear_conv_kernel_dim - 1) : 0),
                .value_heads = (config.gdn ? dimension(config.gdn->linear_num_value_heads) : 0),
                .value_head_dim  = (config.gdn ? dimension(config.gdn->linear_value_head_dim) : 0),
                .key_head_dim    = (config.gdn ? dimension(config.gdn->linear_key_head_dim) : 0),
                .slot_count      = state_image_slots,
                .conv_dtype      = DType::BF16,
                .recurrent_dtype = plan.recurrent_state_dtype,
            },
        .hidden = dimension(config.hidden_size),
    };
    {
        const auto* draft =
            parameters.model.config().draft ? &*parameters.model.config().draft : nullptr;
        if (plan.features.masked_draft()) {
            state_image_spec.dflash_local = qwen3_5::DFlashLocalStateSpec{
                .layers   = draft->local_layer_count(),
                .capacity = draft->sliding_window.value_or(0),
                .kv_heads = dimension(draft->attention.num_key_value_heads),
                .head_dim = dimension(draft->attention.head_dim),
                .int8 = dflash_local_int8(draft->attention.head_dim, draft->dflash2.has_value()),
            };
        }
    }
    out.state_images = qwen3_5::plan_state_image_device_pool(builder, state_image_spec);
    if (plan.speculative_backend != SpeculativeBackend::None) {
        out.replay_records = plan_gdn_replay_records(
            builder,
            GdnReplayRecordSpec{
                .layers          = dimension(config.linear_attention_layers),
                .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
                .width           = static_cast<std::int32_t>(plan.draft_window + 1U),
                .conv_channels   = (config.gdn ? dimension(config.gdn->conv_channels()) : 0),
                .qk_heads        = (config.gdn ? dimension(config.gdn->linear_num_key_heads) : 0),
                .value_heads     = (config.gdn ? dimension(config.gdn->linear_num_value_heads) : 0),
                .key_dim         = (config.gdn ? dimension(config.gdn->linear_key_head_dim) : 0),
                .value_dim       = (config.gdn ? dimension(config.gdn->linear_value_head_dim) : 0),
            });
    }
    {
        const auto* draft =
            parameters.model.config().draft ? &*parameters.model.config().draft : nullptr;
        if (plan.features.masked_draft()) {
            DFlashPersistentLayout& dflash = out.dflash.emplace();
            if (draft->full_layer_count() != 0) {
                // A DFlash drafter's full layers keep per-token G64 int8 K/V at D128 (half the
                // bytes of BF16 K + FP16 V plus 1/32 for scales), like its sliding ring: with
                // both, Qwen3.6-35B-A3B on an RTX 4090 with four 40K requests holds ~129K
                // tokens instead of 90K and its ~30K rounds stop collapsing (516 / 112 -> 655
                // tok/s), while an RTX 5090 is flat at 1-8 requests. DFlash2 keeps BF16 (the
                // int8 full-layer cache was not evaluated for it).
                const bool drafter_int8 =
                    !draft->dflash2.has_value() && draft->attention.head_dim == 128;
                const PagedKVStorageLayout full_storage = paged_kv_storage_layout(
                    drafter_int8 ? KvCacheStorage::Int8Group64 : KvCacheStorage::BFloat16,
                    dimension(draft->attention.head_dim));
                KVPageGeometry full_geometry{
                    .page_tokens        = kPagedKVPageSize,
                    .device_plane_order = PagedKVPlaneOrder::HeadMajor,
                    .planes =
                        {
                            {full_storage.key.data_dtype, full_storage.key.data_leading_extent,
                             dimension(draft->attention.num_key_value_heads), 256},
                            {full_storage.value.data_dtype, full_storage.value.data_leading_extent,
                             dimension(draft->attention.num_key_value_heads), 256},
                        },
                };
                // A layer's planes in PagedKVCache::layer_view order: K, V, then the scales.
                if (full_storage.key.has_scale()) {
                    full_geometry.planes.push_back(
                        {full_storage.key.scale_dtype, full_storage.key.scale_leading_extent,
                         dimension(draft->attention.num_key_value_heads), 256});
                }
                if (full_storage.value.has_scale()) {
                    full_geometry.planes.push_back(
                        {full_storage.value.scale_dtype, full_storage.value.scale_leading_extent,
                         dimension(draft->attention.num_key_value_heads), 256});
                }
                const auto planes = full_geometry.planes;
                for (std::uint32_t layer = 1; layer < draft->full_layer_count(); ++layer) {
                    full_geometry.planes.insert(full_geometry.planes.end(), planes.begin(),
                                                planes.end());
                }
                dflash.full = qwen3_5::PagedKVCacheLayout{
                    .pages = plan_device_kv_page_pool(
                        builder, DeviceKVPagePoolSpec{.page_group_count = physical_pages,
                                                      .geometry = std::move(full_geometry)}),
                    .execution_tables = plan_kv_execution_tables(
                        builder,
                        KVExecutionTableSpec{
                            .logical_page_capacity = logical_pages,
                            .table_rows = static_cast<std::int32_t>(plan.max_concurrency),
                        }),
                    .layers        = draft->full_layer_count(),
                    .max_context   = plan.capacity,
                    .kv_heads      = dimension(draft->attention.num_key_value_heads),
                    .layer_storage = full_storage,
                };
            }
            dflash.prefill_features = add_tensor(
                builder, DType::BF16,
                {dimension(config.hidden_size * std::uint64_t(draft->target_layer_ids.size())),
                 effective_prefill_chunk},
                "DFlash prefill target features");
            dflash.prefill_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                  "DFlash prefill target positions");
            dflash.pending_features  = add_tensor(
                builder, DType::BF16,
                {dimension(config.hidden_size * std::uint64_t(draft->target_layer_ids.size())),
                  static_cast<std::int32_t>(plan.draft_window + 1U),
                  static_cast<std::int32_t>(plan.max_concurrency)},
                "DFlash pending target features");
        }
    }

    plan_program_buffers(builder, plan, plan.model->traits(), out);
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    return out;
}

WorkspacePlan build_workspace_plan(const execution::Parameters& parameters,
                                   const SequencePlanImpl& plan) {
    const DeviceExecutionView device_execution{nullptr, plan.multiprocessor_count};
    const auto& config     = parameters.model.config().text;

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
        for (const auto& block : parameters.text.layers) {
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
                    scratch(layout, execution::gdn_control_workspace_bytes(
                                        gdn, *config.gdn, dimension(config.hidden_size), first,
                                        last));
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
            // A prefill chunk projects only its last column through the head (TextContext::prefill).
            const std::int32_t head_last = phase == qwen3_5::TextPhase::Prefill ? 1 : last;
            linear_scratch(layout, parameters.text.output_head, std::min(first, head_last),
                           head_last);
        }
    };
    const auto mtp_post_mixer = [&](WorkspaceLayoutBuilder& layout, int first, int last) {
        linear_scratch(layout, parameters.mtp->output, first, last);
        scratch(layout, execution::ffn_workspace_bytes(parameters.mtp->ffn, first, last, true));
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, dimension(parameters.proposal->rows), columns);
            linear_scratch(layout, parameters.proposal->head, columns, columns);
        } else {
            linear_scratch(layout, parameters.mtp->output_head, columns, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace::mtp_stem(layout, config, tokens, !preembedded);
        linear_scratch(layout, parameters.mtp->input_projection, 1, tokens);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::CausalAttentionExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace::mtp_attention_projection(layout, config, tokens);
        scratch(layout, execution::mtp_projection_workspace_bytes(parameters.mtp->projection,
                                                                  tokens, tokens));
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
            scratch(layout, execution::mtp_kv_workspace_bytes(parameters.mtp->projection,
                                                              *config.attention, first, last));
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
        }
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        scratch(layout, execution::mtp_query_gate_workspace_bytes(parameters.mtp->projection,
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
        scratch(text_prefill,
                ops::sampling_workspace_capacity_bytes(
                    dimension(parameters.model.resources().public_token_count), 1, 1));
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
        linear_scratch(causal_score, parameters.text.output_head, 1, kCausalScoreTile);
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
                    ops::sampling_workspace_capacity_bytes(
                        dimension(parameters.model.resources().public_token_count), batch, batch));
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
            dimension(parameters.model.resources().public_token_count), drafts, drafts, 1, 1);
        out.mtp_round = std::max({accept, finish(mtp_batch), finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));

        for (std::int32_t batch = 1;
             batch <= static_cast<std::int32_t>(speculative_round_batches(plan));
             ++batch) {
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
                                    parameters.mtp->projection, tokens, tokens));
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
                    dimension(parameters.model.resources().public_token_count), round_drafts,
                    round_drafts, batch, batch);
            out.mtp_round = std::max({out.mtp_round, finish(target), finish(alignment), finish(ar),
                                      finish(proposal), batch_accept});
        }
    }

    if (plan.features.masked_draft()) {
        {
            const auto* draft                  = &*parameters.model.config().draft;
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
                    linear_scratch(layout, parameters.draft->feature_projection,
                                   local_width * batch, local_width * batch);
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
                linear_scratch(layout, parameters.draft->feature_projection, tokens, tokens);
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
                    for (const auto& block : parameters.draft->layers) {
                        for (const auto* conv : {&block.attention_conv, &block.mlp_conv}) {
                            prepare_bytes = std::max(
                                prepare_bytes,
                                ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                                    (*conv)->kernel_projection.weight.qtype, width, width, batch,
                                    batch));
                        }
                        attention_add = std::max(
                            attention_add,
                            ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                block.output.weight.qtype,
                                dimension(draft->attention.query_width()), width, width, batch,
                                batch));
                        mlp_add = std::max(
                            mlp_add, ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                         block.mlp.down.weight.qtype,
                                         dimension(draft->intermediate_size), width, width, batch,
                                         batch));
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
                        for (const auto& block : parameters.draft->layers) {
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
                        for (const auto& block : parameters.draft->layers) {
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
                                           ? parameters.proposal->head
                                           : parameters.draft->output_head;
                    scratch(layout, ops::linear_topk_workspace_capacity_bytes(
                                        head.weight.qtype, head.weight.n, head.weight.k,
                                        mask_columns, mask_columns));
                    matrix(layout, DType::BF16, dimension(draft->dflash2->selector_rank),
                           mask_columns);
                    linear_scratch(layout, parameters.draft->selector->hidden_projection,
                                   mask_columns, mask_columns);
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
                                     ops::context_softmax_attention_workspace_capacity_bytes(
                                         {dimension(draft->attention.head_dim),
                                          dimension(draft->attention.num_attention_heads),
                                          dimension(draft->attention.num_key_value_heads)},
                                         {0, plan.capacity}, width, width, batch)));
                    for (const auto& block : parameters.draft->layers) {
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
                    for (const auto& block : parameters.draft->layers) {
                        const auto& p = block.mlp.gate_up;
                        scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                            p.weight.qtype, p.weight.n, p.weight.k, p.policy,
                                            tokens, tokens));
                    }
                    for (const auto& block : parameters.draft->layers) {
                        add_scratch(layout, block.mlp.down, tokens, tokens);
                    }
                }
                matrix(layout, DType::BF16, dimension(config.hidden_size), drafts * batch);
                matrix(layout, DType::BF16, dimension(config.hidden_size), drafts * batch);
                if (plan.proposal_head == ProposalHead::Optimized) {
                    matrix(layout, DType::BF16, dimension(parameters.proposal->rows),
                           drafts * batch);
                } else {
                    matrix(layout, DType::BF16, dimension(config.vocab_size), drafts * batch);
                }
                const auto& head = plan.proposal_head == ProposalHead::Optimized
                                       ? parameters.proposal->head
                                       : parameters.draft->output_head;
                linear_scratch(layout, head, drafts * batch, drafts * batch);
                return finish(layout);
            };

            out.dflash_context = dflash_context_capacity(chunk, 1, false);
            for (std::int32_t batch = 1;
                 batch <= static_cast<std::int32_t>(speculative_round_batches(plan));
                 ++batch) {
                const std::int32_t aggregate = verify * batch;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, dimension(config.hidden_size), aggregate);
                target_body(target, aggregate, aggregate, qwen3_5::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);
                const std::size_t accept =
                    draft->dflash2.has_value()
                        ? ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                              dimension(parameters.model.resources().public_token_count), {false},
                              drafts, drafts, batch, batch)
                        : ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                              dimension(parameters.model.resources().public_token_count), drafts,
                              drafts, batch, batch);
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
    out.capacity = out.general_capacity;
    if (plan.features.vision) {
        const std::uint32_t merged = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(plan.capacity, kMaximumVisionItemTokens));
        out.vision = execution::VisionContext::plan_workspace(
            *parameters.model.config().vision, *parameters.vision, merged, out.general_capacity);
        out.capacity = std::max(out.capacity, out.vision->capacity_bytes);
    }
    return out;
}

void validate_target_options(const execution::Parameters& parameters, DeviceContext& device,
                             const EngineOptions& options) {
    if (!parameters.model.config().text.attention ||
        parameters.model.config().text.full_attention_layers == 0) {
        throw std::invalid_argument("Qwen3.5 Program requires at least one full-attention layer");
    }
    if (parameters.model.options() != models::load_options(options)) {
        throw std::invalid_argument(
            "loaded components do not match the requested execution options");
    }
    if (parameters.draft &&
        options.max_context > parameters.model.config().draft->max_position_embeddings) {
        throw std::invalid_argument("max_context exceeds the selected draft position capacity");
    }
    if (options.sliding_kv_cache != KvCacheStorage::BFloat16) {
        throw std::invalid_argument(
            "Qwen 3.5 has no sliding-window layers (--sliding-kv-dtype is for their rings)");
    }
    validate_program_options(options, parameters.model.config().text.max_position_embeddings,
                             kMaximumMtpDraftTokens, true);
    if (!options.speculative.drafter.empty()) {
        throw std::invalid_argument("Qwen3.5 artifacts carry their own drafter");
    }
    if (options.speculative.backend == SpeculativeBackend::DFlash ||
        options.speculative.backend == SpeculativeBackend::DFlash2) {
        if (!parameters.draft || (options.speculative.backend == SpeculativeBackend::DFlash2) !=
                                     parameters.model.config().draft->dflash2.has_value()) {
            throw std::invalid_argument(
                "selected masked draft backend is not supported by this target");
        }
        // The DFlash2 kernels (the dynamic grouped convolution, the packed tail norm) are built for
        // a 5120-wide target in groups of 16: another would plan here and fail at warm-up.
        const auto& extra = parameters.model.config().draft->dflash2;
        const std::uint64_t hidden = parameters.model.config().text.hidden_size;
        if (extra && (hidden != 5120 || extra->conv_group_size != 16)) {
            throw std::invalid_argument(
                "this DFlash2 drafter runs on a 5120-wide target in groups of 16 (Qwen3.8 27B); "
                "this target is " + std::to_string(hidden) + " wide: serve it with --spec mtp or "
                "--spec none");
        }
    }
    // The runtime's routes issue BF16, FP8 (.kind::f8f6f4) and NVFP4 (.kind::mxf4nvf4) mma.sync,
    // TMA and PDL; any device with those features and an image in this build is admitted.
    // The Ada build compiles no NVFP4/TMA/PDL route (NINFERNO_WITHOUT_NVFP4): its FP8 MMA is the
    // un-prefixed spelling sm_89 assembles and PDL degrades to stream order, so the gate asks
    // only for what that build can issue; NVFP4 weights still refuse at weight admission.
    try {
        require_device_support(
            device.device_class(),
#ifdef NINFERNO_WITHOUT_NVFP4
            {.mma_bf16 = true},
#else
            {.mma_bf16 = true, .mma_f8f6f4 = true, .mma_mxf4nvf4 = true, .tma = true, .pdl = true},
#endif
            "Qwen3.5 family runtime");
    } catch (const std::runtime_error& e) { throw std::invalid_argument(e.what()); }
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups);

} // namespace

void validate_ring_decoder_storage(const EngineOptions& options, std::string_view family) {
    const auto bf16_or_fp8 = [](KvCacheStorage storage) {
        return storage == KvCacheStorage::BFloat16 || storage == KvCacheStorage::Fp8E4M3Row256;
    };
    if (!bf16_or_fp8(options.kv_cache)) {
        throw std::invalid_argument(std::string(family) +
                                    " keeps keys and values in BF16 or FP8 (--kv-dtype bf16|fp8)");
    }
    if (!bf16_or_fp8(options.sliding_kv_cache)) {
        throw std::invalid_argument(std::string(family) + " keeps its sliding rings in BF16 or FP8 "
                                                          "(--sliding-kv-dtype bf16|fp8)");
    }
}

void validate_program_options(const EngineOptions& options, std::uint64_t max_positions,
                              std::uint32_t maximum_mtp_draft_tokens, bool mtp_copy_rounds) {
    if (maximum_mtp_draft_tokens > kMtpDecodeMaximumDrafts) {
        throw std::logic_error("a family's MTP depth exceeds the Program's MTP round domain");
    }
    if (options.max_context == 0 || options.max_context > max_positions) {
        throw std::invalid_argument("max_context " + std::to_string(options.max_context) +
                                    " exceeds the model's position capacity " +
                                    std::to_string(max_positions) +
                                    " (max_position_embeddings)");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,8]");
    }
    const std::uint32_t logical_pages = paged_kv_capacity_pages(options.max_context);
    const std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(options.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens < options.max_context) {
            throw std::invalid_argument("kv_capacity must be at least max_context");
        }
        const std::uint32_t requested_pages =
            paged_kv_capacity_pages(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages || requested_pages > maximum_pages64) {
            throw std::invalid_argument(
                "kv_capacity is outside the usable range for max_context and max_concurrency");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    const auto depths_within = [&](std::uint32_t maximum) {
        if (options.speculative.draft_tokens == 0 || options.speculative.draft_tokens > maximum) {
            return false;
        }
        return std::all_of(options.speculative.request_draft_tokens.begin(),
                           options.speculative.request_draft_tokens.end(),
                           [&](std::uint32_t k) { return k != 0 && k <= maximum; });
    };
    if (!speculation_steps_descend(options.speculative.draft_tokens, options.speculative.step_down)) {
        throw std::invalid_argument(
            "speculative step_down depths must descend below draft_tokens, down to 1");
    }
    if (options.speculative.ngram_draft_tokens != 0 &&
        options.speculative.ngram_draft_tokens != SpeculativeOptions::kCopyWindowAuto) {
        std::uint32_t deepest = options.speculative.draft_tokens;
        for (const std::uint32_t k : options.speculative.request_draft_tokens) {
            deepest = std::max(deepest, k);
        }
        if (options.speculative.backend != SpeculativeBackend::Mtp || !mtp_copy_rounds) {
            throw std::invalid_argument(
                "copy rounds (--ngram-draft-tokens) need MTP speculation on a model whose MTP "
                "round runs them (Qwen 3.5/3.6, Gemma 4)");
        }
        if (options.speculative.ngram_draft_tokens <= deepest ||
            options.speculative.ngram_draft_tokens > kMtpDecodeMaximumDrafts ||
            options.speculative.ngram_min_match == 0) {
            throw std::invalid_argument(
                "--ngram-draft-tokens must exceed every draft depth and be at most 15, and "
                "--ngram-min-match must be positive");
        }
    }
    if (!speculation_load_valid(options.speculative)) {
        throw std::invalid_argument(
            "speculative load bounds need ascending lanes from 2, depths within the window and "
            "no speculating bound after a bound of 0");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            !options.speculative.request_draft_tokens.empty() ||
            !options.speculative.load.empty() ||
            options.speculative.proposal_head != ProposalHead::Full ||
            names_draft_width(options.speculative)) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0, no request draft depths, "
                "no load bounds, the full proposal head and no DFlash draft width");
        }
        break;
    case SpeculativeBackend::Mtp:
        if (names_draft_width(options.speculative)) {
            throw std::invalid_argument("a DFlash draft width needs a DFlash drafter");
        }
        // spec/engine_support.h tells the products these windows; it must name Qwen 3.5's.
        static_assert(spec::engine_support("qwen3_5_text").mtp_depth == kMaximumMtpDraftTokens &&
                      spec::engine_support("qwen3_5_moe_text").mtp_depth ==
                          kMaximumMtpDraftTokens);
        if (!depths_within(maximum_mtp_draft_tokens)) {
            throw std::invalid_argument("MTP draft window must be in [1," +
                                        std::to_string(maximum_mtp_draft_tokens) +
                                        "] for this model");
        }
        break;
    case SpeculativeBackend::DFlash:
    case SpeculativeBackend::DFlash2:
        static_assert(spec::engine_support("qwen3_5_text").dflash_depth == 15 &&
                      spec::engine_support("qwen3_5_moe_text").dflash_depth == 15);
        if (!depths_within(15)) {
            throw std::invalid_argument("masked draft window must be in [1,15]");
        }
        break;
    }
}

void plan_program_buffers(LayoutBuilder& builder, const SequencePlanImpl& plan,
                          const ModelTraits& traits, PersistentLayout& out) {
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    out.round = qwen3_5::begin_round_state_layout(
        builder, qwen3_5::RoundStateSpec{.hidden         = traits.hidden,
                                         .output_rows    = traits.output_rows,
                                         .batch_capacity = plan.max_concurrency,
                                         .draft_window   = plan.draft_window,
                                         .backend        = plan.speculative_backend,
                                         .causal_scoring = plan.causal_scoring});
    out.prefill_hidden = add_tensor(builder, DType::BF16, {traits.hidden, effective_prefill_chunk},
                                    "step prefill hidden");
    if (plan.causal_scoring) {
        out.score_hidden = add_tensor(builder, DType::BF16,
                                      {traits.hidden, static_cast<std::int32_t>(kCausalScoreTile)},
                                      "causal score hidden staging");
    }
    qwen3_5::complete_round_state_layout(builder, out.round);
    if (!plan.causal_scoring) {
        out.token_counts =
            add_tensor(builder, DType::I32,
                       {traits.public_tokens, static_cast<std::int32_t>(plan.max_concurrency)},
                       "sampling token counts");
        const auto config_words = static_cast<std::int32_t>(
            (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
        out.sampling_config = add_tensor(
            builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
            "sampling config");
    }
    out.bytes = builder.finish(kArenaAlign, "persistent layout");
}

namespace {

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                 = std::make_unique<SequencePlanImpl>();
    impl->model               = inputs.model;
    impl->capacity            = inputs.capacity;
    impl->main_page_groups    = main_page_groups;
    impl->kv_capacity         = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency     = inputs.max_concurrency;
    impl->prefill_chunk       = inputs.prefill_chunk;
    impl->draft_window        = inputs.draft_window;
    impl->launch_windows      = inputs.launch_windows;
    impl->launch_window_min_batch = inputs.launch_window_min_batch;
    impl->launch_window_batches   = inputs.launch_window_batches;
    impl->speculative_batch_limit = inputs.speculative_batch_limit;
    impl->default_draft_depth = inputs.default_draft_depth;
    impl->speculative_backend = inputs.speculative_backend;
    impl->proposal_head       = inputs.proposal_head;
    impl->draft_width         = inputs.draft_width;
    impl->acceptance_floor    = inputs.acceptance_floor;
    impl->step_down           = inputs.step_down;
    impl->load                = inputs.load;
    impl->context_limit       = inputs.context_limit;
    impl->watchdog_window     = inputs.watchdog_window;
    impl->watchdog_min_rounds = inputs.watchdog_min_rounds;
    impl->shared_trip_floor   = inputs.shared_trip_floor;
    impl->ngram_draft_tokens      = inputs.ngram_draft_tokens;
    impl->ngram_min_match         = inputs.ngram_min_match;
    impl->kv_lease_window         = inputs.kv_lease_window;
    impl->features            = inputs.features;
    impl->use_cuda_graph      = inputs.use_cuda_graph;
    impl->causal_scoring      = inputs.causal_scoring;
    impl->device              = inputs.device;
    impl->multiprocessor_count = inputs.multiprocessor_count;
    impl->context_cache       = inputs.context_cache;
    impl->kv_storage          = inputs.kv_storage;
    impl->ring_storage        = inputs.ring_storage;
    impl->recurrent_state_dtype   = inputs.recurrent_state_dtype;
    impl->persistent          = impl->model->persistent_layout(*impl);
    impl->workspace           = impl->model->workspace_plan(*impl);
    if (impl->use_cuda_graph && !inputs.graph_memory_held) {
        // Definitions remain per execution profile, but only one executable is instantiated for
        // each reachable node-topology class: the allowance is a bound per instantiated class.
        // Measured (ninferno-serve's startup ledger: allowance - (free after startup - planned
        // slack); every graph is captured at startup, memory does not grow while serving): an
        // ordinary class takes 2-9 MiB, an MTP or DFlash class 3-7 MiB, whatever the visible
        // length (Qwen 3.6 35B MTP C=8 at 262,144 tokens: 34 MiB in all; Qwen 3.8 27B DFlash2
        // depths 5/11/15 at C=8: 370 MiB for 64 classes; Gemma 4 26B MTP C=8: 103 MiB). The bounds
        // keep about twice the largest. The driver and module state every engine materializes
        // besides (7-31 MiB) is the automatic KV headroom's (kDefaultKvCapacityHeadroomBytes).
        // Every generation engine captures the plain decode graphs (a speculative engine runs a
        // request whose acceptance watchdog tripped through them).
        constexpr std::uint64_t kOrdinaryClassBytes    = 12ULL * kMiB;
        constexpr std::uint64_t kSpeculativeClassBytes = 16ULL * kMiB;
        impl->graph_allowance_bytes = checked_mul(kOrdinaryClassBytes, impl->max_concurrency,
                                                  "ordinary exact-b graph allowance");
        // One set of speculative graphs per launch window, for the speculative batches it runs
        // under the load bounds (launch_window_batches: at most the widest speculative launch,
        // below a bound of 0; a depth only a bound asks from that bound's lanes; a window deeper
        // than every bound from a step's rows on, not at that many rows).
        impl->launch_window_graph_bytes.assign(impl->launch_windows.size(), 0);
        for (std::size_t w = 0; w < impl->launch_windows.size(); ++w) {
            const std::uint32_t window = impl->launch_windows[w];
            const std::uint32_t mask   = impl->launch_window_batches.at(w);
            const std::uint32_t last   = impl->speculative_batch_limit;
            std::size_t& window_bytes  = impl->launch_window_graph_bytes[w];
            if (impl->speculative_backend == SpeculativeBackend::Mtp) {
                const auto profiles = mtp_graph_profiles(impl->capacity, window);
                const std::size_t per_batch_allowance = graph_topology_allowance(
                    profiles, [](GraphExecutionProfile) { return kSpeculativeClassBytes; },
                    "MTP graph allowance");
                std::uint32_t batches = 0;
                for (std::uint32_t batch_size = 1; batch_size <= last; ++batch_size) {
                    batches += window_runs_batch(mask, batch_size) ? 1U : 0U;
                }
                window_bytes =
                    checked_mul(per_batch_allowance, batches, "MTP exact-b graph allowance");
            } else {
                for (std::uint32_t batch_size = 1; batch_size <= last; ++batch_size) {
                    if (!window_runs_batch(mask, batch_size)) { continue; }
                    const auto profiles = dflash_graph_profiles(
                        impl->speculative_backend, impl->capacity, window, batch_size);
                    window_bytes = checked_add(
                        window_bytes,
                        graph_topology_allowance(
                            profiles, [](GraphExecutionProfile) { return kSpeculativeClassBytes; },
                            "DFlash graph allowance"),
                        "DFlash exact-b graph allowance");
                }
            }
            impl->graph_allowance_bytes =
                checked_add(impl->graph_allowance_bytes, window_bytes, "speculative graphs");
        }
    }

    impl->device_reservation_bytes = checked_add(
        checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace

void validate_qwen3_5_options(const execution::Parameters& parameters, DeviceContext& device,
                              const EngineOptions& options) {
    validate_target_options(parameters, device, options);
}

PersistentLayout qwen3_5_persistent_layout(const execution::Parameters& parameters,
                                           const SequencePlanImpl& plan) {
    return persistent_layout(parameters, plan);
}

WorkspacePlan qwen3_5_workspace_plan(const execution::Parameters& parameters,
                                     const SequencePlanImpl& plan) {
    return build_workspace_plan(parameters, plan);
}

namespace {
// The recurrent state's element type a --state-dtype names.
DType recurrent_state_dtype(RecurrentStateStorage storage) {
    switch (storage) {
    case RecurrentStateStorage::BF16:
        return DType::BF16;
    case RecurrentStateStorage::FP16:
        return DType::FP16;
    case RecurrentStateStorage::FP32:
        break;
    }
    return DType::FP32;
}

// The copy window an automatic ngram_draft_tokens takes: the MTP backend's widest window on a
// family whose round runs copy rounds, where it is deeper than every draft depth; else none.
EngineOptions resolve_copy_window(const ModelExecution& model, const EngineOptions& options) {
    EngineOptions out = options;
    if (out.speculative.ngram_draft_tokens != SpeculativeOptions::kCopyWindowAuto) { return out; }
    std::uint32_t deepest = out.speculative.draft_tokens;
    for (const std::uint32_t k : out.speculative.request_draft_tokens) {
        deepest = std::max(deepest, k);
    }
    const bool runs = out.speculative.backend == SpeculativeBackend::Mtp &&
                      model.traits().copy_rounds && deepest < kMtpDecodeMaximumDrafts &&
                      out.speculative.ngram_min_match != 0;
    out.speculative.ngram_draft_tokens = runs ? kMtpDecodeMaximumDrafts : 0U;
    return out;
}
} // namespace

std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const ModelExecution& model, DeviceContext& device,
                           const EngineOptions& requested, bool graph_memory_held) {
    const EngineOptions options = resolve_copy_window(model, requested);
    model.validate_options(device, options);
    std::vector<std::uint32_t> launch_windows;
    std::vector<std::uint32_t> launch_window_min_batch;
    for (const LaunchWindowShape& window :
         launch_window_shapes(options.speculative, options.max_concurrency)) {
        launch_windows.push_back(window.drafts);
        launch_window_min_batch.push_back(window.min_batch);
    }
    const std::uint32_t batch_limit =
        launch_windows.empty()
            ? 0U
            : speculative_batch_limit(options.speculative.load, options.max_concurrency);
    SequencePlanningInputs inputs{
        .model                   = &model,
        .capacity                = options.max_context,
        .max_concurrency         = options.max_concurrency,
        .prefill_chunk           = std::min(options.prefill_chunk, options.max_context),
        .draft_window            = launch_windows.empty() ? 0U : launch_windows.back(),
        .launch_windows          = launch_windows,
        .launch_window_min_batch = launch_window_min_batch,
        .launch_window_batches =
            launch_window_batches(launch_windows, launch_window_min_batch, options.speculative.load,
                                  options.max_concurrency, batch_limit),
        .speculative_batch_limit = batch_limit,
        .default_draft_depth     = options.speculative.draft_tokens,
        .speculative_backend     = options.speculative.backend,
        .kv_storage              = options.kv_cache,
        .ring_storage            = options.sliding_kv_cache,
        .recurrent_state_dtype   = recurrent_state_dtype(options.recurrent_state),
        .proposal_head       = options.speculative.proposal_head,
        .draft_width         = options.speculative.draft_width,
        .acceptance_floor    = options.speculative.acceptance_floor,
        .step_down           = options.speculative.step_down,
        .load                = options.speculative.load,
        .context_limit       = options.speculative.context_limit,
        .watchdog_window     = options.speculative.watchdog_window,
        .watchdog_min_rounds = options.speculative.watchdog_min_rounds,
        .shared_trip_floor   = options.speculative.shared_trip_floor,
        .ngram_draft_tokens  = options.speculative.ngram_draft_tokens,
        .ngram_min_match     = options.speculative.ngram_min_match,
        .kv_lease_window     = options.kv_lease_window,
        .features            = models::load_options(options),
        .use_cuda_graph      = options.use_cuda_graph,
        .causal_scoring      = options.purpose == EnginePurpose::CausalScoring,
        .device              = options.device,
        .multiprocessor_count = device.multiprocessor_count(),
        .context_cache       = options.context_cache,
        .graph_memory_held   = graph_memory_held,
    };
    const std::uint32_t logical_pages = paged_kv_capacity_pages(inputs.capacity);
    const std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(inputs.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    const auto maximum_pages = static_cast<std::uint32_t>(maximum_pages64);

    auto planner     = std::make_unique<qwen3_5::detail::SequencePlannerImpl>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::SequenceCapacityCurve{
          .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
          .minimum_main_page_groups             = minimum_pages,
          .maximum_main_page_groups             = maximum_pages,
          .minimum_device_reservation_bytes     = planner->minimum->device_reservation_bytes,
          .bytes_per_additional_main_page_group = 0,
    };
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen3.5 sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_main_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen3.5 sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(main_page_groups);
    std::unique_ptr<SequencePlanImpl> plan;
    if (main_page_groups == planner->curve.minimum_main_page_groups) {
        plan = std::move(planner->minimum);
    } else {
        plan = build_sequence_candidate(planner->inputs, main_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen3.5 physical sequence layout is not affine in Main KV page capacity");
    }
    return plan;
}

} // namespace ninferno::models::qwen3_5::detail
