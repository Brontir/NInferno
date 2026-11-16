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
#include "models/qwen3_5/program/planning/workspace_plan.h"
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

PersistentLayout persistent_layout(const execution::Parameters& parameters,
                                   const SequencePlanImpl& plan) {
    return plan_persistent_layout(
        qwen3_5_state_shape(parameters.model.config(),
                            dimension(parameters.model.resources().public_token_count)),
        plan);
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
        options.max_context > parameters.model.config().draft->position_capacity()) {
        throw std::invalid_argument("max_context exceeds the selected draft position capacity");
    }
    validate_state_storage(
        qwen3_5_state_shape(parameters.model.config(),
                            dimension(parameters.model.resources().public_token_count)),
        options);
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

namespace {

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    auto impl        = sequence_plan_shell(inputs, main_page_groups);
    impl->persistent = impl->model->persistent_layout(*impl);
    impl->workspace  = impl->model->workspace_plan(*impl);
    plan_graph_allowance(*impl, inputs.graph_memory_held);
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

StateShape qwen3_5_state_shape(const qwen3_5::Config& config, std::int32_t public_tokens) {
    const auto& text = config.text;
    StateShape out{
        .family           = StateFamily::Qwen35,
        .full_layers      = text.full_attention_layers,
        .kv_heads         = text.attention ? dimension(text.attention->num_key_value_heads) : 0,
        .head_dim         = text.attention ? dimension(text.attention->head_dim) : 0,
        .recurrent_layers = text.linear_attention_layers,
        .hidden           = dimension(text.hidden_size),
        .output_rows      = dimension(text.vocab_size),
        .public_tokens    = public_tokens,
    };
    if (text.gdn) {
        out.value_heads = dimension(text.gdn->linear_num_value_heads);
        out.value_dim   = dimension(text.gdn->linear_value_head_dim);
        out.key_heads   = dimension(text.gdn->linear_num_key_heads);
        out.key_dim     = dimension(text.gdn->linear_key_head_dim);
        out.conv_kernel = dimension(text.gdn->linear_conv_kernel_dim);
    }
    if (config.draft) {
        const DraftConfig& draft = *config.draft;
        const std::int32_t features =
            dimension(text.hidden_size * std::uint64_t(draft.target_layer_ids.size()));
        out.drafter = DrafterStateShape{
            .full_layers          = draft.full_layer_count(),
            .local_layers         = draft.local_layer_count(),
            .window               = draft.sliding_window.value_or(0),
            .kv_heads             = dimension(draft.attention.num_key_value_heads),
            .head_dim             = dimension(draft.attention.head_dim),
            .feature_rows         = features,
            .prefill_feature_rows = features,
            .dflash2              = draft.dflash2.has_value(),
        };
    }
    return out;
}

WorkspacePlan prepared_workspace_plan(const qwen3_5::Config& config, std::int32_t public_tokens,
                                      const execution::PreparedParameters& forms,
                                      const SequencePlanImpl& plan) {
    // The text, MTP, proposal and drafter arenas over the prepared forms (workspace_plan.h), then
    // the Vision encoder's region above them.
    WorkspacePlan out = build_workspace_plan(
        workspace_inputs(config, public_tokens, forms.text, forms.mtp, forms.proposal, forms.draft),
        plan);
    if (plan.features.vision) {
        if (!config.vision || !forms.vision) {
            throw std::invalid_argument("workspace inputs: a Vision plan without the encoder");
        }
        const std::uint32_t merged = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(plan.capacity, kMaximumVisionItemTokens));
        out.vision = execution::VisionContext::plan_workspace(*config.vision, *forms.vision, merged,
                                                              out.general_capacity);
        out.capacity = std::max(out.capacity, out.vision->capacity_bytes);
    }
    return out;
}

WorkspacePlan qwen3_5_workspace_plan(const execution::Parameters& parameters,
                                     const SequencePlanImpl& plan) {
    return prepared_workspace_plan(parameters.model.config(),
                                   dimension(parameters.model.resources().public_token_count),
                                   parameters, plan);
}

std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const ModelExecution& model, DeviceContext& device,
                           const EngineOptions& options, bool graph_memory_held) {
    model.validate_options(device, options);
    return make_sequence_planner_impl(model, options, device.multiprocessor_count(),
                                      graph_memory_held);
}

std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const ModelExecution& model, const EngineOptions& options,
                           std::int32_t multiprocessor_count, bool graph_memory_held) {
    SequencePlanningInputs inputs = sequence_planning_inputs(
        options, model.traits().copy_rounds, multiprocessor_count, graph_memory_held);
    inputs.model                  = &model;
    const SequencePageRange range = sequence_page_range(inputs.capacity, inputs.max_concurrency);
    const std::uint32_t minimum_pages = range.minimum;
    const std::uint32_t maximum_pages = range.maximum;

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
