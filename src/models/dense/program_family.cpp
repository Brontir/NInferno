#include "models/dense/program_family.h"

#include "core/device_class.h"
#include "models/dense/program_execution.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/program/ordinary_decode.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/ring_kv_bindings.h"
#include "models/qwen3_5/program/prefill_text_call.h"
#include "ninferno/ops/argmax.h"
#include "ninferno/ops/position.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/scalar.h"
#include "ninferno/ops/scatter.h"
#include "runtime/contract/timing.h"
#include "spec/engine_support.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::dense {
namespace detail {

DenseExecution::DenseExecution(const Model& model, const speculation::DFlashDrafter* drafter,
                               DeviceContext& device, std::int32_t public_tokens,
                               ActivationMode activations)
    : model_(model), c_(model.config()), pass_(model, device), drafter_(drafter),
      invariant_(activations == ActivationMode::Exact),
      traits_{.hidden             = c_.hidden_size,
              .output_rows        = c_.vocab_size,
              .public_tokens      = public_tokens,
              .weight_bytes       = model.device_bytes() + (drafter ? drafter->device_bytes() : 0),
              .draft_feature_rows = drafter ? drafter->weights().feature_projection.weight.k : 0,
              // The target's positions, and its drafter's when the drafter records a limit.
              .max_positions =
                  drafter && drafter->config().max_positions != 0
                      ? std::min<std::uint64_t>(c_.max_positions, drafter->config().max_positions)
                      : static_cast<std::uint64_t>(c_.max_positions),
              .dflash_block_width = drafter != nullptr} {
    if (public_tokens <= 0 || public_tokens > c_.vocab_size) {
        throw std::invalid_argument(
            "dense decoder public token domain exceeds the embedding rows");
    }
    // Each layer's place among its kind: a full layer's plane in the paged pool, a sliding
    // layer's ring in the state image.
    for (const LayerSpec& spec : c_.layer_specs) {
        std::uint32_t& count = spec.window == 0 ? full_layers_ : sliding_layers_;
        index_.push_back(count++);
    }
    if (full_layers_ == 0) {
        throw std::invalid_argument(
            "the dense decoder Program keeps its context in full-attention layers; this "
            "model has none");
    }
    if (drafter_ != nullptr) {
        check_drafter();
        const auto& dc = drafter_->config();
        tap_slots_.assign(static_cast<std::size_t>(c_.layers), -1);
        for (std::size_t i = 0; i < dc.taps.size(); ++i) {
            tap_slots_[static_cast<std::size_t>(dc.taps[i])] = static_cast<std::int32_t>(i);
        }
        for (const std::int32_t slot : tap_slots_) {
            if (slot >= 0) { tap_order_.push_back(slot); }
        }
        split_features_ = !drafter_->weights().feature_slices.empty();
        for (const std::int32_t window : dc.windows) {
            std::uint32_t& count = window == 0 ? drafter_full_layers_ : drafter_local_layers_;
            drafter_index_.push_back(count++);
        }
    }
}

models::LoadOptions DenseExecution::load_options() const {
    return {.speculative = drafter_ ? SpeculativeBackend::DFlash : SpeculativeBackend::None,
            .activations = invariant_ ? ActivationMode::Exact : ActivationMode::Fast};
}

void DenseExecution::validate_options(DeviceContext& device, const EngineOptions& options) const {
    if (options.purpose != EnginePurpose::Generation) {
        throw std::invalid_argument("the dense decoder serves generation only; ninferno-perplexity "
                                    "scores Qwen models (a source build "
                                    "scores this family with ninferno-dense score)");
    }
    if (options.enable_vision) {
        throw std::invalid_argument("dense decoder artifacts carry no Vision encoder");
    }
    if (options.speculative.backend !=
        (drafter_ ? SpeculativeBackend::DFlash : SpeculativeBackend::None)) {
        throw std::invalid_argument(
            "the dense decoder speculates only with a DFlash drafter (--spec dflash --drafter "
            "DRAFTER)");
    }
    if (drafter_ != nullptr) {
        // spec/engine_support.h tells the products the family's DFlash window; it must name the
        // Program's round.
        static_assert(spec::kDFlashBlockColumns ==
                          static_cast<std::int32_t>(qwen3_5::kDFlashDecodeMaximumWidth) &&
                      spec::engine_support("dense_decoder").dflash_depth ==
                          spec::kDFlashBlockColumns - 1 &&
                      spec::engine_support("moe_decoder").dflash_depth ==
                          spec::kDFlashBlockColumns - 1 &&
                      !spec::engine_support("dense_decoder").dflash2 &&
                      !spec::engine_support("dense_decoder").proposal_head &&
                      spec::engine_support("dense_decoder").dflash_block_width &&
                      spec::engine_support("moe_decoder").dflash_block_width);
        // A round verifies at most the drafts one block holds.
        const auto deepest = static_cast<std::uint32_t>(drafter_->config().block_size - 1);
        const auto within  = [&](std::uint32_t k) { return k <= deepest; };
        if (!within(options.speculative.draft_tokens) ||
            !std::all_of(options.speculative.request_draft_tokens.begin(),
                         options.speculative.request_draft_tokens.end(), within)) {
            throw std::invalid_argument(
                "DFlash draft window exceeds the drafter's block (at most " +
                std::to_string(deepest) + " drafts)");
        }
        if (options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "the dense decoder's DFlash drafter scores with the target's own head");
        }
    }
    qwen3_5::detail::validate_ring_decoder_storage(options, "the dense decoder");
    // FP8 rings on a model with no ring would only be logged: refused, as Qwen 3.5 and
    // Nemotron-H refuse it.
    if (sliding_layers_ == 0 && options.sliding_kv_cache != KvCacheStorage::BFloat16) {
        throw std::invalid_argument("this dense decoder has no sliding-window layers "
                                    "(--sliding-kv-dtype is for their rings)");
    }
    qwen3_5::detail::validate_program_options(options, traits_.max_positions, 0);
    try {
        require_device_support(device.device_class(), {.mma_bf16 = true},
                               "dense decoder family runtime");
    } catch (const std::runtime_error& e) { throw std::invalid_argument(e.what()); }
}

PersistentLayout DenseExecution::persistent_layout(const SequencePlanImpl& plan) const {
    if (!plan.context_cache.device_state_slots) {
        throw std::logic_error("dense decoder context cache options are not normalized");
    }
    const std::uint64_t slots = static_cast<std::uint64_t>(plan.max_concurrency) +
                                *plan.context_cache.device_state_slots;
    if (slots > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("dense decoder state image slot count exceeds int32");
    }
    // --kv-dtype fp8: the full layers' pool holds E4M3 codes and an FP32 scale per row (the
    // Gemma4KvLayer FP8 storage the attention reads); the rings follow --sliding-kv-dtype.
    const bool fp8 = plan.kv_storage == KvCacheStorage::Fp8E4M3Row256;
    const PagedKVVectorLayout vector =
        fp8 ? PagedKVVectorLayout{DType::FP8_E4M3FN, c_.head_dim, DType::FP32, 1}
            : PagedKVVectorLayout{DType::BF16, c_.head_dim, DType::U8, 0};
    const KvCacheStorage storage = fp8 ? KvCacheStorage::Fp8E4M3Row256 : KvCacheStorage::BFloat16;

    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen3_5::plan_decoder_state(
        builder, qwen3_5::DecoderStateSpec{
                     .full_attention_layers     = full_layers_,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = c_.kv_heads,
                     .attention_head_dim        = c_.head_dim,
                     .kv_storage                = storage,
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
                     .text_physical_page_groups = plan.main_page_groups,
                     // The attention reads keys and values from a head-major pool.
                     .layer_storage = PagedKVStorageLayout{storage, c_.head_dim, vector, vector},
                     .plane_order   = PagedKVPlaneOrder::HeadMajor,
                 });
    qwen3_5::StateImageSpec images{
        .linear = {.layers = 0, .slot_count = static_cast<std::int32_t>(slots)},
        .hidden = c_.hidden_size,
    };
    if (sliding_layers_ != 0) {
        images.rings.push_back({.layers   = sliding_layers_,
                                .capacity = ring_capacity(plan),
                                .kv_heads = c_.kv_heads,
                                .head_dim = c_.head_dim,
                                .fp8      = plan.ring_storage == KvCacheStorage::Fp8E4M3Row256});
    }
    if (plan.features.masked_draft()) { images.dflash_local = dflash_local_spec(); }
    out.state_images = qwen3_5::plan_state_image_device_pool(builder, images);
    if (plan.features.masked_draft()) { out.dflash = dflash_layout(builder, plan); }
    qwen3_5::detail::plan_program_buffers(builder, plan, traits_, out);
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    return out;
}

WorkspacePlan DenseExecution::workspace_plan(const SequencePlanImpl& plan) const {
    const auto chunk   = static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const auto batch   = static_cast<std::int32_t>(plan.max_concurrency);
    const auto context = static_cast<std::int32_t>(plan.capacity);
    // Every capacity query under the route profiles its calls run under
    // (ninferno/ops/prefill_routes.h): the invariant one under --activations exact, and the prefill
    // one for a prefill chunk, whose routes can take another workspace than a decode width's.
    const ops::ScopedInvariantRoutes routes(invariant_);
    WorkspacePlan out;
    // One call's activations and every Op's transient capacity, plus the sampler of the rows it
    // draws for.
    {
        const ops::ScopedPrefillRoutes prefill(true);
        out.text_prefill = DensePass::workspace_bytes(model_, context, chunk, 1) +
                           ops::sampling_workspace_capacity_bytes(traits_.public_tokens, 1, 1) +
                           (drafter_ != nullptr ? prefill_tap_bytes(chunk) : 0);
    }
    out.ordinary_round = DensePass::workspace_bytes(model_, context, batch, batch) +
                         ops::sampling_workspace_capacity_bytes(traits_.public_tokens, 1, batch);
    if (plan.features.masked_draft()) { dflash_workspace(plan, out); }
    out.general_capacity =
        std::max({out.text_prefill, out.ordinary_round, out.dflash_context, out.dflash_round});
    out.capacity = out.general_capacity;
    return out;
}

std::uint32_t DenseExecution::prefill_call_tokens(std::uint32_t chunk, std::uint32_t cursor,
                                                  std::uint32_t end) const {
    return program::ungridded_call_tokens(chunk, end - cursor);
}

void DenseExecution::output_logits(const Tensor& hidden, Tensor& logits, WorkspaceArena& work,
                                   cudaStream_t) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    pass_.head(work, hidden, logits);
}

void DenseExecution::capture_ordinary_decode_batch(program::OrdinaryBatchContext& state,
                                                   std::int32_t batch_size,
                                                   ops::CausalAttentionExecutionEnvelope envelope,
                                                   DecodeGraphDefinition& definition) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    program::capture_graph(state, definition, ordinary_body(state, batch_size, envelope));
}

void DenseExecution::ordinary_decode_batch(program::OrdinaryBatchContext& state,
                                           std::int32_t batch_size,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           DecodeGraphExecutable* executable) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    // A row that forks from a checkpoint writes its first token in this launch: its destination
    // takes the source's rings first, and the launch then runs in place.
    for (std::int32_t row = 0; row < batch_size; ++row) {
        state.execution.state_images.copy_rings(state.host_ingress.state_source_slots[row],
                                                state.host_ingress.state_destination_slots[row],
                                                state.execution.device.stream);
    }
    program::run_prepared(state, executable, ordinary_body(state, batch_size, envelope));
}

std::int32_t DenseExecution::session_context(const program::ExecutionCore& core) {
    if (core.capacity == 0 ||
        core.capacity > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::logic_error("dense decoder session context is out of range");
    }
    return static_cast<std::int32_t>(core.capacity);
}

std::uint32_t DenseExecution::uncommitted_row_limit(const SequencePlanImpl& plan) const {
    return sliding_layers_ == 0 ? plan.capacity : widest_call(plan);
}

std::uint32_t DenseExecution::ring_capacity(const SequencePlanImpl& plan) const {
    return static_cast<std::uint32_t>(c_.max_window()) + widest_call(plan) - 1U;
}

std::uint32_t DenseExecution::widest_call(const SequencePlanImpl& plan) {
    // A prefill chunk, or a speculative verify of the draft window.
    return std::max(std::min(plan.prefill_chunk, plan.capacity), plan.draft_window + 1U);
}

std::vector<ops::Gemma4KvLayer>
DenseExecution::bindings(const qwen3_5::PagedKVCache& cache,
                         const qwen3_5::StateImageDevicePool& images, const Tensor& tables,
                         const Tensor& rows_full, const Tensor& rows_ring,
                         const Tensor& stored) const {
    return program::ring_decoder_bindings(
        cache, images, index_, [&](std::size_t i) { return c_.layer_specs[i].window; }, tables,
        rows_full, rows_ring, stored);
}

std::vector<ops::Gemma4KvLayer> DenseExecution::sequence_bindings(program::PrefillContext& state,
                                                                  std::int32_t slot,
                                                                  Tensor& rows) const {
    return program::ring_decoder_sequence_bindings(
        state, slot, rows, [&](const auto&... args) { return bindings(args...); });
}

std::function<void()>
DenseExecution::ordinary_body(program::OrdinaryBatchContext& state, std::int32_t batch_size,
                              ops::CausalAttentionExecutionEnvelope envelope) const {
    return program::ordinary_decode_body(
        state, batch_size, traits_.public_tokens, "dense decoder",
        [this, &state, batch_size, envelope](program::ExecutionCore& core, const Tensor& tokens,
                                             const Tensor& positions, const Tensor& table_rows,
                                             const Tensor& destinations) {
            const auto layers = bindings(state.text_cache, core.state_images,
                                         state.text_cache.batch_layer_view(0).block_tables,
                                         table_rows, destinations);
            return pass_.forward(core.work, tokens, positions,
                                 {.layers = layers, .batch = batch_size}, session_context(core),
                                 {.max_visible_keys = envelope.max_visible_keys, .max_tokens = 1});
        },
        [this](auto& work, const Tensor& hidden, Tensor& logits) {
            pass_.head(work, hidden, logits);
        });
}

program::PrefillChunkResult DenseExecution::prefill_text_chunk(
    program::PrefillContext& state, std::span<const TokenId> ids, std::uint32_t nominal_length,
    std::optional<std::uint32_t> split_frontier, bool finalize_at_end) const {
    runtime::ExecutionTimingRecorder timing;
    program::ExecutionCore& core = state.execution;
    const cudaStream_t s         = core.device.stream;
    const program::PrefillTextCall call = program::prefill_text_call(
        state, ids, nominal_length, split_frontier, finalize_at_end, "dense decoder",
        [this](std::uint32_t chunk, std::uint32_t cursor, std::uint32_t frontier) {
            return prefill_call_tokens(chunk, cursor, frontier);
        });
    const auto [base, len, t, is_last] = call;
    // The prefill route profile: a token's result does not depend on the call that carries it
    // (ninferno-dense runs its prefill under the same profile).
    const ops::ScopedPrefillRoutes route_profile(true);
    // --activations exact: every width on one summation order (a prefix-cache frontier may leave a
    // call of a single token).
    const ops::ScopedInvariantRoutes invariant_routes(invariant_);

    core.work.reset();
    if (state.state_source_slot != state.state_destination_slot) {
        // A fork's first write: the destination takes the source's rings, then runs in place.
        core.state_images.copy_rings(state.state_source_slot, state.state_destination_slot, s);
    }
    auto [device_ids, positions] = program::upload_prefill_text(core, ids, call);
    Tensor rows       = core.work.alloc(DType::I32, {2});
    const auto layers = sequence_bindings(state, state.state_destination_slot, rows);
    ops::set_i32_scalar(core.io.rope_delta, 0, s);

    // A DFlash drafter reads the target's residual stream after its tapped layers: the forward
    // copies them into the chunk's feature buffer, and the whole chunk is committed context.
    std::optional<FeatureTap> tap;
    if (state.dflash != nullptr) {
        if (drafter_ == nullptr) { throw std::logic_error("DFlash prefill without a drafter"); }
        tap = prefill_tap(state.dflash->prefill_features.slice(1, 0, t), core.work, s);
    }

    Tensor hidden = core.prefill_hidden.slice(1, 0, t);
    {
        auto scope          = core.work.scope();
        const Tensor normed = pass_.forward(
            core.work, device_ids, positions, {.layers = layers}, session_context(core),
            {.max_visible_keys = base + len, .max_tokens = t}, Activations::Recorded,
            tap ? &*tap : nullptr);
        CUDA_CHECK(cudaMemcpyAsync(hidden.data, normed.data, hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, s));
    }
    if (tap) {
        // The chunk's own bindings, as the target's: its StateImage slot (a checkpoint may have
        // forked it) and its drafter KV row (decode may have used another compact row). The
        // Program sets them per chunk in the context; the decode frame is not prefill's.
        if (!core.io.dflash_prefill || state.dflash_prefill_host_ingress == nullptr) {
            throw std::logic_error("DFlash prefill controls are unavailable");
        }
        auto& frame                        = *core.io.dflash_prefill;
        *state.dflash_prefill_host_ingress = {
            .append_count           = t,
            .state_destination_slot = state.state_destination_slot,
            .full_kv_table_row      = state.dflash_kv_table_row,
        };
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, state.dflash_prefill_host_ingress,
                                   sizeof(qwen3_5::DFlashPrefillIngress), cudaMemcpyHostToDevice,
                                   s));
        const auto exact = static_cast<std::uint32_t>(t);
        append_context(core, *state.dflash, tap->sink.view({prefill_feature_rows(), t, 1}),
                       positions.view({t, 1}), frame.append_count, frame.state_destination_slot,
                       frame.full_kv_table_row, {exact, exact}, split_features_);
    }
    const Tensor last = hidden.slice(1, t - 1, 1);
    program::finish_prefill_text(state, call, split_frontier, last, traits_.public_tokens,
                                 [this](auto& work, const Tensor& x, Tensor& logits) {
                                     pass_.head(work, x, logits);
                                 });
    timing.begin_wait();
    core.device.synchronize();
    timing.end_wait();
    core.work.reset();
    return {.processed_tokens = len, .finalized = is_last, .timing = timing.finish()};
}

} // namespace detail

std::unique_ptr<const qwen3_5::detail::ModelExecution>
make_program_execution(const Model& model, const speculation::DFlashDrafter* drafter,
                       DeviceContext& device, std::int32_t public_tokens,
                       ActivationMode activations) {
    return std::make_unique<const detail::DenseExecution>(model, drafter, device, public_tokens,
                                                          activations);
}

} // namespace ninferno::models::dense
