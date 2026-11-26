#include "models/inkling2/program_family.h"

#include "core/device_class.h"
#include "models/inkling2/pass.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/program/ordinary_decode.h"
#include "models/qwen3_5/program/planning/sequence_plan.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/prefill_text_call.h"
#include "models/qwen3_5/program/ring_kv_bindings.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/sampling.h"
#include "runtime/contract/timing.h"
#include "spec/engine_support.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::inkling2 {
namespace {

using qwen3_5::detail::ModelExecution;
using qwen3_5::detail::ModelTraits;
using qwen3_5::detail::PersistentLayout;
using qwen3_5::detail::SequencePlanImpl;
using qwen3_5::detail::WorkspacePlan;
namespace program = qwen3_5::execution;

// The kind of a visit slot: the kind of the layer its visit runs.
std::vector<LayerKind> slot_kinds(const Config& c) {
    const auto visits = visit_schedule(c);
    std::vector<LayerKind> out;
    for (const Visit& v : visits) {
        if (v.slot >= static_cast<std::int32_t>(out.size())) {
            out.resize(static_cast<std::size_t>(v.slot) + 1, LayerKind::Sliding);
        }
        out[static_cast<std::size_t>(v.slot)] = c.kinds[static_cast<std::size_t>(v.layer)];
    }
    return out;
}

class Inkling2Execution final : public ModelExecution {
public:
    Inkling2Execution(const Model& model, std::int32_t public_tokens, ActivationMode activations,
                      EnginePurpose purpose)
        : model_(model), c_(model.config()), pass_(model), kinds_(slot_kinds(c_)),
          invariant_(activations == ActivationMode::Exact), purpose_(purpose),
          traits_{.hidden        = c_.hidden_size,
                  .output_rows   = c_.vocab_rows,
                  .public_tokens = public_tokens,
                  .weight_bytes  = model.device_bytes(),
                  .max_positions = static_cast<std::uint64_t>(c_.max_positions),
                  .copy_rounds   = qwen3_5::detail::family_copy_rounds(StateFamily::Inkling2)} {
        if (public_tokens <= 0 || public_tokens > c_.vocab_rows) {
            throw std::invalid_argument("Inkling 2 public token domain exceeds the head's rows");
        }
        // Each visit slot's place among the slots of its kind: a full slot's plane in the paged
        // pool, a sliding slot's ring in the state image. Its convolution history is the linear
        // state's layer of the slot's own index.
        for (const LayerKind kind : kinds_) {
            std::uint32_t& count = kind == LayerKind::Full ? full_slots_ : sliding_slots_;
            index_.push_back(count++);
        }
        if (full_slots_ == 0 || sliding_slots_ == 0) {
            throw std::invalid_argument("Inkling 2 Program requires full and sliding visits");
        }
    }

    [[nodiscard]] const ModelTraits& traits() const noexcept override { return traits_; }

    [[nodiscard]] std::vector<std::string> tune_routes(cudaStream_t stream) const override {
        return qwen3_5::detail::tune_moe_layers(stream, [&](const auto& tune) {
            for (const auto& layer : model_.weights().layers) { tune(layer.moe, layer.moe_config); }
        });
    }

    [[nodiscard]] models::LoadOptions load_options() const override {
        return {.purpose     = purpose_,
                .speculative = SpeculativeBackend::None,
                .activations = invariant_ ? ActivationMode::Exact : ActivationMode::Fast};
    }

    void validate_options(DeviceContext& device, const EngineOptions& options) const override {
        if (options.enable_vision) {
            throw std::invalid_argument("Inkling 2 artifacts carry no Vision encoder");
        }
        static_assert(spec::engine_support("inkling2_text").mtp_depth == 0 &&
                      spec::engine_support("inkling2_text").dflash_depth == 0);
        if (options.speculative.backend != SpeculativeBackend::None) {
            throw std::invalid_argument("Inkling 2 runs no speculative backend (--spec off)");
        }
        qwen3_5::detail::validate_state_storage(state_shape(c_, traits_.public_tokens), options);
        qwen3_5::detail::validate_program_options(options, traits_.max_positions, 0);
        try {
            require_device_support(device.device_class(), {.mma_bf16 = true},
                                   "Inkling 2 family runtime");
        } catch (const std::runtime_error& e) { throw std::invalid_argument(e.what()); }
    }

    [[nodiscard]] PersistentLayout persistent_layout(const SequencePlanImpl& plan) const override {
        return qwen3_5::detail::plan_persistent_layout(state_shape(c_, traits_.public_tokens),
                                                       plan);
    }

    [[nodiscard]] WorkspacePlan workspace_plan(const SequencePlanImpl& plan) const override {
        const ops::ScopedInvariantRoutes routes(invariant_);
        return target_workspace_plan(model_, traits_.public_tokens, plan);
    }

    [[nodiscard]] std::uint32_t prefill_call_tokens(std::uint32_t chunk, std::uint32_t cursor,
                                                    std::uint32_t end) const override {
        return program::ungridded_call_tokens(chunk, end - cursor);
    }

    [[nodiscard]] program::PrefillChunkResult
    prefill_text_chunk(program::PrefillContext& state, std::span<const TokenId> ids,
                       std::uint32_t nominal_length, std::optional<std::uint32_t> split_frontier,
                       bool finalize_at_end) const override;

    void output_logits(const Tensor& hidden, Tensor& logits, WorkspaceArena& work,
                       cudaStream_t stream) const override {
        const ops::ScopedInvariantRoutes routes(invariant_);
        pass_.head(work, hidden, logits, stream);
    }

    void capture_ordinary_decode_batch(program::OrdinaryBatchContext& state,
                                       std::int32_t batch_size,
                                       ops::CausalAttentionExecutionEnvelope envelope,
                                       DecodeGraphDefinition& definition) const override {
        (void)envelope;
        const ops::ScopedInvariantRoutes routes(invariant_);
        program::capture_graph(state, definition, ordinary_body(state, batch_size));
    }

    void ordinary_decode_batch(program::OrdinaryBatchContext& state, std::int32_t batch_size,
                               ops::CausalAttentionExecutionEnvelope envelope,
                               DecodeGraphExecutable* executable) const override {
        (void)envelope;
        const ops::ScopedInvariantRoutes routes(invariant_);
        // A row that forks from a checkpoint writes its first token in this launch: its
        // destination takes the source's rings and convolution history first, and the launch
        // then runs in place.
        for (std::int32_t row = 0; row < batch_size; ++row) {
            settle_fork(state.execution, state.host_ingress.state_source_slots[row],
                        state.host_ingress.state_destination_slots[row]);
        }
        program::run_prepared(state, executable, ordinary_body(state, batch_size));
    }

private:
    static void settle_fork(program::ExecutionCore& core, std::int32_t from, std::int32_t to) {
        if (from == to) { return; }
        core.state_images.copy_rings(from, to, core.device.stream);
        core.state_images.linear().copy_slot(from, to, core.device.stream);
    }

    [[nodiscard]] std::vector<ops::Gemma4KvLayer>
    bindings(const qwen3_5::PagedKVCache& cache, const qwen3_5::StateImageDevicePool& images,
             const Tensor& tables, const Tensor& rows_full, const Tensor& rows_ring) const {
        return program::ring_decoder_bindings(
            cache, images, index_,
            [&](std::size_t i) { return kinds_[i] == LayerKind::Full ? 0 : c_.sliding_window; },
            tables, rows_full, rows_ring, Tensor{});
    }

    // Every visit slot's convolution history: the linear state's layer of the slot, the batch
    // row's plane state slot `slots[b]` (FP32 [3, C] a slot, contiguous: the pool's slot axis is
    // its outermost).
    [[nodiscard]] std::vector<ops::Inkling2ConvState>
    conv_bindings(const qwen3_5::StateImageDevicePool& images, const Tensor& slots) const {
        const auto& linear = images.linear();
        std::vector<ops::Inkling2ConvState> out;
        out.reserve(kinds_.size());
        for (std::size_t i = 0; i < kinds_.size(); ++i) {
            const Tensor conv = linear.layer_view(static_cast<std::uint32_t>(i)).conv;
            out.push_back({.data        = static_cast<float*>(conv.data),
                           .slot_stride = static_cast<std::int64_t>(conv.ne[0]) * conv.ne[1],
                           .slots       = static_cast<const std::int32_t*>(slots.data)});
        }
        return out;
    }

    std::function<void()> ordinary_body(program::OrdinaryBatchContext& state,
                                        std::int32_t batch_size) const {
        return program::ordinary_decode_body(
            state, batch_size, traits_.public_tokens, "Inkling 2",
            [this, &state, batch_size](program::ExecutionCore& core, const Tensor& tokens,
                                       const Tensor& positions, const Tensor& table_rows,
                                       const Tensor& destinations) {
                const auto kv   = bindings(state.text_cache, core.state_images,
                                           state.text_cache.batch_layer_view(0).block_tables,
                                           table_rows, destinations);
                const auto conv = conv_bindings(core.state_images, destinations);
                // One token a row: column b is batch row b.
                return pass_.forward(core.work, tokens, positions,
                                     {.kv = kv, .conv = conv, .batch = batch_size},
                                     core.device.stream);
            },
            [this, &state](auto& work, const Tensor& hidden, Tensor& logits) {
                pass_.head(work, hidden, logits, state.execution.device.stream);
            });
    }

    const Model& model_;
    const Config& c_;
    mutable Pass pass_;
    std::vector<LayerKind> kinds_; // a visit slot's kind
    bool invariant_;               // ActivationMode::Exact: the invariant route profile
    EnginePurpose purpose_;
    ModelTraits traits_;
    std::vector<std::uint32_t> index_; // a visit slot's place among the slots of its kind
    std::uint32_t full_slots_    = 0;
    std::uint32_t sliding_slots_ = 0;
};

program::PrefillChunkResult Inkling2Execution::prefill_text_chunk(
    program::PrefillContext& state, std::span<const TokenId> ids, std::uint32_t nominal_length,
    std::optional<std::uint32_t> split_frontier, bool finalize_at_end) const {
    const program::PrefillTextCall call = program::prefill_text_call(
        state, ids, nominal_length, split_frontier, finalize_at_end, "Inkling 2",
        [this](std::uint32_t chunk, std::uint32_t cursor, std::uint32_t frontier) {
            return prefill_call_tokens(chunk, cursor, frontier);
        });
    runtime::ExecutionTimingRecorder timing;
    program::ExecutionCore& core       = state.execution;
    const cudaStream_t s               = core.device.stream;
    const auto [base, len, t, is_last] = call;
    const ops::ScopedInvariantRoutes invariant_routes(invariant_);
    const ops::ScopedPrefillRoutes prefill_routes(true);

    core.work.reset();
    // A fork's first write: the destination takes the source's rings and convolution history,
    // then runs in place.
    settle_fork(core, state.state_source_slot, state.state_destination_slot);
    auto [device_ids, positions] = program::upload_prefill_text(core, ids, call);
    // rows[0]: the one-row table's row; rows[1]: the sequence's state slot.
    Tensor rows   = core.work.alloc(DType::I32, {2});
    const auto kv = program::ring_decoder_sequence_bindings(
        state, state.state_destination_slot, rows,
        [&](const auto& cache, const auto& images, const Tensor& tables, const Tensor& rows_full,
            const Tensor& rows_ring) {
            return bindings(cache, images, tables, rows_full, rows_ring);
        });
    const auto conv = conv_bindings(core.state_images, rows.slice(0, 1, 1));

    Tensor hidden = core.prefill_hidden.slice(1, 0, t);
    {
        auto scope          = core.work.scope();
        const Tensor normed = pass_.forward(core.work, device_ids, positions,
                                            {.kv = kv, .conv = conv, .batch = 1}, s);
        CUDA_CHECK(
            cudaMemcpyAsync(hidden.data, normed.data, hidden.bytes(), cudaMemcpyDeviceToDevice, s));
    }
    const Tensor last = hidden.slice(1, t - 1, 1);
    program::finish_prefill_text(
        state, call, split_frontier, last, traits_.public_tokens,
        [this, s](auto& work, const Tensor& x, Tensor& logits) { pass_.head(work, x, logits, s); });
    timing.begin_wait();
    core.device.synchronize();
    timing.end_wait();
    core.work.reset();
    return {.processed_tokens = len, .finalized = is_last, .timing = timing.finish()};
}

} // namespace

StateShape state_shape(const Config& config, std::int32_t public_tokens) {
    StateShape out{
        .family           = StateFamily::Inkling2,
        .kv_heads         = config.kv_heads,
        .head_dim         = config.head_dim,
        .sliding_kv_heads = config.kv_heads,
        .sliding_head_dim = config.head_dim,
        .window           = static_cast<std::uint32_t>(config.sliding_window),
        .conv_channels    = config.conv_channels(),
        .hidden           = config.hidden_size,
        .output_rows      = config.vocab_rows,
        .public_tokens    = public_tokens,
    };
    for (const LayerKind kind : slot_kinds(config)) {
        ++(kind == LayerKind::Full ? out.full_layers : out.sliding_layers);
        ++out.recurrent_layers;
    }
    return out;
}

std::unique_ptr<const ModelExecution>
make_program_execution(const Model& model, DeviceContext& device, std::int32_t public_tokens,
                       ActivationMode activations, EnginePurpose purpose) {
    (void)device;
    return std::make_unique<const Inkling2Execution>(model, public_tokens, activations, purpose);
}

WorkspacePlan target_workspace_plan(const Model& model, std::int32_t public_tokens,
                                    const SequencePlanImpl& plan) {
    const auto chunk = static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const auto batch = static_cast<std::int32_t>(plan.max_concurrency);
    WorkspacePlan out;
    {
        const ops::ScopedPrefillRoutes prefill(true);
        out.text_prefill = Pass::work_bytes(model, chunk) +
                           ops::sampling_workspace_capacity_bytes(public_tokens, 1, 1);
    }
    out.ordinary_round = Pass::work_bytes(model, batch) +
                         ops::sampling_workspace_capacity_bytes(public_tokens, 1, batch);
    if (plan.causal_scoring) {
        out.causal_score = qwen3_5::detail::causal_score_workspace_bytes(
            model.config().vocab_rows, model.weights().output_head);
    }
    out.general_capacity = std::max({out.text_prefill, out.ordinary_round, out.causal_score});
    out.capacity         = out.general_capacity;
    return out;
}

} // namespace ninferno::models::inkling2
