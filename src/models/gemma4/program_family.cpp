#include "models/gemma4/program_family.h"

#include "core/device_class.h"
#include "models/gemma4/mtp_round.h"
#include "models/gemma4/target.h"
#include "models/gemma4/vision.h"
#include "models/qwen3_5/program/vision_prefill.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/program/ordinary_decode.h"
#include "models/qwen3_5/program/speculative/mtp_verify_accept.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/ring_kv_bindings.h"
#include "models/qwen3_5/program/prefill_text_call.h"
#include "ninferno/ops/argmax.h"
#include "ninferno/ops/position.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/scalar.h"
#include "ninferno/ops/scatter.h"
#include "ninferno/ops/mtp_round.h"
#include "ninferno/ops/speculative_round.h"
#include "runtime/contract/timing.h"
#include "spec/engine_support.h"

#include <algorithm>
#include <limits>
#include <array>
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ninferno::models::gemma4 {
namespace {

using qwen3_5::detail::ModelExecution;
using qwen3_5::detail::ModelTraits;
using qwen3_5::detail::PersistentLayout;
using qwen3_5::detail::SequencePlanImpl;
using qwen3_5::detail::VisionPrefillPlan;
using qwen3_5::detail::VisionUseSpan;
using qwen3_5::detail::VisionWorkspacePlan;
using qwen3_5::detail::WorkspacePlan;
namespace program = qwen3_5::execution;

constexpr std::size_t kVisionAlignment = 256;

std::size_t align_vision(std::size_t bytes) {
    return (bytes + kVisionAlignment - 1) / kVisionAlignment * kVisionAlignment;
}

// One request's images through the vision tower. Each image is one prefill call of its own soft
// tokens (they see one another on the sliding layers); it is encoded when its call begins, into
// the handoff region above the encoder's workspace.
class Gemma4VisionPrefill final : public program::VisionPrefill {
public:
    Gemma4VisionPrefill(DeviceContext& device, const VisionEncoder& encoder, DeviceSpan workspace,
                        const VisionWorkspacePlan& workspace_plan,
                        qwen3_5::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                        std::size_t& handoff_peak_bytes)
        : device_(device), encoder_(encoder), workspace_(workspace), workspace_plan_(workspace_plan),
          prompt_(prompt), plan_(plan), handoff_peak_bytes_(handoff_peak_bytes) {
        if (workspace_.data == nullptr || workspace_.bytes < workspace_plan_.capacity_bytes ||
            plan_.uses.empty()) {
            throw std::invalid_argument("Gemma 4 Vision prefill plan or workspace is invalid");
        }
        std::uint32_t previous_end = 0;
        for (const VisionUseSpan& use : plan_.uses) {
            if (use.begin >= use.end || use.begin < previous_end ||
                use.end > prompt_.token_ids.size() ||
                use.prepared_item_index >= prompt_.vision_items.size() ||
                use.prepared_item_index >= prompt_.media_payloads.size() ||
                use.end - use.begin > workspace_plan_.max_merged_tokens) {
                throw std::invalid_argument("Gemma 4 Vision item spans are invalid or unordered");
            }
            previous_end = use.end;
        }
        timers_.reserve(plan_.uses.size());
    }

    // An image is one call: a unit whose end would cut one ends where it begins (a prompt of 1900
    // text tokens and a 282-token image no longer asks for a larger --prefill-chunk at a 2048 one).
    [[nodiscard]] std::uint32_t unit_end(std::uint32_t cursor,
                                         std::uint32_t end) const noexcept override {
        for (const VisionUseSpan& use : plan_.uses) {
            if (use.begin > cursor && use.begin < end && end < use.end) { return use.begin; }
        }
        return end;
    }

    // The image whose soft tokens hold `position`, or null.
    [[nodiscard]] const VisionUseSpan* use_at(std::uint32_t position) const noexcept {
        for (const VisionUseSpan& use : plan_.uses) {
            if (use.begin <= position && position < use.end) { return &use; }
        }
        return nullptr;
    }

    // The first image beginning at or after `position`, or null.
    [[nodiscard]] const VisionUseSpan* next_use(std::uint32_t position) const noexcept {
        for (const VisionUseSpan& use : plan_.uses) {
            if (use.begin >= position) { return &use; }
        }
        return nullptr;
    }

    // The image's soft tokens, BF16 [text hidden, its tokens], encoded on its first call.
    [[nodiscard]] Tensor soft_tokens(const VisionUseSpan& use) {
        const auto count = static_cast<std::int32_t>(use.end - use.begin);
        Tensor output(static_cast<unsigned char*>(workspace_.data) +
                          workspace_plan_.handoff_offset_bytes,
                      DType::BF16, {workspace_plan_.output_hidden, count});
        if (active_item_ != use.prepared_item_index) {
            const qwen3_5::VisionItem& item = prompt_.vision_items[use.prepared_item_index];
            const auto& payload             = prompt_.media_payloads[use.prepared_item_index];
            if (!payload) { throw std::logic_error("Gemma 4 Vision item payload was released"); }
            DeviceSpan scratch = workspace_;
            scratch.bytes      = workspace_plan_.handoff_offset_bytes;
            timers_.emplace_back(device_);
            timers_.back().start();
            encoder_.encode(VisionImage{.patches = payload->span(),
                                        .width   = item.grid.width,
                                        .rows    = item.grid.height},
                            output, scratch, device_.stream);
            timers_.back().record_stop();
            active_item_          = use.prepared_item_index;
            active_handoff_bytes_ = static_cast<std::size_t>(output.bytes());
            handoff_peak_bytes_   = std::max(handoff_peak_bytes_, active_handoff_bytes_);
            encoded_.push_back(use.prepared_item_index);
        }
        return output;
    }

    void release_encoded_media_payloads() noexcept override {
        for (const std::uint32_t index : encoded_) {
            if (index < prompt_.media_payloads.size()) { prompt_.media_payloads[index].reset(); }
        }
        encoded_.clear();
    }

    void retire_handoff() noexcept override {
        active_item_.reset();
        active_handoff_bytes_ = 0;
    }

    [[nodiscard]] double elapsed_seconds() const override {
        double milliseconds = 0.0;
        for (const CudaEventTimer& timer : timers_) { milliseconds += timer.elapsed_ms(); }
        return milliseconds / 1000.0;
    }

    [[nodiscard]] std::size_t active_handoff_bytes() const noexcept override {
        return active_handoff_bytes_;
    }

private:
    DeviceContext& device_;
    const VisionEncoder& encoder_;
    DeviceSpan workspace_;
    const VisionWorkspacePlan& workspace_plan_;
    qwen3_5::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    std::size_t& handoff_peak_bytes_;
    std::optional<std::uint32_t> active_item_;
    std::size_t active_handoff_bytes_ = 0;
    std::vector<std::uint32_t> encoded_;
    std::vector<CudaEventTimer> timers_;
};

static_assert(gemma4::kMaximumMtpDraftTokens + 1 <= program::kNarrowCallTokens &&
                  gemma4::kMaximumMtpDraftTokens <= qwen3_5::kMtpDecodeMaximumDrafts,
              "a Gemma 4 MTP verify must stay a narrow call inside the Program's MTP round domain");

class Gemma4Execution final : public ModelExecution {
public:
    Gemma4Execution(const Model& model, const Drafter* drafter, DeviceContext& device,
                    std::int32_t public_tokens, ActivationMode activations, EnginePurpose purpose)
        : model_(model), c_(model.config()), pass_(model, device), drafter_(drafter),
          invariant_(activations == ActivationMode::Exact), purpose_(purpose),
          traits_{.hidden        = c_.hidden_size,
                  .output_rows   = c_.vocab_size,
                  .public_tokens = public_tokens,
                  .weight_bytes  = model.device_bytes(),
                  .max_positions = static_cast<std::uint64_t>(c_.max_positions),
                  .copy_rounds   = qwen3_5::detail::family_copy_rounds(StateFamily::Gemma4)} {
        if (public_tokens <= 0 || public_tokens > c_.vocab_size) {
            throw std::invalid_argument("Gemma 4 public token domain exceeds the embedding rows");
        }
        // Each layer's place among its kind: a full layer's plane in the paged pool, a sliding
        // layer's ring in the state image. A layer that reads shared keys and values takes its
        // source's place and adds none.
        for (std::int32_t i = 0; i < c_.layers; ++i) {
            if (!c_.owns_kv(i)) {
                index_.push_back(index_[static_cast<std::size_t>(c_.kv_source[i])]);
                continue;
            }
            std::uint32_t& count =
                c_.kinds[i] == LayerKind::Full ? full_layers_ : sliding_layers_;
            index_.push_back(count++);
        }
        if (full_layers_ == 0 || sliding_layers_ == 0) {
            throw std::invalid_argument("Gemma 4 Program requires full and sliding layers");
        }
        if (drafter_ != nullptr) { drafter_pass_.emplace(*drafter_, model, device); }
        if (model.vision_config()) {
            encoder_.emplace(*model.vision_config(), *model.vision_weights());
        }
    }

    [[nodiscard]] const ModelTraits& traits() const noexcept override { return traits_; }
    [[nodiscard]] std::vector<std::string> tune_routes(cudaStream_t stream) const override {
        return qwen3_5::detail::tune_moe_layers(stream, [&](const auto& tune) {
            for (const auto& layer : model_.weights().layers) {
                if (layer.generic_moe) { tune(*layer.generic_moe, layer.generic_moe_config); }
            }
        });
    }

    [[nodiscard]] models::LoadOptions load_options() const override {
        return {.purpose     = purpose_,
                .vision      = encoder_.has_value(),
                .speculative = drafter_ ? SpeculativeBackend::Mtp : SpeculativeBackend::None,
                .activations = invariant_ ? ActivationMode::Exact : ActivationMode::Fast};
    }

    void validate_options(DeviceContext& device, const EngineOptions& options) const override {
        if (options.enable_vision && !encoder_) {
            throw std::invalid_argument("Gemma 4: this artifact was loaded without its vision "
                                        "tower; serve it without --vision");
        }
        if (encoder_ && c_.ple()) {
            throw std::invalid_argument("Gemma 4: images on a model with per-layer inputs (E2B, "
                                        "E4B) are not supported; serve it without --vision");
        }
        // spec/engine_support.h tells the products this family runs MTP only, up to its own
        // window and without the proposal head; it must say so.
        static_assert(spec::engine_support("gemma4_text").mtp_depth ==
                          static_cast<std::int32_t>(gemma4::kMaximumMtpDraftTokens) &&
                      spec::engine_support("gemma4_text").dflash_depth == 0 &&
                      !spec::engine_support("gemma4_text").proposal_head);
        if (options.speculative.backend !=
            (drafter_ ? SpeculativeBackend::Mtp : SpeculativeBackend::None)) {
            throw std::invalid_argument(
                "Gemma 4 speculates with its MTP assistant only (--spec mtp --drafter ASSISTANT)");
        }
        if (options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument("Gemma 4's assistant scores with the full head "
                                        "(--lm-head-draft is for Qwen models)");
        }
        qwen3_5::detail::validate_state_storage(gemma4::state_shape(c_, traits_.public_tokens),
                                                options);
        qwen3_5::detail::validate_program_options(
            options, traits_.max_positions, gemma4::kMaximumMtpDraftTokens, traits_.copy_rounds);
        try {
            // The Ada build issues no .kind::mxf4nvf4 MMA (NINFERNO_WITHOUT_NVFP4); its Gemma route
            // is the A16 register/tiled MMA, so the gate asks for BF16 only there.
            require_device_support(
                device.device_class(),
#ifdef NINFERNO_WITHOUT_NVFP4
                {.mma_bf16 = true},
#else
                {.mma_bf16 = true, .mma_mxf4nvf4 = true},
#endif
                "Gemma 4 family runtime");
        } catch (const std::runtime_error& e) { throw std::invalid_argument(e.what()); }
    }

    [[nodiscard]] PersistentLayout persistent_layout(const SequencePlanImpl& plan) const override {
        return qwen3_5::detail::plan_persistent_layout(
            gemma4::state_shape(c_, traits_.public_tokens), plan);
    }

    [[nodiscard]] WorkspacePlan workspace_plan(const SequencePlanImpl& plan) const override {
        const auto chunk   = static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
        const auto batch   = static_cast<std::int32_t>(plan.max_concurrency);
        const auto context = static_cast<std::int32_t>(plan.capacity);
        // Every capacity query under the route profiles its calls run under
        // (ninferno/ops/prefill_routes.h): the invariant one under --activations exact, and the
        // prefill one for a prefill chunk.
        const ops::ScopedInvariantRoutes routes(invariant_);
        WorkspacePlan out;
        // One call's activations and every Op's transient capacity (Target::workspace_bytes),
        // plus the sampler of the rows it draws for.
        {
            const ops::ScopedPrefillRoutes prefill(true);
            out.text_prefill = Target::workspace_bytes(model_, context, chunk, 1) +
                               ops::sampling_workspace_capacity_bytes(traits_.public_tokens, 1, 1);
        }
        out.ordinary_round =
            Target::workspace_bytes(model_, context, batch, batch) +
            ops::sampling_workspace_capacity_bytes(traits_.public_tokens, 1, batch);
        if (drafter_ != nullptr) {
            const auto width = static_cast<std::int32_t>(plan.draft_window) + 1;
            // A speculative round runs at most the widest speculative launch's rows.
            const auto rows = static_cast<std::int32_t>(
                qwen3_5::detail::speculative_round_batches(plan));
            // The most verify columns a round runs: under a load bound, a batch at least its lanes
            // wide runs no window deeper than the bound.
            const auto columns = static_cast<std::int32_t>(
                qwen3_5::detail::speculative_round_columns(plan));
            const std::size_t draft_one =
                DrafterPass::workspace_bytes(*drafter_, context, 1) + 1024;
            out.mtp_prefill = out.text_prefill + draft_one;
            out.mtp_round =
                Target::workspace_bytes(model_, context, columns, columns) +
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    traits_.public_tokens, 1, width - 1, 1, rows) +
                DrafterPass::workspace_bytes(*drafter_, context, rows) + 1024;
        }
        // Causal scoring (ninferno-perplexity): the Program's scoring tiles over the output head.
        if (plan.causal_scoring) {
            out.causal_score = qwen3_5::detail::causal_score_workspace_bytes(
                traits_.output_rows, model_.weights().output_head);
        }
        out.general_capacity = std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill,
                                         out.mtp_round, out.causal_score});
        out.capacity         = out.general_capacity;
        if (plan.features.vision) {
            const VisionConfig& v = vision_config();
            if (static_cast<std::int32_t>(chunk) < v.default_tokens) {
                throw std::invalid_argument(
                    "Gemma 4 vision: a prefill call holds a whole image (" +
                    std::to_string(v.default_tokens) +
                    " soft tokens see one another); raise --prefill-chunk");
            }
            VisionWorkspacePlan vision;
            vision.output_hidden          = c_.hidden_size;
            vision.max_merged_tokens      = static_cast<std::uint32_t>(v.default_tokens);
            vision.general_capacity_bytes = out.general_capacity;
            vision.encode_peak_bytes =
                encoder_->workspace_bytes(v.default_tokens * v.pooling * v.pooling);
            vision.handoff_offset_bytes =
                align_vision(std::max(out.general_capacity, vision.encode_peak_bytes));
            vision.handoff_capacity_bytes =
                align_vision(static_cast<std::size_t>(c_.hidden_size) * v.default_tokens * 2);
            vision.capacity_bytes = vision.handoff_offset_bytes + vision.handoff_capacity_bytes;
            out.vision            = vision;
            out.capacity          = std::max(out.capacity, vision.capacity_bytes);
        }
        return out;
    }

    [[nodiscard]] qwen3_5::VisionControlPlan
    plan_vision_control(const qwen3_5::PreparedPromptData& prompt) const override {
        const VisionConfig& v = vision_config();
        const std::int32_t pool = v.pooling;
        qwen3_5::VisionControlPlan plan{.spatial_merge_size = pool, .position_grid_side = 1};
        std::size_t patch_cursor = 0, tokens = 0, next = 0;
        for (const qwen3_5::VisionItem& item : prompt.vision_items) {
            const std::int32_t h = item.grid.height, w = item.grid.width;
            if (item.modality != qwen3_5::PromptModality::Image || item.grid.temporal != 1 ||
                h <= 0 || w <= 0 || h % pool != 0 || w % pool != 0) {
                throw std::invalid_argument(
                    "Gemma 4 vision: an image's patch grid must be whole pooling cells");
            }
            if (item.patch_features != static_cast<std::size_t>(v.patch_width())) {
                throw std::invalid_argument(
                    "Gemma 4 vision: the image was prepared for another vision encoder");
            }
            const auto patches = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
            const auto count   = patches / static_cast<std::size_t>(pool * pool);
            if (item.patch_begin != patch_cursor || item.patch_count != patches ||
                item.token_spans.size() != 1 || item.token_spans.front().count != count ||
                item.token_spans.front().begin < next ||
                item.token_spans.front().begin + count > prompt.token_ids.size() ||
                count > static_cast<std::size_t>(v.default_tokens)) {
                throw std::invalid_argument("Gemma 4 vision: the prompt's image spans are invalid");
            }
            const auto begin = static_cast<std::uint32_t>(item.token_spans.front().begin);
            plan.items.push_back(qwen3_5::VisionItemControlPlan{
                .token_begin  = begin,
                .token_end    = begin + static_cast<std::uint32_t>(count),
                .merged_count = count,
            });
            next = begin + count;
            patch_cursor += patches;
            tokens += count;
        }
        if (patch_cursor != prompt.prepare.raw_patches || tokens != prompt.prepare.vision_tokens) {
            throw std::invalid_argument("Gemma 4 vision: the image spans do not cover the prompt's");
        }
        return plan;
    }

    [[nodiscard]] std::size_t
    vision_item_workspace_bytes(std::size_t patches, std::size_t merged_tokens,
                                const VisionWorkspacePlan& plan) const override {
        (void)merged_tokens;
        (void)plan;
        if (patches > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("Gemma 4 vision: too many patches");
        }
        return vision_encoder().workspace_bytes(static_cast<std::int32_t>(patches));
    }

    [[nodiscard]] std::unique_ptr<program::VisionPrefill>
    begin_vision_prefill(DeviceContext& device, DeviceSpan workspace,
                         const VisionWorkspacePlan& workspace_plan,
                         qwen3_5::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                         std::size_t& handoff_peak_bytes) const override {
        return std::make_unique<Gemma4VisionPrefill>(device, vision_encoder(), workspace,
                                                     workspace_plan, prompt, plan,
                                                     handoff_peak_bytes);
    }

    [[nodiscard]] program::PrefillChunkResult
    prefill_multimodal_chunk(program::PrefillContext& state,
                             const qwen3_5::PreparedPromptData& prompt,
                             program::VisionPrefill& vision, std::uint32_t nominal_length,
                             std::optional<std::uint32_t> split_frontier,
                             bool finalize_at_end) const override;

    // The drafter keeps no keys or values: a bridge before an image suffix has nothing to fill
    // (and builds no proposal, which the suffix's last call makes).
    void mtp_bridge_multimodal(program::PrefillContext&, const qwen3_5::PreparedPromptData&,
                               program::VisionPrefill&,
                               const program::MtpBridgeInput&) const override {}

    [[nodiscard]] std::uint32_t prefill_call_tokens(std::uint32_t chunk, std::uint32_t cursor,
                                                    std::uint32_t end) const override {
        return program::ungridded_call_tokens(chunk, end - cursor);
    }

    [[nodiscard]] program::PrefillChunkResult
    prefill_text_chunk(program::PrefillContext& state, std::span<const TokenId> ids,
                       std::uint32_t nominal_length, std::optional<std::uint32_t> split_frontier,
                       bool finalize_at_end) const override;

    void output_logits(const Tensor& hidden, Tensor& logits, WorkspaceArena& work,
                       cudaStream_t) const override {
        const ops::ScopedInvariantRoutes routes(invariant_);
        pass_.head(work, hidden, logits);
    }

    // Full layers never read a key past the reader; a sliding ring holds the widest call's rows
    // beyond the window (ring_capacity).
    [[nodiscard]] std::uint32_t
    uncommitted_row_limit(const SequencePlanImpl& plan) const override {
        return widest_call(plan);
    }

    void capture_ordinary_decode_batch(program::OrdinaryBatchContext& state,
                                       std::int32_t batch_size,
                                       ops::CausalAttentionExecutionEnvelope envelope,
                                       DecodeGraphDefinition& definition) const override {
        const ops::ScopedInvariantRoutes routes(invariant_);
        program::capture_graph(state, definition, ordinary_body(state, batch_size, envelope));
    }

    void ordinary_decode_batch(program::OrdinaryBatchContext& state, std::int32_t batch_size,
                               ops::CausalAttentionExecutionEnvelope envelope,
                               DecodeGraphExecutable* executable) const override {
        const ops::ScopedInvariantRoutes routes(invariant_);
        settle_forks(state.execution, state.host_ingress, batch_size);
        program::run_prepared(state, executable, ordinary_body(state, batch_size, envelope));
    }

    void mtp_bridge_and_propose(program::PrefillContext& state, const Tensor& next_token,
                                const Tensor& previous_hidden, std::int32_t position,
                                std::span<const std::int32_t>,
                                bool build_proposal) const override;

    void capture_mtp_decode_batch(program::MtpBatchContext& state, std::int32_t batch_size,
                                  std::uint32_t k, program::MtpCausalAttentionEnvelopes envelopes,
                                  DecodeGraphDefinition& definition) const override {
        const ops::ScopedInvariantRoutes routes(invariant_);
        const ops::ScopedRoundLanes lanes(batch_size);
        program::capture_graph(state, definition, mtp_body(state, batch_size, k, envelopes));
    }

    void mtp_decode_batch(program::MtpBatchContext& state, std::int32_t batch_size,
                          std::uint32_t k, program::MtpCausalAttentionEnvelopes envelopes,
                          DecodeGraphExecutable* executable) const override {
        const ops::ScopedInvariantRoutes routes(invariant_);
        const ops::ScopedRoundLanes lanes(batch_size);
        settle_forks(state.execution, state.host_ingress, batch_size);
        program::run_prepared(state, executable, mtp_body(state, batch_size, k, envelopes));
    }

private:
    [[nodiscard]] const VisionConfig& vision_config() const {
        if (!encoder_ || !model_.vision_config()) {
            throw std::invalid_argument("Gemma 4: this artifact was loaded without its vision tower");
        }
        return *model_.vision_config();
    }

    [[nodiscard]] const VisionEncoder& vision_encoder() const {
        (void)vision_config();
        return *encoder_;
    }

    // One prefill call over `call` of the prompt `ids`: the text embedding, or `embeddings` (one
    // image's soft tokens, attended to one another on the sliding layers).
    program::PrefillChunkResult run_prefill_call(program::PrefillContext& state,
                                                 std::span<const TokenId> ids,
                                                 const program::PrefillTextCall& call,
                                                 std::optional<std::uint32_t> split_frontier,
                                                 const Tensor* embeddings) const;

    // A row that forks from a checkpoint writes its first token in this launch: its destination
    // takes the source's rings first, and the launch then runs in place. The copy precedes the
    // graph, from the host ingress the launch was filled with.
    template <class Ingress>
    static void settle_forks(program::ExecutionCore& core, const Ingress& ingress,
                             std::int32_t batch_size) {
        for (std::int32_t row = 0; row < batch_size; ++row) {
            core.state_images.copy_rings(ingress.state_source_slots[row],
                                         ingress.state_destination_slots[row], core.device.stream);
        }
    }

    // The one-sequence bindings of a prefill or bridge (ring_decoder_sequence_bindings).
    [[nodiscard]] std::vector<ops::Gemma4KvLayer> sequence_bindings(program::PrefillContext& state,
                                                                    std::int32_t slot,
                                                                    Tensor& rows) const {
        return program::ring_decoder_sequence_bindings(
            state, slot, rows, [&](const auto&... args) { return bindings(args...); });
    }

    // Drafts `extent` tokens for one sequence into the prefill proposal: the first step reads the
    // pending `token` at `position` and the target state that predicted it; every step sees the
    // target's keys through position - 1.
    void propose_one(program::ExecutionCore& core, const std::vector<ops::Gemma4KvLayer>& layers,
                     const Tensor& token, const Tensor& hidden, std::int32_t position,
                     std::uint32_t extent) const {
        if (!drafter_pass_ || !core.io.mtp ||
            extent > static_cast<std::uint32_t>(core.io.mtp->draft_tokens.ne[0])) {
            throw std::logic_error("Gemma 4 MTP proposal has no drafter or exceeds its window");
        }
        const cudaStream_t s = core.device.stream;
        auto scope           = core.work.scope();
        Tensor positions     = core.work.alloc(DType::I32, {2});
        const std::array<std::int32_t, 2> host{position, position - 1};
        CUDA_CHECK(cudaMemcpyAsync(positions.data, host.data(), sizeof(host),
                                   cudaMemcpyHostToDevice, s));
        Tensor carry        = core.work.alloc(DType::BF16, {c_.hidden_size, 1});
        const Tensor drafts = core.io.mtp->draft_tokens;
        const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(position), 1};
        for (std::uint32_t i = 0; i < extent; ++i) {
            const auto step    = static_cast<std::int32_t>(i);
            const Tensor input = i == 0 ? token : drafts.slice(0, step - 1, 1);
            Tensor next        = drafts.slice(0, step, 1);
            drafter_pass_->step(core.work, input, i == 0 ? hidden : carry, positions.slice(0, 0, 1),
                                positions.slice(0, 1, 1),
                                layers[drafter_pass_->sliding_source()],
                                layers[drafter_pass_->full_source()], envelope, next, carry);
        }
    }

    std::function<void()> mtp_body(program::MtpBatchContext& state, std::int32_t batch_size,
                                   std::uint32_t k,
                                   program::MtpCausalAttentionEnvelopes envelopes) const {
        return [this, &state, batch_size, k, envelopes] {
            // The drafter's depth: below k in a copy window
            // (SpeculativeOptions::ngram_draft_tokens).
            const std::uint32_t depth = state.draft_depth == 0 ? k : state.draft_depth;
            if (!drafter_pass_ || batch_size <= 0 ||
                batch_size > static_cast<std::int32_t>(kMaximumConcurrency) || k == 0 ||
                k > qwen3_5::kMtpDecodeMaximumDrafts ||
                depth > std::min<std::uint32_t>(k, gemma4::kMaximumMtpDraftTokens)) {
                throw std::logic_error("Gemma 4 MTP decode batch state is incomplete");
            }
            program::ExecutionCore& core   = state.execution;
            const cudaStream_t s           = core.device.stream;
            qwen3_5::MtpDecodeState& frame = state.frame;
            const std::int32_t b           = batch_size;
            const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
            CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                       sizeof(qwen3_5::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                       s));
            // The frame's first b rows.
            qwen3_5::MtpDecodeState fb = frame.rows(b);

            // Verify: the pending token and the current drafts of every row, B runs of k + 1
            // columns through the target, every activation in BF16 (a verify computes what k + 1
            // single steps would).
            ops::speculative_prepare_verify_inputs(fb.anchors, fb.current_drafts, fb.base_frontiers,
                                                   fb.current_extents, fb.verify_ids,
                                                   fb.target_positions, s);
            // A row with fewer drafts than k repeats its last position in the padding columns;
            // only its extent + 1 real columns write keys and values (the ingress's valid
            // columns), so the padding never races the last real column for its slot.
            const auto layers =
                bindings(state.text_cache, core.state_images,
                         state.text_cache.batch_layer_view(0).block_tables, fb.text_kv_table_rows,
                         fb.state_destination_slots, fb.target_valid_columns);
            const std::int32_t columns = width * b;
            {
                auto scope          = core.work.scope();
                const Tensor normed = pass_.forward(
                    core.work, fb.verify_ids.view({columns}), fb.target_positions.view({columns}),
                    {.layers = layers, .batch = b},
                    {.max_visible_keys = envelopes.target_verify.max_visible_keys,
                     .max_tokens       = width},
                    Activations::A16Only);
                CUDA_CHECK(cudaMemcpyAsync(fb.target_hidden.data, normed.data,
                                           fb.target_hidden.bytes(), cudaMemcpyDeviceToDevice, s));
            }
            program::mtp_verify_accept(fb, frame.sampling, c_.hidden_size, columns,
                                       traits_.public_tokens,
                                       static_cast<std::int32_t>(state.text_cache.max_context()),
                                       state.continuation_hidden_store, core.work, s,
                                       [this](auto& work, const Tensor& hidden, Tensor& logits) {
                                           pass_.head(work, hidden, logits);
                                       },
                                       static_cast<std::int32_t>(depth));

            // Draft the next round from each row's new pending token (the correction or bonus,
            // now its anchor) at its new frontier, and the target state that predicted it.
            auto scope             = core.work.scope();
            Tensor key_positions   = core.work.alloc(DType::I32, {b});
            Tensor minus_one       = core.work.alloc(DType::I32, {1});
            ops::set_i32_scalar(minus_one, -1, s);
            ops::offset_i32_positions(fb.base_frontiers, minus_one, key_positions, s);
            const ops::Gemma4AttentionEnvelope envelope{envelopes.target_verify.max_visible_keys,
                                                        1};
            for (std::uint32_t step = 0; step < depth; ++step) {
                const auto i = static_cast<std::int32_t>(step);
                const Tensor input =
                    i == 0 ? fb.anchors : fb.next_drafts.slice(1, i - 1, 1).view({b});
                Tensor next = fb.next_drafts.slice(1, i, 1).view({b});
                drafter_pass_->step(
                    core.work, input, i == 0 ? fb.target_continuation_hidden : fb.ar_hidden,
                    fb.base_frontiers, key_positions, layers[drafter_pass_->sliding_source()],
                    layers[drafter_pass_->full_source()], envelope, next, fb.ar_hidden);
            }
            ops::copy_to_pinned_host(&state.host_egress, frame.egress.data,
                                     sizeof(qwen3_5::MtpDecodeEgress), s);
        };
    }

    [[nodiscard]] static std::uint32_t widest_call(const SequencePlanImpl& plan) {
        return qwen3_5::detail::ring_decoder_widest_call(plan);
    }

    // One binding per layer (ring_decoder_bindings): a full layer reads the paged pool through
    // `tables` (row rows_full[b] of batch row b), a sliding layer its ring in state image
    // rows_ring[b].
    [[nodiscard]] std::vector<ops::Gemma4KvLayer>
    bindings(const qwen3_5::PagedKVCache& cache, const qwen3_5::StateImageDevicePool& images,
             const Tensor& tables, const Tensor& rows_full, const Tensor& rows_ring,
             const Tensor& stored = {}) const {
        return program::ring_decoder_bindings(
            cache, images, index_,
            [&](std::size_t i) { return c_.kinds[i] == LayerKind::Full ? 0 : c_.sliding_window; },
            tables, rows_full, rows_ring, stored);
    }

    std::function<void()> ordinary_body(program::OrdinaryBatchContext& state,
                                        std::int32_t batch_size,
                                        ops::CausalAttentionExecutionEnvelope envelope) const {
        return program::ordinary_decode_body(
            state, batch_size, traits_.public_tokens, "Gemma 4",
            [this, &state, batch_size, envelope](program::ExecutionCore& core, const Tensor& tokens,
                                                 const Tensor& positions, const Tensor& table_rows,
                                                 const Tensor& destinations) {
                const auto layers = bindings(state.text_cache, core.state_images,
                                             state.text_cache.batch_layer_view(0).block_tables,
                                             table_rows, destinations);
                return pass_.forward(
                    core.work, tokens, positions, {.layers = layers, .batch = batch_size},
                    {.max_visible_keys = envelope.max_visible_keys, .max_tokens = 1});
            },
            [this](auto& work, const Tensor& hidden, Tensor& logits) {
                pass_.head(work, hidden, logits);
            });
    }

    const Model& model_;
    const Config& c_;
    TargetPass pass_;
    const Drafter* drafter_ = nullptr;
    std::optional<DrafterPass> drafter_pass_;
    bool invariant_; // ActivationMode::Exact: the invariant route profile
    EnginePurpose purpose_;
    ModelTraits traits_;
    std::vector<std::uint32_t> index_;
    std::uint32_t full_layers_    = 0;
    std::uint32_t sliding_layers_ = 0;
    std::optional<VisionEncoder> encoder_;
};

program::PrefillChunkResult Gemma4Execution::prefill_text_chunk(
    program::PrefillContext& state, std::span<const TokenId> ids, std::uint32_t nominal_length,
    std::optional<std::uint32_t> split_frontier, bool finalize_at_end) const {
    const program::PrefillTextCall call = program::prefill_text_call(
        state, ids, nominal_length, split_frontier, finalize_at_end, "Gemma 4",
        [this](std::uint32_t chunk, std::uint32_t cursor, std::uint32_t frontier) {
            return prefill_call_tokens(chunk, cursor, frontier);
        });
    return run_prefill_call(state, ids, call, split_frontier, nullptr);
}

program::PrefillChunkResult Gemma4Execution::prefill_multimodal_chunk(
    program::PrefillContext& state, const qwen3_5::PreparedPromptData& prompt,
    program::VisionPrefill& vision, std::uint32_t nominal_length,
    std::optional<std::uint32_t> split_frontier, bool finalize_at_end) const {
    auto& images = static_cast<Gemma4VisionPrefill&>(vision);
    const std::span<const TokenId> ids(prompt.token_ids);
    program::PrefillTextCall call = program::prefill_text_call(
        state, ids, nominal_length, split_frontier, finalize_at_end, "Gemma 4",
        [this](std::uint32_t chunk, std::uint32_t cursor, std::uint32_t frontier) {
            return prefill_call_tokens(chunk, cursor, frontier);
        });
    // A call is text alone or one whole image: an image's soft tokens see one another, so they
    // run as one call, and a text call stops where the next image begins.
    const VisionUseSpan* image = images.use_at(call.base);
    if (image != nullptr) {
        if (image->begin != call.base) {
            throw std::logic_error("Gemma 4 vision: a prefill resumes inside an image");
        }
        if (call.len < image->end - image->begin) {
            throw std::invalid_argument(
                "Gemma 4 vision: a prefill call must hold a whole image; raise --prefill-chunk");
        }
        call.len = image->end - image->begin;
    } else if (const VisionUseSpan* next = images.next_use(call.base);
               next != nullptr && next->begin < call.base + call.len) {
        call.len = next->begin - call.base;
    }
    call.t       = static_cast<std::int32_t>(call.len);
    call.is_last = finalize_at_end && call.len == nominal_length;
    if (image == nullptr) { return run_prefill_call(state, ids, call, split_frontier, nullptr); }
    // The vision tower runs under the invariant profile too (--activations exact), as Qwen's
    // encode does: run_prefill_call scopes only the decoder's call.
    const Tensor embeddings = [&] {
        const ops::ScopedInvariantRoutes routes(invariant_);
        return images.soft_tokens(*image);
    }();
    return run_prefill_call(state, ids, call, split_frontier, &embeddings);
}

program::PrefillChunkResult Gemma4Execution::run_prefill_call(
    program::PrefillContext& state, std::span<const TokenId> ids,
    const program::PrefillTextCall& call, std::optional<std::uint32_t> split_frontier,
    const Tensor* embeddings) const {
    runtime::ExecutionTimingRecorder timing;
    program::ExecutionCore& core       = state.execution;
    const cudaStream_t s               = core.device.stream;
    const auto [base, len, t, is_last] = call;
    // --activations exact: every width on one summation order, and attention on the prefill
    // kernel however narrow the call (a prefix-cache frontier may leave a single token).
    const ops::ScopedInvariantRoutes invariant_routes(invariant_);
    // Every prefill call takes the prefill route profile, as the Qwen, dense and Nemotron-H
    // families do: a split frontier cuts a call to another width than the cold run's, and only
    // under the profile does a projection on the generic path (GGUF weights on every card, NVFP4
    // outside sm_120a) give a column the same bytes at every width of 17 tokens or more. The
    // sm_120a NVFP4 shape tables are invariant with or without it.
    const ops::ScopedPrefillRoutes prefill_routes(true);

    core.work.reset();
    if (state.state_source_slot != state.state_destination_slot) {
        // A fork's first write: the destination takes the source's rings, then runs in place.
        core.state_images.copy_rings(state.state_source_slot, state.state_destination_slot, s);
    }
    auto [device_ids, positions] = program::upload_prefill_text(core, ids, call);
    Tensor rows       = core.work.alloc(DType::I32, {2});
    const auto layers = sequence_bindings(state, state.state_destination_slot, rows);
    ops::set_i32_scalar(core.io.rope_delta, 0, s);

    Tensor hidden = core.prefill_hidden.slice(1, 0, t);
    {
        auto scope          = core.work.scope();
        const Tensor normed = pass_.forward(
            core.work, device_ids, positions,
            {.layers = layers, .image_block = embeddings != nullptr},
            {.max_visible_keys = base + len, .max_tokens = t}, Activations::Recorded, nullptr,
            embeddings);
        CUDA_CHECK(cudaMemcpyAsync(hidden.data, normed.data, hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, s));
    }
    const Tensor last = hidden.slice(1, t - 1, 1);
    program::finish_prefill_text(state, call, split_frontier, last, traits_.public_tokens,
                                 [this](auto& work, const Tensor& x, Tensor& logits) {
                                     pass_.head(work, x, logits);
                                 });
    if (is_last && state.mtp_proposal_extent != 0) {
        propose_one(core, layers, core.io.token, last, static_cast<std::int32_t>(base + len),
                    state.mtp_proposal_extent);
    }
    timing.begin_wait();
    core.device.synchronize();
    timing.end_wait();
    core.work.reset();
    return {.processed_tokens = len, .finalized = is_last, .timing = timing.finish()};
}

void Gemma4Execution::mtp_bridge_and_propose(program::PrefillContext& state,
                                             const Tensor& next_token,
                                             const Tensor& previous_hidden, std::int32_t position,
                                             std::span<const std::int32_t>,
                                             bool build_proposal) const {
    // The drafter keeps no keys or values: a bridge has nothing to fill, only a proposal to make.
    if (!build_proposal) { return; }
    const ops::ScopedInvariantRoutes routes(invariant_);
    program::ExecutionCore& core = state.execution;
    core.work.reset();
    Tensor rows = core.work.alloc(DType::I32, {2});
    // An exact hit still reads its source image: the destination is written by the first step.
    const auto layers = sequence_bindings(state, state.state_source_slot, rows);
    propose_one(core, layers, next_token, previous_hidden, position + 1,
                state.mtp_proposal_extent);
    core.work.reset();
}

} // namespace

StateShape state_shape(const Config& config, std::int32_t public_tokens) {
    // A layer that reads shared keys and values takes its source's place and adds none.
    StateShape out{
        .family           = StateFamily::Gemma4,
        .kv_heads         = config.full.kv_heads,
        .head_dim         = config.full.head_dim,
        .sliding_kv_heads = config.sliding.kv_heads,
        .sliding_head_dim = config.sliding.head_dim,
        .window           = static_cast<std::uint32_t>(config.sliding_window),
        .hidden           = config.hidden_size,
        .output_rows      = config.vocab_size,
        .public_tokens    = public_tokens,
    };
    for (std::int32_t i = 0; i < config.layers; ++i) {
        if (!config.owns_kv(i)) { continue; }
        ++(config.kinds[static_cast<std::size_t>(i)] == LayerKind::Full ? out.full_layers
                                                                        : out.sliding_layers);
    }
    return out;
}

std::unique_ptr<const ModelExecution> make_program_execution(const Model& model,
                                                             const Drafter* drafter,
                                                             DeviceContext& device,
                                                             std::int32_t public_tokens,
                                                             ActivationMode activations,
                                                             EnginePurpose purpose) {
    return std::make_unique<const Gemma4Execution>(model, drafter, device, public_tokens,
                                                   activations, purpose);
}

} // namespace ninferno::models::gemma4
