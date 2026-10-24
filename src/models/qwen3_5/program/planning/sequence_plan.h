#pragma once

// The plan of one sequence configuration (SequencePlanImpl) and its persistent and workspace
// layouts, without the model: a family's ModelExecution fills them at a start, and the plan
// without the weights lays the persistent share out from a StateShape (sequence_bytes.h).

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/load_options.h"
#include "models/qwen3_5/program/planning/speculation_load.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/state/state_image.h"
#include "ninferno/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninferno::models::qwen3_5::detail {

class ModelExecution;

using TensorLayout                              = TensorRegion;
inline constexpr std::uint32_t kCausalScoreTile = 1024;

struct DFlashPersistentLayout {
    std::optional<qwen3_5::PagedKVCacheLayout> full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return full ? full->payload_bytes() : 0;
    }
};

struct PersistentLayout {
    qwen3_5::DecoderStateLayout decoder;
    qwen3_5::StateImageDeviceLayout state_images;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3_5::RoundStateLayout round;
    TensorLayout prefill_hidden;
    std::optional<TensorLayout> score_hidden;
    std::optional<TensorLayout> token_counts;
    std::optional<TensorLayout> sampling_config;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

struct VisionWorkspacePlan {
    std::int32_t output_hidden         = 0;
    std::uint32_t max_merged_tokens    = 0;
    std::size_t general_capacity_bytes = 0;
    std::size_t encode_peak_bytes      = 0;
    std::size_t handoff_offset_bytes   = 0;
    std::size_t handoff_capacity_bytes = 0;
    std::size_t capacity_bytes         = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill   = 0;
    std::size_t ordinary_round = 0;
    std::size_t mtp_prefill    = 0;
    std::size_t mtp_round      = 0;
    std::size_t dflash_context = 0; // a prefill chunk's append of its features
    std::size_t dflash_round   = 0;
    // A round's append of its compacted features, the widest batch (within dflash_round).
    std::size_t dflash_round_append = 0;
    std::size_t causal_score        = 0;
    std::size_t general_capacity    = 0;
    std::optional<VisionWorkspacePlan> vision;
    std::size_t capacity = 0;
};

struct SequencePlanningInputs {
    const ModelExecution* model   = nullptr;
    std::uint32_t capacity        = 0;
    std::uint32_t max_concurrency = 1;
    std::uint32_t prefill_chunk   = 0;
    // The speculative window (the largest launch window) and every launch window, ascending: the
    // draft depths whose speculative graphs are captured (SpeculativeOptions::draft_tokens,
    // request_draft_tokens and the nonzero load bounds). default_draft_depth: a request's depth
    // unless it chooses one.
    std::uint32_t draft_window = 0;
    std::vector<std::uint32_t> launch_windows;
    // Per launch window, the smallest speculative batch its graphs are captured for: 1 for the
    // menu's depths, the lanes of a load bound for a depth only a bound asks (a step of fewer rows
    // runs the next wider window). Every window stops at speculative_batch_limit.
    std::vector<std::uint32_t> launch_window_min_batch;
    // Per launch window, the speculative batches it runs under the load bounds, as bit (batch - 1)
    // (launch_window_batches): its graphs are captured and allowed for those batches only. A
    // DFlash row still appending the features of a wider round from before its step grew may need
    // a window at a batch outside its mask: that launch runs without a graph.
    std::vector<std::uint32_t> launch_window_batches;
    // The widest speculative launch: a load bound of 0 turns every row of a step that wide plain,
    // so no speculative graph or round workspace is planned from its lanes on (max_concurrency
    // without one).
    std::uint32_t speculative_batch_limit  = 0;
    std::uint32_t default_draft_depth      = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    KvCacheStorage kv_storage              = KvCacheStorage::BFloat16;
    KvCacheStorage ring_storage            = KvCacheStorage::BFloat16; // sliding rings
    DType recurrent_state_dtype            = DType::FP32;              // --state-dtype, GDN layers
    ProposalHead proposal_head             = ProposalHead::Full;
    DFlashDraftWidth draft_width           = DFlashDraftWidth::Default; // SpeculativeOptions
    double acceptance_floor                = 0.0; // SpeculativeOptions watchdog
    std::vector<SpeculationStep> step_down;       // SpeculativeOptions::step_down
    std::vector<SpeculationLoad> load;            // SpeculativeOptions::load
    std::uint32_t context_limit       = 0;        // SpeculativeOptions::context_limit
    std::uint32_t watchdog_window     = 32;
    std::uint32_t watchdog_min_rounds = 16;
    double shared_trip_floor          = 1.4; // SpeculativeOptions::shared_trip_floor
    std::uint32_t ngram_draft_tokens  = 0;   // SpeculativeOptions: the copy window
    std::uint32_t ngram_min_match     = 0;   // SpeculativeOptions::ngram_min_match
    std::uint32_t kv_lease_window     = 0;   // EngineOptions::kv_lease_window
    DecodePacingOptions decode_pacing;       // EngineOptions::pacing
    models::LoadOptions features;
    bool use_cuda_graph               = true;
    bool causal_scoring               = false;
    int device                        = 0;
    std::int32_t multiprocessor_count = 0;
    ContextCacheOptions context_cache;
    // The driver already holds the memory of this plan's CUDA graphs: construction's kernel warm-up
    // instantiated the same graph set in a probe Program, and the driver keeps a destroyed graph
    // executable's memory for the next one (rebuilding a destroyed set takes nothing more).
    // The plan then reserves no graph allowance.
    bool graph_memory_held = false;
};

struct SequencePlanImpl {
    const ModelExecution* model    = nullptr;
    std::uint32_t capacity         = 0;
    std::uint32_t kv_capacity      = 0;
    std::uint32_t main_page_groups = 0;
    std::uint32_t max_concurrency  = 1;
    std::uint32_t prefill_chunk    = 0;
    std::uint32_t draft_window     = 0;                 // the largest launch window
    std::vector<std::uint32_t> launch_windows;          // ascending; the last is draft_window
    std::vector<std::uint32_t> launch_window_min_batch; // SequencePlanningInputs
    std::vector<std::uint32_t> launch_window_batches;   // SequencePlanningInputs
    std::uint32_t speculative_batch_limit  = 0;         // SequencePlanningInputs
    std::uint32_t default_draft_depth      = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    KvCacheStorage kv_storage              = KvCacheStorage::BFloat16;
    KvCacheStorage ring_storage            = KvCacheStorage::BFloat16; // sliding rings
    DType recurrent_state_dtype            = DType::FP32;              // --state-dtype, GDN layers
    ProposalHead proposal_head             = ProposalHead::Full;
    DFlashDraftWidth draft_width           = DFlashDraftWidth::Default; // SpeculativeOptions
    double acceptance_floor                = 0.0; // SpeculativeOptions watchdog
    std::vector<SpeculationStep> step_down;       // SpeculativeOptions::step_down
    std::vector<SpeculationLoad> load;            // SpeculativeOptions::load
    std::uint32_t context_limit       = 0;        // SpeculativeOptions::context_limit
    std::uint32_t watchdog_window     = 32;
    std::uint32_t watchdog_min_rounds = 16;
    double shared_trip_floor          = 1.4; // SpeculativeOptions::shared_trip_floor
    std::uint32_t ngram_draft_tokens  = 0;   // SpeculativeOptions: the copy window
    std::uint32_t ngram_min_match     = 0;   // SpeculativeOptions::ngram_min_match
    std::uint32_t kv_lease_window     = 0;   // EngineOptions::kv_lease_window
    DecodePacingOptions decode_pacing;       // EngineOptions::pacing
    models::LoadOptions features;
    bool use_cuda_graph               = true;
    bool causal_scoring               = false;
    int device                        = 0;
    std::int32_t multiprocessor_count = 0;
    ContextCacheOptions context_cache;
    PersistentLayout persistent;
    WorkspacePlan workspace;
    std::size_t graph_allowance_bytes = 0;
    // The part of it each launch window's speculative graphs take, in launch_windows' order.
    std::vector<std::size_t> launch_window_graph_bytes;
    std::size_t device_reservation_bytes = 0;
};

// How many rows a speculative round's workspace is planned for: the widest speculative launch.
[[nodiscard]] inline std::uint32_t
speculative_round_batches(const SequencePlanImpl& plan) noexcept {
    return plan.speculative_batch_limit != 0 ? plan.speculative_batch_limit : plan.max_concurrency;
}

// The widest launch window a speculative round of `batch` rows is planned for (0: none runs). An
// MTP launch's rows are all capped by the step's load bound, so a batch runs only the windows its
// mask holds (launch_window_batches). A DFlash launch keeps the engine's window at every batch: a
// row may still append the features of a wider round from before its step grew.
[[nodiscard]] inline std::uint32_t speculative_round_window(const SequencePlanImpl& plan,
                                                            std::uint32_t batch) noexcept {
    if (batch == 0 || batch > speculative_round_batches(plan)) { return 0; }
    if (plan.speculative_backend != SpeculativeBackend::Mtp ||
        plan.launch_window_batches.size() != plan.launch_windows.size()) {
        return plan.draft_window;
    }
    std::uint32_t widest = 0;
    for (std::size_t w = 0; w < plan.launch_windows.size(); ++w) {
        if (window_runs_batch(plan.launch_window_batches[w], batch)) {
            widest = std::max(widest, plan.launch_windows[w]);
        }
    }
    return widest;
}

// The most verify columns (rows times the window's width) any speculative round is planned for.
[[nodiscard]] inline std::uint32_t
speculative_round_columns(const SequencePlanImpl& plan) noexcept {
    std::uint32_t columns = 0;
    for (std::uint32_t batch = 1; batch <= speculative_round_batches(plan); ++batch) {
        const std::uint32_t window = speculative_round_window(plan, batch);
        if (window != 0) { columns = std::max(columns, batch * (window + 1U)); }
    }
    return columns;
}

} // namespace ninferno::models::qwen3_5::detail
