#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/qwen3_5_execution.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "models/qwen3_5/program/planning/copy_proposal.h"
#include "core/debug_poison.h"
#include "core/nvtx.h"
#include "core/device.h"
#include "ninferno/ops/position.h"
#include "ninferno/ops/scalar.h"
#include "ninferno/ops/prepare_ragged_prefix.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/scatter.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::qwen3_5 {
namespace {

// Debug poisoning (NINFERNO_DEBUG_POISON). A round's graph is captured for
// exactly B rows and must read only rows [0, B) of the transfer object copied to the device each
// round; the rows past B hold whatever an earlier, wider round left. Under poisoning they hold the
// poison byte instead (-1 as an index or a slot, NaN as a sampling parameter), so a kernel that
// reads past its batch fails loudly instead of reusing a neighbour's stale row.
template <class T, std::size_t N>
void poison_tail(std::array<T, N>& values, std::size_t used, unsigned char byte) {
    if (used < N) {
        std::memset(static_cast<void*>(values.data() + used), byte, (N - used) * sizeof(T));
    }
}

void poison_ingress_tail(OrdinaryDecodeIngress& in, std::size_t rows) {
    const std::optional<unsigned char> byte = debug_poison_byte_for("ingress");
    if (!byte) { return; }
    poison_tail(in.tokens, rows, *byte);
    poison_tail(in.cache_positions, rows, *byte);
    poison_tail(in.rope_positions, rows, *byte);
    poison_tail(in.text_kv_table_rows, rows, *byte);
    poison_tail(in.state_source_slots, rows, *byte);
    poison_tail(in.state_destination_slots, rows, *byte);
    poison_tail(in.sampling, rows, *byte);
}

void poison_ingress_tail(MtpDecodeIngress& in, std::size_t rows, std::size_t draft_window,
                         std::size_t width) {
    const std::optional<unsigned char> byte = debug_poison_byte_for("ingress");
    if (!byte) { return; }
    poison_tail(in.anchors, rows, *byte);
    poison_tail(in.base_frontiers, rows, *byte);
    poison_tail(in.remaining_budgets, rows, *byte);
    poison_tail(in.current_extents, rows, *byte);
    poison_tail(in.target_valid_columns, rows, *byte);
    poison_tail(in.current_drafts, rows * draft_window, *byte);
    poison_tail(in.target_rope_positions, rows * width, *byte);
    poison_tail(in.text_kv_table_rows, rows, *byte);
    poison_tail(in.mtp_kv_table_rows, rows, *byte);
    poison_tail(in.state_source_slots, rows, *byte);
    poison_tail(in.state_destination_slots, rows, *byte);
    poison_tail(in.rope_deltas, rows, *byte);
    poison_tail(in.sampling, rows, *byte);
}

} // namespace
} // namespace ninferno::models::qwen3_5

namespace ninferno::models::qwen3_5::execution {
namespace {

auto ordinary_batch_body(OrdinaryBatchContext& state, std::int32_t batch_size,
                         ops::CausalAttentionExecutionEnvelope envelope) {
    return [&state, batch_size, envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency)) {
            throw std::logic_error("ordinary decode batch state is incomplete");
        }

        qwen3_5::OrdinaryDecodeState& ordinary = state.frame;
        CUDA_CHECK(cudaMemcpyAsync(ordinary.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_5::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, qwen3_5::detail::qwen_parameters(state.execution),
                         state.execution.work, {}, state.execution.state_images.linear(),
                         state.execution.io, state.execution.prefill_hidden,
                         state.execution.prefill_chunk, 0, {}, &state.text_cache);

        Tensor tokens             = ordinary.tokens.slice(0, 0, batch_size);
        Tensor cache_positions    = ordinary.cache_positions.slice(0, 0, batch_size);
        Tensor rope_positions     = ordinary.rope_positions.slice(0, 0, batch_size);
        Tensor kv_rows            = ordinary.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = ordinary.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = ordinary.state_destination_slots.slice(0, 0, batch_size);
        Tensor hidden             = ordinary.hidden.slice(1, 0, batch_size);
        Tensor logits             = ordinary.logits.slice(1, 0, batch_size);
        Tensor sampled            = ordinary.sampled_tokens.slice(0, 0, batch_size);

        card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows, state_sources,
                                   state_destinations, envelope, hidden, logits);
        ops::scatter(hidden, state_destinations, state.continuation_hidden_store,
                     state.execution.device.stream);
        ops::sample(logits, sampled,
                    dimension(qwen3_5::detail::qwen_parameters(state.execution)
                                  .model.resources()
                                  .public_token_count),
                    ordinary.sampling, cache_positions, ops::kSamplePurposeDecode,
                    state.execution.work, state.execution.device.stream);
        ops::copy_to_pinned_host(&state.host_egress, ordinary.egress.data,
                                 sizeof(qwen3_5::OrdinaryDecodeEgress),
                                 state.execution.device.stream);
    };
}

} // namespace

void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    capture_graph(state, definition, body);
}

void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::CausalAttentionExecutionEnvelope envelope,
                           DecodeGraphExecutable* executable) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    run_prepared(state, executable, body);
}

} // namespace ninferno::models::qwen3_5::execution

namespace ninferno::models::qwen3_5::detail {

namespace {

// Flags the speculative rows whose verified target hidden [H,width,B] is not finite, over the
// columns each row verified.
template <typename Frame>
void check_target_hidden(NumericStatus& status, const Frame& frame, std::uint32_t batch,
                         cudaStream_t stream) {
    status.check(NumericStatus::Launch::Speculative,
                 frame.target_hidden.slice(2, 0, static_cast<std::int32_t>(batch)),
                 static_cast<const std::int32_t*>(frame.target_valid_columns.data), stream);
}

// The verify columns a step's stopped rows may take in the speculative launch (their count times
// its width) before they move to the ordinary launch instead: a second launch reads the weights
// once more, which a dense target's verify of 16 columns costs about as much as (Qwen3.8 27B: T=16
// verifies at 1.25x a decode step) and a MoE target's in fewer (Qwen3.6 35B-A3B: 1.46x at T=4).
// The budget follows from those verify costs; riding against latching plain at C=8 with forced
// trips is not measured (a trip needs a profile floor the server cannot force).
constexpr std::uint32_t kRideColumns = 16;


DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label);

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label);

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label);

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label) {
    const auto it = std::find_if(
        family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& profile) {
            return profile.batch_size == batch_size && profile.min_execution_frontier <= frontier &&
                   frontier <= profile.max_execution_frontier;
        });
    if (it == family.profiles.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label) {
    const auto it = std::find_if(family.topologies.begin(), family.topologies.end(),
                                 [topology_class](const DecodeGraphTopology& topology) {
                                     return topology.topology_class == topology_class;
                                 });
    if (it == family.topologies.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph topology is unavailable");
    }
    return *it;
}

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label) {
    DecodeGraphTopology& topology   = select_graph_topology(family, profile.topology_class, label);
    const std::size_t profile_index = static_cast<std::size_t>(&profile - family.profiles.data());
    if (topology.installed_profile != profile_index) {
        // CUDA may refuse an update it accepted when the topology was grouped (a profiler that
        // instruments graph nodes, a driver change): the profile is instantiated instead, at the
        // cost of one instantiation, rather than failing the Engine.
        if (!topology.executable.try_update(profile.definition)) {
            if (!topology.update_refused) {
                topology.update_refused = true;
                std::fprintf(stderr,
                             "[decode] %s CUDA Graph update refused: profiles are instantiated "
                             "when they change\n",
                             label);
            }
            topology.executable.instantiate(profile.definition);
        }
        topology.installed_profile = profile_index;
    }
    return topology.executable;
}

// Feeds the request's judged counters (the rounds at its own depth) to its acceptance watchdog.
// True when it trips (apply_watchdog_trip() then acts on it).
bool observe_watchdog(RequestControl& request) {
    if (request.speculative_stats.watchdog_tripped) { return false; }
    return request.watchdog.observe(static_cast<std::int64_t>(request.judged_rounds),
                                    static_cast<std::int64_t>(request.judged_accepted));
}

// A trip moves the request to its next shallower depth that can still pay (steps_that_cannot_pay,
// spec/watchdog.h, skips the others), judged afresh against that depth's floor, or, with none
// left, stops its drafting for the rest of the request (decode_raw() runs it without drafts);
// either lands in its stats (step_downs counts the menu steps to the depth it lands on; skipped
// to plain, final_draft_window stays the last depth that ran). A watchdog floor is the break-even
// of a round against plain decode of one request (spec-tune at C=1). When another row of the round
// drafted and keeps drafting, the next steps run a speculative launch whatever this row does:
// stopped, it would verify its columns there without drafts and save only its share of the draft
// phase while emitting one token per step instead of its tokens per round (Nemotron-H MTP on gb202
// tripped at 58-66 % acceptance and rode at 142 tok/s against 216 plain; Qwen3.6 35B-A3B DFlash
// rows tripped at about 2.8 tokens per round at C=2..8 and rode the rest of their requests). Its
// drafting then stops only below `shared_floor` (SpeculativeOptions::shared_trip_floor; 0 when the
// round was not shared), else the watchdog judges again after the next round.
void apply_watchdog_trip(RequestControl& request, double shared_floor) {
    auto& stats         = request.speculative_stats;
    const auto rounds   = static_cast<std::int64_t>(request.judged_rounds);
    const auto accepted = static_cast<std::int64_t>(request.judged_accepted);
    std::vector<double> floors;
    floors.reserve(request.step_down.size());
    for (const SpeculationStep& step : request.step_down) {
        floors.push_back(step.acceptance_floor);
    }
    const std::size_t skipped = spec::steps_that_cannot_pay(
        request.watchdog.depth_tokens_per_round(), request.watchdog.policy().trip_margin, floors);
    if (skipped < request.step_down.size()) {
        request.step_down.erase(request.step_down.begin(),
                                request.step_down.begin() + static_cast<std::ptrdiff_t>(skipped));
        stats.step_downs += static_cast<std::uint32_t>(skipped);
    } else {
        request.step_down.clear();
    }
    if (!request.step_down.empty()) {
        const SpeculationStep step = request.step_down.front();
        request.step_down.erase(request.step_down.begin());
        request.draft_depth = step.draft_tokens;
        request.watchdog.restart(step.acceptance_floor, rounds, accepted);
        stats.step_downs += 1;
        stats.final_draft_window = step.draft_tokens;
        return;
    }
    if (shared_floor > 1.0 && request.watchdog.rolling_tokens_per_round() >= shared_floor) {
        request.watchdog.rearm();
        stats.shared_trip_deferrals += 1;
        return;
    }
    stats.watchdog_tripped          = true;
    stats.watchdog_round            = stats.rounds;
    stats.watchdog_tokens_per_round = request.watchdog.rolling_tokens_per_round();
    request.drafting_stopped        = true;
}

} // namespace

void ProgramImpl::install_sampling(SequenceState& sequence, RequestControl& request,
                                   const ops::SamplingConfig& config,
                                   const RequestSpeculationPlan& speculation) {
    Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(sequence.lane), 1)
                        .view({model.traits().public_tokens});
    const bool speculative    = speculative_backend != SpeculativeBackend::None;
    request.draft_depth       = speculative ? speculation.draft_depth : 0U;
    request.sampling_host     = config;
    request.speculative_stats = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = request.draft_depth != 0,
        .draft_window          = request.draft_depth,
        .accepted_per_position = std::vector<std::uint64_t>(request.draft_depth, 0),
        .final_draft_window    = request.draft_depth,
    };
    request.step_down = speculative ? speculation.step_down : std::vector<SpeculationStep>{};
    request.judged_rounds    = 0;
    request.judged_accepted  = 0;
    request.drafting_stopped = false;
    request.watchdog         = spec::AcceptanceWatchdog(spec::WatchdogPolicy{
                .floor_tokens_per_round =
            speculation.acceptance_floor,
                .window_rounds = watchdog_policy.window_rounds,
                .min_rounds    = watchdog_policy.min_rounds});
    // Depth 0 in a speculative engine: plain decode from the first step.
    request.plain        = speculative && request.draft_depth == 0;
    request.load_latch_origin = false;
    const bool penalties = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F;
    if (penalties) { CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream)); }
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &request.sampling_host,
                               sizeof(request.sampling_host), cudaMemcpyHostToDevice,
                               device.stream));
}

void ProgramImpl::copy_tail(SequenceState& sequence, const Tensor& source) {
    if (source.dtype != DType::BF16 || source.ne[0] != model.traits().hidden || source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(sequence.tail_hidden.data, source.data, sequence.tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    sequence.tail_hidden_valid = true;
}

void ProgramImpl::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

void ProgramImpl::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

void ProgramImpl::enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                                std::span<const std::uint32_t> starts,
                                                std::span<const std::uint32_t> counts) {
    if (!is_masked_draft_backend(speculative_backend) || !dflash || !io.dflash_decode ||
        lanes.empty() || lanes.size() > max_concurrency || starts.size() != lanes.size() ||
        counts.size() != lanes.size()) {
        throw std::logic_error("DFlash context append has invalid membership");
    }

    std::uint32_t minimum_count = draft_window + 1U;
    std::uint32_t maximum_count = 0;
    *dflash_host_ingress        = {};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || counts[row] == 0 || counts[row] > draft_window + 1U ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("DFlash context append contains an invalid row");
        }
        SequenceState& sequence   = active_sequence(lane);
        const std::uint32_t start = starts[row];
        const std::uint64_t end64 = static_cast<std::uint64_t>(start) + counts[row];
        const std::uint32_t end   = static_cast<std::uint32_t>(end64);
        if (!sequence.kv || text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            (backend_kv_cache() && (!sequence.kv->backend ||
                                    backend_kv_addresses->bound_row(*sequence.kv->backend) < 0)) ||
            end64 > capacity) {
            throw std::logic_error("DFlash context append is outside retained target storage");
        }
        dflash_host_ingress->context_frontiers[row] =
            checked_i32(start, "DFlash append context frontier");
        dflash_host_ingress->execution_frontiers[row] =
            checked_i32(end, "DFlash append target frontier");
        dflash_host_ingress->dflash_kv_table_rows[row] =
            sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
        dflash_host_ingress->active_lanes[row]            = static_cast<std::int32_t>(lane);
        const StateImageSelectors selectors               = state_selectors(sequence);
        dflash_host_ingress->state_source_slots[row]      = selectors.source;
        dflash_host_ingress->state_destination_slots[row] = selectors.destination;
        // Context append writes only the draft caches. Target execution owns Main KV
        // coverage, including any uncommitted suffix that survives until the round is settled.
        // DFlash2 has only fixed cyclic state; DFlash also grows its Full backend KV here.
        if (sequence.kv->backend) {
            backend_kv_addresses->ensure_mapped_to_tokens(*sequence.kv->backend, end,
                                                          device.stream);
        }
        minimum_count = std::min(minimum_count, counts[row]);
        maximum_count = std::max(maximum_count, counts[row]);
    }

    qwen3_5::DFlashDecodeState& frame = *io.dflash_decode;
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, dflash_host_ingress,
                               sizeof(qwen3_5::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                               device.stream));
    const auto batch                = static_cast<std::int32_t>(lanes.size());
    Tensor active_lane_tensor       = frame.active_lanes.slice(0, 0, batch);
    Tensor state_destination_tensor = frame.state_destination_slots.slice(0, 0, batch);
    Tensor device_starts            = frame.context_frontiers.slice(0, 0, batch);
    Tensor device_ends              = frame.execution_frontiers.slice(0, 0, batch);
    Tensor table_rows               = frame.dflash_kv_table_rows.slice(0, 0, batch);
    Tensor positions                = frame.append_positions.slice(1, 0, batch);
    Tensor device_counts            = frame.append_counts.slice(0, 0, batch);

    work.reset();
    Tensor features =
        work.alloc(DType::BF16, {model.traits().draft_feature_rows,
                                 static_cast<std::int32_t>(draft_window + 1U), batch});
    ops::prepare_ragged_prefix(dflash->pending_features, active_lane_tensor, device_starts,
                               device_ends, features, positions, device_counts, device.stream);

    execution::DFlashAppendContext state{
        {device, model, work, *state_images, replay_records ? &*replay_records : nullptr, io,
         prefill_hidden, prefill_chunk, capacity, proposal_head, draft_width, &speculation_load},
        *dflash};
    mark_workspace_usage(workspace_plan.dflash_round_append);
    model.dflash_append_context(state, features, positions, device_counts, state_destination_tensor,
                                table_rows, {minimum_count, maximum_count});
}

void ProgramImpl::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= model.traits().public_tokens) {
            throw std::runtime_error("target returned a token outside the public token domain");
        }
    }
}

// A decode step runs its rows in at most two launches that share one synchronization: the rows
// that speculate verify their drafts in one speculative launch (MTP or DFlash), and the rows that
// decode plainly (every row of an ordinary engine; in a speculative engine, a request whose
// acceptance watchdog tripped) run the ordinary launch, whose CUDA graphs a speculative engine
// captures as well. Launch rows index the launch's frames and egress; step rows (LaunchRows::rows)
// index the step's token masks and the published round.
runtime::BatchedGeneratedRound
ProgramImpl::decode_raw(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed_timing) {
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("decode step membership is invalid");
    }
    // A load bound of 0 turns the step's speculating rows plain for the rest of their requests. A
    // request whose watchdog stopped its drafting verifies without drafts in the speculative launch
    // while another row of the step drafts (one launch instead of a second, ordinary one, and its
    // drafter state follows), as long as the columns the stopped rows verify for nothing, their
    // count times the width of the launch they would ride (the captured window launch_mtp and
    // launch_dflash pick for the deepest drafting row and the riding batch), stay within
    // kRideColumns; beyond it, or in a step where no row drafts, they turn plain (the ordinary
    // launch) for the rest of their requests. A row a load bound turned plain in a step at least
    // the bound's lanes wide (RequestControl::load_latch_origin) keeps the step at bound 0: a
    // request that joins it after the load fell decodes plainly as well, rather than draft in a
    // second launch beside it (5080, Muse Glimmer 30B + DFlash, 3:0 at C=3: the last request,
    // plain beside a newcomer that drafted, decoded at 35 tok/s against 50 alone). A joiner does
    // not pass this on, so under overlapping traffic speculation resumes once the rows the bound
    // itself turned plain have finished.
    const std::uint32_t natural = load_bound(lanes.size());
    bool latched                = false;
    for (const std::uint32_t lane : lanes) {
        latched = latched || (lane < max_concurrency && requests[lane].load_latch_origin);
    }
    const std::uint32_t bound   = latched ? 0U : natural;
    std::uint32_t drafting_rows = 0;
    std::uint32_t stopped       = 0;
    std::uint32_t widest        = 1; // the deepest launch depth of the rows that would ride
    for (const std::uint32_t lane : lanes) {
        if (lane >= max_concurrency || requests[lane].plain) { continue; }
        if (requests[lane].drafting_stopped) {
            ++stopped;
        } else {
            ++drafting_rows;
        }
        if (speculative_backend != SpeculativeBackend::None && bound > 0) {
            widest = std::max(widest, launch_depth(lane, bound));
        }
    }
    bool ride = drafting_rows > 0;
    if (ride && stopped > 0 && bound > 0 && speculative_backend != SpeculativeBackend::None) {
        ride = stopped_rows_ride(launch_windows, launch_window_for(widest, drafting_rows + stopped),
                                 stopped, kRideColumns);
    }
    LaunchRows speculative;
    LaunchRows plain;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("decode step contains an invalid or duplicate lane");
        }
        if (bound == 0 && speculative_backend != SpeculativeBackend::None &&
            !requests[lane].plain) {
            requests[lane].plain                        = true;
            requests[lane].load_latch_origin            = natural == 0;
            requests[lane].speculative_stats.load_plain = !requests[lane].drafting_stopped;
        }
        if (!ride && requests[lane].drafting_stopped) { requests[lane].plain = true; }
        LaunchRows& part = speculative_backend == SpeculativeBackend::None || requests[lane].plain
                               ? plain
                               : speculative;
        part.add(lane, budgets[row], static_cast<std::uint32_t>(row));
    }

    const bool mtp = speculative_backend == SpeculativeBackend::Mtp;
    nvtx::ScopedRange round_range(speculative.size == 0 ? nvtx::Name::DecodeOrdinaryRound
                                  : mtp                 ? nvtx::Name::DecodeMtpRound
                                                        : nvtx::Name::DecodeDFlashRound,
                                  speculative.size == 0 ? nvtx::Category::Decode
                                  : mtp                 ? nvtx::Category::Mtp
                                                        : nvtx::Category::DFlash,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    runtime::MaskedUnit masks(token_masks.get(), active_masks, lanes);
    const auto started = Clock::now();
    // A round is timed on the device, so its pacing sees how long the host lagged behind it (a
    // speculative engine's pacing chooses only the wait). A constrained row needs the host at every
    // step (and its mask work would read as lag).
    const bool timed_round              = round_pacing.has_value() && !masks.any();
    // Masks are computed while the launched kernels run, and the samplers wait for them only up to
    // a bound (kTokenMaskReleaseWaitNs). A unit whose masks are slow to compute (a deeply nested
    // grammar takes hundreds of milliseconds a row, an MTP row a column per draft) computes and
    // releases them before its launch: the round waits for the host either way, and no sampler
    // waits on the release.
    std::uint64_t mask_cost_ns = 0;
    std::uint64_t drafted_cost_ns = 0; // the same with DFlash rows masked over every draft column
    for (std::size_t i = 0; i < speculative.size; ++i) {
        mask_cost_ns += masks.fill_cost_ns(speculative.rows[i], mtp ? draft_window + 1U : 1U);
        drafted_cost_ns += masks.fill_cost_ns(speculative.rows[i], draft_window + 1U);
    }
    for (std::size_t i = 0; i < plain.size; ++i) {
        mask_cost_ns += masks.fill_cost_ns(plain.rows[i], 1U);
        drafted_cost_ns += masks.fill_cost_ns(plain.rows[i], 1U);
    }
    const bool masks_first              = mask_cost_ns > ops::kTokenMaskReleaseWaitNs / 8U;
    // A constrained DFlash row drafts when its masks over every draft column are cheap: the
    // drafts reach the host mid-round and the masks follow them along the grammar while the target
    // verifies (fill_dflash_draft_masks). Otherwise it runs with a zero draft extent, its single
    // column masked.
    dflash_masks_follow_drafts = !mtp && speculative.size != 0 && masks.any() &&
                                 drafted_cost_ns <= ops::kTokenMaskReleaseWaitNs / 8U;
    const std::uint32_t requested_steps = timed_round ? round_pacing->steps(started) : 1U;
    const bool follows_round            = !round_after_other_work;
    round_after_other_work              = !timed_round;
    auto launched                       = started;
    std::uint32_t ordinary_steps        = 1;
    try {
        {
            nvtx::ScopedRange submit_range(speculative.size == 0 ? nvtx::Name::DecodeOrdinarySubmit
                                           : mtp ? nvtx::Name::DecodeMtpSubmit
                                                 : nvtx::Name::DecodeDFlashSubmit,
                                           nvtx::Category::Decode,
                                           static_cast<std::uint64_t>(lanes.size()));
            // The timer starts before any launch of the round, the speculative one's host
            // preparation included: the stream is idle, so that preparation counts as device time
            // and only a late wake-up from the wait reads as lag.
            if (timed_round) {
                launched = Clock::now();
                round_timer->start();
            }
            numeric_status.clear();
            if (masks_first) {
                // Plain rows and DFlash rows (a constrained one runs no drafts) are host-known
                // now; MTP rows once launch_mtp has set their draft extents.
                fill_launch_masks(plain, false, masks);
                if (!mtp) { fill_launch_masks(speculative, false, masks); }
                if (!mtp || speculative.size == 0) { masks.release(); }
            }
            if (speculative.size != 0) {
                if (mtp) {
                    launch_mtp(speculative, masks, bound, masks_first);
                } else {
                    launch_dflash(speculative, masks, bound);
                }
            }
            if (plain.size != 0) {
                ordinary_steps = launch_ordinary(plain, masks, requested_steps);
            }
            if (timed_round) { round_timer->record_stop(); }
            if (masks.any() && !masks_first) {
                // The launches run while the host computes this step's masks; the samplers
                // wait for the release.
                (void)cudaStreamQuery(device.stream);
                if (dflash_masks_follow_drafts) {
                    fill_dflash_draft_masks(speculative, masks);
                } else {
                    fill_launch_masks(speculative, mtp, masks);
                }
                fill_launch_masks(plain, false, masks);
                masks.release();
            }
        }
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(speculative.size == 0 ? nvtx::Name::DecodeOrdinaryWait
                                         : mtp ? nvtx::Name::DecodeMtpWait
                                               : nvtx::Name::DecodeDFlashWait,
                                         nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            // A stalling host may block on the round's last kernel rather than spin on it.
            if (timed_round && round_pacing->block(launched)) { round_timer->wait_stop(); }
            device.synchronize();
        }
        timing.end_wait();
        masks.collect_faults();

        const auto finished  = Clock::now();
        const double seconds = std::chrono::duration<double>(finished - started).count();
        if (timed_round) {
            round_pacing->observe(ordinary_steps,
                                  std::chrono::duration<double>(finished - launched).count(),
                                  round_timer->elapsed_ms() / 1000.0, finished, follows_round);
        }
        if (speculative.size != 0) { collect_speculative(speculative, seconds); }
        if (plain.size != 0) { collect_ordinary(plain, seconds, ordinary_steps); }
        return publish_step(lanes.size(), speculative, plain, ordinary_steps, timing.finish());
    } catch (...) {
        masks.release(); // a launched sampler may be waiting for it
        timing.begin_wait();
        try {
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

void ProgramImpl::fill_launch_masks(const LaunchRows& part, bool mtp_drafts,
                                    runtime::MaskedUnit& masks) {
    for (std::size_t i = 0; i < part.size; ++i) {
        if (!mtp_drafts) {
            masks.fill(part.rows[i], {});
            continue;
        }
        // The MTP drafts verified this round are host-known, so every verify column's mask is
        // computed here while the target forward runs. (DFlash drafts reach the host only
        // mid-round: fill_dflash_draft_masks, or a zero draft extent.)
        const SequenceState& sequence = active_sequence(part.lanes[i]);
        const auto extent = static_cast<std::size_t>(mtp_host_ingress->current_extents[i]);
        masks.fill(part.rows[i], std::span<const TokenId>(sequence.mtp_drafts.data(), extent));
    }
}

void ProgramImpl::fill_dflash_draft_masks(const LaunchRows& part, runtime::MaskedUnit& masks) {
    bool constrained = false;
    for (std::size_t i = 0; i < part.size; ++i) { constrained |= masks.active(part.rows[i]); }
    if (!constrained) { return; }
    // The round copies its drafts to the host and then stamps them with its epoch; the target
    // forward that follows leaves the host its time to compute the masks.
    const volatile std::uint32_t* stamp = &dflash_host_egress->drafts_epoch;
    const auto deadline                 = Clock::now() + std::chrono::seconds(30);
    for (std::uint32_t spin = 0; *stamp != dflash_round_epoch; ++spin) {
        if ((spin & 1023U) != 0U) { continue; }
        const cudaError_t status = cudaStreamQuery(device.stream);
        if (status == cudaSuccess) {
            throw std::logic_error("DFlash round finished before its drafts reached the host");
        }
        if (status != cudaErrorNotReady) { CUDA_CHECK(status); }
        if (Clock::now() > deadline) {
            throw std::runtime_error("DFlash round drafts did not reach the host");
        }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    const std::uint32_t k = launch_windows[launch_window_index].drafts;
    for (std::size_t i = 0; i < part.size; ++i) {
        const auto extent = static_cast<std::size_t>(dflash_host_ingress->proposal_extents[i]);
        masks.fill(part.rows[i],
                   std::span<const TokenId>(dflash_host_egress->drafts.data() + i * k, extent));
    }
}

std::uint32_t ProgramImpl::launch_depth(std::uint32_t lane, std::uint32_t bound) const {
    const RequestControl& request = requests[lane];
    if (speculative_backend == SpeculativeBackend::Mtp) {
        return request.drafting_stopped ? 0U : std::min(request.draft_depth, bound);
    }
    const SequenceState& sequence = active_sequence(lane);
    const std::uint32_t frontier  = sequence.execution_frontier;
    const std::uint32_t context   = sequence.dflash_context_frontier;
    const std::uint32_t pending   = frontier >= context ? frontier - context : 0U;
    return std::max(request.drafting_stopped ? 1U : std::min(request.draft_depth, bound),
                    pending > 0 ? pending - 1U : 0U);
}

std::uint32_t ProgramImpl::load_bound(std::size_t rows) const noexcept {
    return speculation_load_bound(speculation_load, rows);
}

DFlashDraftWidth ProgramImpl::speculation_draft_width(std::size_t rows) const noexcept {
    return dflash_draft_width(draft_width, speculation_load, rows);
}

// The narrowest window covering `drafts` whose graphs cover a speculative batch of `batch` rows (a
// depth only a load bound asks is captured from the bound's lanes on; the widest window, a menu
// depth, covers every batch up to the widest speculative launch).
std::size_t ProgramImpl::launch_window_for(std::uint32_t drafts, std::uint32_t batch) const {
    if (batch == 0 || batch > speculative_batch_limit) {
        throw std::logic_error("speculative launch is wider than the plan's widest");
    }
    if (const auto window = launch_window_of(launch_windows, drafts, batch)) { return *window; }
    throw std::logic_error("no speculative launch window covers the step's rows");
}

runtime::BatchedGeneratedRound ProgramImpl::publish_step(std::size_t rows,
                                                         const LaunchRows& speculative,
                                                         const LaunchRows& plain,
                                                         std::uint32_t ordinary_steps,
                                                         runtime::ExecutionTiming timing) {
    bool faulted = false;
    for (std::size_t i = 0; i < speculative.size; ++i) {
        step_faults[speculative.rows[i]] =
            numeric_status.faulted(NumericStatus::Launch::Speculative, i) ? 1U : 0U;
        faulted = faulted || step_faults[speculative.rows[i]] != 0U;
    }
    for (std::size_t i = 0; i < plain.size; ++i) {
        step_faults[plain.rows[i]] =
            numeric_status.faulted(NumericStatus::Launch::Plain, i) ? 1U : 0U;
        faulted = faulted || step_faults[plain.rows[i]] != 0U;
    }
    const auto row_faults = faulted ? std::span<const std::uint8_t>(step_faults.data(), rows)
                                    : std::span<const std::uint8_t>{};
    if (speculative.size == 0 && ordinary_steps == 1) {
        return runtime::BatchedGeneratedRound{
            .tokens = std::span<const TokenId>(ordinary_host_egress->sampled_tokens.data(), rows),
            .row_faults = row_faults,
            .timing     = timing,
        };
    }
    if (speculative.size == 0) {
        // A round of several ordinary steps: every row produced one token per step.
        for (std::size_t i = 0; i < plain.size; ++i) {
            step_counts[plain.rows[i]] = static_cast<std::int32_t>(ordinary_steps);
        }
        return runtime::BatchedGeneratedRound{
            .tokens = std::span<const TokenId>(ordinary_run_tokens.data(), rows * ordinary_steps),
            .row_counts = std::span<const std::int32_t>(step_counts.data(), rows),
            .row_faults = row_faults,
            .row_stride = ordinary_steps,
            .timing     = timing,
        };
    }
    const bool mtp               = speculative_backend == SpeculativeBackend::Mtp;
    const std::uint32_t width    = launch_windows[launch_window_index].drafts + 1U;
    const TokenId* tokens        = mtp ? mtp_host_egress->licensed_tokens.data()
                                       : dflash_host_egress->licensed_tokens.data();
    const std::int32_t* counts   = mtp ? mtp_host_egress->licensed_counts.data()
                                       : dflash_host_egress->licensed_counts.data();
    if (plain.size == 0) {
        // Every row speculated: the launch rows are the step rows.
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(tokens, rows * width),
            .row_counts = std::span<const std::int32_t>(counts, rows),
            .row_faults = row_faults,
            .row_stride = width,
            .timing     = timing,
        };
    }
    for (std::size_t i = 0; i < speculative.size; ++i) {
        const std::uint32_t row = speculative.rows[i];
        step_counts[row]        = counts[i];
        std::copy_n(tokens + i * width, width, step_tokens.data() + row * width);
    }
    for (std::size_t i = 0; i < plain.size; ++i) {
        const std::uint32_t row          = plain.rows[i];
        step_counts[row]                 = 1;
        step_tokens[row * width]         = ordinary_host_egress->sampled_tokens[i];
    }
    return runtime::BatchedGeneratedRound{
        .tokens     = std::span<const TokenId>(step_tokens.data(), rows * width),
        .row_counts = std::span<const std::int32_t>(step_counts.data(), rows),
        .row_faults = row_faults,
        .row_stride = width,
        .timing     = timing,
    };
}

std::uint32_t ProgramImpl::launch_ordinary(const LaunchRows& part, runtime::MaskedUnit& masks,
                                           std::uint32_t steps) {
    std::uint32_t maximum_frontier = 0;
    // Every step of a round executes one more position of every row; a row that stops mid-round
    // keeps a prefix, and its executed rows past it must leave nothing a later row reads.
    steps = std::min({steps, qwen3_5::kMaximumOrdinaryDecodeSteps, uncommitted_row_limit + 1U});
    if (!use_cuda_graph) { steps = 1; }
    for (std::size_t row = 0; row < part.size; ++row) {
        const SequenceState& sequence = active_sequence(part.lanes[row]);
        const RequestControl& request = requests[part.lanes[row]];
        if (request.lifecycle != Lifecycle::Active ||
            part.budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("ordinary batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
        steps            = std::min({steps, part.budgets[row].generated_tokens_remaining,
                                     capacity - sequence.execution_frontier});
        // A forking row copies its source's rings before its first step only.
        if (sequence.state.fork_pending) { steps = 1; }
    }

    const auto batch                  = static_cast<std::uint32_t>(part.size);
    DecodeGraphExecutable* executable = nullptr;
    ops::CausalAttentionExecutionEnvelope envelope{maximum_frontier + 1, maximum_frontier + 1};
    if (use_cuda_graph) {
        DecodeGraphProfile& profile =
            select_graph_profile(ordinary_graphs, batch, maximum_frontier, "ordinary batch");
        executable = &install_graph_profile(ordinary_graphs, profile, "ordinary batch");
        envelope   = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
        // Every step runs the one installed graph, so its envelope must cover the last step too.
        steps = std::min(steps, profile.max_execution_frontier - maximum_frontier + 1U);
    }

    for (std::size_t row = 0; row < part.size; ++row) {
        SequenceState& sequence            = active_sequence(part.lanes[row]);
        const RequestControl& request      = requests[part.lanes[row]];
        const std::uint32_t frontier       = sequence.execution_frontier;
        ordinary_host_ingress->tokens[row] = sequence.ledger.back();
        ordinary_host_ingress->cache_positions[row] =
            checked_i32(frontier, "ordinary batch position");
        ordinary_host_ingress->rope_positions[row] =
            checked_i32(frontier, "ordinary batch RoPE position") + sequence.rope_delta;
        ordinary_host_ingress->text_kv_table_rows[row] =
            text_kv_addresses->bound_row(sequence.kv->text);
        const StateImageSelectors selectors                 = state_selectors(sequence);
        ordinary_host_ingress->state_source_slots[row]      = selectors.source;
        ordinary_host_ingress->state_destination_slots[row] = selectors.destination;
        ordinary_host_ingress->sampling[row] = masks.config(part.rows[row], request.sampling_host);
        ensure_sequence_kv_mapped(sequence, frontier + steps, 0);
    }
    poison_ingress_tail(*ordinary_host_ingress, part.size);

    execution::OrdinaryBatchContext schedule_state{
        {device, model, work, *state_images, replay_records ? &*replay_records : nullptr, io,
         prefill_hidden, prefill_chunk, capacity, proposal_head, draft_width, &speculation_load},
        decoder->text_kv,
        *io.ordinary,
        *ordinary_host_ingress,
        *ordinary_host_egress,
        state_images->continuation_hidden_store()};

    mark_workspace_usage(workspace_plan.ordinary_round);
    model.ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(batch), envelope,
                                executable);
    // Later steps feed on the device: a step's sampled tokens become the next step's inputs one
    // position later (in the device ingress, then in the pinned ingress the next graph copies
    // in), and are recorded in the egress, which the last step copies out whole.
    qwen3_5::OrdinaryDecodeState& frame = *io.ordinary;
    const auto rows                     = static_cast<std::int32_t>(batch);
    for (std::uint32_t step = 1; step < steps; ++step) {
        Tensor record = frame.step_tokens.slice(
            0, static_cast<std::int32_t>((step - 1U) * kMaximumConcurrency), rows);
        Tensor tokens          = frame.tokens.slice(0, 0, rows);
        Tensor cache_positions = frame.cache_positions.slice(0, 0, rows);
        Tensor rope_positions  = frame.rope_positions.slice(0, 0, rows);
        ops::feed_sampled_tokens(frame.sampled_tokens.slice(0, 0, rows), record, tokens,
                                 cache_positions, rope_positions, device.stream);
        CUDA_CHECK(cudaMemcpyAsync(ordinary_host_ingress, frame.ingress.data,
                                   qwen3_5::kOrdinaryDecodeFeedBytes, cudaMemcpyDeviceToHost,
                                   device.stream));
        model.ordinary_decode_batch(schedule_state, rows, envelope, executable);
    }
    // A NaN in the KV or the recurrent state carries into every later step, so the last step's
    // hidden stands for the round.
    numeric_status.check(NumericStatus::Launch::Plain,
                         frame.hidden.slice(1, 0, rows).view({frame.hidden.ne[0], 1, rows}),
                         nullptr, device.stream);
    return steps;
}

void ProgramImpl::collect_ordinary(const LaunchRows& part, double seconds, std::uint32_t steps) {
    const bool speculative_engine = speculative_backend != SpeculativeBackend::None;
    for (std::size_t row = 0; row < part.size; ++row) {
        SequenceState& sequence    = active_sequence(part.lanes[row]);
        RequestControl& request    = requests[part.lanes[row]];
        const std::uint32_t base_E = sequence.execution_frontier;
        const std::uint32_t base_S = sequence.ledger_frontier;
        if (steps == 1) {
            const TokenId token = ordinary_host_egress->sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            sequence.text_kv_valid = base_E + 1;
            // A speculative engine's drafter state stays where the request stopped speculating.
            commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            sequence.tail_hidden_valid = true;
            sequence.ledger.push_back(token);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
            sequence.prefix_digests.append_generated(std::span<const TokenId>(&token, 1),
                                                     sequence.rope_delta);
        } else {
            // The round's tokens, row-major at `steps` in the step's row: the recorded steps,
            // then the last one. Their KV is committed when the round resolves, up to the prefix
            // the request keeps (resolve_non_speculative_pending).
            TokenId* tokens = ordinary_run_tokens.data() + part.rows[row] * steps;
            for (std::uint32_t step = 0; step + 1U < steps; ++step) {
                tokens[step] = ordinary_host_egress->step_tokens[step * kMaximumConcurrency + row];
            }
            tokens[steps - 1U] = ordinary_host_egress->sampled_tokens[row];
            const std::span<const TokenId> produced(tokens, steps);
            validate_licensed_tokens(produced);
            sequence.ledger.insert(sequence.ledger.end(), produced.begin(), produced.end());
            sequence.prefix_identity.append_generated(steps, sequence.rope_delta);
            sequence.prefix_digests.append_generated(produced, sequence.rope_delta);
        }
        request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                             .base_E        = base_E,
                                             .base_S        = base_S,
                                             .prompt_tokens = 0,
                                             .produced      = steps};
        request.lifecycle = Lifecycle::Pending;
        request.timings.decode_seconds += seconds;
        if (speculative_engine) { request.speculative_stats.plain_steps += 1; }
    }
}

void ProgramImpl::launch_mtp(const LaunchRows& part, runtime::MaskedUnit& masks,
                             std::uint32_t bound, bool masks_first) {
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode) {
        throw std::logic_error("MTP batch execution requires the MTP backend");
    }
    // The launch window: the smallest captured depth that covers every row's depth, each capped
    // by the step's load bound (itself a captured window).
    std::uint32_t deepest = 1;
    for (std::size_t row = 0; row < part.size; ++row) {
        deepest = std::max(deepest, launch_depth(part.lanes[row], bound));
    }
    // Copy rounds: a drafting row whose context repeats verifies the copy proposal instead of its
    // MTP drafts when it holds more tokens; it replaces the row's drafts (the verify masks read
    // them too). Only where the copy window runs the batch and the step's load bound allows it,
    // and only for a launch of one row: the launch window is the batch's, so one copying row
    // verifies every row at the copy's width (at C=4, 64 columns instead of 16). 27B MTP 3 with
    // agreeing copies, C=4 against none: RTX 3090 -3.9 %, V100 -2.2 %, RTX 4090 +0.3 %; one row
    // +0.4 % (3090) to +2.6 % (4090) on short answers, +36 % on 24K summaries.
    if (copy_window_drafts != 0 && bound >= copy_window_drafts && part.size == 1 &&
        window_runs_batch(launch_windows.back().batches, static_cast<std::uint32_t>(part.size))) {
        for (std::size_t row = 0; row < part.size; ++row) {
            RequestControl& request = requests[part.lanes[row]];
            SequenceState& sequence = active_sequence(part.lanes[row]);
            request.copy_round      = false;
            if (request.drafting_stopped || sequence.execution_frontier + 1 >= capacity) {
                continue;
            }
            const std::uint32_t remaining = part.budgets[row].generated_tokens_remaining;
            const std::size_t limit =
                std::min({copy_window_drafts, remaining > 1 ? remaining - 1 : 0U,
                          capacity - sequence.execution_frontier - 1});
            std::array<TokenId, kMtpDecodeMaximumDrafts> copy{};
            const std::size_t copied = detail::copy_proposal(
                std::span<const TokenId>(sequence.ledger.data(), sequence.ledger.size()),
                copy_min_match, std::min(limit, copy.size()), copy.data());
            if (copied <= sequence.mtp_draft_count) { continue; }
            // The copy runs only where the MTP drafter's first draft equals the copy's first
            // token: a recurring suffix the drafter does not follow at all is a coincidence, and
            // its wide round is wasted (on an RTX 4090, 27B C=4 lost 7 % with no agreement). The
            // drafter's later drafts need not agree: they are where a copy beats it. Against every
            // draft agreeing, 24K summaries at C=1 (MTP 3 + head): RTX 5090 NVFP4 +2.9 %, RTX 4090
            // +0.5..1.5 %, RTX 5070 Ti IQ3_S +0.7 %, short answers even to +1.1 %; with no agreement
            // at all the RTX 5070 Ti lost 3.7 % on short answers.
            if (sequence.mtp_draft_count != 0 && sequence.mtp_drafts[0] != copy[0]) { continue; }
            std::copy_n(copy.begin(), copied, sequence.mtp_drafts.begin());
            sequence.mtp_draft_count = static_cast<std::uint32_t>(copied);
            request.copy_round       = true;
            deepest                  = std::max(deepest, static_cast<std::uint32_t>(copied));
        }
    }
    launch_window_index        = launch_window_for(deepest, static_cast<std::uint32_t>(part.size));
    LaunchWindow& window       = launch_windows[launch_window_index];
    const std::uint32_t k      = window.drafts;
    const std::uint32_t width  = k + 1;
    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < part.size; ++row) {
        const SequenceState& sequence = active_sequence(part.lanes[row]);
        const RequestControl& request = requests[part.lanes[row]];
        if (request.lifecycle != Lifecycle::Active ||
            part.budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            (backend_kv_cache() && (!sequence.kv->backend ||
                                    backend_kv_addresses->bound_row(*sequence.kv->backend) < 0)) ||
            sequence.execution_frontier >= capacity ||
            sequence.mtp_kv_valid != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier ||
            sequence.mtp_draft_count > draft_window) {
            throw std::logic_error("MTP batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto batch                  = static_cast<std::uint32_t>(part.size);
    DecodeGraphExecutable* executable = nullptr;
    execution::MtpCausalAttentionEnvelopes envelopes =
        mtp_causal_attention_envelopes(maximum_frontier, k, capacity);
    // The load bounds cap every row of an MTP launch, so its window runs this batch
    // (launch_window_batches); the round workspace and graphs are planned for no other.
    if (!window_runs_batch(window.batches, batch)) {
        throw std::logic_error("MTP launch window does not run a batch this wide");
    }
    if (use_cuda_graph) {
        DecodeGraphProfile& profile =
            select_graph_profile(window.graphs, batch, maximum_frontier, "MTP batch");
        executable = &install_graph_profile(window.graphs, profile, "MTP batch");
        envelopes  = mtp_causal_attention_envelopes(profile.max_execution_frontier, k, capacity);
    }

    for (std::size_t row = 0; row < part.size; ++row) {
        SequenceState& sequence           = active_sequence(part.lanes[row]);
        const RequestControl& request     = requests[part.lanes[row]];
        const std::uint32_t frontier      = sequence.execution_frontier;
        const std::uint32_t max_by_budget = part.budgets[row].generated_tokens_remaining > 1
                                                ? part.budgets[row].generated_tokens_remaining - 1
                                                : 0;
        // Every row drafts at the launch's width: the columns are verified for the whole batch
        // anyway, and a longer accepted run only saves rounds. A row's own depth decides only how
        // wide a launch it needs.
        const std::uint32_t extent =
            request.drafting_stopped
                ? 0U
                : std::min({sequence.mtp_draft_count, k, max_by_budget,
                            capacity - sequence.execution_frontier - 1});
        mtp_host_ingress->anchors[row]        = sequence.ledger.back();
        mtp_host_ingress->base_frontiers[row] = checked_i32(frontier, "MTP batch frontier");
        mtp_host_ingress->remaining_budgets[row] = checked_i32(
            part.budgets[row].generated_tokens_remaining, "MTP batch remaining budget");
        mtp_host_ingress->current_extents[row]      = static_cast<std::int32_t>(extent);
        mtp_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1);
        for (std::uint32_t j = 0; j < k; ++j) {
            mtp_host_ingress->current_drafts[row * k + j] =
                j < extent ? sequence.mtp_drafts[j] : sequence.ledger.back();
        }
        for (std::uint32_t j = 0; j < width; ++j) {
            const std::uint32_t position = frontier + std::min(j, extent);
            mtp_host_ingress->target_rope_positions[row * width + j] =
                checked_i32(position, "MTP batch RoPE position") + sequence.rope_delta;
        }
        mtp_host_ingress->text_kv_table_rows[row] = text_kv_addresses->bound_row(sequence.kv->text);
        mtp_host_ingress->mtp_kv_table_rows[row] =
            sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
        const StateImageSelectors selectors            = state_selectors(sequence);
        mtp_host_ingress->state_source_slots[row]      = selectors.source;
        mtp_host_ingress->state_destination_slots[row] = selectors.destination;
        mtp_host_ingress->rope_deltas[row]             = sequence.rope_delta;
        mtp_host_ingress->sampling[row] = masks.config(part.rows[row], request.sampling_host);
        ensure_sequence_kv_mapped(
            sequence, frontier + extent + 1,
            backend_kv_cache() != nullptr ? std::min(capacity, frontier + extent + k) : 0U);
    }

    poison_ingress_tail(*mtp_host_ingress, part.size, k, width);
    if (masks_first) {
        fill_launch_masks(part, true, masks);
        masks.release();
    }

    execution::MtpBatchContext schedule_state{
        {device, model, work, *state_images, window.records ? &*window.records : nullptr, io,
         prefill_hidden, prefill_chunk, capacity, proposal_head, draft_width, &speculation_load},
        decoder->text_kv,
        decoder->mtp_cache(),
        *window.mtp,
        *mtp_host_ingress,
        *mtp_host_egress,
        state_images->continuation_hidden_store(),
        window.depth};

    mark_workspace_usage(workspace_plan.mtp_round);
    model.mtp_decode_batch(schedule_state, static_cast<std::int32_t>(batch), k, envelopes,
                           executable);
    check_target_hidden(numeric_status, *window.mtp, batch, device.stream);
}

void ProgramImpl::launch_dflash(const LaunchRows& part, runtime::MaskedUnit& masks,
                                std::uint32_t bound) {
    if (!is_masked_draft_backend(speculative_backend) || !io.dflash_decode || !dflash) {
        throw std::logic_error("DFlash batch execution requires the DFlash backend");
    }
    // The launch window: the smallest captured depth that covers every row's depth (capped by the
    // step's load bound, itself a captured window) and holds the target features each row still
    // has to append to its drafter context (at most the width of the round that produced them:
    // the first bounded step after a wider round may run wider than the bound).
    std::uint32_t deepest = 1;
    for (std::size_t row = 0; row < part.size; ++row) {
        deepest = std::max(deepest, launch_depth(part.lanes[row], bound));
    }
    launch_window_index = launch_window_for(deepest, static_cast<std::uint32_t>(part.size));
    LaunchWindow& window                = launch_windows[launch_window_index];
    const std::uint32_t k               = window.drafts;
    const std::uint32_t width           = k + 1U;
    std::uint32_t maximum_frontier      = 0;
    // DFlash drafts appear on the host only mid-round: a constrained row drafts when its masks
    // follow the drafts (fill_dflash_draft_masks), else it runs with a zero draft extent, its
    // single verify column masked exactly.
    const auto extent_of = [&](std::size_t row, const SequenceState& sequence) {
        const std::uint32_t max_by_budget = part.budgets[row].generated_tokens_remaining > 1
                                                ? part.budgets[row].generated_tokens_remaining - 1U
                                                : 0U;
        return (masks.active(part.rows[row]) && !dflash_masks_follow_drafts) ||
                       requests[part.lanes[row]].drafting_stopped
                   ? 0U
                   : std::min({k, max_by_budget, capacity - sequence.execution_frontier - 1U});
    };
    std::uint32_t maximum_target_tokens = 1;
    for (std::size_t row = 0; row < part.size; ++row) {
        const SequenceState& sequence = active_sequence(part.lanes[row]);
        const RequestControl& request = requests[part.lanes[row]];
        if (request.lifecycle != Lifecycle::Active ||
            part.budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            (backend_kv_cache() && (!sequence.kv->backend ||
                                    backend_kv_addresses->bound_row(*sequence.kv->backend) < 0)) ||
            sequence.execution_frontier >= capacity ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            sequence.dflash_context_frontier > sequence.execution_frontier ||
            sequence.execution_frontier - sequence.dflash_context_frontier > width ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("DFlash batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
        maximum_target_tokens =
            std::max(maximum_target_tokens,
                     sequence.execution_frontier + extent_of(row, sequence) + 1U);
    }

    const auto batch                     = static_cast<std::uint32_t>(part.size);
    DecodeGraphExecutable* executable    = nullptr;
    execution::DFlashEnvelopes envelopes = dflash_envelopes(maximum_frontier, k);
    ops::CausalAttentionExecutionEnvelope target_envelope{1, maximum_target_tokens};
    if (use_cuda_graph) {
        // A window the load bounds leave out at this batch (a row still appending the features of
        // a wider round from before its step grew) runs without a graph, over the envelopes the
        // graph of its profile would have, so its kernels are the same.
        GraphExecutionProfile covering{};
        if (window_runs_batch(window.batches, batch)) {
            DecodeGraphProfile& profile =
                select_graph_profile(window.graphs, batch, maximum_frontier, "DFlash batch");
            executable   = &install_graph_profile(window.graphs, profile, "DFlash batch");
            covering.min = profile.min_execution_frontier;
            covering.max = profile.max_execution_frontier;
        } else {
            bool covered = false;
            for (const GraphExecutionProfile& planned :
                 dflash_graph_profiles(speculative_backend, capacity, k, batch)) {
                if (planned.min <= maximum_frontier && maximum_frontier <= planned.max) {
                    covering = planned;
                    covered  = true;
                    break;
                }
            }
            if (!covered) {
                throw std::logic_error("DFlash batch graph profiles do not cover the frontier");
            }
            for (std::size_t row = 0; row < part.size; ++row) {
                requests[part.lanes[row]].speculative_stats.ungraphed_rounds += 1;
            }
        }
        envelopes       = dflash_envelopes(covering.max, k);
        target_envelope = {
            1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                   capacity, static_cast<std::uint64_t>(covering.max) + k + 1ULL))};
    }

    for (std::size_t row = 0; row < part.size; ++row) {
        SequenceState& sequence           = active_sequence(part.lanes[row]);
        const RequestControl& request     = requests[part.lanes[row]];
        const std::uint32_t frontier      = sequence.execution_frontier;
        const std::uint32_t extent        = extent_of(row, sequence);
        dflash_host_ingress->anchors[row] = sequence.ledger.back();
        dflash_host_ingress->execution_frontiers[row] =
            checked_i32(frontier, "DFlash batch frontier");
        dflash_host_ingress->context_frontiers[row] =
            checked_i32(sequence.dflash_context_frontier, "DFlash context frontier");
        dflash_host_ingress->proposal_valid_columns[row] = static_cast<std::int32_t>(width);
        dflash_host_ingress->proposal_extents[row]       = static_cast<std::int32_t>(extent);
        dflash_host_ingress->target_valid_columns[row]   = static_cast<std::int32_t>(extent + 1U);
        for (std::uint32_t column = 0; column < width; ++column) {
            const std::uint32_t position = frontier + std::min(column, extent);
            dflash_host_ingress->target_rope_positions[row * width + column] =
                checked_i32(position, "DFlash target RoPE position") + sequence.rope_delta;
        }
        dflash_host_ingress->text_kv_table_rows[row] =
            text_kv_addresses->bound_row(sequence.kv->text);
        dflash_host_ingress->dflash_kv_table_rows[row] =
            sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
        dflash_host_ingress->active_lanes[row]       = static_cast<std::int32_t>(sequence.lane);
        const StateImageSelectors selectors          = state_selectors(sequence);
        dflash_host_ingress->state_source_slots[row] = selectors.source;
        dflash_host_ingress->state_destination_slots[row] = selectors.destination;
        dflash_host_ingress->sampling[row] = masks.config(part.rows[row], request.sampling_host);
        ensure_sequence_kv_mapped(sequence, frontier + extent + 1U,
                                  backend_kv_cache() ? frontier : 0U);
    }

    execution::DFlashBatchContext schedule_state{
        {device, model, work, *state_images, window.records ? &*window.records : nullptr, io,
         prefill_hidden, prefill_chunk, capacity, proposal_head, draft_width, &speculation_load},
        decoder->text_kv,
        *dflash,
        *window.dflash,
        *dflash_host_ingress,
        *dflash_host_egress,
        state_images->continuation_hidden_store()};

    dflash_host_ingress->round_epoch = ++dflash_round_epoch;
    mark_workspace_usage(workspace_plan.dflash_round);
    model.dflash_decode_batch(schedule_state, static_cast<std::int32_t>(batch), k, envelopes,
                              target_envelope, executable);
    check_target_hidden(numeric_status, *window.dflash, batch, device.stream);
}

void ProgramImpl::collect_speculative(const LaunchRows& part, double seconds) {
    const bool mtp            = speculative_backend == SpeculativeBackend::Mtp;
    const std::uint32_t k     = launch_windows[launch_window_index].drafts;
    const std::uint32_t width = k + 1U;
    // The launch's DFlash drafter drafted only the anchor and its k drafts (dflash_draft_width, by
    // the launch's rows, as the model's round resolved it).
    const bool narrow = !mtp && (!model.traits().dflash_block_width ||
                                 speculation_draft_width(part.size) == DFlashDraftWidth::Drafts);
    std::vector<std::size_t> tripping; // rows whose watchdog tripped on this round
    std::uint32_t drafting_on = 0;     // rows that drafted and whose watchdog held
    for (std::size_t row = 0; row < part.size; ++row) {
        SequenceState& sequence    = active_sequence(part.lanes[row]);
        RequestControl& request    = requests[part.lanes[row]];
        const std::uint32_t base_E = sequence.execution_frontier;
        const std::uint32_t base_S = sequence.ledger_frontier;
        const std::int32_t count_i =
            mtp ? mtp_host_egress->licensed_counts[row] : dflash_host_egress->licensed_counts[row];
        const std::int32_t accepted_i =
            mtp ? mtp_host_egress->accepted_drafts[row] : dflash_host_egress->accepted_drafts[row];
        const std::uint32_t extent =
            mtp ? static_cast<std::uint32_t>(mtp_host_ingress->current_extents[row])
                : static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]);
        const std::int32_t next_i = mtp ? mtp_host_egress->next_extents[row] : 0;
        if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
            accepted_i + 1 != count_i || accepted_i > static_cast<std::int32_t>(extent) ||
            next_i < 0 || next_i > static_cast<std::int32_t>(k) ||
            static_cast<std::uint32_t>(count_i) > part.budgets[row].generated_tokens_remaining ||
            static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) > capacity) {
            throw std::runtime_error(mtp ? "MTP batch returned invalid row metadata"
                                         : "DFlash batch returned invalid row metadata");
        }
        const TokenId* tokens = mtp ? mtp_host_egress->licensed_tokens.data()
                                    : dflash_host_egress->licensed_tokens.data();
        validate_licensed_tokens(
            std::span<const TokenId>(tokens + row * width, static_cast<std::size_t>(count_i)));
        const bool copy_round = mtp && request.copy_round;
        request.copy_round    = false;
        if (extent == 0) {
            request.speculative_stats.fallback_steps += 1;
        } else {
            request.speculative_stats.rounds += 1;
            request.speculative_stats.wide_rounds += k > request.draft_depth ? 1U : 0U;
            request.speculative_stats.narrow_rounds += narrow ? 1U : 0U;
            if (copy_round) {
                request.speculative_stats.copy_rounds += 1;
                request.speculative_stats.copy_accepted_tokens +=
                    static_cast<std::uint32_t>(accepted_i);
            } else if (k < request.draft_depth) {
                request.speculative_stats.capped_rounds += 1;
            } else {
                request.judged_rounds += 1;
                request.judged_accepted += static_cast<std::uint32_t>(accepted_i);
            }
            request.speculative_stats.drafted_tokens += extent;
            request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
            auto& per_position = request.speculative_stats.accepted_per_position;
            if (per_position.size() < static_cast<std::size_t>(accepted_i)) {
                per_position.resize(static_cast<std::size_t>(accepted_i), 0);
            }
            for (std::int32_t i = 0; i < accepted_i; ++i) {
                request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] += 1;
            }
        }
        if (copy_round) {
            ++drafting_on; // its MTP drafts were not judged: its watchdog stays where it was
        } else if (observe_watchdog(request)) {
            tripping.push_back(row);
        } else if (extent > 0) {
            ++drafting_on;
        }
        if (!mtp) { sequence.dflash_context_frontier = base_E; }
        request.pending   = PendingCandidate{
              .kind          = PendingKind::Speculative,
              .base_E        = base_E,
              .base_S        = base_S,
              .prompt_tokens = 0,
              .produced      = static_cast<std::uint32_t>(count_i),
              .launch_row    = static_cast<std::uint32_t>(row),
              .launch_window = k,
        };
        request.lifecycle = Lifecycle::Pending;
        request.timings.decode_seconds += seconds;
    }
    // The rows that keep drafting next step whatever a tripping row does: those whose watchdog
    // held, and tripping rows that step down to a shallower depth.
    std::uint32_t keep_drafting = drafting_on;
    for (const std::size_t row : tripping) {
        keep_drafting += requests[part.lanes[row]].step_down.empty() ? 0U : 1U;
    }
    for (const std::size_t row : tripping) {
        RequestControl& request = requests[part.lanes[row]];
        const std::uint32_t self = request.step_down.empty() ? 0U : 1U;
        apply_watchdog_trip(request, keep_drafting > self ? shared_trip_floor : 0.0);
    }
}

runtime::ExecutionTiming ProgramImpl::resolve_non_speculative_pending(
    SequenceState& sequence, RequestControl& request, std::uint32_t accepted_tokens, bool terminal,
    std::optional<std::uint32_t> prefix_execution_split_after,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (request.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("pending resolution requires a pending generated round");
    }
    // Begin commits its single token; an ordinary round all it produced, or a prefix at a stop.
    const std::uint32_t produced = request.pending.produced;
    const bool valid_prefix =
        request.pending.kind == PendingKind::Begin
            ? produced == 1 && accepted_tokens == 1
            : request.pending.kind == PendingKind::Ordinary && accepted_tokens != 0 &&
                  accepted_tokens <= produced && (terminal || accepted_tokens == produced);
    if (!valid_prefix) {
        throw std::logic_error("non-speculative pending round has an invalid committed prefix");
    }

    const std::uint32_t base_ledger_frontier = request.pending.kind == PendingKind::Begin
                                                   ? request.pending.prompt_tokens
                                                   : request.pending.base_S;
    if (request.pending.kind == PendingKind::Ordinary && produced > 1) {
        // A round of several steps commits its KV here, to the prefix the request keeps. A stop
        // mid-round drops the tokens past it; the rows it executed past it stay invisible
        // (uncommitted_row_limit), but the continuation hidden is the last executed row's.
        if (accepted_tokens < produced) {
            sequence.ledger.resize(base_ledger_frontier + accepted_tokens);
            sequence.prefix_identity.truncate(base_ledger_frontier);
            sequence.prefix_digests.truncate(base_ledger_frontier);
        }
        sequence.tail_hidden_valid = accepted_tokens == produced;
        sequence.text_kv_valid     = request.pending.base_E + accepted_tokens;
        commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    }
    commit_generated_prefix_identity(
        sequence, base_ledger_frontier,
        std::span<const TokenId>(sequence.ledger).subspan(base_ledger_frontier, accepted_tokens),
        prefix_execution_split_after);

    switch (request.pending.kind) {
    case PendingKind::Begin:
        sequence.execution_frontier = request.pending.prompt_tokens;
        sequence.ledger_frontier    = request.pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
        advance_rebuild_work(sequence, request.pending.base_E + accepted_tokens, prefill_chunk);
        sequence.execution_frontier = request.pending.base_E + accepted_tokens;
        sequence.ledger_frontier    = request.pending.base_S + accepted_tokens;
        break;
    case PendingKind::Speculative:
    case PendingKind::None:
        throw std::logic_error("non-speculative pending round has an invalid kind");
    }
    if (sequence.ledger_frontier != sequence.execution_frontier + 1 ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier ||
        sequence.prefix_digests.size() != sequence.ledger_frontier) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    // Begin publishes a sampled token but does not execute it through the target. An exact-hit
    // Fork therefore still names an immutable read source and an unwritten destination here; the
    // first state-mutating decode commit closes it. A suffix prefill already closed its Fork at
    // the committed prefill frontier.
    if (request.pending.kind == PendingKind::Begin && terminal && sequence.state.fork_pending) {
        const StateImageSelectors selectors = state_selectors(sequence);
        timing.resume_submit();
        state_images->copy_slot(selectors.source, selectors.destination, device.stream);
        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        settle_state_fork(sequence);
    } else if (request.pending.kind == PendingKind::Ordinary) {
        settle_state_fork(sequence);
    }
    trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    if (terminal) { sequence.mtp_draft_count = 0; }
    request.lifecycle = terminal ? Lifecycle::Finishable : Lifecycle::Active;
    request.pending   = {};
    return timing.finish();
}


} // namespace ninferno::models::qwen3_5::detail
