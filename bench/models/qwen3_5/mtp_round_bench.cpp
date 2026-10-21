#include "runtime/engine/model_instance.h"

#include "core/device.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace qwen = ninferno::models::qwen3_5;

struct Options {
    std::filesystem::path artifact = "out/qwen3_6_27b.ninferno";
    int device                     = 0;
    int warmup                     = 2;
    int repetitions                = 10;
    std::uint32_t draft_tokens     = 5;
    ninferno::ProposalHead proposal  = ninferno::ProposalHead::Optimized;
    bool use_cuda_graph            = true;
};

void print_usage(const char* executable) {
    std::cout << "usage: " << executable
              << " [--artifact <model.ninferno>] [--device <N|pci:BUSID|GPU-UUID>] [--warmup <n>] "
                 "[--reps <n>]"
                 " [--draft-tokens <1..5>] [--proposal-head full|optimized]"
                 " [--no-cuda-graph]\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&](const char* name) -> const char* {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(name) + " needs value");
            }
            return argv[index];
        };
        if (argument == "--artifact") {
            options.artifact = value("--artifact");
        } else if (argument == "--device") {
            // As ninferno-serve takes it: N counts GPUs in PCI bus order.
            options.device =
                ninferno::select_device(ninferno::parse_device_option(value("--device")));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(value("--warmup"));
        } else if (argument == "--reps") {
            options.repetitions = std::stoi(value("--reps"));
        } else if (argument == "--draft-tokens") {
            options.draft_tokens = static_cast<std::uint32_t>(std::stoul(value("--draft-tokens")));
        } else if (argument == "--proposal-head") {
            const std::string_view head(value("--proposal-head"));
            if (head == "full") {
                options.proposal = ninferno::ProposalHead::Full;
            } else if (head == "optimized") {
                options.proposal = ninferno::ProposalHead::Optimized;
            } else {
                throw std::invalid_argument("--proposal-head must be full or optimized");
            }
        } else if (argument == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (argument == "-h" || argument == "--help") {
            print_usage(argc > 0 ? argv[0] : "ninferno_qwen3_5_mtp_round_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.warmup < 0) { throw std::invalid_argument("--warmup must be nonnegative"); }
    if (options.repetitions <= 0) { throw std::invalid_argument("--reps must be positive"); }
    if (options.draft_tokens == 0 || options.draft_tokens > 5) {
        throw std::invalid_argument("--draft-tokens must be in [1,5]");
    }
    return options;
}

struct RoundMeasurement {
    float milliseconds            = 0.0F;
    std::uint32_t licensed_tokens = 0;
};

RoundMeasurement measure_round(qwen::Program& program, ninferno::DeviceContext& device,
                               qwen::SequenceHandle sequence, std::uint32_t draft_tokens) {
    const std::array<qwen::SequenceHandle, 1> sequences{sequence};
    const std::array<ninferno::runtime::RoundBudget, 1> budgets{
        ninferno::runtime::RoundBudget{.generated_tokens_remaining = draft_tokens + 1}};
    ninferno::CudaEventTimer timer(device);
    timer.start();
    auto pending                 = program.decode(sequences, budgets);
    const std::uint32_t licensed = pending.row_counts().empty()
                                       ? 1U
                                       : static_cast<std::uint32_t>(pending.row_counts().front());
    const std::array<ninferno::runtime::CommitDecision, 1> decisions{
        ninferno::runtime::CommitDecision{.accepted_tokens = licensed}};
    (void)program.commit(std::move(pending), decisions);
    const float milliseconds = timer.stop_ms();
    return RoundMeasurement{.milliseconds = milliseconds, .licensed_tokens = licensed};
}

int run(const Options& options) {
    if (!std::filesystem::exists(options.artifact)) {
        std::cout << "SKIP: artifact not present: " << options.artifact.string() << '\n';
        return 0;
    }

    const std::vector<ninferno::TokenId> seed{248045, 846, 198, 5834, 248046, 198};
    const std::uint32_t measured_rounds =
        static_cast<std::uint32_t>(options.warmup + options.repetitions);
    ninferno::EngineOptions engine;
    engine.artifact_path       = options.artifact;
    engine.device              = options.device;
    engine.max_context         = static_cast<std::uint32_t>(seed.size() + 64ULL +
                                                            static_cast<std::uint64_t>(measured_rounds) *
                                                                (options.draft_tokens + 1ULL) +
                                                            2ULL * options.draft_tokens);
    engine.kv_capacity         = ninferno::KvCapacityPolicy::explicit_capacity(engine.max_context);
    engine.prefill_chunk       = 128;
    engine.kv_cache            = ninferno::KvCacheStorage::BFloat16;
    engine.speculative.backend = ninferno::SpeculativeBackend::Mtp;
    engine.speculative.draft_tokens  = options.draft_tokens;
    engine.speculative.proposal_head = options.proposal;
    engine.use_cuda_graph            = options.use_cuda_graph;

    engine = ninferno::runtime::normalize_engine_options(std::move(engine));
    ninferno::DeviceContext device(options.device);
    auto constructed = ninferno::runtime::construct_model(engine, device);
    auto& frontend   = constructed.instance->frontend;
    auto& program    = constructed.instance->program;
    auto prompt      = frontend.prepare_tokens(seed, false);
    ninferno::runtime::ResolvedExecutionOptions execution;
    execution.requested_output_tokens = 1 + measured_rounds * (options.draft_tokens + 1);
    execution.allow_prefix_reuse      = false;
    auto request_base                 = program->plan_request(prompt, execution);
    auto request_plan                 = program->inspect_admission(
        prompt, request_base, ninferno::runtime::LaneId{0}, nullptr, nullptr, std::nullopt, false);
    if (!request_plan) { throw std::runtime_error("benchmark root admission was rejected"); }
    auto resource_plan = program->seal_identity(*request_plan, prompt, {});
    if (!resource_plan) { throw std::runtime_error("benchmark root resources were not sealed"); }
    const auto reserved =
        program->start_resource_transaction(std::move(*resource_plan), std::move(prompt), {});
    if (reserved != ninferno::runtime::ContextTransactionReserveStatus::Reserved) {
        throw std::runtime_error("benchmark root materialization was not reserved");
    }
    std::optional<qwen::MaterializationResult> published;
    for (;;) {
        auto transaction = program->progress_context_transaction({});
        if (std::holds_alternative<ninferno::runtime::ContextTransactionInProgress>(transaction)) {
            continue;
        }
        if (!std::holds_alternative<qwen::MaterializationResult>(transaction)) {
            program->finalize_context_transaction();
            throw std::runtime_error("benchmark root returned the wrong transaction result");
        }
        published.emplace(std::get<qwen::MaterializationResult>(std::move(transaction)));
        break;
    }
    if (published->status != ninferno::runtime::ContextTransactionStatus::Published ||
        !published->published) {
        program->finalize_context_transaction();
        throw std::runtime_error("benchmark root materialization was not published");
    }
    auto started = std::move(*published->published);
    program->finalize_context_transaction();
    auto progress = program->advance_prefill(started.sequence);
    if (!progress.complete || !progress.pending || progress.pending->tokens().size() != 1) {
        throw std::runtime_error("benchmark seed prefill did not complete in one scheduling unit");
    }
    const std::array<ninferno::runtime::CommitDecision, 1> begin_decision{
        ninferno::runtime::CommitDecision{.accepted_tokens = 1}};
    (void)program->commit(std::move(*progress.pending), begin_decision);
    const auto active_sequence = started.sequence;

    constexpr std::uint64_t rounds_before = 0;
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        (void)measure_round(*program, device, active_sequence, options.draft_tokens);
    }

    std::vector<RoundMeasurement> measurements;
    measurements.reserve(static_cast<std::size_t>(options.repetitions));
    for (int iteration = 0; iteration < options.repetitions; ++iteration) {
        measurements.push_back(
            measure_round(*program, device, active_sequence, options.draft_tokens));
    }
    const auto aborted = program->abort(active_sequence);
    if (aborted.status != ninferno::runtime::ConsumeStatus::Consumed) {
        throw std::runtime_error("benchmark could not release its active sequence");
    }
    const ninferno::SpeculativeStats stats = aborted.speculative;
    if (stats.rounds - rounds_before != measured_rounds || stats.fallback_steps != 0) {
        throw std::runtime_error("benchmark did not stay on the native MTP proposal/verify path");
    }

    std::vector<float> milliseconds;
    milliseconds.reserve(measurements.size());
    std::uint64_t licensed_tokens = 0;
    for (const RoundMeasurement& measurement : measurements) {
        milliseconds.push_back(measurement.milliseconds);
        licensed_tokens += measurement.licensed_tokens;
    }
    const double mean_ms =
        std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0) / measurements.size();
    const auto [minimum, maximum] = std::minmax_element(milliseconds.begin(), milliseconds.end());
    const double mean_licensed =
        static_cast<double>(licensed_tokens) / static_cast<double>(measurements.size());

    std::cout << "format,ninferno_qwen3_5_mtp_round_bench_v2\n";
    std::cout << "artifact," << options.artifact.string() << '\n';
    std::cout << "device," << device.props.name << '\n';
    std::cout << "draft_tokens," << options.draft_tokens << '\n';
    std::cout << "proposal_head,"
              << (options.proposal == ninferno::ProposalHead::Optimized ? "optimized" : "full")
              << '\n';
    std::cout << "cuda_graph," << (options.use_cuda_graph ? "true" : "false") << '\n';
    std::cout << "warmup," << options.warmup << '\n';
    std::cout << "repetitions," << options.repetitions << '\n';
    std::cout << "mtp_round_mean_ms," << mean_ms << '\n';
    std::cout << "mtp_round_min_ms," << *minimum << '\n';
    std::cout << "mtp_round_max_ms," << *maximum << '\n';
    std::cout << "mean_licensed_tokens," << mean_licensed << '\n';
    std::cout << "accepted_draft_tokens," << stats.accepted_tokens << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ninferno_qwen3_5_mtp_round_bench: " << error.what() << '\n';
        return 1;
    }
}
