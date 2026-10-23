#include "core/removed_environment.h"
#include "options.h"
#include "spec_tune.h"

#include "artifact/source_identity.h"
#include "product/plan/plan_choice.h"
#include "product/plan/plan_facts.h"
#include "product/plan/plan_quants.h"
#include "product/plan/plan_speculation.h"
#include "spec/gpu_identity.h"
#include "core/command_line.h"
#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"
#include "product/logging/startup_log.h"
#include "product/prompt_input/prompt_input.h"
#include "product/speculation/speculation.h"
#include "product/speculative_options.h"
#include "spec/registry.h"

#include "ninferno/engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include <spdlog/logger.h>

namespace {

std::string format_seconds(double seconds) {
    return ninferno::product::format_pretty_duration(seconds);
}

std::string format_rate(double tokens, double seconds) {
    if (tokens <= 0.0 || seconds <= 0.0) { return "n/a"; }
    return ninferno::product::format_pretty_rate(tokens / seconds, "tok");
}

std::string format_percent(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) { return "n/a"; }
    return ninferno::product::format_pretty_percent(static_cast<double>(numerator) /
                                                  static_cast<double>(denominator));
}

std::string format_bytes(std::uint64_t bytes) {
    return ninferno::product::format_pretty_bytes(bytes);
}

std::string format_arena_used(const ninferno::ArenaMemorySummary& arena) {
    return format_bytes(arena.used_bytes) + " / " + format_bytes(arena.capacity_bytes);
}

std::string format_arena_peak(const ninferno::ArenaMemorySummary& arena) {
    return format_bytes(arena.peak_used_bytes) + " / " + format_bytes(arena.capacity_bytes);
}

std::string format_sampling(const ninferno::ResolvedSamplingParameters& sampling) {
    if (sampling.temperature <= 0.0F) { return "greedy (temperature 0)"; }
    std::ostringstream output;
    output << std::fixed << std::setprecision(2) << "temp=" << sampling.temperature
           << " top_p=" << sampling.top_p << " top_k=" << sampling.top_k
           << " min_p=" << sampling.min_p << " presence=" << sampling.presence_penalty
           << " freq=" << sampling.frequency_penalty << " seed=" << sampling.seed;
    return output.str();
}

std::string format_finish(ninferno::FinishReason reason) {
    std::string name(ninferno::finish_reason_name(reason));
    std::replace(name.begin(), name.end(), '_', '-');
    return name;
}

void print_stage(std::string_view group, std::string_view detail, double seconds) {
    std::cerr << std::left << std::setw(12) << group << std::setw(26) << detail << std::right
              << std::setw(12) << format_seconds(seconds) << '\n';
}

void print_metric(std::string_view label, std::string_view value) {
    std::cerr << std::left << std::setw(12) << "summary" << std::setw(26) << label << value << '\n';
}

class StreamingSink final : public ninferno::OutputSink {
public:
    void start(ninferno::GenerationStart) override {}

    void progress(ninferno::PromptProgress) override {}

    void timing(ninferno::GenerationTimingObservation) override {}

    void publish(ninferno::OutputDelta delta) override {
        std::ostream& output =
            delta.channel == ninferno::OutputChannel::Reasoning ? std::cerr : std::cout;
        output << delta.text;
        output.flush();
        if (delta.channel == ninferno::OutputChannel::Reasoning) {
            reasoning_seen_ = reasoning_seen_ || !delta.text.empty();
            if (!delta.text.empty()) { reasoning_ends_in_newline_ = delta.text.back() == '\n'; }
        } else {
            content_seen_ = content_seen_ || !delta.text.empty();
            if (!delta.text.empty()) { content_ends_in_newline_ = delta.text.back() == '\n'; }
        }
    }

    void finish_streams(bool successful = true) {
        if (finished_) { return; }
        finished_ = true;
        if ((successful && !content_seen_) || (content_seen_ && !content_ends_in_newline_)) {
            std::cout << '\n';
        }
        std::cout.flush();
        if (reasoning_seen_ && !reasoning_ends_in_newline_) { std::cerr << '\n'; }
    }

private:
    bool content_seen_              = false;
    bool content_ends_in_newline_   = false;
    bool reasoning_seen_            = false;
    bool reasoning_ends_in_newline_ = false;
    bool finished_                  = false;
};

void print_generation_summary(const ninferno::GenerationResult& result,
                              const ninferno::ResolvedSamplingParameters& sampling,
                              const ninferno::MemorySummary& memory) {
    print_stage("prepare", "render/preprocess", result.timings.prepare_seconds);
    print_stage("generate", "vision", result.timings.vision_seconds);
    print_stage("generate", "text prefill", result.timings.prefill_seconds);
    print_stage("generate", "decode", result.timings.decode_seconds);
    print_stage("generate", "total", result.timings.total_seconds);

    const std::size_t generated = result.generated_token_ids.size();
    const std::size_t decoded   = generated == 0 ? 0 : generated - 1;
    const double model_seconds  = result.timings.vision_seconds + result.timings.prefill_seconds +
                                 result.timings.decode_seconds;
    print_metric("sampling", format_sampling(sampling));
    print_metric("finish reason", format_finish(result.finish_reason));
    print_metric("prompt tokens", std::to_string(result.prompt.prompt_tokens));
    print_metric("reused prompt tokens", std::to_string(result.reused_prompt_tokens));
    print_metric("generated tokens", std::to_string(generated));
    if (result.thinking.configured_budget) {
        print_metric("thinking budget", std::to_string(*result.thinking.configured_budget));
        print_metric("model thinking tokens",
                     std::to_string(result.thinking.model_thinking_tokens));
        print_metric("thinking control tokens", std::to_string(result.thinking.injected_tokens));
        print_metric("thinking control", result.thinking.applied ? "applied" : "not applied");
    }
    print_metric("model elapsed", format_seconds(model_seconds));
    print_metric("prefill speed", format_rate(static_cast<double>(result.prompt.prompt_tokens),
                                              result.timings.prefill_seconds));
    print_metric("decode speed",
                 format_rate(static_cast<double>(decoded), result.timings.decode_seconds));
    print_metric("throughput (overall)",
                 format_rate(static_cast<double>(generated), model_seconds));

    const std::uint64_t reserved = static_cast<std::uint64_t>(memory.weights.capacity_bytes) +
                                   memory.runtime_reservation_bytes;
    print_metric("device", std::to_string(memory.device));
    print_metric("max context", std::to_string(memory.max_context));
    print_metric("KV capacity policy",
                 std::string(ninferno::kv_capacity_mode_name(memory.kv_capacity_mode)));
    print_metric("KV capacity", std::to_string(memory.kv_capacity));
    print_metric("KV page groups", std::to_string(memory.kv_capacity_page_groups) + " / " +
                                       std::to_string(memory.kv_capacity_max_page_groups));
    print_metric("gpu weights used", format_arena_used(memory.weights));
    print_metric("gpu sequence used", format_arena_used(memory.sequence));
    print_metric("kv cache dtype", std::string(ninferno::kv_cache_storage_layout(memory.kv_cache)));
    print_metric("kv cache payload", format_bytes(memory.kv_payload_bytes));
    print_metric("gpu workspace peak", format_arena_peak(memory.workspace));
    print_metric("runtime reservation", format_bytes(memory.runtime_reservation_bytes));
    print_metric("free after weights", format_bytes(memory.available_after_weights_bytes));
    print_metric("free after startup", format_bytes(memory.available_after_startup_bytes));
    print_metric("KV capacity headroom", format_bytes(memory.kv_capacity_headroom_bytes));
    print_metric("planned slack", format_bytes(memory.planned_slack_bytes));
    print_metric("CUDA Graph allowance", format_bytes(memory.cuda_graph_allowance_bytes));
    print_metric("planned device total", format_bytes(reserved));

    const ninferno::SpeculativeStats& speculative = result.speculative;
    if (speculative.enabled) {
        const std::string backend =
            ninferno::product::speculative_backend_name(speculative.backend);
        print_metric(backend + " draft window", std::to_string(speculative.draft_window));
        print_metric(backend + " rounds", std::to_string(speculative.rounds));
        print_metric(backend + " fallback steps", std::to_string(speculative.fallback_steps));
        print_metric(backend + " drafted tokens", std::to_string(speculative.drafted_tokens));
        print_metric(backend + " accepted tokens", std::to_string(speculative.accepted_tokens));
        print_metric(backend + " acceptance rate",
                     format_percent(speculative.accepted_tokens, speculative.drafted_tokens));
        if (speculative.rounds != 0) {
            std::ostringstream length;
            length << std::fixed << std::setprecision(2)
                   << 1.0 + static_cast<double>(speculative.accepted_tokens) /
                                static_cast<double>(speculative.rounds)
                   << " tok/round";
            print_metric(backend + " acceptance length", length.str());
        }
        if (speculative.copy_rounds != 0) {
            print_metric(backend + " copy rounds",
                         std::to_string(speculative.copy_rounds) + " (" +
                             std::to_string(speculative.copy_accepted_tokens) + " accepted)");
        }
        if (speculative.step_downs != 0) {
            print_metric(backend + " watchdog steps",
                         std::to_string(speculative.step_downs) + " down to depth " +
                             std::to_string(speculative.final_draft_window));
        }
        if (speculative.watchdog_tripped) {
            std::ostringstream verdict;
            verdict << std::fixed << std::setprecision(2) << "tripped at round "
                    << speculative.watchdog_round << " (" << speculative.watchdog_tokens_per_round
                    << " tok/round, below the profile's break-even)";
            print_metric(backend + " watchdog", verdict.str());
        }
        if (!speculative.accepted_per_position.empty()) {
            std::ostringstream positions;
            for (std::size_t i = 0; i < speculative.accepted_per_position.size(); ++i) {
                if (i != 0) { positions << ','; }
                positions << speculative.accepted_per_position[i];
            }
            print_metric(backend + " accepted by pos", positions.str());
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (ninferno::version_requested(argc, argv)) {
        std::cout << "NInferno " << NINFERNO_VERSION_STRING << '\n';
        return 0;
    }
    if (argc >= 2 && std::string_view(argv[1]) == "spec-tune") {
        ninferno::warn_removed_environment("ninferno spec-tune");
        try {
            return ninferno::cli::run_spec_tune(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "error: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc >= 2 && std::string_view(argv[1]) == "spec-reselect") {
        ninferno::warn_removed_environment("ninferno spec-reselect");
        try {
            return ninferno::cli::run_spec_reselect(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "error: " << error.what() << '\n';
            return 1;
        }
    }
    ninferno::cli::Options cli;
    try {
        cli = ninferno::cli::parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        std::cerr << ninferno::cli::usage_text(argv[0]);
        return 1;
    }
    if (cli.help_requested) {
        std::cout << ninferno::cli::usage_text(argv[0]);
        return 0;
    }
    // A removed NINFERNO_* decision variable changes nothing: say so once.
    ninferno::warn_removed_environment("ninferno");

    if (!cli.drafter_registry.empty()) { ninferno::spec::Registry::use_file(cli.drafter_registry); }

    ninferno::product::LoggingRuntime logging(
        {.logger_name  = "ninferno",
         .level        = cli.log_level,
         .presentation = ninferno::product::LogPresentation::Tool});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninferno::product::StartupLogRenderer startup_log(logging);

    try {

        ninferno::PromptInput input =
            cli.messages_path.empty()
                ? ninferno::product::prompt_from_text(cli.prompt, cli.enable_thinking)
                : ninferno::product::prompt_from_messages(cli.messages_path, cli.enable_thinking,
                                                        cli.enable_vision);
        input.options.reasoning_effort = cli.reasoning_effort;

        ninferno::RequestOptions request;
        request.execution.sampling                = cli.sampling;
        request.execution.requested_output_tokens = cli.max_new;
        request.execution.thinking.budget         = cli.thinking_budget;
        request.stop.token_ids                    = cli.stop_token_ids;
        request.stop.strings                      = cli.stop_strings;
        request.output.raw                        = cli.raw_output;

        ninferno::EngineOptions engine_options;
        engine_options.artifact_path      = cli.artifact_path;
        engine_options.chat_template_path = cli.chat_template_path;
        engine_options.device             = cli.device;
        engine_options.max_context        = cli.max_context;
        engine_options.kv_capacity        = cli.kv_capacity;
        engine_options.prefill_chunk      = cli.prefill_chunk;
        engine_options.kv_cache = cli.kv_cache.value_or(ninferno::KvCacheStorage::BFloat16);
        engine_options.kv_cache_automatic = !cli.kv_cache.has_value();
        engine_options.sliding_kv_cache   = cli.sliding_kv_cache;
        engine_options.recurrent_state =
            cli.recurrent_state.value_or(ninferno::RecurrentStateStorage::FP32);
        engine_options.recurrent_state_automatic = !cli.recurrent_state.has_value();
        engine_options.activations        = cli.activations;
        engine_options.rope_yarn_factor   = cli.rope_yarn_factor;
        engine_options.speculative        = cli.speculative;
        // A tune named by --spec-profile serves or the run does not start (whatever --spec says).
        ninferno::product::require_named_tune(cli.artifact_path, cli.device, cli.spec_profile);
        if (!cli.spec_explicit) {
            // The speculation the model came with, for this GPU: the profile's default for the
            // thinking flag (on by default, as the profile's).
            std::string source;
            std::vector<std::string> notes;
            const auto chosen = ninferno::product::profile_speculation(
                cli.artifact_path, cli.device, cli.enable_thinking, &source, &notes,
                cli.spec_profile);
            for (const auto& note : notes) { logger->info("speculation: {}", note); }
            if (chosen) {
                engine_options.speculative =
                    ninferno::product::with_given_settings(*chosen, cli.speculative);
                logger->info("speculation: {}", source);
            }
        } else if (engine_options.speculative.drafter.empty()) {
            // An explicit --spec drafts with the drafter that came with the model, as the server.
            engine_options.speculative.drafter = ninferno::product::model_drafter(
                cli.artifact_path, engine_options.speculative.backend);
        }
        if (cli.acceptance_floor) {
            engine_options.speculative.acceptance_floor = *cli.acceptance_floor;
        }
        engine_options.enable_vision      = cli.enable_vision;
        engine_options.use_cuda_graph     = cli.use_cuda_graph;
        engine_options.pacing             = cli.pacing;
        engine_options.maintainer         = cli.maintainer;
        // One CLI invocation owns exactly one request, so retained cross-request context has no
        // consumer and must not reserve an extra Device StateImage or run terminal capture.
        engine_options.context_cache.disable();
        engine_options.startup_observer                     = startup_log.observer();
        // --plan --no-load: the plan from the artifact's own facts, as ninferno-serve's.
        if (cli.plan_only) {
            namespace product = ninferno::product;
            ninferno::runtime::PlanDeviceFacts device;
            ninferno::spec::GpuKey gpu;
            if (!cli.plan_device.empty()) {
                std::string why;
                const auto board = product::plan_device_of_board(cli.plan_device, &why);
                if (!board) { throw std::runtime_error("--plan-device: " + why); }
                device = *board;
                gpu    = product::gpu_key_of(device);
            } else {
                device = product::plan_device_of_cuda(
                    cli.device, true, ninferno::runtime::plan_automatic_fit(engine_options));
                gpu    = ninferno::spec::gpu_identity(cli.device).key;
            }
            const product::PlanSourceFacts facts =
                product::plan_facts_of_artifact(cli.artifact_path);
            const product::SpeculationSubject subject =
                product::speculation_subject(cli.artifact_path, gpu, cli.spec_profile);
            product::DryRunInputs inputs;
            inputs.device              = device;
            inputs.gpu                 = gpu;
            inputs.options             = engine_options;
            inputs.ask                 = product::SpeculationAsk{.automatic = !cli.spec_explicit,
                                                                 .given     = cli.speculative,
                                                                 .thinking  = cli.enable_thinking.value_or(true)};
            inputs.options.speculative = cli.speculative;
            try {
                inputs.ask.source_identity =
                    ninferno::artifact::source_identity(cli.artifact_path).value_or(std::string());
            } catch (const std::exception&) {}
            inputs.subject                            = &subject;
            const product::PlannedSpeculation planned = product::plan_model(facts, inputs);
            const std::string source                  = cli.artifact_path.string();
            // The configuration choice for the axes left automatic, as a start would take it.
            const ninferno::product::ChoicePins pins{.kv          = cli.kv_cache.has_value(),
                                                     .activations = cli.activations_given,
                                                     .state       = cli.recurrent_state.has_value(),
                                                     .ring        = cli.sliding_kv_given,
                                                     .chunk       = true}; // the CLI's chunk is fixed
            const auto served = product::choose_at_weights(facts, inputs, pins);
            if (cli.plan_json) {
                nlohmann::json doc =
                    product::dry_plan_document(facts, source, "artifact", inputs, planned, nullptr);
                if (served) { doc["choice"] = product::choice_json(*served); }
                std::cout << doc.dump(1) << '\n';
            } else {
                std::cout << product::render_dry_plan_text(facts, source, inputs, planned, nullptr);
                if (served) { std::cout << product::choice_line(*served) << '\n'; }
            }
            return planned.plan.fits ? 0 : 2;
        }
        // The axes the user left to the engine: the configuration choice for this artifact on this
        // GPU (product/plan/plan_choice.h), as ninferno-serve's start takes it.
        {
            const ninferno::product::ChoicePins pins{.kv          = cli.kv_cache.has_value(),
                                                     .activations = cli.activations_given,
                                                     .state       = cli.recurrent_state.has_value(),
                                                     .ring        = cli.sliding_kv_given,
                                                     .chunk       = true}; // the CLI's chunk is fixed
            if (!(pins.kv && pins.activations && pins.state && pins.ring)) {
                ninferno::product::SpeculationAsk ask;
                ask.automatic = false;
                ask.given     = engine_options.speculative;
                ask.thinking  = cli.enable_thinking.value_or(true);
                std::string why;
                const auto served = ninferno::product::choose_for_artifact(
                    cli.artifact_path, cli.device, engine_options, ask, cli.spec_profile, pins, &why);
                if (served) {
                    logger->info("{}", ninferno::product::choice_line(*served));
                    if (served->choice.chosen) {
                        ninferno::product::apply_start_axes(
                            engine_options, ninferno::product::start_axes(*served, pins));
                    }
                } else if (!why.empty()) {
                    logger->info("choice | not planned: {}", why);
                }
            }
        }

        // A profile's choice that does not fit the requested context on this device gives way as
        // the server's does (GenerationService): to the profile's fit fallback
        // (spec::Profile::fit_fallback), then the deepest shallower draft with the same drafter
        // (product::deepest_fitting_shallower), then the model's own MTP head, then plain decode,
        // the switch logged. An explicit --spec, or neither, keeps the refusal.
        const bool from_profile = !cli.spec_explicit && engine_options.speculative.backend !=
                                                            ninferno::SpeculativeBackend::None;
        const auto construct = [&]() -> ninferno::Engine {
            try {
                return ninferno::Engine(engine_options);
            } catch (const ninferno::CapacityError& refused) {
                if (!from_profile) { throw; }
                namespace product = ninferno::product;
                const std::string reason =
                    product::fit_refusal(product::speculation_label(engine_options.speculative),
                                         engine_options.max_context, refused.what(),
                                         engine_options.max_context_automatic);
                std::optional<ninferno::Engine> built;
                // Builds the Engine with `speculative` (the given settings kept): true when it
                // fits.
                const auto fits = [&](ninferno::SpeculativeOptions speculative,
                                      const std::string& source) {
                    if (cli.acceptance_floor) {
                        speculative.acceptance_floor = *cli.acceptance_floor;
                    }
                    engine_options.speculative =
                        product::with_given_settings(std::move(speculative), cli.speculative);
                    built.reset();
                    try {
                        built.emplace(engine_options);
                    } catch (const ninferno::CapacityError&) { return false; }
                    logger->warn("speculation: {}; running {}", reason, source);
                    return true;
                };
                auto served = product::served_fit_fallback(cli.artifact_path, cli.device, false,
                                                           cli.spec_profile);
                if (served && fits(served->engine, served->source)) { return std::move(*built); }
                if (!served) {
                    served = product::served_speculation(cli.artifact_path, cli.device,
                                                         cli.enable_thinking.value_or(true), false,
                                                         cli.spec_profile);
                }
                if (served) {
                    const auto shallower = [&](const ninferno::spec::SpecConfig& config,
                                               double floor) {
                        return product::served_shallower(cli.artifact_path, cli.device, false,
                                                         config, floor, cli.spec_profile);
                    };
                    if (product::deepest_fitting_shallower(
                            served->menu.primary, shallower,
                            [&](const product::ServedSpeculation& candidate) {
                                return fits(candidate.engine, candidate.source);
                            })) {
                        return std::move(*built);
                    }
                }
                if (auto head = product::served_head_fallback(cli.artifact_path, cli.device, false,
                                                              cli.spec_profile);
                    head && fits(head->engine, head->source)) {
                    return std::move(*built);
                }
                if (fits(ninferno::SpeculativeOptions{}, "plain decode")) {
                    return std::move(*built);
                }
                throw;
            }
        };
        ninferno::Engine engine = construct();
        startup_log.engine_ready(engine.load_summary());
        engine.reset_memory_peaks();

        ninferno::PreparedPrompt prompt = engine.prepare(std::move(input));

        StreamingSink sink;
        ninferno::GenerationHandle generation = engine.submit(
            std::move(prompt), std::move(request), ninferno::OutputConsumerMode::Streaming);
        const ninferno::ResolvedSamplingParameters sampling = generation.resolved_sampling();
        ninferno::GenerationResult result;
        try {
            result = generation.wait(&sink);
            sink.finish_streams();
        } catch (...) {
            sink.finish_streams(false);
            throw;
        }

        std::cerr << "phase       detail                      elapsed/progress\n";
        if (cli.print_token_ids) {
            std::cerr << std::left << std::setw(12) << "tokens" << std::setw(26) << "generated ids";
            for (std::size_t i = 0; i < result.generated_token_ids.size(); ++i) {
                if (i != 0) { std::cerr << ' '; }
                std::cerr << result.generated_token_ids[i];
            }
            std::cerr << '\n';
        }
        print_generation_summary(result, sampling, engine.memory_summary());
        return 0;
    } catch (const std::exception& error) {
        logger->error("{}", ninferno::product::format_pretty_text(error.what()));
        return 1;
    }
}
