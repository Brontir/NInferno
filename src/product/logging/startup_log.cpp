#include "product/logging/startup_log.h"

#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"

#include <spdlog/logger.h>

#include "core/sys.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace ninferno::product {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kStartupPhaseCount =
    static_cast<std::size_t>(StartupPhase::EngineFinalize) + 1;
constexpr auto kInteractiveRefresh = std::chrono::milliseconds(200);
constexpr auto kPersistentRefresh  = std::chrono::seconds(10);

enum class PhaseVisibility : std::uint8_t {
    Info,
    Debug,
};

struct PhasePresentation {
    const char* active;
    const char* complete;
    PhaseVisibility visibility;
    bool potentially_long;
    bool show_rate;
};

PhasePresentation phase_presentation(StartupPhase phase) noexcept {
    switch (phase) {
    case StartupPhase::EngineStartup:
        return {"starting engine", "engine startup", PhaseVisibility::Info, true, false};
    case StartupPhase::CudaInitialize:
        return {"initializing CUDA", "CUDA initialized", PhaseVisibility::Debug, false, false};
    case StartupPhase::ArtifactInspect:
        return {"inspecting artifact", "artifact inspected", PhaseVisibility::Debug, false, false};
    case StartupPhase::TargetPlan:
        return {"planning runtime", "runtime planned", PhaseVisibility::Debug, false, false};
    case StartupPhase::WeightsMaterialize:
        return {"loading weights", "weights ready", PhaseVisibility::Info, true, true};
    case StartupPhase::WeightsStagingPin:
        return {"pinning staging buffers", "staging buffers pinned", PhaseVisibility::Debug, false,
                false};
    case StartupPhase::TargetFinalize:
        return {"finalizing target", "target finalized", PhaseVisibility::Debug, false, false};
    case StartupPhase::FrontendInitialize:
        return {"initializing frontend", "frontend ready", PhaseVisibility::Debug, false, false};
    case StartupPhase::ProgramInitialize:
        return {"initializing runtime", "runtime initialized", PhaseVisibility::Debug, false,
                false};
    case StartupPhase::HostStatePin:
        return {"pinning host state", "host state pinned", PhaseVisibility::Info, true, false};
    case StartupPhase::HostKvPin:
        return {"pinning host KV", "host KV pinned", PhaseVisibility::Info, true, false};
    case StartupPhase::CudaGraphPrepare:
        return {"preparing CUDA graphs", "CUDA graphs ready", PhaseVisibility::Info, false, false};
    case StartupPhase::KernelWarmUp:
        return {"warming up kernels", "kernels warmed up", PhaseVisibility::Info, true, false};
    case StartupPhase::EngineFinalize:
        return {"finalizing engine", "engine finalized", PhaseVisibility::Debug, false, false};
    }
    return {"starting unknown phase", "unknown phase complete", PhaseVisibility::Debug, false,
            false};
}

std::size_t terminal_columns() noexcept {
    const int columns = sys::terminal_columns();
    return columns > 0 ? static_cast<std::size_t>(columns) : 120;
}

std::string progress_bar(double ratio, std::size_t width) {
    const std::size_t completed = static_cast<std::size_t>(ratio * static_cast<double>(width));
    std::string bar(width, '.');
    for (std::size_t index = 0; index < std::min(completed, width); ++index) { bar[index] = '='; }
    if (completed < width) { bar[completed] = '>'; }
    return bar;
}

struct PhaseProgress {
    Clock::time_point last_persistent;
    Clock::time_point last_interactive;
    Clock::time_point sample_time;
    std::uint64_t sample_bytes       = 0;
    double smoothed_bytes_per_second = 0.0;
};

void update_rate(PhaseProgress& state, std::uint64_t current, Clock::time_point now) {
    const double seconds = std::chrono::duration<double>(now - state.sample_time).count();
    if (current >= state.sample_bytes && seconds > 0.0) {
        const double instantaneous = static_cast<double>(current - state.sample_bytes) / seconds;
        if (instantaneous > 0.0) {
            constexpr double alpha = 0.25;
            state.smoothed_bytes_per_second =
                state.smoothed_bytes_per_second > 0.0
                    ? alpha * instantaneous + (1.0 - alpha) * state.smoothed_bytes_per_second
                    : instantaneous;
        }
    }
    state.sample_time  = now;
    state.sample_bytes = current;
}

std::string begin_line(const PhasePresentation& phase, const StartupEvent& event) {
    std::string line = phase.active;
    if (event.progress_unit == StartupProgressUnit::Bytes && event.total != 0) {
        line += " | " + format_pretty_bytes(event.total);
    }
    return line;
}

std::string progress_line_candidate(const PhasePresentation& phase, const StartupEvent& event,
                                    const PhaseProgress& state, double ratio, std::size_t bar_width,
                                    bool include_bytes, bool include_rate, bool include_eta) {
    std::ostringstream out;
    out << "  " << phase.active << ' ';
    if (bar_width != 0) { out << '[' << progress_bar(ratio, bar_width) << "] "; }
    out << format_pretty_percent(ratio);
    if (include_bytes) {
        out << " | " << format_pretty_bytes(event.current) << '/'
            << format_pretty_bytes(event.total);
    }
    if (include_rate && state.smoothed_bytes_per_second > 0.0) {
        out << " | "
            << format_pretty_bytes(static_cast<std::uint64_t>(state.smoothed_bytes_per_second))
            << "/s";
        if (include_eta && event.current < event.total) {
            const double eta =
                static_cast<double>(event.total - event.current) / state.smoothed_bytes_per_second;
            out << " | ETA " << format_pretty_duration(eta);
        }
    }
    return out.str();
}

std::string interactive_progress_line(const PhasePresentation& phase, const StartupEvent& event,
                                      const PhaseProgress& state) {
    const double ratio =
        event.total == 0
            ? 0.0
            : std::clamp(static_cast<double>(event.current) / static_cast<double>(event.total), 0.0,
                         1.0);
    const std::size_t columns   = terminal_columns();
    const std::array candidates = {
        progress_line_candidate(phase, event, state, ratio, 20, true, true, true),
        progress_line_candidate(phase, event, state, ratio, 16, true, true, false),
        progress_line_candidate(phase, event, state, ratio, 12, true, false, false),
        progress_line_candidate(phase, event, state, ratio, 0, true, false, false),
    };
    for (const std::string& candidate : candidates) {
        if (candidate.size() < columns) { return candidate; }
    }
    std::string compact = "  " + std::string(phase.active) + " " + format_pretty_percent(ratio);
    if (columns > 1 && compact.size() >= columns) { compact.resize(columns - 1); }
    return compact;
}

std::string complete_line(const PhasePresentation& phase, const StartupEvent& event) {
    std::ostringstream out;
    out << phase.complete;
    if (event.progress_unit == StartupProgressUnit::Bytes) {
        const std::uint64_t bytes = event.current != 0 ? event.current : event.total;
        out << " | " << format_pretty_bytes(bytes);
    }
    const double seconds = static_cast<double>(event.elapsed_ns) * 1.0e-9;
    out << " | " << format_pretty_duration(seconds);
    if (phase.show_rate && seconds > 0.0 && event.current != 0) {
        out << " | "
            << format_pretty_bytes(
                   static_cast<std::uint64_t>(static_cast<double>(event.current) / seconds))
            << "/s";
    }
    return out.str();
}

} // namespace

struct StartupLogRenderer::Impl {
    Impl(std::shared_ptr<spdlog::logger> logger_in, std::shared_ptr<TerminalProgress> progress_in)
        : logger(std::move(logger_in)), progress(std::move(progress_in)) {}

    ~Impl() { progress->clear(); }

    void render(const StartupEvent& event) {
        const PhasePresentation presentation = phase_presentation(event.phase);
        const std::size_t index              = static_cast<std::size_t>(event.phase);
        switch (event.status) {
        case StartupStatus::Begin: {
            const Clock::time_point now = Clock::now();
            phases[index]               = PhaseProgress{
                              .last_persistent  = now,
                              .last_interactive = now - kInteractiveRefresh,
                              .sample_time      = now,
            };
            if (event.phase == StartupPhase::EngineStartup) {
                failure_logged = false;
                logger->info("starting engine");
            } else if (presentation.visibility == PhaseVisibility::Debug) {
                logger->debug("{}", begin_line(presentation, event));
            } else if (progress->enabled()) {
                progress->update("  " + begin_line(presentation, event));
            } else if (presentation.potentially_long) {
                logger->info("{}", begin_line(presentation, event));
            } else {
                logger->debug("{}", begin_line(presentation, event));
            }
            return;
        }
        case StartupStatus::Progress: {
            const Clock::time_point now = Clock::now();
            PhaseProgress& state        = phases[index];
            update_rate(state, event.current, now);
            if (progress->enabled()) {
                if (now - state.last_interactive < kInteractiveRefresh) { return; }
                state.last_interactive = now;
                if (event.progress_unit == StartupProgressUnit::Bytes && event.total != 0) {
                    progress->update(interactive_progress_line(presentation, event, state));
                }
                return;
            }
            if (now - state.last_persistent < kPersistentRefresh) { return; }
            state.last_persistent = now;
            if (event.progress_unit == StartupProgressUnit::Bytes && event.total != 0) {
                logger->info("{}", progress_line_candidate(presentation, event, state,
                                                           static_cast<double>(event.current) /
                                                               static_cast<double>(event.total),
                                                           0, true, true, true));
            }
            return;
        }
        case StartupStatus::Complete: {
            progress->clear();
            if (event.phase == StartupPhase::EngineStartup) {
                engine_elapsed_ns = event.elapsed_ns;
                return;
            }
            const std::string line = complete_line(presentation, event);
            if (presentation.visibility == PhaseVisibility::Info) {
                logger->info("{}", line);
            } else {
                logger->debug("{}", line);
            }
            return;
        }
        case StartupStatus::Failed:
            progress->clear();
            if (failure_logged) { return; }
            failure_logged = true;
            // An attempt, not the outcome: ninferno-serve builds the Engine again with less when
            // one is refused for memory (a fit fallback, shallower drafts, plain decode), and an
            // RTX 5090 with 19 GB taken started gemma-4-31B after an "ERROR startup failed" line.
            // A start that fails for good ends with the application's own error and its reason.
            logger->info("engine start refused | {} | {}", presentation.active,
                         format_pretty_duration(static_cast<double>(event.elapsed_ns) * 1.0e-9));
            return;
        }
    }

    std::shared_ptr<spdlog::logger> logger;
    std::shared_ptr<TerminalProgress> progress;
    std::array<PhaseProgress, kStartupPhaseCount> phases;
    std::uint64_t engine_elapsed_ns = 0;
    bool failure_logged             = false;
};

StartupLogRenderer::StartupLogRenderer(LoggingRuntime& logging)
    : impl_(std::make_shared<Impl>(logging.logger(), logging.terminal_progress())) {}

StartupLogRenderer::~StartupLogRenderer() = default;

StartupObserver StartupLogRenderer::observer() {
    const std::shared_ptr<Impl> state = impl_;
    return StartupObserver{
        .callback = [state](const StartupEvent& event) { state->render(event); }};
}

void StartupLogRenderer::engine_ready(const LoadSummary& load) {
    impl_->progress->clear();
    const double total_seconds = impl_->engine_elapsed_ns != 0
                                     ? static_cast<double>(impl_->engine_elapsed_ns) * 1.0e-9
                                     : load.load_seconds;
    // The decode pacing beside the CUDA synchronization schedule, on the same line (a compact
    // start): the steps per round, and the CPU quota that chose them when there is one.
    std::string quota;
    if (load.cpu_quota_us != 0 && load.cpu_quota_period_us != 0) {
        quota = fmt::format(" | CPU quota {:.2f} CPUs",
                            static_cast<double>(load.cpu_quota_us) /
                                static_cast<double>(load.cpu_quota_period_us));
    }
    impl_->logger->info(
        "engine ready | {} | total {} | weights {} | CUDA sync {} | decode {} step(s) per round{}",
        format_pretty_text(load.model_name), format_pretty_duration(total_seconds),
        format_pretty_bytes(load.host_to_device_bytes), load.cuda_sync_mode,
        load.decode_steps_per_round, quota);
    // Which per-GPU-class tuning the routes run with: a refused or missing table is not silent.
    if (!load.gpu_class.empty() || !load.route_table.empty()) {
        impl_->logger->info("routes | gpu class {} | {}", format_pretty_text(load.gpu_class),
                            format_pretty_text(load.route_table));
    }
    for (const auto& line : load.moe_tune) {
        impl_->logger->info("routes | {}", format_pretty_text(line));
    }
    for (const auto& line : load.device_facts) {
        impl_->logger->warn("device | {}", format_pretty_text(line));
    }
    if (!load.drafter.empty()) { impl_->logger->info("drafter | {}", load.drafter); }
    // The decision fingerprint (docs/maintainer/deterministic-decisions.md): two runs that print
    // the same one were asked the same thing on the same kind of machine.
    if (!load.decision_fingerprint.empty()) {
        // The board beside it, never in it: two boards of one chip type share a fingerprint.
        impl_->logger->info(
            "fingerprint | {} | manifest {} | ninferno-fingerprint/1 | device {} {}",
            load.decision_fingerprint, load.decision_manifest,
            load.metadata.device_pci_bus_id.empty() ? std::string("unknown")
                                                    : load.metadata.device_pci_bus_id,
            load.metadata.device_uuid);
        std::string taint;
        for (const auto& item : load.decision_taint) {
            taint += (taint.empty() ? "" : ", ") + item;
        }
        impl_->logger->debug("fingerprint taint | {}", taint.empty() ? "none" : taint);
        // The observed sizes (manifest only) after the hashed inputs.
        const std::string all      = load.decision_canonical + load.decision_observed;
        std::string_view canonical = all;
        while (!canonical.empty()) {
            const std::size_t end = canonical.find('\n');
            impl_->logger->debug("fingerprint input | {}", canonical.substr(0, end));
            if (end == std::string_view::npos) { break; }
            canonical.remove_prefix(end + 1);
        }
    }
    impl_->logger->debug(
        "load detail | architecture {} | artifact read {} | H2D {} | staging peak {} | device "
        "objects {} | host objects {}",
        format_pretty_text(load.architecture), format_pretty_bytes(load.artifact_bytes_read),
        format_pretty_bytes(load.host_to_device_bytes),
        format_pretty_bytes(load.peak_staging_bytes), format_pretty_count(load.device_object_count),
        format_pretty_count(load.host_object_count));
    impl_->logger->debug("context cost | transfer {} | prefill {} | machine {} | signature {}",
                         context_cost_preset_source_name(load.context_cost.transfer_source),
                         context_cost_preset_source_name(load.context_cost.prefill_source),
                         load.context_cost.machine, load.prefill_signature);
}

} // namespace ninferno::product
