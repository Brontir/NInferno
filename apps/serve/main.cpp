#include "core/removed_environment.h"
#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "product/update/update_check.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/model_source.h"
#include "serve/plan_offline.h"
#include "serve/plan_report.h"
#include "serve/serve_options.h"
#include "spec/registry.h"

#include <cuda_runtime_api.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <timeapi.h>
#endif
#include <spdlog/logger.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>
#include <utility>

namespace {

std::atomic<ninferno::serve::HttpServer*> g_server{nullptr};
std::atomic<spdlog::logger*> g_logger{nullptr};

// The first SIGTERM/SIGINT drains (in-flight requests finish, /health and new requests get 503);
// a second one stops at once. request_drain() only stores atomics.
void handle_signal(int) {
    ninferno::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->request_drain(); }
}

#ifdef _WIN32
// Ctrl+Break drains as SIGTERM does (the C runtime turns only Ctrl+C into SIGINT). Closing the
// console window, logoff and shutdown end the process once the handler returns, and Windows
// waits about five seconds for it: those stop at once (as a second signal does) and the handler
// waits for the server to return from listen().
BOOL WINAPI handle_console_event(DWORD event) {
    if (event == CTRL_C_EVENT) { return FALSE; }
    handle_signal(SIGTERM);
    if (event != CTRL_BREAK_EVENT) {
        handle_signal(SIGTERM);
        for (int waited = 0; g_server.load() != nullptr && waited < 4500; waited += 50) {
            Sleep(50);
        }
    }
    return TRUE;
}
#endif

// An exception that escapes a request boundary ends the process through std::terminate, and the
// default handler writes which one only to stderr. In a container the server is pid 1: the kernel
// drops the SIGABRT abort() raises against it, glibc falls through to its abort instruction, and
// the kernel reports a bare protection fault in libc. So the exception is named in the server's
// own log first.
[[noreturn]] void log_terminate() {
    std::string detail = "terminate called with no active exception";
    if (std::current_exception() != nullptr) {
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception& error) {
            detail = std::string("terminate called after throwing ") + typeid(error).name() + ": " +
                     error.what();
        } catch (...) { detail = "terminate called after throwing a non-std exception"; }
    }
    if (spdlog::logger* logger = g_logger.load(); logger != nullptr) {
        logger->critical("{}", detail);
        logger->flush();
    }
    std::cerr << "ninferno-serve: " << detail << std::endl;
    std::abort();
}

// Names the server's log to log_terminate for as long as the logger lives.
class TerminateLog {
public:
    explicit TerminateLog(spdlog::logger& logger) noexcept { g_logger.store(&logger); }

    ~TerminateLog() { g_logger.store(nullptr); }

    TerminateLog(const TerminateLog&)            = delete;
    TerminateLog& operator=(const TerminateLog&) = delete;
};

// --version: what this executable was built for, and whether the machine's driver and GPUs run
// it. The installer calls it to check a download before saying it works.
int print_version() {
    const auto cuda = [](int v) { return std::to_string(v / 1000) + '.' + std::to_string(v % 1000 / 10); };
    int runtime = 0;
    int driver  = 0;
    (void)cudaRuntimeGetVersion(&runtime);
    (void)cudaDriverGetVersion(&driver);
    std::cout << "NInferno " << NINFERNO_VERSION_STRING << " (built for sm_" << NINFERNO_BUILD_ARCH
              << ", CUDA " << cuda(runtime) << ")\n";
    if (driver == 0) {
        std::cout << "no NVIDIA driver found\n";
        return 1;
    }
    std::cout << "driver: CUDA " << cuda(driver) << '\n';
    int count                = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        std::cout << "CUDA cannot start: " << cudaGetErrorString(status) << '\n';
        return 1;
    }
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, device) != cudaSuccess) { continue; }
        // SASS for sm_XY runs on the same major version from minor Y up; an architecture-specific
        // build (120a) only on its own.
        const int sm = props.major * 10 + props.minor;
#ifdef NINFERNO_BUILD_ARCH_SPECIFIC
        const bool runs = sm == NINFERNO_BUILD_ARCH;
#else
        const bool runs = sm / 10 == NINFERNO_BUILD_ARCH / 10 && sm >= NINFERNO_BUILD_ARCH;
#endif
        std::cout << "GPU " << device << ": " << props.name << ", sm_" << sm << ", "
                  << (props.totalGlobalMem >> 20) << " MiB"
                  << (runs ? "" : " (this build cannot run on it)") << '\n';
    }
    return count > 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    std::set_terminate(log_terminate);
#ifdef _WIN32
    // Waits of a millisecond (the scheduler's polls) last the 15.6 ms system tick otherwise.
    timeBeginPeriod(1);
#endif
    ninferno::serve::ServeOptions options;
    try {
        options = ninferno::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninferno-serve: " << exception.what() << '\n';
        std::cerr << ninferno::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninferno-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninferno::serve::serve_usage_text(argv[0]);
        return 0;
    }
    if (options.version_requested) { return print_version(); }
    // A removed NINFERNO_* decision variable changes nothing: say so once.
    ninferno::warn_removed_environment("ninferno-serve");
    if (!options.drafter_registry.empty()) {
        ninferno::spec::Registry::use_file(options.drafter_registry);
    }

    ninferno::product::LoggingRuntime logging(
        {.logger_name  = "ninferno-serve",
         .level        = options.log_level,
         .presentation = ninferno::product::LogPresentation::Service});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    const TerminateLog terminate_log(*logger);
    ninferno::product::StartupLogRenderer startup_log(logging);
    ninferno::serve::OperationalLog operational_log(logger);
    bool serving = false;

    try {
        // --plan --no-load: the plan from the model's header or config and the device's facts,
        // on stdout; no weights are loaded or downloaded and no GPU is needed.
        if (options.plan_only && options.plan_no_load) {
            const ninferno::serve::OfflinePlan plan = ninferno::serve::plan_without_load(options);
            std::cout << (options.plan_json ? plan.json.dump(1) + '\n' : plan.text);
            std::cout.flush();
            return plan.fits ? 0 : 2;
        }
        // The model named on the command line, as an artifact: fetched and converted first when
        // it is a hub id or a checkpoint (serve/model_source.h).
        const ninferno::serve::ModelSource source = ninferno::serve::resolve_model_source(
            options, [&](const std::string& line) { logger->info("model | {}", line); });
        options.artifact_path = source.artifact_path;
        // The quantization was chosen for each stream's own context: the start keeps it as the
        // ceiling of its automatic context (a busy card still steps down a rung).
        if (source.planned_context != 0 && options.max_context == 0 &&
            options.max_context_cap == 0) {
            options.max_context_cap = source.planned_context;
        }
        if (source.kind == ninferno::serve::ModelSource::Kind::Hub &&
            !options.model_id_override.has_value()) {
            options.model_id_override = source.repo; // served under its hub id
        }
        if (options.prepare_only) {
            logger->info("model | prepared {}", options.artifact_path);
            return 0;
        }
        // The axes the user left to the engine (--kv-dtype auto, no --activations, no
        // --state-dtype): the configuration choice for this artifact on this GPU
        // (product/choice/choice.h through serve/plan_offline.h).
        if (const auto axes = ninferno::serve::choose_served_axes(
                options, [&](const std::string& line) { logger->info("choice | {}", line); })) {
            if (axes->kv) { options.kv_cache = axes->kv; }
            if (axes->activations) { options.activations = *axes->activations; }
            if (axes->state) { options.recurrent_state = axes->state; }
        }
        // --plan: the plan ladder and the rung the free memory picks now, on stdout; nothing is
        // bound or served.
        if (options.plan_only) {
            const ninferno::serve::ServePlan plan =
                ninferno::serve::GenerationService::plan(options);
            std::cout << (options.plan_json
                              ? ninferno::serve::render_plan_json(plan) + '\n'
                              : ninferno::serve::render_plan_text(plan, options.max_concurrency));
            std::cout.flush();
            return plan.plan.rung != 0 || (plan.plain && plan.plain->rung != 0) ? 0 : 2;
        }

        ninferno::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

        ninferno::serve::GenerationService service(options, startup_log.observer());
        startup_log.engine_ready(service.load_summary());
        operational_log.engine_capacity(service);

        using Clock                            = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        operational_log.warmup_started();
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - warmup_started).count();
            operational_log.warmup_failure(seconds, exception.what());
            return 1;
        }
        operational_log.warmup_complete(
            std::chrono::duration<double>(Clock::now() - warmup_started).count());
        server.attach(service);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
#ifdef _WIN32
        SetConsoleCtrlHandler(handle_console_event, TRUE); // handle_console_event's rules
#endif

        serving = true;
        operational_log.server_ready(options.host, options.port, server.public_model_id(),
                                     !options.api_key.empty());
        // A newer release, said once in the log; off the serving path (a dead network costs
        // nothing but this thread's few seconds).
        std::thread([logger] {
            try {
                const auto release = ninferno::update::check({.current = NINFERNO_VERSION_STRING});
                if (!release) { return; }
                if (const auto line = ninferno::update::notice(*release); !line.empty()) {
                    logger->warn("update | {}", line);
                }
            } catch (const std::exception& exception) {
                // Only a notice: a damaged update cache must not end the server.
                logger->debug("update | check failed: {}", exception.what());
            }
        }).detach();

        const bool ok = server.listen();
        g_server.store(nullptr);
        // Teardown: a signal now ends the process, as before serving.
        std::signal(SIGINT, SIG_DFL);
        std::signal(SIGTERM, SIG_DFL);
#ifdef _WIN32
        SetConsoleCtrlHandler(handle_console_event, FALSE);
#endif
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        if (server.engine_failed()) {
            // The Engine ended, usually on a failed device: unwinding would free Device memory
            // through a context with a sticky error, which can block or abort before the exit
            // code reaches the supervisor. The driver reclaims a process that just ends.
            logger->flush();
            std::fflush(nullptr);
            std::_Exit(1);
        }
        return 0;
    } catch (const std::exception& exception) {
        g_server.store(nullptr);
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}
