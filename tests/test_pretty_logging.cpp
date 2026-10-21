#include "product/logging/logging.h"
#include "product/logging/startup_log.h"

#include <spdlog/logger.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

namespace {

class StderrCapture {
public:
    StderrCapture() {
        if (::pipe(pipe_) != 0) { throw std::runtime_error(std::strerror(errno)); }
        saved_ = ::dup(STDERR_FILENO);
        if (saved_ < 0 || ::dup2(pipe_[1], STDERR_FILENO) < 0) {
            throw std::runtime_error(std::strerror(errno));
        }
        ::close(pipe_[1]);
        pipe_[1] = -1;
    }

    ~StderrCapture() {
        if (saved_ >= 0) {
            (void)::dup2(saved_, STDERR_FILENO);
            ::close(saved_);
        }
        if (pipe_[0] >= 0) { ::close(pipe_[0]); }
    }

    std::string finish() {
        std::fflush(stderr);
        if (::dup2(saved_, STDERR_FILENO) < 0) { throw std::runtime_error(std::strerror(errno)); }
        ::close(saved_);
        saved_ = -1;

        std::string output;
        std::array<char, 4096> buffer{};
        for (;;) {
            const ssize_t count = ::read(pipe_[0], buffer.data(), buffer.size());
            if (count == 0) { break; }
            if (count < 0) {
                if (errno == EINTR) { continue; }
                throw std::runtime_error(std::strerror(errno));
            }
            output.append(buffer.data(), static_cast<std::size_t>(count));
        }
        ::close(pipe_[0]);
        pipe_[0] = -1;
        return output;
    }

private:
    int pipe_[2]{-1, -1};
    int saved_ = -1;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::size_t line_count(std::string_view value) {
    return static_cast<std::size_t>(std::count(value.begin(), value.end(), '\n'));
}

} // namespace

int main() {
    int failures = 0;
    std::string service_output;
    {
        StderrCapture capture;
        {
            ninferno::product::LoggingRuntime logging(
                {.logger_name  = "ninferno-serve",
                 .color        = ninferno::product::LogColorMode::Auto,
                 .presentation = ninferno::product::LogPresentation::Service});
            logging.logger()->info("throughput | sample");
            logging.flush();
        }
        service_output = capture.finish();
    }
    failures += check(
        std::regex_match(
            service_output,
            std::regex(
                R"(^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}  INFO  throughput \| sample\n$)")),
        "service pretty prefix mismatch");
    failures += check(service_output.find("ninferno-serve") == std::string::npos,
                      "service pretty output repeated the executable name");
    failures += check(service_output.find("\x1b[") == std::string::npos,
                      "redirected service output contains ANSI escapes");

    std::string startup_output;
    {
        StderrCapture capture;
        {
            ninferno::product::LoggingRuntime logging(
                {.logger_name  = "ninferno-serve",
                 .color        = ninferno::product::LogColorMode::Never,
                 .presentation = ninferno::product::LogPresentation::Service});
            ninferno::product::StartupLogRenderer startup(logging);
            ninferno::StartupObserver observer = startup.observer();
            observer.callback({.phase  = ninferno::StartupPhase::EngineStartup,
                               .status = ninferno::StartupStatus::Begin});
            observer.callback({.phase  = ninferno::StartupPhase::CudaInitialize,
                               .status = ninferno::StartupStatus::Begin});
            observer.callback({.phase      = ninferno::StartupPhase::CudaInitialize,
                               .status     = ninferno::StartupStatus::Complete,
                               .elapsed_ns = 1'000'000'000});
            observer.callback({.phase         = ninferno::StartupPhase::WeightsMaterialize,
                               .status        = ninferno::StartupStatus::Begin,
                               .progress_unit = ninferno::StartupProgressUnit::Bytes,
                               .total         = 16ULL << 30});
            observer.callback({.phase         = ninferno::StartupPhase::WeightsMaterialize,
                               .status        = ninferno::StartupStatus::Complete,
                               .progress_unit = ninferno::StartupProgressUnit::Bytes,
                               .current       = 16ULL << 30,
                               .total         = 16ULL << 30,
                               .elapsed_ns    = 2'000'000'000});
            observer.callback({.phase      = ninferno::StartupPhase::EngineStartup,
                               .status     = ninferno::StartupStatus::Complete,
                               .elapsed_ns = 3'000'000'000});
            startup.engine_ready({.model_name             = "qwen3.6-27b",
                                  .cuda_sync_mode         = "blocking",
                                  .weight_formats         = {"q4_g64_fp16", "q8_g32_fp16"},
                                  .gpu_class              = "gb202:170sm",
                                  .route_table            = "route table t.json refused (x)",
                                  .decode_steps_per_round = 16,
                                  .cpu_quota_us           = 200000,
                                  .cpu_quota_period_us    = 100000,
                                  .host_to_device_bytes   = 16ULL << 30});
            logging.flush();
        }
        startup_output = capture.finish();
    }
    failures += check(
        line_count(startup_output) == 5 &&
            startup_output.find("starting engine") != std::string::npos &&
            startup_output.find("loading weights | 16.0 GiB") != std::string::npos &&
            startup_output.find("weights ready | 16.0 GiB | 2.0s | 8.00 GiB/s") !=
                std::string::npos &&
            startup_output.find("engine ready | qwen3.6-27b | total 3.0s | weights 16.0 GiB") !=
                std::string::npos &&
            // The decode pacing rides on the same line: no line of its own.
            startup_output.find("CUDA sync blocking | decode 16 step(s) per round | CPU quota "
                                "2.00 CPUs") != std::string::npos &&
            // The route table's state is part of startup, refused or not.
            startup_output.find(
                "routes | gpu class gb202:170sm | route table t.json refused (x)") !=
                std::string::npos &&
            startup_output.find("CUDA initialized") == std::string::npos,
        "normal startup pretty output is noisy or incomplete");

    std::string tool_output;
    {
        StderrCapture capture;
        {
            ninferno::product::LoggingRuntime logging(
                {.logger_name  = "ninferno",
                 .color        = ninferno::product::LogColorMode::Never,
                 .presentation = ninferno::product::LogPresentation::Tool});
            logging.logger()->info("engine ready");
            logging.logger()->error("failed");
            logging.flush();
        }
        tool_output = capture.finish();
    }
    failures +=
        check(tool_output == "engine ready\nerror: failed\n", "tool pretty prefix mismatch");
    return failures == 0 ? 0 : 1;
}
