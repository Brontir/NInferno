#pragma once

// The processes ninferno-ui starts: one ninferno-serve per running model (on the GPU it was given) and
// the download/convert jobs (ninferno-serve --prepare-only). Each is a child in its own process
// group, its stdout and stderr kept in a ring of the last lines; stopping sends SIGTERM to the
// group (ninferno-serve drains) and SIGKILL after a grace period. On Windows each child is in a job
// object (its whole tree ends with it, and with ninferno-ui) and its own console process group:
// stopping sends it Ctrl+Break (ninferno-serve drains) and ends the job after the grace period.

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ninferno::ui {

struct LaunchSpec {
    std::string kind;                       // "serve" | "prepare"
    std::string model;                      // what was asked for (hub id, path)
    int gpu  = -1;                          // NVML index; -1: no GPU binding
    int port = 0;                           // serve: the port it listens on
    std::vector<std::string> argv;          // argv[0] is the executable
    std::map<std::string, std::string> env; // added to the UI's environment
    // Lines the UI writes to the process log before its output ("ninferno-ui | ..."), such as
    // why a tune beside the model is not named.
    std::vector<std::string> notes;
};

struct ProcessInfo {
    std::uint64_t id = 0;
    LaunchSpec spec;
    int pid = 0;
    std::string state; // starting | running | ready | exited | failed | stopping
    int exit_code = 0;
    std::chrono::system_clock::time_point started;
    std::chrono::system_clock::time_point ended;
    std::string last_line;
    // What a ninferno-serve start decided, from its log: the context per request ("context | ...")
    // and the speculation it runs ("speculation | ...", the first such line).
    std::string context;
    std::string speculation;
    // The last speculation warning ("WARN  speculation | ..."): a fit fallback or plain decode.
    std::string speculation_fallback;
    // The converted artifact it runs ("cached artifact PATH (" / "converted artifact PATH (").
    std::string artifact;
    bool converted = false; // "converted artifact": made now (its original may go)
    // The download under way: fetch's progress line ("  NAME: 42% (X of Y), R/s, about N min
    // left"), its share done (-1: none seen) and the file.
    int percent = -1;
    std::string progress, progress_file;

    [[nodiscard]] nlohmann::json to_json() const;
};

class ProcessPool {
public:
    explicit ProcessPool(std::size_t log_lines = 2000);
    ~ProcessPool(); // stops every child
    ProcessPool(const ProcessPool&)            = delete;
    ProcessPool& operator=(const ProcessPool&) = delete;

    // Throws std::runtime_error when the process cannot be started.
    std::uint64_t start(LaunchSpec spec);
    // false when no such process or it already ended.
    bool stop(std::uint64_t id, std::chrono::milliseconds grace = std::chrono::seconds(20));
    // Marks a serve process ready (its /health answered).
    void mark_ready(std::uint64_t id);

    [[nodiscard]] std::vector<ProcessInfo> list() const;
    [[nodiscard]] bool find(std::uint64_t id, ProcessInfo& out) const;
    // The last `max` lines (0: all kept) and the total count written, for incremental reads.
    [[nodiscard]] std::vector<std::string> log(std::uint64_t id, std::size_t max = 0) const;
    // Drops ended processes older than `age`.
    void forget_ended(std::chrono::seconds age);

    // A TCP port on 127.0.0.1 nobody listens on and no running child was given, from `first` up.
    [[nodiscard]] int free_port(int first = 8100) const;

private:
    struct Child {
        ProcessInfo info;
        std::deque<std::string> lines;
        std::string partial;
        int out_fd = -1;
#ifdef _WIN32
        void* process = nullptr; // HANDLEs: the child, the job holding its tree, its output
        void* job     = nullptr;
        void* output  = nullptr;
        ~Child(); // closes them (the job's last handle ends what is left of the tree)
#endif
        std::thread reader;
    };

    void read_loop(Child* child);

    mutable std::mutex mutex_;
    std::map<std::uint64_t, std::unique_ptr<Child>> children_;
    std::uint64_t next_id_ = 1;
    std::size_t log_lines_;
};

} // namespace ninferno::ui
