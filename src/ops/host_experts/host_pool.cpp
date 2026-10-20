// HostPool (host_pool.h).
//
// One 64-bit word carries a run: generation (32 bits), task count (16) and the next unclaimed task
// (16). A worker claims with compare-exchange against the generation it woke for, so a worker that
// arrives late never takes a task of a later run, and no other shared field is read before a
// claim succeeds; fn_ and context_ are stable while a claimed task of their run is unfinished.
#include "ops/host_experts/host_pool.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace ninferno::ops::host {

namespace {

constexpr std::uint64_t pack(std::uint32_t generation, std::uint32_t tasks, std::uint32_t next) {
    return (static_cast<std::uint64_t>(generation) << 32) |
           (static_cast<std::uint64_t>(tasks) << 16) | next;
}
constexpr std::uint32_t generation_of(std::uint64_t v) { return static_cast<std::uint32_t>(v >> 32); }
constexpr std::uint32_t tasks_of(std::uint64_t v) { return static_cast<std::uint32_t>((v >> 16) & 0xffffU); }
constexpr std::uint32_t next_of(std::uint64_t v) { return static_cast<std::uint32_t>(v & 0xffffU); }

// How long an idle worker spins before it sleeps: decode calls come about a layer apart.
constexpr auto kSpin = std::chrono::milliseconds(2);

int read_int(const std::string& path) {
    std::ifstream in(path);
    int value = -1;
    in >> value;
    return in ? value : -1;
}

} // namespace

std::vector<int> physical_core_cpus() {
    std::vector<int> cpus;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        throw std::runtime_error("host experts: sched_getaffinity failed");
    }
    std::map<std::pair<int, int>, int> first; // (package, core) -> lowest CPU
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &set)) { continue; }
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        const int package      = read_int(base + "physical_package_id");
        const int core         = read_int(base + "core_id");
        // Without topology every CPU counts as its own core.
        const auto key = package < 0 || core < 0 ? std::make_pair(-1, cpu) : std::make_pair(package, core);
        first.try_emplace(key, cpu);
    }
    for (const auto& [key, cpu] : first) { cpus.push_back(cpu); }
    std::sort(cpus.begin(), cpus.end());
#else
    const unsigned n = std::max(1U, std::thread::hardware_concurrency());
    for (unsigned i = 0; i < n; ++i) { cpus.push_back(static_cast<int>(i)); }
#endif
    if (cpus.empty()) { throw std::runtime_error("host experts: no CPU in the process affinity"); }
    return cpus;
}

HostPool::HostPool(int threads) {
    const std::vector<int> cores = physical_core_cpus();
    if (threads < 0 || threads > static_cast<int>(cores.size())) {
        throw std::invalid_argument("host experts: " + std::to_string(threads) +
                                    " threads asked, the process may use " +
                                    std::to_string(cores.size()) + " physical cores");
    }
    const int n = threads == 0 ? static_cast<int>(cores.size()) : threads;
    cpus_.assign(cores.begin(), cores.begin() + n);
    // The caller is the n-th worker: it runs tasks too, so n - 1 threads are started.
    workers_.reserve(static_cast<std::size_t>(n - 1));
    for (int i = 1; i < n; ++i) { workers_.emplace_back([this, cpu = cpus_[static_cast<std::size_t>(i)]] { worker(cpu); }); }
}

HostPool::~HostPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_.store(true);
    }
    wake_.notify_all();
    for (auto& t : workers_) { t.join(); }
}

void HostPool::drain(std::uint32_t generation) {
    std::uint64_t v = state_.load(std::memory_order_acquire);
    while (true) {
        if (generation_of(v) != generation || next_of(v) >= tasks_of(v)) { return; }
        if (!state_.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel,
                                          std::memory_order_acquire)) {
            continue;
        }
        fn_(context_, static_cast<int>(next_of(v)));
        finished_.fetch_add(1, std::memory_order_acq_rel);
        v = state_.load(std::memory_order_acquire);
    }
}

void HostPool::worker(int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    // A worker that cannot be pinned still computes the same bytes; it only loses its core.
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)cpu;
#endif
    std::uint32_t seen = generation_of(state_.load(std::memory_order_acquire));
    while (true) {
        auto idle_since = std::chrono::steady_clock::now();
        std::uint32_t now_gen = seen;
        for (int spin = 0;; ++spin) {
            if (stop_.load(std::memory_order_acquire)) { return; }
            now_gen = generation_of(state_.load(std::memory_order_acquire));
            if (now_gen != seen) { break; }
            if ((spin & 255) == 255 && std::chrono::steady_clock::now() - idle_since > kSpin) {
                std::unique_lock<std::mutex> lock(mutex_);
                sleepers_.fetch_add(1);
                wake_.wait(lock, [&] {
                    return stop_.load() || generation_of(state_.load()) != seen;
                });
                sleepers_.fetch_sub(1);
                if (stop_.load()) { return; }
                now_gen     = generation_of(state_.load(std::memory_order_acquire));
                idle_since  = std::chrono::steady_clock::now();
                break;
            }
            relax();
        }
        seen = now_gen;
        drain(seen);
    }
}

void HostPool::run(int tasks, Task fn, void* context) {
    if (tasks < 0 || tasks > 0xffff) {
        throw std::invalid_argument("host experts: a run takes 0 to 65535 tasks");
    }
    if (tasks == 0) { return; }
    if (busy_.exchange(true)) {
        throw std::logic_error("host experts: HostPool::run entered while a run is in flight");
    }
    std::uint32_t generation = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        fn_        = fn;
        context_   = context;
        finished_.store(0, std::memory_order_relaxed);
        generation = generation_of(state_.load(std::memory_order_relaxed)) + 1;
        state_.store(pack(generation, static_cast<std::uint32_t>(tasks), 0), std::memory_order_release);
    }
    if (sleepers_.load() > 0) { wake_.notify_all(); }
    drain(generation);
    while (finished_.load(std::memory_order_acquire) < tasks) { relax(); }
    busy_.store(false);
}

} // namespace ninferno::ops::host
