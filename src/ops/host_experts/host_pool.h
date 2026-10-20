#pragma once

// The CPU workers of MoE expert offload's decode path: one thread per physical core of the
// process's CPU affinity, each pinned to its core. run() hands out task indices from one counter
// and returns when every task has finished; the caller works too. A worker spins for a while after
// its last task (decode calls come a layer apart), then sleeps until the next run().
//
// Tasks write disjoint outputs, so which thread runs a task never changes a result.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#endif

namespace ninferno::ops::host {

// One step of a spin wait: the CPU's pause hint on x86-64, a yield elsewhere.
inline void relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}


// CPUs of this process's affinity, the first of each physical core (package, core id), in order.
[[nodiscard]] std::vector<int> physical_core_cpus();

class HostPool {
public:
    // threads: 0 = one per physical core (physical_core_cpus); otherwise that many, pinned to the
    // first `threads` of those cores (more than there are is refused).
    explicit HostPool(int threads = 0);
    ~HostPool();
    HostPool(const HostPool&)            = delete;
    HostPool& operator=(const HostPool&) = delete;

    // Workers plus the caller.
    [[nodiscard]] int size() const noexcept { return static_cast<int>(workers_.size()) + 1; }
    [[nodiscard]] const std::vector<int>& cpus() const noexcept { return cpus_; }

    // fn(context, task) for every task in [0, tasks). Not reentrant: one run at a time.
    using Task = void (*)(void* context, int task);
    void run(int tasks, Task fn, void* context);

    template <class F>
    void run(int tasks, F& f) {
        run(tasks, [](void* c, int task) { (*static_cast<F*>(c))(task); }, &f);
    }

private:
    void worker(int cpu);
    void drain(std::uint32_t generation);

    std::vector<int> cpus_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    // generation << 32 | tasks << 16 | next unclaimed task (host_pool.cpp).
    std::atomic<std::uint64_t> state_{0};
    std::atomic<int> finished_{0};
    std::atomic<int> sleepers_{0};
    std::atomic<bool> busy_{false};
    std::atomic<bool> stop_{false};
    Task fn_       = nullptr;
    void* context_ = nullptr;
};

} // namespace ninferno::ops::host
