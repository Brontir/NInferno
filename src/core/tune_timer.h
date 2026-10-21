#pragma once

// The load-time tuners' timer (ops/generic/generic_tune.cpp, ops/moe/moe_tune.cpp): a call after a
// cold-L2 flush, the median of its timed repetitions after untimed warm-up ones.
//
// The flush reads a buffer of 2x L2 (at most 256 MiB) zeroed once. A cudaMemsetAsync over it would
// evict the timed call's lines too but leave up to the whole L2 dirty, and the timed call would pay
// the write-back to DRAM (RTX 4090: a 15 MB streaming read 27.6 us after a memset flush, 19.6 us
// after a read flush; the benches' L2FlushBuffer reads too). Where the device has no room for the
// buffer, the times read a warm L2: the ranking mostly holds.

#include "core/arena.h"
#include "ninferno/option_enums.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno {

// A tuner summary's time: microseconds with one decimal.
[[nodiscard]] inline std::string format_us(double us) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f", us);
    return text;
}

class TuneTimer {
public:
    TuneTimer(cudaStream_t stream, int warm, int reps);
    ~TuneTimer();
    TuneTimer(const TuneTimer&)            = delete;
    TuneTimer& operator=(const TuneTimer&) = delete;

    // The median of `reps` timed run(stream) calls past `warm` untimed ones, each after a flush.
    template <class Run>
    float median_us(Run&& run) {
        for (int r = 0; r < warm_ + reps_; ++r) {
            flush();
            record(start_[static_cast<std::size_t>(r)]);
            run(stream_);
            record(stop_[static_cast<std::size_t>(r)]);
        }
        return median();
    }

private:
    void flush();
    void record(cudaEvent_t event);
    float median();

    cudaStream_t stream_;
    int warm_, reps_;
    DeviceBuffer flush_;
    std::vector<cudaEvent_t> start_, stop_;
};

} // namespace ninferno
