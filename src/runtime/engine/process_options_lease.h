#pragma once

#include "ninferno/types.h"

#include <mutex>
#include <stdexcept>

namespace ninferno::runtime {

// The process-wide options: MaintainerOptions (the route tables, the lab tuners, the debug aids)
// and the CUDA synchronization schedule. One record per process holds the values the engines alive
// run under. An engine built while none is alive applies its own values afresh, so nothing an
// earlier engine set survives it; one built while others run must ask for the same values (it
// would change their decisions otherwise) and is refused when it does not. A lease holds the
// record's mutex from its construction until the device is up and the values are applied, so
// concurrent constructions apply one at a time.
class ProcessOptionsLease {
public:
    explicit ProcessOptionsLease(const EngineOptions& options) : lock_(mutex()) {
        Record& record = this->record();
        if (record.live > 0 && (!(record.maintainer == options.maintainer) ||
                                record.cuda_sync != options.pacing.cuda_sync)) {
            throw std::invalid_argument(
                "another Engine of this process runs with other maintainer options or --cuda-sync "
                "(process-wide: route tables, tuners, debug aids, the CUDA synchronization "
                "schedule); every live Engine must use the same");
        }
        first_ = record.live == 0;
        if (first_) {
            record.maintainer = options.maintainer;
            record.cuda_sync  = options.pacing.cuda_sync;
        }
        ++record.live;
    }

    ~ProcessOptionsLease() {
        if (!lock_.owns_lock()) { lock_.lock(); }
        --record().live;
    }

    ProcessOptionsLease(const ProcessOptionsLease&)            = delete;
    ProcessOptionsLease& operator=(const ProcessOptionsLease&) = delete;

    // No other engine was alive: this one applies the process-wide values.
    [[nodiscard]] bool first() const noexcept { return first_; }

    // The values are in place: other constructions may proceed.
    void applied() { lock_.unlock(); }

private:
    struct Record {
        MaintainerOptions maintainer;
        CudaSyncSchedule cuda_sync = CudaSyncSchedule::Spin;
        int live                   = 0;
    };

    static std::mutex& mutex() {
        static std::mutex m;
        return m;
    }

    static Record& record() {
        static Record r;
        return r;
    }

    std::unique_lock<std::mutex> lock_;
    bool first_ = false;
};

} // namespace ninferno::runtime
