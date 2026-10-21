// The process-wide options lease (runtime/engine/process_options_lease.h): the first Engine of the
// process applies its maintainer options and --cuda-sync; one built while others are alive must ask
// for the same values or is refused; once the last one is gone the next applies its own afresh.

#include "runtime/engine/process_options_lease.h"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

bool refused(const ninferno::EngineOptions& options) {
    try {
        ninferno::runtime::ProcessOptionsLease lease(options);
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

} // namespace

int main() {
    using ninferno::runtime::ProcessOptionsLease;
    ninferno::EngineOptions release;
    ninferno::EngineOptions lab;
    lab.maintainer.route_dir  = "routes";
    lab.maintainer.dense_tune = ninferno::TuneMode::On;
    ninferno::EngineOptions blocking;
    blocking.pacing.cuda_sync = ninferno::CudaSyncSchedule::Blocking;
    ninferno::EngineOptions paced;
    paced.pacing.steps = 8; // not process-wide

    {
        std::optional<ProcessOptionsLease> first(std::in_place, release);
        check(first->first(), "the process's first engine does not apply its values");
        first->applied();
        check(refused(lab), "a live engine's maintainer options were overridden");
        check(refused(blocking), "a live engine's --cuda-sync was overridden");
        {
            ProcessOptionsLease second(paced);
            check(!second.first(), "a second engine re-applied the process-wide values");
            second.applied();
        }
        first.reset();
        // The last engine is gone: the next one applies its own values.
        ProcessOptionsLease fresh(lab);
        check(fresh.first(), "the values of a dead engine outlived it");
        fresh.applied();
        check(refused(release), "the new values were not recorded");
        ProcessOptionsLease same(lab);
        check(!same.first(), "an engine with the live values was not admitted as a follower");
        same.applied();
    }
    {
        ProcessOptionsLease after(blocking);
        check(after.first(), "the lease was not released when its engines died");
        after.applied();
    }
    std::cout << (failures ? "FAIL" : "OK") << " process_options_lease\n";
    return failures ? 1 : 0;
}
