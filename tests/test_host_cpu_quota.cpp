// The CPU bandwidth quota fact (host_cpu_quota.h) parses the cgroup v2 and v1 files exactly: a
// quota and a period, or none for "max", "-1" and anything unreadable.

#include "core/host_cpu_quota.h"

#include <iostream>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

} // namespace

int main() {
    using ninferno::parse_cgroup_v1_cfs;
    using ninferno::parse_cgroup_v2_cpu_max;
    const auto two = parse_cgroup_v2_cpu_max("200000 100000\n");
    check(two.set() && two.quota_us == 200'000 && two.period_us == 100'000, "v2 quota not parsed");
    check(!parse_cgroup_v2_cpu_max("max 100000\n").set(), "v2 max read as a quota");
    check(!parse_cgroup_v2_cpu_max("").set() && !parse_cgroup_v2_cpu_max("garbage").set() &&
              !parse_cgroup_v2_cpu_max("0 100000").set() &&
              !parse_cgroup_v2_cpu_max("12x 100000").set(),
          "unreadable v2 text read as a quota");
    const auto one = parse_cgroup_v1_cfs("150000\n", "100000\n");
    check(one.set() && one.quota_us == 150'000 && one.period_us == 100'000, "v1 quota not parsed");
    check(!parse_cgroup_v1_cfs("-1\n", "100000\n").set() && !parse_cgroup_v1_cfs("", "").set(),
          "v1 unlimited read as a quota");
    // The reader runs on any host and returns a well-formed fact.
    const auto host = ninferno::read_host_cpu_quota();
    check(host.set() == (host.quota_us != 0 && host.period_us != 0), "host fact malformed");
    std::cout << (failures ? "FAIL" : "OK") << " host_cpu_quota\n";
    return failures ? 1 : 0;
}
