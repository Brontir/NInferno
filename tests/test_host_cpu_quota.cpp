// The CPU bandwidth quota fact (host_cpu_quota.h) parses the cgroup v2 and v1 files exactly: a
// quota and a period, or none for "max", "-1" and anything unreadable.

#include "core/host_cpu_quota.h"

#include <iostream>
#include <map>
#include <optional>
#include <string>

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
    // The tightest quota along the cgroup and its parents binds (v2): a parent's 1 CPU beats the
    // own 3; a container's namespace root is the last ancestor.
    using Files = std::map<std::string, std::string>;
    const auto reader = [](const Files& files) {
        return [&files](const std::string& path) -> std::optional<std::string> {
            const auto it = files.find(path);
            return it == files.end() ? std::nullopt : std::optional<std::string>(it->second);
        };
    };
    const Files v2_files{{"/sys/fs/cgroup/kubepods/pod/c/cpu.max", "300000 100000\n"},
                         {"/sys/fs/cgroup/kubepods/pod/cpu.max", "100000 100000\n"},
                         {"/sys/fs/cgroup/kubepods/cpu.max", "max 100000\n"}};
    const auto walked = ninferno::read_cgroup_cpu_quota("0::/kubepods/pod/c\n", reader(v2_files));
    check(walked.quota_us == 100'000 && walked.period_us == 100'000,
          "v2: a parent's tighter quota was not taken");
    const Files namespaced{{"/sys/fs/cgroup/cpu.max", "200000 100000\n"}};
    check(ninferno::read_cgroup_cpu_quota("0::/\n", reader(namespaced)).quota_us == 200'000,
          "v2: a namespaced container's own cgroup at the mount root was not read");
    check(ninferno::read_cgroup_cpu_quota("0::/docker/abc\n", reader(namespaced)).quota_us ==
              200'000,
          "v2: a host path absent from the container's mount did not fall back to its root");
    check(!ninferno::read_cgroup_cpu_quota("0::/a\n", reader(Files{})).set(),
          "v2: no cpu.max read as a quota");
    // v1: the cpu controller's line names the path; a parent with a different period still
    // compares by quota / period (150 % against 50 %).
    const Files v1_files{{"/sys/fs/cgroup/cpu,cpuacct/docker/x/cpu.cfs_quota_us", "150000\n"},
                         {"/sys/fs/cgroup/cpu,cpuacct/docker/x/cpu.cfs_period_us", "100000\n"},
                         {"/sys/fs/cgroup/cpu,cpuacct/docker/cpu.cfs_quota_us", "25000\n"},
                         {"/sys/fs/cgroup/cpu,cpuacct/docker/cpu.cfs_period_us", "50000\n"},
                         {"/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us", "-1\n"},
                         {"/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us", "100000\n"}};
    const auto v1_walked = ninferno::read_cgroup_cpu_quota(
        "12:memory:/docker/x\n4:cpu,cpuacct:/docker/x\n", reader(v1_files));
    check(v1_walked.quota_us == 25'000 && v1_walked.period_us == 50'000,
          "v1: the parent's tighter quota was not taken");

    // Run-ahead follows the quota's ratio to min(4, logical cores) CPUs.
    const ninferno::HostCpuQuota three{.quota_us = 300'000, .period_us = 100'000};
    const ninferno::HostCpuQuota four{.quota_us = 400'000, .period_us = 100'000};
    check(three.limits_run_ahead(64) && !four.limits_run_ahead(64),
          "a 3-CPU quota must run ahead and a 4-CPU one must not");
    check(!three.limits_run_ahead(2) && !three.limits_run_ahead(3) && three.limits_run_ahead(4),
          "a quota at or above the logical cores is no limit");
    check(three.limits_run_ahead(0) && !ninferno::HostCpuQuota{}.limits_run_ahead(64),
          "unknown cores read as many; no quota never runs ahead");

    // The reader runs on any host and returns a well-formed fact.
    const auto host = ninferno::read_host_cpu_quota();
    check(host.set() == (host.quota_us != 0 && host.period_us != 0), "host fact malformed");
    std::cout << (failures ? "FAIL" : "OK") << " host_cpu_quota\n";
    return failures ? 1 : 0;
}
