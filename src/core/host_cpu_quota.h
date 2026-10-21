#pragma once

// The process's CPU bandwidth quota: a discrete machine fact read once at startup. A container's
// CFS bandwidth limit stops every thread of its cgroup for the rest of a quota period once the
// quota is used, so a decode that waits for each step leaves the GPU idle while the host is
// stopped (round_pacing.h). The rules read the parsed fact, never the files.

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__linux__)
#    include <sched.h>
#endif

namespace ninferno {

struct HostCpuQuota {
    std::uint64_t quota_us  = 0; // 0: no quota
    std::uint64_t period_us = 0;

    // A run-ahead round needs a quota below this many CPUs (and below the logical cores).
    static constexpr std::uint64_t kRunAheadCpus = 4;

    [[nodiscard]] bool set() const noexcept { return quota_us != 0 && period_us != 0; }

    // Whether the quota stops the host often enough that decode rounds should run ahead
    // (round_pacing.h): quota / period < min(kRunAheadCpus, logical_cores) CPUs, from the facts
    // alone. The engine keeps about three host threads busy while it decodes at load (the worker
    // that submits rounds and spins on their end, the serving front end's request and streaming
    // threads, detokenization); a quota of four CPUs or more covers them within each period, and
    // the throttling stall was seen on lab containers with quotas of one to three CPUs on hosts of
    // 32 to 128 cores. A quota at or above the cores the process can run on can never be used up
    // (its threads cannot run on more CPUs than there are), so it is no limit at all; half the
    // cores (another threshold proposed) would leave a 2-CPU quota on a 4-core host, where those
    // three threads do stall, without run-ahead. logical_cores 0 (unknown) reads as many cores.
    // The threshold is a declared rule awaiting calibration on lab hosts, like kQuotaSteps.
    [[nodiscard]] bool limits_run_ahead(unsigned logical_cores) const noexcept {
        if (!set()) { return false; }
        const std::uint64_t cpus = logical_cores == 0
                                       ? kRunAheadCpus
                                       : std::min<std::uint64_t>(kRunAheadCpus, logical_cores);
        // quota / period < cpus, without division.
        return quota_us < cpus * period_us;
    }
};

// The logical CPUs the process may run on: its affinity mask (sched_getaffinity, which a
// container's cpuset and taskset narrow) on Linux, std::thread::hardware_concurrency() elsewhere or
// when the mask cannot be read (0: unknown). A machine fact read once at startup, like the quota.
[[nodiscard]] inline unsigned usable_logical_cores() noexcept {
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        const int count = CPU_COUNT(&mask);
        if (count > 0) { return static_cast<unsigned>(count); }
    }
#endif
    return std::thread::hardware_concurrency();
}

// The tighter of two quotas (the lower quota / period); an unset one is no limit.
[[nodiscard]] inline HostCpuQuota tighter_cpu_quota(const HostCpuQuota& a,
                                                    const HostCpuQuota& b) noexcept {
    if (!a.set()) { return b; }
    if (!b.set()) { return a; }
    // a.quota / a.period <= b.quota / b.period; cgroup periods are at most one second (10^6 us),
    // so the products of a quota below 2^44 us and a period do not overflow.
    const long double left  = static_cast<long double>(a.quota_us) * b.period_us;
    const long double right = static_cast<long double>(b.quota_us) * a.period_us;
    return left <= right ? a : b;
}

namespace detail {

[[nodiscard]] inline std::optional<std::uint64_t> parse_quota_number(std::string_view text) {
    while (!text.empty() && (text.back() == '\n' || text.back() == ' ' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    while (!text.empty() && text.front() == ' ') { text.remove_prefix(1); }
    if (text.empty()) { return std::nullopt; }
    std::uint64_t value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') { return std::nullopt; }
        if (value > (UINT64_MAX - static_cast<std::uint64_t>(c - '0')) / 10U) {
            return std::nullopt;
        }
        value = value * 10U + static_cast<std::uint64_t>(c - '0');
    }
    return value;
}

[[nodiscard]] inline std::optional<std::string> read_small_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) { return std::nullopt; }
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

} // namespace detail

// cgroup v2 cpu.max: "max PERIOD" (no quota) or "QUOTA PERIOD". Unreadable text: no quota.
[[nodiscard]] inline HostCpuQuota parse_cgroup_v2_cpu_max(std::string_view text) {
    const std::size_t space = text.find(' ');
    if (space == std::string_view::npos) { return {}; }
    const auto quota  = detail::parse_quota_number(text.substr(0, space));
    const auto period = detail::parse_quota_number(text.substr(space + 1));
    if (!quota || !period || *quota == 0 || *period == 0) { return {}; }
    return {.quota_us = *quota, .period_us = *period};
}

// cgroup v1 cpu.cfs_quota_us ("-1": no quota) and cpu.cfs_period_us.
[[nodiscard]] inline HostCpuQuota parse_cgroup_v1_cfs(std::string_view quota_text,
                                                      std::string_view period_text) {
    const auto quota  = detail::parse_quota_number(quota_text);
    const auto period = detail::parse_quota_number(period_text);
    if (!quota || !period || *quota == 0 || *period == 0) { return {}; }
    return {.quota_us = *quota, .period_us = *period};
}

namespace detail {

// `path` and each of its parents up to the root ("/a/b" gives "/a/b", "/a" and ""), as suffixes
// of a cgroup mount.
[[nodiscard]] inline std::vector<std::string> cgroup_lineage(std::string path) {
    while (!path.empty() && (path.back() == '/' || path.back() == '\n' || path.back() == '\r')) {
        path.pop_back();
    }
    std::vector<std::string> out;
    for (;;) {
        out.push_back(path);
        if (path.empty()) { break; }
        const std::size_t slash = path.rfind('/');
        path.resize(slash == std::string::npos ? 0 : slash);
    }
    return out;
}

} // namespace detail

// The quota that binds the process, from its /proc/self/cgroup text and a file reader
// (`read(path)` returns the file's text or nothing): every ancestor's limit applies to its
// descendants, so it is the tightest quota along the process's cgroup and its parents. cgroup v2
// ("0::PATH"): cpu.max under /sys/fs/cgroup; a container whose namespace hides the host's path
// sees its own cgroup at the mount's root, which the walk reaches as the last ancestor. cgroup v1
// (the line whose controllers include "cpu"): cpu.cfs_quota_us and cpu.cfs_period_us along the
// same walk under the cpu controller's mount. A hybrid host lists both and mounts the unified
// hierarchy at /sys/fs/cgroup/unified; its cpu controller is almost always on v1, so a v1 cpu line
// whose files exist wins. In the rare hybrid setup that attaches the cpu controller to the unified
// hierarchy instead, the v1 files are absent and the v2 walk reads cpu.max under
// /sys/fs/cgroup/unified as well as /sys/fs/cgroup.
template <class Reader>
[[nodiscard]] HostCpuQuota read_cgroup_cpu_quota(std::string_view proc_self_cgroup, Reader&& read) {
    std::optional<std::string> v2;
    std::optional<std::string> v1;
    std::istringstream lines{std::string(proc_self_cgroup)};
    for (std::string line; std::getline(lines, line);) {
        if (line.rfind("0::", 0) == 0) {
            v2 = line.substr(3);
            continue;
        }
        const std::size_t first  = line.find(':');
        const std::size_t second = first == std::string::npos ? first : line.find(':', first + 1);
        if (second == std::string::npos) { continue; }
        const std::string controllers = "," + line.substr(first + 1, second - first - 1) + ",";
        if (controllers.find(",cpu,") != std::string::npos) { v1 = line.substr(second + 1); }
    }
    HostCpuQuota tightest;
    if (v1) {
        for (const char* mount :
             {"/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct", "/sys/fs/cgroup/cpuacct,cpu"}) {
            bool found = false;
            for (const std::string& suffix : detail::cgroup_lineage(*v1)) {
                const std::string directory = std::string(mount) + suffix;
                const auto quota            = read(directory + "/cpu.cfs_quota_us");
                const auto period           = read(directory + "/cpu.cfs_period_us");
                if (!quota || !period) { continue; }
                found    = true;
                tightest = tighter_cpu_quota(tightest, parse_cgroup_v1_cfs(*quota, *period));
            }
            if (found) { return tightest; }
        }
    }
    if (v2) {
        for (const char* mount : {"/sys/fs/cgroup", "/sys/fs/cgroup/unified"}) {
            for (const std::string& suffix : detail::cgroup_lineage(*v2)) {
                if (const auto text = read(std::string(mount) + suffix + "/cpu.max")) {
                    tightest = tighter_cpu_quota(tightest, parse_cgroup_v2_cpu_max(*text));
                }
            }
        }
    }
    return tightest;
}

// The quota that binds the process (read_cgroup_cpu_quota on this host's files). Windows and
// other hosts: no quota.
[[nodiscard]] inline HostCpuQuota read_host_cpu_quota() {
#if defined(__linux__)
    const auto groups = detail::read_small_file("/proc/self/cgroup");
    return read_cgroup_cpu_quota(
        groups.value_or(""), [](const std::string& path) { return detail::read_small_file(path); });
#else
    return {};
#endif
}

} // namespace ninferno
