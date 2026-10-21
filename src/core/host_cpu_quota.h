#pragma once

// The process's CPU bandwidth quota: a discrete machine fact read once at startup. A container's
// CFS bandwidth limit stops every thread of its cgroup for the rest of a quota period once the
// quota is used, so a decode that waits for each step leaves the GPU idle while the host is
// stopped (round_pacing.h). The rules read the parsed fact, never the files.

#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace ninferno {

struct HostCpuQuota {
    std::uint64_t quota_us  = 0; // 0: no quota
    std::uint64_t period_us = 0;

    [[nodiscard]] bool set() const noexcept { return quota_us != 0 && period_us != 0; }
};

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

// The quota of the process's own cgroup (v2: its path in /proc/self/cgroup, then the namespace
// root a container sees; v1: the cpu controller's root). Windows and other hosts: no quota.
[[nodiscard]] inline HostCpuQuota read_host_cpu_quota() {
#if defined(__linux__)
    std::string own;
    if (const auto groups = detail::read_small_file("/proc/self/cgroup")) {
        std::istringstream lines(*groups);
        for (std::string line; std::getline(lines, line);) {
            if (line.rfind("0::", 0) == 0) { own = line.substr(3); }
        }
    }
    for (const std::string& directory : {"/sys/fs/cgroup" + own, std::string("/sys/fs/cgroup")}) {
        if (const auto text = detail::read_small_file(directory + "/cpu.max")) {
            return parse_cgroup_v2_cpu_max(*text);
        }
    }
    for (const char* directory : {"/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"}) {
        const auto quota  = detail::read_small_file(std::string(directory) + "/cpu.cfs_quota_us");
        const auto period = detail::read_small_file(std::string(directory) + "/cpu.cfs_period_us");
        if (quota && period) { return parse_cgroup_v1_cfs(*quota, *period); }
    }
#endif
    return {};
}

} // namespace ninferno
