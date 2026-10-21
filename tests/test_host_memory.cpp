// Host memory for pinned buffers: /proc/meminfo parsing, the headroom kept for the rest of the
// host, the memory cgroup that limits the process, and the refusal (with the numbers) of a pinned
// total the host cannot spare. No GPU.

#include "core/host_memory.h"
#include "core/units.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "support/posix.h"

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

using ninferno::kGiB;

void write(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << text;
}

void write_v2(const std::filesystem::path& dir, const std::string& max, const std::string& high,
              std::uint64_t current, std::uint64_t inactive_file, std::uint64_t active_file) {
    write(dir / "memory.max", max + "\n");
    write(dir / "memory.high", high + "\n");
    write(dir / "memory.current", std::to_string(current) + "\n");
    write(dir / "memory.stat", "anon 1\nfile 2\nactive_file " + std::to_string(active_file) +
                                   "\ninactive_file " + std::to_string(inactive_file) +
                                   "\nshmem 3\n");
}

// A container's /proc/meminfo is the host's; the memory cgroup is what the kernel enforces.
void cgroup_checks() {
    using namespace ninferno;
    namespace fs = std::filesystem;
    // A container as on the 5080 stand: /proc/meminfo 30 GiB with 20 available, a cgroup of 24 GiB with 20
    // charged, 6 of them page cache: 14 unreclaimable, 10 left.
    const auto meminfo = parse_meminfo("MemTotal: 31457280 kB\nMemAvailable: 20971520 kB\n");
    check(meminfo.has_value(), "meminfo for the cgroup checks");
    if (!meminfo) { return; }
    const CgroupMemory cgroup{.path              = "/sys/fs/cgroup",
                              .limit_bytes       = 24 * kGiB,
                              .usage_bytes       = 20 * kGiB,
                              .reclaimable_bytes = 6 * kGiB};
    check(cgroup.available_bytes() == 10 * kGiB, "cgroup available = limit - unreclaimable");
    const HostMemory held = within_cgroup(*meminfo, cgroup);
    check(held.total_bytes == 24 * kGiB && held.available_bytes == 10 * kGiB && held.cgroup,
          "the cgroup lowers total and available");
    // Headroom max(4, 24/8) = 4 GiB; cap 3/5 of 24 GiB; limit min(10 - 4, cap) = 6 GiB.
    check(pinned_host_limit(held) == 6 * kGiB, "pinned limit within the cgroup");
    std::string message;
    try {
        require_pinned_host_fits(8 * kGiB, held, "the experts", "");
    } catch (const std::runtime_error& error) { message = error.what(); }
    check(message.find("at most 6.00 GiB: available 10.00 GiB of 24.00 GiB (MemAvailable 20.00 "
                       "GiB of MemTotal 30.00 GiB, within the memory cgroup /sys/fs/cgroup: "
                       "limit 24.00 GiB, 20.00 GiB charged, 6.00 GiB of it reclaimable page "
                       "cache)") != std::string::npos,
          "refusal names the cgroup: " + message);
    // A cgroup with room to spare changes neither number beyond its limit.
    const HostMemory roomy = within_cgroup(
        *meminfo, {.path = "x", .limit_bytes = 64 * kGiB, .usage_bytes = kGiB, .reclaimable_bytes = 0});
    check(roomy.total_bytes == 30 * kGiB && roomy.available_bytes == 20 * kGiB,
          "a larger cgroup leaves /proc/meminfo's numbers");
    // Page cache above the charge (accounting skew) is no negative unreclaimable.
    check(CgroupMemory{.limit_bytes = kGiB, .usage_bytes = 1, .reclaimable_bytes = 2}
                  .available_bytes() == kGiB,
          "reclaimable above usage");

    // Files: a v2 mount with a limited parent and a leaf limited only by memory.high.
    const fs::path root =
        fs::temp_directory_path() / ("ninferno_cgroup_test_" + std::to_string(::getpid()));
    fs::remove_all(root);
    const fs::path mount = root / "cgroup";
    write(root / "meminfo", "MemTotal: 31457280 kB\nMemAvailable: 20971520 kB\n");
    write(root / "self_v2", "0::/job/leaf\n");
    write_v2(mount / "job", "25769803776", "max", 20 * kGiB, 4 * kGiB, 2 * kGiB);
    write_v2(mount / "job" / "leaf", "max", std::to_string(16 * kGiB), 12 * kGiB, 3 * kGiB, kGiB);
    const auto levels = read_cgroup_memory(root / "self_v2", mount);
    check(levels.size() == 2 && levels[0].limit_bytes == 16 * kGiB &&
              levels[0].reclaimable_bytes == 4 * kGiB && levels[1].limit_bytes == 24 * kGiB,
          "v2: the leaf (memory.high) and its parent, leaf first");
    const auto v2 = read_host_memory(root / "meminfo", root / "self_v2", mount);
    // Leaf: 16 - (12 - 4) = 8; parent: 24 - (20 - 6) = 10; total min(30, 16, 24) = 16.
    check(v2 && v2->total_bytes == 16 * kGiB && v2->available_bytes == 8 * kGiB && v2->cgroup &&
              fs::path(v2->cgroup->path) == mount / "job" / "leaf",
          "v2: the tightest level wins");
    // A container: /proc/self/cgroup names a path its mount does not show; the mount's root is
    // its own cgroup.
    write(root / "self_ns", "0::/system.slice/docker-1.scope\n");
    write_v2(root / "ns", "31335645184", "max", 27317997568ULL, 11394158592ULL, 7961894912ULL);
    const auto ns = read_cgroup_memory(root / "self_ns", root / "ns");
    check(ns.size() == 1 && ns[0].available_bytes() == 31335645184ULL - 27317997568ULL +
                                                          11394158592ULL + 7961894912ULL,
          "a container's own cgroup at the mount root");
    // No limit anywhere: /proc/meminfo as it is.
    write_v2(root / "free", "max", "max", kGiB, 0, 0);
    write(root / "self_root", "0::/\n");
    check(read_cgroup_memory(root / "self_root", root / "free").empty(), "memory.max max: no limit");
    // v1: the memory hierarchy, and no limit as a page-rounded LONG_MAX.
    write(root / "self_v1", "12:cpu,cpuacct:/a\n9:memory:/docker/abc\n");
    write(root / "v1" / "memory" / "docker" / "abc" / "memory.limit_in_bytes", "8589934592\n");
    write(root / "v1" / "memory" / "docker" / "abc" / "memory.usage_in_bytes", "6442450944\n");
    write(root / "v1" / "memory" / "docker" / "abc" / "memory.stat",
          "cache 1\ntotal_inactive_file 1073741824\ntotal_active_file 1073741824\n");
    write(root / "v1" / "memory" / "docker" / "memory.limit_in_bytes", "9223372036854771712\n");
    const auto v1 = read_cgroup_memory(root / "self_v1", root / "v1");
    check(v1.size() == 1 && v1[0].limit_bytes == 8 * kGiB && v1[0].available_bytes() == 4 * kGiB,
          "v1: the limited level, not the unlimited parent");
    // A limit without its charge is an error, not a silently ignored limit.
    write(root / "broken" / "memory.max", "1073741824\n");
    write(root / "self_broken", "0::/\n");
    bool threw = false;
    try {
        (void)read_cgroup_memory(root / "self_broken", root / "broken");
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()).find("memory.current is missing") != std::string::npos;
    }
    check(threw, "a limit without memory.current throws, naming the file");
    fs::remove_all(root);
    // This host, for the log.
    if (const auto here = read_host_memory()) {
        std::cout << "this host: total " << format_gib(here->total_bytes) << ", available "
                  << format_gib(here->available_bytes) << ", pinned limit "
                  << format_gib(pinned_host_limit(*here))
                  << (here->cgroup ? ", memory cgroup " + here->cgroup->path +
                                         " limit " + format_gib(here->cgroup->limit_bytes)
                                   : std::string(", no memory cgroup limit"))
                  << "\n";
    }
}

} // namespace

int main() {
    using namespace ninferno;
    // A 30 GiB host with 20 GiB available, as /proc/meminfo writes it.
    const char* meminfo = "MemTotal:       31457280 kB\n"
                          "MemFree:         9437184 kB\n"
                          "MemAvailable:   20971520 kB\n"
                          "Buffers:          123456 kB\n";
    const auto memory = parse_meminfo(meminfo);
    check(memory.has_value(), "parse a meminfo text");
    if (memory) {
        check(memory->total_bytes == 30 * kGiB, "MemTotal in bytes");
        check(memory->available_bytes == 20 * kGiB, "MemAvailable in bytes");
        // max(4 GiB, 30/8 GiB) = 4 GiB of headroom: at most 16 GiB may be pinned (the cap, 3/5 of
        // 30 GiB = 18 GiB, is above it).
        check(pinned_host_headroom(*memory) == 4 * kGiB, "headroom floor");
        check(pinned_host_cap(*memory) == 18 * kGiB, "cap is 3/5 of MemTotal");
        check(pinned_host_limit(*memory) == 16 * kGiB, "pinned limit");
        bool threw = false;
        try {
            require_pinned_host_fits(16 * kGiB, *memory, "sixteen", "");
        } catch (const std::exception&) { threw = true; }
        check(!threw, "the limit itself fits");
        // 904/906 on the 5080: 8 Host StateImages of 2.25 GiB + 8 GiB Host KV = 26 GiB.
        std::string message;
        try {
            require_pinned_host_fits(26 * kGiB, *memory, "the Host tier", "Ask for less.");
        } catch (const std::runtime_error& error) { message = error.what(); }
        check(message.find("the Host tier needs 26.00 GiB") != std::string::npos &&
                  message.find("at most 16.00 GiB") != std::string::npos &&
                  message.find("MemAvailable 20.00 GiB of MemTotal 30.00 GiB") !=
                      std::string::npos &&
                  message.find("minus 4.00 GiB") != std::string::npos &&
                  message.find("at most 18.00 GiB pinned in all") != std::string::npos &&
                  message.ends_with("Ask for less."),
              "refusal carries the numbers: " + message);
        const HostMemory idle{.total_bytes = 30 * kGiB, .available_bytes = 29 * kGiB};
        check(pinned_host_limit(idle) == 18 * kGiB, "an idle host still stops at the cap");
    }
    // A large host keeps an eighth: 512 GiB total, 500 available -> 436 GiB may be pinned, under
    // the cap of 3/5 (307.2 GiB).
    const HostMemory big{.total_bytes = 512 * kGiB, .available_bytes = 500 * kGiB};
    check(pinned_host_headroom(big) == 64 * kGiB, "headroom is an eighth of a large host");
    check(pinned_host_limit(big) == 512 * kGiB / 5 * 3, "large host limit is the cap");
    // Less available than the headroom: nothing may be pinned.
    check(pinned_host_limit({.total_bytes = 8 * kGiB, .available_bytes = 3 * kGiB}) == 0,
          "no pinned memory below the headroom");
    // Missing or malformed values are no answer, not zero.
    check(!parse_meminfo("MemTotal: 1024 kB\n"), "MemAvailable missing");
    check(!parse_meminfo("MemTotal: x kB\nMemAvailable: 1 kB\n"), "malformed value");
    check(!parse_meminfo("MemTotal: 1 MB\nMemAvailable: 1 kB\n"), "unknown unit");
    check(!parse_meminfo("MemTotalX: 1 kB\nMemAvailable: 1 kB\n"), "a longer key is not MemTotal");
    check(!read_host_memory("/nonexistent/meminfo"), "no file, no answer");
    cgroup_checks();
    // An automatic Host tier: kept when it fits, else both parts scaled by one factor.
    const auto kept = fit_host_tier(kGiB, 8, 8 * kGiB, 16 * kGiB);
    check(!kept.reduced && kept.state_slots == 8 && kept.kv_bytes == 8 * kGiB,
          "a Host tier within the budget is kept");
    // OLMo 3 7B on the 30 GiB host (907): 8 x 2.25 GiB + 8 GiB against a budget of 20.04 GiB.
    const std::uint64_t image  = 9 * kGiB / 4;
    const std::uint64_t budget = static_cast<std::uint64_t>(22.27 * 0.9 * static_cast<double>(kGiB));
    const auto fitted          = fit_host_tier(image, 8, 8 * kGiB, budget);
    check(fitted.reduced && fitted.state_slots == 6 &&
              fitted.state_slots * image + fitted.kv_bytes <= budget &&
              fitted.kv_bytes % (1ULL << 20) == 0 && fitted.kv_bytes > 6 * kGiB,
          "an oversized Host tier is scaled to the budget: " + std::to_string(fitted.state_slots) +
              " slots, " + std::to_string(fitted.kv_bytes) + " bytes");
    const auto none = fit_host_tier(image, 8, 8 * kGiB, 0);
    check(none.reduced && none.state_slots == 0 && none.kv_bytes == 0,
          "no budget, no Host tier");
    if (failures == 0) { std::cout << "host memory: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
