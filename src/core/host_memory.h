#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno {

// A memory cgroup's limit on this process, in bytes: the lower of memory.max and memory.high (v2)
// or memory.limit_in_bytes (v1); what it has charged (memory.current / memory.usage_in_bytes); and
// the page cache of that charge the kernel reclaims before it OOM-kills (the file LRU of
// memory.stat). A container's /proc/meminfo is the host's: without this a process in a cgroup
// smaller than the host pins what the kernel will not give it, and the cgroup OOM-kills it.
struct CgroupMemory {
    std::string path; // the cgroup directory the numbers were read from, for messages
    std::uint64_t limit_bytes       = 0;
    std::uint64_t usage_bytes       = 0;
    std::uint64_t reclaimable_bytes = 0;

    // What the cgroup can still give without an OOM kill: limit - (usage - reclaimable), >= 0.
    [[nodiscard]] std::uint64_t available_bytes() const noexcept;
};

// The memory this process may use: MemTotal and MemAvailable of /proc/meminfo, lowered to the
// tightest memory cgroup the process is in (`cgroup`, when one limits it).
struct HostMemory {
    std::uint64_t total_bytes     = 0;
    std::uint64_t available_bytes = 0;
    std::uint64_t meminfo_total_bytes     = 0;
    std::uint64_t meminfo_available_bytes = 0;
    std::optional<CgroupMemory> cgroup;
};

// MemTotal and MemAvailable of a /proc/meminfo text; nullopt when either is missing or malformed.
[[nodiscard]] std::optional<HostMemory> parse_meminfo(std::string_view text);

// `memory` within `cgroup`: total at most the limit, available at most what the cgroup can still
// give. `cgroup` is recorded (for messages) unless an earlier level leaves less available.
[[nodiscard]] HostMemory within_cgroup(HostMemory memory, const CgroupMemory& cgroup);

// One cgroup directory's memory limit: v2 (memory.max, memory.high, memory.current, memory.stat
// inactive_file + active_file) or v1 (memory.limit_in_bytes, memory.usage_in_bytes, memory.stat
// total_inactive_file + total_active_file). nullopt when the directory sets no limit ("max", or no
// such files: the root cgroup, a controller not enabled there). Throws std::runtime_error, naming
// the file, when a file that should be there is missing or malformed.
[[nodiscard]] std::optional<CgroupMemory> read_cgroup_level(const std::filesystem::path& dir);

// Every memory cgroup level that limits this process, from /proc/self/cgroup (v2 "0::<path>", v1
// "<id>:<controllers with memory>:<path>") under the cgroup mount: each level from the process's
// cgroup up to the mount's root, skipping levels the mount does not show (a container sees its
// own cgroup as the root).
[[nodiscard]] std::vector<CgroupMemory>
read_cgroup_memory(const std::filesystem::path& self_cgroup  = "/proc/self/cgroup",
                   const std::filesystem::path& cgroup_mount = "/sys/fs/cgroup");

// The running process's memory: /proc/meminfo within every memory cgroup that limits it; nullopt
// where /proc/meminfo does not exist or does not carry both values. On Windows the default paths
// ask the system instead (physical memory, no cgroups).
[[nodiscard]] std::optional<HostMemory>
read_host_memory(const std::filesystem::path& meminfo      = "/proc/meminfo",
                 const std::filesystem::path& self_cgroup  = "/proc/self/cgroup",
                 const std::filesystem::path& cgroup_mount = "/sys/fs/cgroup");

// Memory left to the rest of the host beside pinned (page-locked) buffers: max(4 GiB, total / 8).
// Pinned pages cannot be reclaimed or swapped, so pinning up to MemAvailable itself leaves the
// kernel, this process's own heap and every other process nothing, and the host stalls or reboots.
[[nodiscard]] std::uint64_t pinned_host_headroom(const HostMemory& memory) noexcept;

// The most this process pins in all, whatever is available: 3/5 of the total (a 32 GB host rebooted
// with more pinned; 18 GiB of its 30).
[[nodiscard]] std::uint64_t pinned_host_cap(const HostMemory& memory) noexcept;

// The most bytes that may be pinned now: available - headroom, within the cap; 0 when the
// available memory is within the headroom.
[[nodiscard]] std::uint64_t pinned_host_limit(const HostMemory& memory) noexcept;

// Throws std::runtime_error, with every number, when `pinned_bytes` exceed pinned_host_limit.
// `what` names the pinned buffers (and how they add up); `remedy` says how to ask for fewer.
void require_pinned_host_fits(std::uint64_t pinned_bytes, const HostMemory& memory,
                              std::string_view what, std::string_view remedy);

// A Host tier of the context cache within a pinning budget: `state_slots` StateImages of
// `image_bytes` each and `kv_bytes` of Host KV, both scaled down by the same factor (StateImages
// rounded down, KV to whole MiB) when together they exceed `budget`; unchanged when they fit.
struct HostTierFit {
    std::uint32_t state_slots = 0;
    std::uint64_t kv_bytes    = 0;
    bool reduced              = false;
};

[[nodiscard]] HostTierFit fit_host_tier(std::uint64_t image_bytes, std::uint32_t state_slots,
                                        std::uint64_t kv_bytes, std::uint64_t budget) noexcept;

// The share of pinned_host_limit an automatic Host tier takes: half, so a server started with
// defaults leaves the host at least half of what it could spare (and absorbs what MemAvailable
// loses between the choice and the pinning). An explicit Host tier may take the whole limit.
inline constexpr double kAutomaticHostTierShare = 0.5;

// "18.00 GiB" and the like, for messages.
[[nodiscard]] std::string format_gib(std::uint64_t bytes);

} // namespace ninferno
