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
    // MemFree and Cached of /proc/meminfo, for messages only.
    std::uint64_t meminfo_free_bytes   = 0;
    std::uint64_t meminfo_cached_bytes = 0;
    // The pages a pin can take at once (#480): the free pages of the NUMA nodes this process may
    // allocate on (Mems_allowed_list of /proc/self/status) above those zones' high watermarks
    // (/proc/zoneinfo). Page cache is not in it: cudaMallocHost is served from free pages, not from
    // the file cache MemAvailable counts. nullopt where the zones cannot be read (Windows, no
    // /proc/zoneinfo): the pin limit then rests on MemAvailable alone.
    std::optional<std::uint64_t> free_above_watermark_bytes;
    std::optional<CgroupMemory> cgroup;
};

// The free pages and the "high" watermark of every zone of the listed NUMA nodes (all nodes when
// `nodes` is empty) in a /proc/zoneinfo text: sum over zones of max(0, free - high), in bytes
// (`page_size` per page). nullopt when no zone of those nodes is there.
[[nodiscard]] std::optional<std::uint64_t>
zoneinfo_free_above_watermark(std::string_view text, const std::vector<int>& nodes,
                              std::uint64_t page_size = 4096);

// The NUMA nodes of a cpulist ("0-1,3"), as Mems_allowed_list of /proc/self/status prints them;
// nullopt when malformed.
[[nodiscard]] std::optional<std::vector<int>> parse_node_list(std::string_view text);

// The nodes of the "Mems_allowed_list:" line of a /proc/self/status text; nullopt when the line is
// missing or malformed (the caller then counts every node).
[[nodiscard]] std::optional<std::vector<int>> mems_allowed_nodes(std::string_view status_text);

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

// What to tell the user when the driver refused a pin: MemFree, Cached and the free pages above the
// watermarks as they are now, why a full page cache refuses a pin MemAvailable allows (#480), and
// what to do. Empty where the host memory cannot be read.
[[nodiscard]] std::string pin_refusal_hint();

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

// The pinning budget of an automatic Host tier: kAutomaticHostTierShare of pinned_host_limit,
// shared by the host's GPUs (#473). A host of G GPUs commonly runs one server per GPU, started
// together; taking half of what is left each, they pinned nearly the whole limit and left their
// own unpinned memory (weights staging, the CUDA context: ~2-3 GiB each) the headroom alone, and
// a memory cgroup OOM-killed them. A share per GPU leaves every server room; a host whose budget
// covers the default tier anyway (8 GiB of KV and the StateImages) is unchanged. `host_gpus`: the
// host's NVIDIA GPUs (host_gpu_count), at least 1.
[[nodiscard]] std::uint64_t automatic_host_tier_budget(const HostMemory& memory,
                                                       int host_gpus) noexcept;

// Serializes, across the processes of one host, the stretch from reading what the host can pin to
// pinning it (#473): eight servers started together otherwise each read the same MemAvailable and
// pin together, and a memory cgroup OOM-kills or refuses the late ones. One process at a time
// holds it (an exclusive flock on a file in /run/lock, /dev/shm or the temporary directory,
// which the processes of one host or container share), at most kHostPinLockWaitSeconds; what it
// pinned is in MemAvailable when the next one reads it. A no-op where the lock cannot be taken (no
// such directory, a platform without flock) and on Windows: the start then proceeds as before.
inline constexpr int kHostPinLockWaitSeconds = 120;

class HostPinLock {
public:
    HostPinLock();
    ~HostPinLock();
    HostPinLock(const HostPinLock&)            = delete;
    HostPinLock& operator=(const HostPinLock&) = delete;

    [[nodiscard]] bool held() const noexcept { return fd_ >= 0; }

    // Whether the wait ran out (another process held the lock for kHostPinLockWaitSeconds).
    [[nodiscard]] bool timed_out() const noexcept { return timed_out_; }

private:
    int fd_         = -1;
    bool timed_out_ = false;
};

// "18.00 GiB" and the like, for messages.
[[nodiscard]] std::string format_gib(std::uint64_t bytes);

} // namespace ninferno
