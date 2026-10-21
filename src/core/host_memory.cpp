#include "core/host_memory.h"

#include "core/parse.h"
#include "core/units.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ninferno {
namespace {

constexpr std::uint64_t kMinimumHeadroom = 4 * kGiB;

// The value of a "Key:   123 kB" line, in bytes.
std::optional<std::uint64_t> meminfo_value(std::string_view text, std::string_view key) {
    std::size_t at = 0;
    while (at < text.size()) {
        const std::size_t end = std::min(text.find('\n', at), text.size());
        std::string_view line = text.substr(at, end - at);
        at                    = end + 1;
        if (!line.starts_with(key) || line.size() <= key.size() || line[key.size()] != ':') {
            continue;
        }
        line.remove_prefix(key.size() + 1);
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) {
            line.remove_prefix(1);
        }
        std::uint64_t value = 0;
        const auto [rest, error] = std::from_chars(line.data(), line.data() + line.size(), value);
        if (error != std::errc{}) { return std::nullopt; }
        std::string_view unit(rest, line.data() + line.size() - rest);
        while (!unit.empty() && (unit.front() == ' ' || unit.front() == '\t')) {
            unit.remove_prefix(1);
        }
        while (!unit.empty() && (unit.back() == ' ' || unit.back() == '\r')) { unit.remove_suffix(1); }
        if (unit == "kB") {
            if (value > std::numeric_limits<std::uint64_t>::max() / 1024) { return std::nullopt; }
            return value * 1024;
        }
        return unit.empty() ? std::optional<std::uint64_t>(value) : std::nullopt;
    }
    return std::nullopt;
}

} // namespace

std::optional<HostMemory> parse_meminfo(std::string_view text) {
    const auto total     = meminfo_value(text, "MemTotal");
    const auto available = meminfo_value(text, "MemAvailable");
    if (!total || !available) { return std::nullopt; }
    return HostMemory{.total_bytes             = *total,
                      .available_bytes         = *available,
                      .meminfo_total_bytes     = *total,
                      .meminfo_available_bytes = *available};
}

std::uint64_t CgroupMemory::available_bytes() const noexcept {
    const std::uint64_t unreclaimable =
        usage_bytes > reclaimable_bytes ? usage_bytes - reclaimable_bytes : 0;
    return limit_bytes > unreclaimable ? limit_bytes - unreclaimable : 0;
}

HostMemory within_cgroup(HostMemory memory, const CgroupMemory& cgroup) {
    memory.total_bytes            = std::min(memory.total_bytes, cgroup.limit_bytes);
    const std::uint64_t available = cgroup.available_bytes();
    if (!memory.cgroup || available <= memory.cgroup->available_bytes()) {
        memory.cgroup = cgroup;
    }
    memory.available_bytes = std::min(memory.available_bytes, available);
    return memory;
}

namespace {

// The whole of a small kernel file; nullopt when it does not exist.
std::optional<std::string> read_text(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) { return std::nullopt; }
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

[[noreturn]] void malformed(const std::filesystem::path& path, const std::string& why) {
    throw std::runtime_error("cgroup memory: " + path.string() + " " + why);
}

// A file holding one number, or "max" (nullopt) when `max_allowed`.
std::optional<std::uint64_t> read_number(const std::filesystem::path& path, bool max_allowed) {
    const auto text = read_text(path);
    if (!text) { malformed(path, "is missing"); }
    std::string_view value(*text);
    while (!value.empty() && (value.back() == '\n' || value.back() == ' ' || value.back() == '\r')) {
        value.remove_suffix(1);
    }
    if (max_allowed && value == "max") { return std::nullopt; }
    const auto number = parse_exact<std::uint64_t>(value);
    if (!number) { malformed(path, "is not a number: \"" + std::string(value) + "\""); }
    return *number;
}

// The sum of `keys` in a memory.stat ("key value" lines); every key must be there.
std::uint64_t stat_sum(const std::filesystem::path& path,
                       std::initializer_list<std::string_view> keys) {
    const auto text = read_text(path);
    if (!text) { malformed(path, "is missing"); }
    const std::string_view all(*text);
    std::uint64_t sum = 0;
    for (const std::string_view key : keys) {
        bool found     = false;
        std::size_t at = 0;
        while (at < all.size() && !found) {
            const std::size_t end = std::min(all.find('\n', at), all.size());
            std::string_view line = all.substr(at, end - at);
            at                    = end + 1;
            if (!line.starts_with(key) || line.size() <= key.size() || line[key.size()] != ' ') {
                continue;
            }
            line.remove_prefix(key.size() + 1);
            std::uint64_t value      = 0;
            const auto [rest, error] = std::from_chars(line.data(), line.data() + line.size(), value);
            if (error != std::errc{}) { malformed(path, "has a malformed " + std::string(key)); }
            sum += value;
            found = true;
        }
        if (!found) { malformed(path, "has no " + std::string(key)); }
    }
    return sum;
}

} // namespace

std::optional<CgroupMemory> read_cgroup_level(const std::filesystem::path& dir) {
    namespace fs = std::filesystem;
    if (fs::exists(dir / "memory.max")) {
        // cgroup v2. memory.high limits too: past it the kernel holds the cgroup in reclaim, and
        // pinned pages cannot be reclaimed.
        const auto max  = read_number(dir / "memory.max", true);
        const auto high = fs::exists(dir / "memory.high") ? read_number(dir / "memory.high", true)
                                                          : std::nullopt;
        if (!max && !high) { return std::nullopt; }
        return CgroupMemory{
            .path              = dir.string(),
            .limit_bytes       = std::min(max.value_or(UINT64_MAX), high.value_or(UINT64_MAX)),
            .usage_bytes       = *read_number(dir / "memory.current", false),
            .reclaimable_bytes = stat_sum(dir / "memory.stat", {"inactive_file", "active_file"})};
    }
    if (fs::exists(dir / "memory.limit_in_bytes")) {
        // cgroup v1: no limit reads as a page-rounded LONG_MAX, beyond any host.
        const std::uint64_t limit = *read_number(dir / "memory.limit_in_bytes", false);
        if (limit >= (1ULL << 62)) { return std::nullopt; }
        return CgroupMemory{
            .path              = dir.string(),
            .limit_bytes       = limit,
            .usage_bytes       = *read_number(dir / "memory.usage_in_bytes", false),
            .reclaimable_bytes = stat_sum(dir / "memory.stat",
                                          {"total_inactive_file", "total_active_file"})};
    }
    return std::nullopt;
}

std::vector<CgroupMemory> read_cgroup_memory(const std::filesystem::path& self_cgroup,
                                             const std::filesystem::path& cgroup_mount) {
    namespace fs = std::filesystem;
    std::vector<CgroupMemory> out;
    const auto text = read_text(self_cgroup);
    if (!text) { return out; }
    // Each level from `path` (as /proc/self/cgroup names it, under `root`) up to `root` itself.
    const auto walk = [&](const fs::path& root, std::string_view path) {
        fs::path relative = fs::path(std::string(path)).relative_path();
        while (true) {
            const fs::path dir = relative.empty() ? root : root / relative;
            if (fs::is_directory(dir)) {
                if (auto level = read_cgroup_level(dir)) { out.push_back(std::move(*level)); }
            }
            if (relative.empty()) { break; }
            relative = relative.parent_path();
        }
    };
    const std::string_view all(*text);
    std::size_t at = 0;
    while (at < all.size()) {
        const std::size_t end       = std::min(all.find('\n', at), all.size());
        const std::string_view line = all.substr(at, end - at);
        at                          = end + 1;
        const std::size_t first     = line.find(':');
        if (first == std::string_view::npos) { continue; }
        const std::size_t second = line.find(':', first + 1);
        if (second == std::string_view::npos) { continue; }
        const std::string_view id          = line.substr(0, first);
        const std::string_view controllers = line.substr(first + 1, second - first - 1);
        const std::string_view path        = line.substr(second + 1);
        if (id == "0" && controllers.empty()) {
            walk(cgroup_mount, path);
            continue;
        }
        // v1: the hierarchy whose controllers include memory, mounted under its controller list.
        std::size_t from = 0;
        while (from <= controllers.size()) {
            const std::size_t comma = std::min(controllers.find(',', from), controllers.size());
            if (controllers.substr(from, comma - from) == "memory") {
                walk(cgroup_mount / std::string(controllers), path);
                break;
            }
            from = comma + 1;
        }
    }
    return out;
}

std::optional<HostMemory> read_host_memory(const std::filesystem::path& meminfo,
                                           const std::filesystem::path& self_cgroup,
                                           const std::filesystem::path& cgroup_mount) {
#ifdef _WIN32
    // Windows has no /proc: asked for the running process (the default paths), the physical memory
    // and what is available of it; given other files (the tests), they are read as on Linux.
    if (meminfo == "/proc/meminfo") {
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status)) { return std::nullopt; }
        HostMemory memory;
        memory.total_bytes = memory.meminfo_total_bytes = status.ullTotalPhys;
        memory.available_bytes = memory.meminfo_available_bytes = status.ullAvailPhys;
        return memory;
    }
#endif
    const auto text = read_text(meminfo);
    if (!text) { return std::nullopt; }
    std::optional<HostMemory> memory = parse_meminfo(*text);
    if (!memory) { return std::nullopt; }
    for (const CgroupMemory& level : read_cgroup_memory(self_cgroup, cgroup_mount)) {
        *memory = within_cgroup(*memory, level);
    }
    return memory;
}

std::uint64_t pinned_host_headroom(const HostMemory& memory) noexcept {
    return std::max(kMinimumHeadroom, memory.total_bytes / 8);
}

std::uint64_t pinned_host_cap(const HostMemory& memory) noexcept {
    return memory.total_bytes / 5 * 3;
}

std::uint64_t pinned_host_limit(const HostMemory& memory) noexcept {
    const std::uint64_t headroom  = pinned_host_headroom(memory);
    const std::uint64_t available = memory.available_bytes > headroom
                                        ? memory.available_bytes - headroom
                                        : 0;
    return std::min(available, pinned_host_cap(memory));
}

namespace {

// "MemAvailable 20.00 GiB of MemTotal 30.00 GiB", within the memory cgroup when one lowers either.
std::string describe_available(const HostMemory& memory) {
    std::string meminfo = "MemAvailable " + format_gib(memory.meminfo_available_bytes) +
                          " of MemTotal " + format_gib(memory.meminfo_total_bytes);
    if (!memory.cgroup) { return meminfo; }
    const CgroupMemory& cgroup = *memory.cgroup;
    return "available " + format_gib(memory.available_bytes) + " of " +
           format_gib(memory.total_bytes) + " (" + meminfo + ", within the memory cgroup " +
           cgroup.path + ": limit " + format_gib(cgroup.limit_bytes) + ", " +
           format_gib(cgroup.usage_bytes) + " charged, " + format_gib(cgroup.reclaimable_bytes) +
           " of it reclaimable page cache)";
}

} // namespace

void require_pinned_host_fits(std::uint64_t pinned_bytes, const HostMemory& memory,
                              std::string_view what, std::string_view remedy) {
    const std::uint64_t limit = pinned_host_limit(memory);
    if (pinned_bytes <= limit) { return; }
    std::string message =
        std::string(what) + " needs " + format_gib(pinned_bytes) +
        " of pinned Host memory, but this host can pin at most " + format_gib(limit) + ": " +
        describe_available(memory) + " minus " + format_gib(pinned_host_headroom(memory)) +
        " kept for the rest of the host (max(4 GiB, total/8); pinned pages "
        "cannot be reclaimed or swapped), and at most " +
        format_gib(pinned_host_cap(memory)) + " pinned in all (3/5 of the total).";
    if (!remedy.empty()) { message += " " + std::string(remedy); }
    throw std::runtime_error(message);
}

HostTierFit fit_host_tier(std::uint64_t image_bytes, std::uint32_t state_slots,
                          std::uint64_t kv_bytes, std::uint64_t budget) noexcept {
    const std::uint64_t images = image_bytes * state_slots;
    if (images + kv_bytes <= budget) {
        return {.state_slots = state_slots, .kv_bytes = kv_bytes, .reduced = false};
    }
    const long double factor =
        static_cast<long double>(budget) / static_cast<long double>(images + kv_bytes);
    HostTierFit out{.reduced = true};
    out.state_slots = static_cast<std::uint32_t>(static_cast<long double>(state_slots) * factor);
    out.kv_bytes = static_cast<std::uint64_t>(static_cast<long double>(kv_bytes) * factor) / kMiB *
                   kMiB;
    // Rounding keeps both at or under their scaled shares, so the sum stays within the budget.
    return out;
}

std::string format_gib(std::uint64_t bytes) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f GiB", to_gib(bytes));
    return buffer;
}

} // namespace ninferno
