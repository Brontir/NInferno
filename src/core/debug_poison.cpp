#include "core/debug_poison.h"

#include "core/sys.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

namespace ninferno {
namespace {

constexpr std::array<std::string_view, 4> kSites = {"kv", "ingress", "state", "weights"};

std::atomic<int> g_byte{-1}; // -1: off
std::mutex g_skip_mutex;
std::vector<std::string> g_skip;

} // namespace

std::optional<std::uint8_t> parse_debug_poison(std::string_view text) {
    const std::string value(text);
    if (value == "0") { return std::nullopt; }
    if (value == "1" || value == "nan") { return static_cast<std::uint8_t>(0xFF); }
    std::size_t used     = 0;
    unsigned long parsed = 0;
    try {
        parsed = std::stoul(value, &used, 0);
    } catch (const std::exception&) { used = 0; }
    if (value.empty() || used != value.size() || parsed == 0 || parsed > 0xFF) {
        throw std::invalid_argument("--debug-poison must be 0, 1, nan or a byte in 1..255, not '" +
                                    value + "'");
    }
    return static_cast<std::uint8_t>(parsed);
}

void set_debug_poison(std::optional<std::uint8_t> byte, const std::vector<std::string>& skip) {
    for (const auto& site : skip) {
        if (std::find(kSites.begin(), kSites.end(), site) == kSites.end()) {
            throw std::invalid_argument("--debug-poison-skip: unknown site '" + site +
                                        "' (kv, ingress, state, weights)");
        }
    }
    {
        std::lock_guard lock(g_skip_mutex);
        g_skip = skip;
    }
    g_byte.store(byte ? static_cast<int>(*byte) : -1, std::memory_order_release);
}

std::optional<unsigned char> debug_poison_byte() {
    const int byte = g_byte.load(std::memory_order_acquire);
    if (byte < 0) { return std::nullopt; }
    return static_cast<unsigned char>(byte);
}

std::optional<unsigned char> debug_poison_byte_for(const char* site) {
    const std::optional<unsigned char> byte = debug_poison_byte();
    if (!byte) { return byte; }
    std::lock_guard lock(g_skip_mutex);
    if (std::find(g_skip.begin(), g_skip.end(), site) != g_skip.end()) { return std::nullopt; }
    return byte;
}

void debug_freeze_point() noexcept {
    static const std::string path = sys::env("NINFERNO_DEBUG_FREEZE_FILE");
    if (path.empty()) { return; }
    std::error_code error;
    while (std::filesystem::exists(path, error)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

} // namespace ninferno
