#include "core/debug_poison.h"

#include "core/sys.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

namespace ninferno {
namespace {

std::optional<unsigned char> parse_poison(const char* raw) {
    if (raw == nullptr) { return std::nullopt; }
    const std::string value(raw);
    if (value.empty() || value == "0") { return std::nullopt; }
    if (value == "1" || value == "nan") { return static_cast<unsigned char>(0xFF); }
    std::size_t used = 0;
    unsigned long parsed = 0;
    try {
        parsed = std::stoul(value, &used, 0);
    } catch (const std::exception&) { used = 0; }
    if (used != value.size() || parsed == 0 || parsed > 0xFF) {
        throw std::invalid_argument("NINFERNO_DEBUG_POISON must be 0, 1, nan or a byte in 1..255");
    }
    return static_cast<unsigned char>(parsed);
}

} // namespace

std::optional<unsigned char> debug_poison_byte() {
    static const std::optional<unsigned char> byte =
        parse_poison(std::getenv("NINFERNO_DEBUG_POISON"));
    return byte;
}

std::optional<unsigned char> debug_poison_byte_for(const char* site) {
    const std::optional<unsigned char> byte = debug_poison_byte();
    if (!byte) { return byte; }
    static const std::string skip = [] {
        const std::string sites = sys::env("NINFERNO_DEBUG_POISON_SKIP");
        return sites.empty() ? sites : "," + sites + ",";
    }();
    if (!skip.empty() && skip.find("," + std::string(site) + ",") != std::string::npos) {
        return std::nullopt;
    }
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
