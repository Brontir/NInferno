#include "runtime/engine/warmup_cache.h"

#include "core/ascii.h"
#include "core/sha256.h"
#include "core/sys.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <system_error>

namespace ninferno::runtime {
namespace {

constexpr std::string_view kSchema = "ninferno-warm-up/1";

std::filesystem::path record_path(std::string_view key_text) {
    const auto dir = warm_up_cache_dir();
    if (dir.empty()) { return {}; }
    return dir / (sha256_hex(key_text) + ".json");
}

} // namespace

bool warm_up_cache_enabled() {
    const auto value = ascii_lower(sys::env("NINFERNO_WARMUP_CACHE"));
    return value != "0" && value != "off" && value != "no" && value != "false";
}

std::filesystem::path warm_up_cache_dir() {
    const auto root = sys::cache_dir();
    return root.empty() ? root : root / "warm-up";
}

std::optional<WarmUpRecord> read_warm_up_record(std::string_view key_text) {
    if (!warm_up_cache_enabled()) { return std::nullopt; }
    const auto path = record_path(key_text);
    if (path.empty()) { return std::nullopt; }
    try {
        std::ifstream in(path, std::ios::binary);
        if (!in) { return std::nullopt; }
        const auto json = nlohmann::json::parse(in);
        if (json.value("schema", std::string()) != kSchema) { return std::nullopt; }
        // The key text itself, so that a hash collision or a hand-edited file is a miss.
        if (json.value("key", std::string()) != key_text) { return std::nullopt; }
        WarmUpRecord record;
        record.stack_limit_bytes  = json.at("stack_limit_bytes").get<std::size_t>();
        record.loop_bytes         = json.at("loop_bytes").get<std::size_t>();
        record.stack_limit_source = json.value("stack_limit_source", std::string());
        return record;
    } catch (const std::exception&) { return std::nullopt; }
}

void write_warm_up_record(std::string_view key_text, const WarmUpRecord& record) {
    if (!warm_up_cache_enabled()) { return; }
    const auto path = record_path(key_text);
    if (path.empty()) { return; }
    try {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        const nlohmann::json json = {{"schema", kSchema},
                                     {"key", std::string(key_text)},
                                     {"stack_limit_bytes", record.stack_limit_bytes},
                                     {"loop_bytes", record.loop_bytes},
                                     {"stack_limit_source", record.stack_limit_source}};
        // Written beside, then renamed: a concurrent start reads the old record or the new one.
        auto partial = path;
        partial += ".partial";
        {
            std::ofstream out(partial, std::ios::binary | std::ios::trunc);
            if (!out) { return; }
            out << json.dump(1) << '\n';
            if (!out) { return; }
        }
        std::filesystem::rename(partial, path, error);
        if (error) { std::filesystem::remove(partial, error); }
    } catch (const std::exception&) {}
}

} // namespace ninferno::runtime
