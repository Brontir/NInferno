#pragma once

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include "core/file_io.h"
#include "core/sys.h"

namespace ninferno {

// Replaces `path` with `text` through a temporary of this process's own (its pid and a clock
// reading in the name), so a reader sees the old file or the new one and concurrent writers never
// share a temporary. A failed open or write throws and leaves `path` as it was.
inline void write_file_atomic(const std::filesystem::path& path, std::string_view text) {
    std::filesystem::path temporary = path;
    temporary += ".tmp." + std::to_string(sys::pid()) + "." +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    try {
        {
            std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
            if (!out) { throw std::runtime_error("cannot open " + temporary.string()); }
            out.write(text.data(), static_cast<std::streamsize>(text.size()));
            out.close();
            if (!out) { throw std::runtime_error("cannot write " + temporary.string()); }
        }
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

// A cache file's document when it parses, names `schema` and holds an object `section`; nullopt
// for a missing, unreadable or foreign file, which the caller treats as a cache miss.
inline std::optional<nlohmann::json> read_cache_file(const std::filesystem::path& path,
                                                     std::string_view schema, const char* section) {
    if (path.empty() || !std::filesystem::exists(path)) { return std::nullopt; }
    try {
        std::ifstream in(path);
        nlohmann::json document;
        in >> document;
        if (document.value("schema", std::string{}) == schema && document.contains(section) &&
            document[section].is_object()) {
            return document;
        }
    } catch (const std::exception&) {}
    return std::nullopt;
}

// Writes `cache` to `path` after taking in the `section` entries another process wrote since this
// one read the file (this process's own entries keep precedence).
inline void save_cache_file(const std::filesystem::path& path, nlohmann::json& cache,
                            std::string_view schema, const char* section) {
    create_parent_directories(path);
    if (const std::optional<nlohmann::json> disk = read_cache_file(path, schema, section)) {
        for (const auto& [key, value] : (*disk)[section].items()) {
            if (!cache[section].contains(key)) { cache[section][key] = value; }
        }
    }
    write_file_atomic(path, cache.dump(1) + '\n');
}

} // namespace ninferno
