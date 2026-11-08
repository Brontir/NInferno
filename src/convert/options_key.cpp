#include "convert/options_key.h"

#include "core/sha256.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <vector>

namespace ninferno::convert {
namespace {

namespace fs = std::filesystem;

constexpr std::uint64_t kWholeFileBytes = 64ULL << 20;
constexpr std::uint64_t kHeadBytes      = 16ULL << 20;
constexpr std::uint64_t kTailBytes      = 1ULL << 20;

std::string file_identity(const fs::path& path) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec) { return "unreadable:" + utf8_path(path); }
    if (size <= kWholeFileBytes) {
        if (const auto digest = file_sha256_hex(path)) { return "sha256:" + *digest; }
        return "unreadable:" + utf8_path(path);
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) { return "unreadable:" + utf8_path(path); }
    const auto part = [&](std::uint64_t at, std::uint64_t bytes) {
        std::string data(static_cast<std::size_t>(bytes), '\0');
        in.seekg(static_cast<std::streamoff>(at));
        in.read(data.data(), static_cast<std::streamsize>(bytes));
        data.resize(static_cast<std::size_t>(in.gcount()));
        return sha256_hex(data);
    };
    // The modification time too: most edits of a large file land outside the parts hashed.
    const auto written = fs::last_write_time(path, ec);
    return "size:" + std::to_string(size) +
           ",mtime:" + std::to_string(ec ? 0 : written.time_since_epoch().count()) +
           ",head:" + part(0, kHeadBytes) + ",tail:" + part(size - kTailBytes, kTailBytes);
}

} // namespace

std::string content_identity(const fs::path& path) {
    std::error_code ec;
    if (fs::is_regular_file(path, ec)) { return file_identity(path); }
    if (!fs::is_directory(path, ec)) { return "unreadable:" + utf8_path(path); }
    std::vector<fs::path> files;
    for (auto it = fs::recursive_directory_iterator(path, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        std::error_code e;
        if (it->is_regular_file(e)) { files.push_back(it->path()); }
    }
    std::sort(files.begin(), files.end());
    std::string text;
    for (const auto& file : files) {
        text += utf8_path(file.lexically_relative(path)) + "=" + file_identity(file) + ";";
    }
    return "tree:" + sha256_hex(text);
}

std::string options_key(const std::vector<OptionPart>& parts, std::string* described) {
    if (parts.empty()) { return {}; }
    std::string identity;
    std::string text;
    for (const auto& part : parts) {
        identity += part.name + "=" + part.value + '\n';
        text += part.name + (part.shown.empty() ? "" : " " + part.shown) + '\n';
    }
    if (described) { *described = text; }
    return sha256_hex(identity).substr(0, 12);
}

} // namespace ninferno::convert
