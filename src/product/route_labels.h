#pragma once

// How a measurement record (a spec-tune profile's settings) names the route data a maintainer
// pointed it at: by content, never by a path local to the machine that measured it, so that a
// resumed tune refuses cells measured with other tables.

#include "core/sha256.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace ninferno::product {

// --route-table FILE: "sha256:<hex of its bytes>", or "unreadable".
[[nodiscard]] inline std::string route_table_label(const std::filesystem::path& table) {
    const auto digest = file_sha256_hex(table);
    return digest ? "sha256:" + *digest : std::string("unreadable");
}

// --route-dir DIR: "sha256:<hex>" over every regular file under it in byte order of its path
// relative to DIR (each as "<relative path>\0<sha256 of its bytes>\n"), so two directories with the
// same tables carry one label wherever they are; "unreadable" when it cannot be listed or read.
[[nodiscard]] inline std::string route_dir_label(const std::filesystem::path& dir) {
    std::error_code error;
    std::vector<std::pair<std::string, std::filesystem::path>> files;
    for (std::filesystem::recursive_directory_iterator it(dir, error), end; !error && it != end;
         it.increment(error)) {
        if (it->is_regular_file(error)) {
            files.emplace_back(std::filesystem::relative(it->path(), dir, error).generic_string(),
                               it->path());
        }
    }
    if (error) { return "unreadable"; }
    std::sort(files.begin(), files.end());
    std::string manifest;
    for (const auto& [relative, path] : files) {
        const auto digest = file_sha256_hex(path);
        if (!digest) { return "unreadable"; }
        manifest += relative;
        manifest += '\0';
        manifest += *digest;
        manifest += '\n';
    }
    return "sha256:" + sha256_hex(std::string_view(manifest));
}

} // namespace ninferno::product
