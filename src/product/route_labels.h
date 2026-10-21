#pragma once

// How a measurement record (a spec-tune profile's settings) names the data files a maintainer
// pointed it at (a route table, a route directory, a drafter registry): by content, never by a
// path local to the machine that measured it, so that a resumed tune refuses cells measured with
// other data. A file or directory that cannot be read is refused (std::invalid_argument), never
// labelled: two unreadable inputs would otherwise carry one label.

#include "core/sha256.h"
#include "ops/route_options.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::product {

// "sha256:<hex of the file's bytes>"; `option` names the flag in a refusal.
[[nodiscard]] inline std::string file_content_label(const std::filesystem::path& file,
                                                    std::string_view option) {
    const auto digest = file_sha256_hex(file);
    if (!digest) {
        throw std::invalid_argument(std::string(option) + " " + file.string() + ": cannot be read");
    }
    return "sha256:" + *digest;
}

// --route-table FILE.
[[nodiscard]] inline std::string route_table_label(const std::filesystem::path& table) {
    return file_content_label(table, "--route-table");
}

// --drafter-registry FILE.
[[nodiscard]] inline std::string drafter_registry_label(const std::filesystem::path& registry) {
    return file_content_label(registry, "--drafter-registry");
}

// --route-dir DIR: "sha256:<hex>" over the files the engine reads from it (ops::route_dir_files:
// the per-GPU-class tables and tuner/dense_tune.json, symlinks followed) in byte order of their
// paths relative to DIR, each as "<relative path>\0<sha256 of its bytes>\n". Two directories with
// the same tables carry one label wherever they are; a file the engine never reads changes nothing.
[[nodiscard]] inline std::string route_dir_label(const std::filesystem::path& dir) {
    std::string manifest;
    for (const auto& file : ops::route_dir_files(dir)) {
        manifest += file.relative;
        manifest += '\0';
        manifest += file_content_label(file.path, "--route-dir");
        manifest += '\n';
    }
    return "sha256:" + sha256_hex(std::string_view(manifest));
}

} // namespace ninferno::product
