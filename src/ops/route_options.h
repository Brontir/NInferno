#pragma once

// The maintainer's route options, --route-dir DIR and --route-table FILE, checked as the Engine and
// the family tools take them: a directory or table that cannot be read is refused, never read as
// "no table" (which would run the heuristic and look like a measurement of DIR).

#include "ninferno/ops/route_table.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace ninferno::ops {

// A file the engine reads from a route directory, by its path relative to the directory.
struct RouteDirFile {
    std::string relative;
    std::filesystem::path path;
};

// The files the engine reads from --route-dir DIR, in byte order of their relative paths: each
// DIR/*.json that is a regular file once symlinks are followed (the per-GPU-class tables,
// ops/route_table.cpp) and DIR/tuner/dense_tune.json (the release's dense-tuner measurements,
// ops/generic/generic_tune.cpp). Nothing else under DIR is read. Throws std::invalid_argument when
// DIR is not a directory that can be listed, or one of those names is a dangling symlink or cannot
// be opened.
[[nodiscard]] inline std::vector<RouteDirFile> route_dir_files(const std::filesystem::path& dir) {
    namespace fs      = std::filesystem;
    const auto refuse = [&](const std::string& why) {
        throw std::invalid_argument("--route-dir " + dir.string() + ": " + why);
    };
    std::error_code error;
    if (!fs::is_directory(dir, error)) {
        refuse(error ? error.message() : std::string("not a directory"));
    }
    std::vector<RouteDirFile> out;
    const auto take = [&](const fs::path& path, std::string relative) {
        std::error_code ec;
        const fs::file_status target = fs::status(path, ec);
        if (!fs::exists(target)) { refuse(relative + " is a dangling symlink"); }
        if (!fs::is_regular_file(target)) { return; } // the engine skips it too
        if (!std::ifstream(path, std::ios::binary)) { refuse(relative + " cannot be read"); }
        out.push_back({std::move(relative), path});
    };
    for (fs::directory_iterator it(dir, error), end; !error && it != end; it.increment(error)) {
        if (it->path().extension() == ".json") {
            take(it->path(), it->path().filename().generic_string());
        }
    }
    if (error) { refuse(error.message()); }
    const fs::path tuner = dir / "tuner" / "dense_tune.json";
    if (fs::exists(fs::symlink_status(tuner, error))) { take(tuner, "tuner/dense_tune.json"); }
    std::sort(out.begin(), out.end(),
              [](const RouteDirFile& a, const RouteDirFile& b) { return a.relative < b.relative; });
    return out;
}

// Both options checked without a device (before anything is converted or loaded): DIR as
// route_dir_files lists it, FILE parsed and validated (inspect_route_table). Empty: not given.
inline void check_route_options(const std::filesystem::path& table,
                                const std::filesystem::path& dir) {
    if (!dir.empty()) { (void)route_dir_files(dir); }
    if (table.empty()) { return; }
    try {
        (void)inspect_route_table(table.string());
    } catch (const std::exception& error) {
        throw std::invalid_argument("--route-table " + table.string() + ": " + error.what());
    }
}

// --route-table FILE loaded on the current device: a table that cannot be read, is malformed or
// is for another GPU class throws std::invalid_argument naming the option.
inline RouteTableStatus load_route_table_option(const std::filesystem::path& table) {
    RouteTableStatus status;
    try {
        status = load_route_table(table.string());
    } catch (const std::exception& error) {
        throw std::invalid_argument("--route-table " + table.string() + ": " + error.what());
    }
    if (!status.loaded) { throw std::invalid_argument("--route-table: " + status.message); }
    return status;
}

} // namespace ninferno::ops
