#pragma once

// Whole-file reads (a file's bytes, or an error that names it) and the directory a file goes in.

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno {

// The whole file, as bytes. `what` names the file in the error: "cannot read <what> <path>".
[[nodiscard]] inline std::string read_file(const std::filesystem::path& path,
                                           std::string_view what = {}) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot read " +
                                 (what.empty() ? std::string() : std::string(what) + " ") +
                                 path.string());
    }
    std::ostringstream bytes;
    bytes << stream.rdbuf();
    return bytes.str();
}

// The directory a file is (or will be) in: its parent, or the working directory.
[[nodiscard]] inline std::filesystem::path directory_of(const std::filesystem::path& file) {
    return file.parent_path().empty() ? std::filesystem::path(".") : file.parent_path();
}

// Creates the directory `file` goes in, with its parents (throws std::filesystem_error).
inline void create_parent_directories(const std::filesystem::path& file) {
    std::filesystem::create_directories(directory_of(file));
}

} // namespace ninferno
