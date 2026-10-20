#pragma once

// The converter's small file helpers: whole-file reads, the directory beside an output path, its
// free space and durability, byte counts as people read them, and the default file size cap.

#include "core/file_io.h"
#include "core/sys.h"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

namespace ninferno::convert {

// The largest file of an artifact a conversion writes by default (--max-file-bytes): larger
// payloads continue in sibling files.
inline constexpr std::uint64_t kDefaultMaxFileBytes = 32'000'000'000ULL;

using ninferno::create_parent_directories;
using ninferno::directory_of;
using ninferno::read_file;

// Makes a file's creation, rename or removal durable: syncs the directory it is in.
inline void sync_directory_of(const std::filesystem::path& file) {
    sys::sync_directory(directory_of(file));
}

// The bytes free to an unprivileged writer in the directory a file is in.
[[nodiscard]] inline std::uint64_t free_bytes_beside(const std::filesystem::path& file) {
    const std::filesystem::path dir = directory_of(file);
    std::error_code error;
    const auto space = std::filesystem::space(dir, error);
    if (error) { throw std::runtime_error("cannot query free space of " + dir.string()); }
    return space.available;
}

} // namespace ninferno::convert
