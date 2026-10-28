#pragma once

#include "ninferno/option_enums.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace ninferno::artifact {

// Direct reads require aligned offsets and buffers. A short final direct block is allowed;
// read_exact always requires the complete requested byte range. Payload ("direct") reads use
// O_DIRECT on local file systems and the page cache on network, FUSE and host-shared ones;
// set_artifact_io (--artifact-io direct|buffered) forces either.
// The process's payload I/O mode, read when a file opens its payload reads (the Engine applies
// --artifact-io at construction; a tool before it loads). Not a decision: the bytes are the same.
void set_artifact_io(ArtifactIo io) noexcept;

class InputFile {
public:
    explicit InputFile(std::filesystem::path path);
    ~InputFile();
    InputFile(const InputFile&)            = delete;
    InputFile& operator=(const InputFile&) = delete;

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::size_t read_direct(std::uint64_t offset,
                                          std::span<std::byte> destination) const;

private:
    std::filesystem::path path_;
    int fd_                = -1;
    mutable int direct_fd_     = -1;
    mutable bool direct_mode_  = false; // direct_fd_ was opened with O_DIRECT
    std::uint64_t bytes_   = 0;
};

} // namespace ninferno::artifact
