#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace ninferno::artifact {

// Direct reads require aligned offsets and buffers. A short final direct block is allowed;
// read_exact always requires the complete requested byte range. Payload ("direct") reads use
// O_DIRECT on local file systems and the page cache on network, FUSE and host-shared ones;
// NINFERNO_ARTIFACT_DIRECT_IO=1/0 forces either.
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
