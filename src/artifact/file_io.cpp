#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "core/ascii.h"
#include "core/sys.h"
#include "core/units.h"

#ifndef _WIN32
#include <sys/vfs.h>
#endif

namespace ninferno::artifact {
namespace {

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + sys::error_text(errno));
}

std::int64_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<std::int64_t>(offset);
}

// NINFERNO_ARTIFACT_DIRECT_IO: 1/true/yes/on forces O_DIRECT payload reads, 0/false/no/off forces
// buffered ones (case-insensitive, surrounding blanks ignored). Unset or any other value: automatic.
enum class DirectIoChoice { Automatic, Direct, Buffered };

DirectIoChoice direct_io_choice() {
    const char* raw = std::getenv("NINFERNO_ARTIFACT_DIRECT_IO");
    if (raw == nullptr) { return DirectIoChoice::Automatic; }
    std::string value(raw);
    const auto blank = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!value.empty() && blank(value.back())) { value.pop_back(); }
    value.erase(value.begin(), std::find_if_not(value.begin(), value.end(), blank));
    value = ascii_lower(value);
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return DirectIoChoice::Direct;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return DirectIoChoice::Buffered;
    }
    return DirectIoChoice::Automatic;
}

// Network, FUSE and host-shared file systems: O_DIRECT bypasses their cache layers (an object
// store behind FUSE read 13.8 MiB/s direct against 177 MiB/s buffered), so the
// automatic choice reads them through the page cache. Local disks keep O_DIRECT, which keeps a
// 20+ GB payload out of the page cache.
bool remote_file_system(int fd) {
#ifdef _WIN32
    (void)fd; // Windows reads local disks here; NO_BUFFERING suits them
    return false;
#else
    struct statfs status {};
    if (::fstatfs(fd, &status) != 0) { return false; }
    switch (static_cast<unsigned long>(status.f_type)) {
    case 0x65735546UL: // FUSE
    case 0x6969UL:     // NFS
    case 0x517BUL:     // SMB
    case 0xFF534D42UL: // CIFS
    case 0xFE534D42UL: // SMB2
    case 0x01021997UL: // 9P (WSL drives, virtio shares)
    case 0x00C36400UL: // Ceph
    case 0x47504653UL: // GPFS
    case 0x0BD00BD0UL: // Lustre
        return true;
    default:
        return false;
    }
#endif
}

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path_, ec) && std::filesystem::exists(path_, ec)) {
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    fd_ = sys::open(path_, O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }
    const std::int64_t size = sys::file_size(fd_);
    if (size < 0) {
        const auto error = errno;
        sys::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    bytes_ = static_cast<std::uint64_t>(size);
}

InputFile::~InputFile() {
    if (direct_fd_ >= 0) { sys::close(direct_fd_); }
    if (fd_ >= 0) { sys::close(fd_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64 * kMiB);
        const auto read  = sys::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ < 0) {
        const DirectIoChoice choice = direct_io_choice();
        const bool direct           = choice == DirectIoChoice::Direct ||
                            (choice == DirectIoChoice::Automatic && !remote_file_system(fd_));
        if (direct) {
            direct_fd_   = sys::open(path_, O_RDONLY | O_CLOEXEC | O_DIRECT);
            direct_mode_ = direct_fd_ >= 0;
            // A file system without O_DIRECT support refuses the flag: read it buffered unless
            // direct reads were asked for.
            if (direct_fd_ < 0 && (errno != EINVAL || choice == DirectIoChoice::Direct)) {
                fail(path_, "open direct");
            }
        }
        if (direct_fd_ < 0) {
            direct_fd_ = sys::open(path_, O_RDONLY | O_CLOEXEC);
            if (direct_fd_ < 0) { fail(path_, "open"); }
        }
    }
    // One pread may return less than asked without an error (network and FUSE file systems cut a
    // read at their own maximum), so read until the block is full or the file ends.
    std::size_t total = 0;
    while (total < destination.size()) {
        const auto read = sys::pread(direct_fd_, destination.data() + total, destination.size() - total,
                                  file_offset(offset + total));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "direct pread");
        }
        if (read == 0) { break; }
        total += static_cast<std::size_t>(read);
        // A direct read short of a whole block ended at the file's tail; the next offset would be
        // unaligned.
        if (direct_mode_ && total % kPayloadAlignment != 0) { break; }
    }
    return total;
}

} // namespace ninferno::artifact
