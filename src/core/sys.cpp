#include "core/sys.h"

#include <cerrno>
#include <cstdlib>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <bcrypt.h>
#include <io.h>
#include <malloc.h>
#include <winioctl.h>
#else
#include <linux/falloc.h>
#include <pwd.h>
#include <sys/random.h>
#include <unistd.h>

#include <vector>
#endif

namespace ninferno::sys {

std::filesystem::path home_dir() {
#ifdef _WIN32
    const wchar_t* home = _wgetenv(L"USERPROFILE");
    return home != nullptr && *home ? std::filesystem::path(home) : std::filesystem::path();
#else
    if (const char* home = std::getenv("HOME"); home != nullptr && *home) {
        return home;
    }
    // A service started without HOME (a systemd unit, a container's runner): the account's own.
    const long hint = sysconf(_SC_GETPW_R_SIZE_MAX);
    std::vector<char> buffer(hint > 0 ? static_cast<std::size_t>(hint) : 16384);
    passwd entry{};
    passwd* found = nullptr;
    if (getpwuid_r(getuid(), &entry, buffer.data(), buffer.size(), &found) == 0 && found != nullptr &&
        found->pw_dir != nullptr && *found->pw_dir) {
        return found->pw_dir;
    }
    return {};
#endif
}

std::filesystem::path cache_dir() {
#ifdef _WIN32
    if (const auto local = env("LOCALAPPDATA"); !local.empty()) {
        return std::filesystem::path(local) / "NInferno";
    }
#endif
    if (const auto xdg = env("XDG_CACHE_HOME"); !xdg.empty()) {
        return std::filesystem::path(xdg) / "ninferno";
    }
    if (const auto home = home_dir(); !home.empty()) { return home / ".cache" / "ninferno"; }
    return {};
}

std::string env(const char* name) {
#ifdef _WIN32
    const wchar_t* value = _wgetenv(std::filesystem::path(name).wstring().c_str());
    if (value == nullptr) { return {}; }
    const std::u8string text = std::filesystem::path(value).u8string();
    return std::string(text.begin(), text.end());
#else
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string();
#endif
}

int punch_hole(int fd, std::int64_t offset, std::int64_t bytes) {
#ifdef _WIN32
    HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    DWORD returned = 0;
    FILE_SET_SPARSE_BUFFER sparse{TRUE};
    DeviceIoControl(handle, FSCTL_SET_SPARSE, &sparse, sizeof(sparse), nullptr, 0, &returned, nullptr);
    FILE_ZERO_DATA_INFORMATION range{};
    range.FileOffset.QuadPart      = offset;
    range.BeyondFinalZero.QuadPart = offset + bytes;
    if (!DeviceIoControl(handle, FSCTL_SET_ZERO_DATA, &range, sizeof(range), nullptr, 0, &returned, nullptr)) {
        errno = EIO;
        return -1;
    }
    return 0;
#else
    return ::fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, static_cast<off_t>(offset),
                       static_cast<off_t>(bytes));
#endif
}

int create_unique(std::string& name_template, int suffix_length) {
#ifdef _WIN32
    const std::size_t end = name_template.size() - static_cast<std::size_t>(suffix_length);
    if (end < 6 || name_template.compare(end - 6, 6, "XXXXXX") != 0) {
        errno = EINVAL;
        return -1;
    }
    static const char letters[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (int attempt = 0; attempt < 100; ++attempt) {
        unsigned char noise[6];
        if (!random_bytes(noise, sizeof(noise))) { return -1; }
        for (int i = 0; i < 6; ++i) { name_template[end - 6 + i] = letters[noise[i] % 62]; }
        const int fd = open(std::filesystem::u8path(name_template), O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd >= 0 || errno != EEXIST) { return fd; }
    }
    errno = EEXIST;
    return -1;
#else
    return ::mkostemps(name_template.data(), suffix_length, O_CLOEXEC);
#endif
}

bool random_bytes(void* buffer, std::size_t bytes) {
#ifdef _WIN32
    return BCryptGenRandom(nullptr, static_cast<PUCHAR>(buffer), static_cast<ULONG>(bytes),
                           BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#else
    return ::getrandom(buffer, bytes, 0) == static_cast<ssize_t>(bytes);
#endif
}

void* aligned_alloc(std::size_t alignment, std::size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(bytes, alignment);
#else
    return std::aligned_alloc(alignment, bytes);
#endif
}

void aligned_free(void* data) noexcept {
#ifdef _WIN32
    _aligned_free(data);
#else
    std::free(data);
#endif
}

void sync_directory(const std::filesystem::path& dir) {
#ifndef _WIN32
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        (void)::fsync(fd);
        ::close(fd);
    }
#else
    (void)dir;
#endif
}

} // namespace ninferno::sys
