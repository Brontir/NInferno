// core/sys.h on Windows: descriptors are the C runtime's (so std::FILE and _close work on them),
// each wrapping a Win32 handle opened by CreateFileW (UTF-8 paths), which does the offset I/O.

#include "core/sys.h"

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <io.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace ninferno::sys {

namespace {

std::wstring wide(const char* text) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    std::wstring out(n > 0 ? static_cast<std::size_t>(n - 1) : 0, L'\0');
    if (n > 1) { MultiByteToWideChar(CP_UTF8, 0, text, -1, out.data(), n); }
    return out;
}

int set_errno_from(DWORD error) {
    switch (error) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND: errno = ENOENT; break;
    case ERROR_FILE_EXISTS:
    case ERROR_ALREADY_EXISTS: errno = EEXIST; break;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION: errno = EACCES; break;
    case ERROR_DISK_FULL:
    case ERROR_HANDLE_DISK_FULL: errno = ENOSPC; break;
    case ERROR_LOCK_VIOLATION: errno = EWOULDBLOCK; break;
    case ERROR_NOT_SUPPORTED:
    case ERROR_INVALID_PARAMETER: errno = EINVAL; break;
    case ERROR_FILENAME_EXCED_RANGE: errno = ENAMETOOLONG; break;
    default: errno = EIO; break;
    }
    return -1;
}

HANDLE handle_of(int fd) { return reinterpret_cast<HANDLE>(_get_osfhandle(fd)); }

// A path Win32 opens past MAX_PATH (260) without the system-wide LongPathsEnabled switch: the
// absolute path behind the \\?\ prefix (\\?\UNC\ for a share). Short paths are left as they are.
std::wstring long_path(const std::filesystem::path& path) {
    std::wstring text = path.wstring();
    if (text.size() < MAX_PATH - 12 || text.starts_with(L"\\\\?\\")) return text;
    std::error_code ec;
    const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
    if (ec) return text;
    text = absolute.lexically_normal().wstring();
    for (wchar_t& c : text) {
        if (c == L'/') c = L'\\';
    }
    if (text.starts_with(L"\\\\")) return L"\\\\?\\UNC\\" + text.substr(2);
    return L"\\\\?\\" + text;
}

} // namespace

int open(const std::filesystem::path& path, int flags, int /*mode*/) {
    const int access_mode = flags & (O_RDONLY | O_WRONLY | O_RDWR);
    DWORD access          = access_mode == O_WRONLY ? GENERIC_WRITE
                          : access_mode == O_RDWR   ? GENERIC_READ | GENERIC_WRITE
                                                    : GENERIC_READ;
    DWORD disposition = OPEN_EXISTING;
    if ((flags & O_CREAT) && (flags & O_EXCL)) {
        disposition = CREATE_NEW;
    } else if ((flags & O_CREAT) && (flags & O_TRUNC)) {
        disposition = CREATE_ALWAYS;
    } else if (flags & O_CREAT) {
        disposition = OPEN_ALWAYS;
    } else if (flags & O_TRUNC) {
        disposition = TRUNCATE_EXISTING;
    }
    DWORD attributes = FILE_ATTRIBUTE_NORMAL;
    if (flags & O_DIRECT) { attributes |= FILE_FLAG_NO_BUFFERING; }
    if (flags & O_DIRECTORY) {
        attributes |= FILE_FLAG_BACKUP_SEMANTICS;
        access = GENERIC_READ;
    }
    // Others may read, write, rename and delete it meanwhile, as on POSIX.
    HANDLE handle = CreateFileW(long_path(path).c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr, disposition, attributes, nullptr);
    if (handle == INVALID_HANDLE_VALUE) { return set_errno_from(GetLastError()); }
    const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(handle),
                                   (access_mode == O_RDONLY ? _O_RDONLY : 0) | _O_BINARY);
    if (fd < 0) {
        CloseHandle(handle);
        errno = EMFILE;
    }
    return fd;
}

int close(int fd) { return _close(fd); }

std::int64_t pread(int fd, void* buffer, std::size_t bytes, std::int64_t offset) {
    HANDLE handle = handle_of(fd);
    if (handle == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    // One ReadFile moves at most 4 GiB - 1: callers loop on short reads as on POSIX.
    const DWORD want = static_cast<DWORD>(bytes > 0x40000000u ? 0x40000000u : bytes);
    OVERLAPPED at{};
    at.Offset     = static_cast<DWORD>(offset & 0xffffffffu);
    at.OffsetHigh = static_cast<DWORD>(static_cast<std::uint64_t>(offset) >> 32);
    DWORD done    = 0;
    if (!ReadFile(handle, buffer, want, &done, &at)) {
        const DWORD error = GetLastError();
        if (error == ERROR_HANDLE_EOF) { return 0; }
        return set_errno_from(error);
    }
    return static_cast<std::int64_t>(done);
}

std::int64_t pwrite(int fd, const void* buffer, std::size_t bytes, std::int64_t offset) {
    HANDLE handle = handle_of(fd);
    if (handle == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    const DWORD want = static_cast<DWORD>(bytes > 0x40000000u ? 0x40000000u : bytes);
    OVERLAPPED at{};
    at.Offset     = static_cast<DWORD>(offset & 0xffffffffu);
    at.OffsetHigh = static_cast<DWORD>(static_cast<std::uint64_t>(offset) >> 32);
    DWORD done    = 0;
    if (!WriteFile(handle, buffer, want, &done, &at)) { return set_errno_from(GetLastError()); }
    return static_cast<std::int64_t>(done);
}

std::int64_t file_size(int fd) {
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(handle_of(fd), &size)) { return set_errno_from(GetLastError()); }
    return size.QuadPart;
}

std::int64_t write(int fd, const void* buffer, std::size_t bytes) {
    return _write(fd, buffer, static_cast<unsigned>(bytes > 0x40000000u ? 0x40000000u : bytes));
}

std::int64_t seek(int fd, std::int64_t offset, int whence) { return _lseeki64(fd, offset, whence); }

int truncate(int fd, std::int64_t length) {
    FILE_END_OF_FILE_INFO end{};
    end.EndOfFile.QuadPart = length;
    if (!SetFileInformationByHandle(handle_of(fd), FileEndOfFileInfo, &end, sizeof(end))) {
        return set_errno_from(GetLastError());
    }
    return 0;
}

int fsync(int fd) { return FlushFileBuffers(handle_of(fd)) ? 0 : set_errno_from(GetLastError()); }
int fdatasync(int fd) { return fsync(fd); }

// One byte far past any file's end: Windows byte-range locks are mandatory, so a lock over the
// file's contents would refuse the locking process's own other handles (the relayout reads the
// artifact it holds locked) where POSIX flock is advisory. A lock no read or write ever reaches
// keeps flock's meaning: only other lockers wait.
constexpr DWORD kLockOffsetHigh = 0x7FFFFFFFu;

int lock(int fd, bool wait, bool shared) {
    OVERLAPPED at{};
    at.OffsetHigh     = kLockOffsetHigh;
    const DWORD flags = (shared ? 0 : LOCKFILE_EXCLUSIVE_LOCK) | (wait ? 0 : LOCKFILE_FAIL_IMMEDIATELY);
    if (!LockFileEx(handle_of(fd), flags, 0, 1, 0, &at)) {
        return set_errno_from(GetLastError());
    }
    return 0;
}

void* map(int fd, std::size_t bytes, Map mode) {
    if (bytes == 0) { errno = EINVAL; return nullptr; }
    // ReadPrivate is read-only (PROT_READ) as on POSIX, so no page is ever copied: a plain
    // read-only view. A copy-on-write one (PAGE_WRITECOPY) charges the commit limit for the whole
    // view, which a multi-GB GGUF read for its metadata can exceed.
    const DWORD protect = mode == Map::ReadWriteShared ? PAGE_READWRITE : PAGE_READONLY;
    const std::uint64_t size = bytes;
    HANDLE mapping = CreateFileMappingW(handle_of(fd), nullptr, protect, static_cast<DWORD>(size >> 32),
                                        static_cast<DWORD>(size & 0xffffffffu), nullptr);
    if (!mapping) { set_errno_from(GetLastError()); return nullptr; }
    const DWORD access = mode == Map::ReadWriteShared ? FILE_MAP_WRITE : FILE_MAP_READ;
    void* base = MapViewOfFile(mapping, access, 0, 0, bytes);
    const DWORD error = GetLastError();
    CloseHandle(mapping); // the view keeps the mapping
    if (!base) { set_errno_from(error); }
    return base;
}

int unmap(void* base, std::size_t /*bytes*/) { return UnmapViewOfFile(base) ? 0 : set_errno_from(GetLastError()); }

void advise(void* base, std::size_t bytes, Advice advice) {
    if (advice == Advice::WillNeed && base && bytes) {
        WIN32_MEMORY_RANGE_ENTRY range{base, bytes};
        (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
    }
    // DontNeed: Windows trims a mapped file's pages from the working set under pressure itself.
}

void advise_file(int, std::int64_t, std::int64_t, Advice) {}
void start_writeback(int) {}

std::size_t page_size() {
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return info.dwPageSize;
}

long long pid() { return static_cast<long long>(GetCurrentProcessId()); }

std::filesystem::path self_exe() {
    std::wstring path(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    path.resize(n);
    return std::filesystem::path(path);
}

bool gmtime(std::time_t t, std::tm& out) { return ::gmtime_s(&out, &t) == 0; }
bool localtime(std::time_t t, std::tm& out) { return ::localtime_s(&out, &t) == 0; }

bool stderr_is_terminal() { return _isatty(_fileno(stderr)) != 0; }

int terminal_columns() {
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (!GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &info)) { return 0; }
    return info.srWindow.Right - info.srWindow.Left + 1;
}

void* load_library(const char* /*linux_name*/, const char* windows_name) {
    return reinterpret_cast<void*>(LoadLibraryExA(windows_name, nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS));
}

void* symbol(void* library, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
}

void free_library(void* library) { FreeLibrary(static_cast<HMODULE>(library)); }

std::string error_text(int error) {
    char text[256];
    strerror_s(text, sizeof(text), error);
    return text;
}

} // namespace ninferno::sys

#endif
