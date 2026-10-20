#pragma once

// The operating system calls NInferno makes outside the C++ library: files by descriptor (offset
// reads and writes, O_DIRECT, sync, locks), file mappings, the process (its id, executable,
// terminal), time, and shared libraries. On Linux each is the POSIX call itself, inline; on
// Windows sys_win.cpp does the same with Win32 (O_DIRECT is FILE_FLAG_NO_BUFFERING, with the same
// 4096-byte alignment contract; mappings are file views; advice is a no-op; a directory needs no
// sync). Errors come back the POSIX way: -1 (or nullptr) and errno.
//
// Flags are POSIX's O_* names (fcntl.h); on Windows O_CLOEXEC is 0 (handles are not inherited
// unless asked), and O_DIRECT and O_DIRECTORY are defined here.

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <string>

#include <fcntl.h>

#ifdef _WIN32
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_DIRECT
#define O_DIRECT 0x40000000
#endif
#ifndef O_DIRECTORY
#define O_DIRECTORY 0x20000000
#endif
#else
#include <dlfcn.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#endif

namespace ninferno::sys {

enum class Map {
    ReadShared,      // PROT_READ, MAP_SHARED | MAP_NORESERVE: a file read in place
    ReadPrivate,     // PROT_READ, MAP_PRIVATE
    ReadWriteShared, // PROT_READ | PROT_WRITE, MAP_SHARED: writes reach the file
};

enum class Advice { Normal, WillNeed, DontNeed };

#ifdef _WIN32

int open(const std::filesystem::path& path, int flags, int mode = 0644);
int close(int fd);
std::int64_t pread(int fd, void* buffer, std::size_t bytes, std::int64_t offset);
std::int64_t pwrite(int fd, const void* buffer, std::size_t bytes, std::int64_t offset);
std::int64_t file_size(int fd);
std::int64_t write(int fd, const void* buffer, std::size_t bytes);
std::int64_t seek(int fd, std::int64_t offset, int whence);
int truncate(int fd, std::int64_t length);
int fsync(int fd);
int fdatasync(int fd);
int lock(int fd, bool wait, bool shared = false);
void* map(int fd, std::size_t bytes, Map mode);
int unmap(void* base, std::size_t bytes);
void advise(void* base, std::size_t bytes, Advice advice);
void advise_file(int fd, std::int64_t offset, std::int64_t bytes, Advice advice);
void start_writeback(int fd);
std::size_t page_size();
long long pid();
std::filesystem::path self_exe();
bool gmtime(std::time_t t, std::tm& out);
bool localtime(std::time_t t, std::tm& out);
bool stderr_is_terminal();
int terminal_columns();
void* load_library(const char* linux_name, const char* windows_name);
void* symbol(void* library, const char* name);
void free_library(void* library);
std::string error_text(int error);

#else

inline int open(const std::filesystem::path& path, int flags, int mode = 0644) {
    return ::open(path.c_str(), flags, mode);
}
inline int close(int fd) { return ::close(fd); }
inline std::int64_t pread(int fd, void* buffer, std::size_t bytes, std::int64_t offset) {
    return ::pread(fd, buffer, bytes, static_cast<off_t>(offset));
}
inline std::int64_t pwrite(int fd, const void* buffer, std::size_t bytes, std::int64_t offset) {
    return ::pwrite(fd, buffer, bytes, static_cast<off_t>(offset));
}
inline std::int64_t file_size(int fd) {
    struct stat status {};
    return ::fstat(fd, &status) == 0 ? static_cast<std::int64_t>(status.st_size) : -1;
}
// write(2) at the descriptor's position, and lseek(2).
inline std::int64_t write(int fd, const void* buffer, std::size_t bytes) { return ::write(fd, buffer, bytes); }
inline std::int64_t seek(int fd, std::int64_t offset, int whence) {
    return ::lseek(fd, static_cast<off_t>(offset), whence);
}
inline int truncate(int fd, std::int64_t length) { return ::ftruncate(fd, static_cast<off_t>(length)); }
inline int fsync(int fd) { return ::fsync(fd); }
inline int fdatasync(int fd) { return ::fdatasync(fd); }
// A lock on the whole file (flock), exclusive or shared: `wait` blocks for it, else -1 with
// EWOULDBLOCK. Closing the descriptor releases it.
inline int lock(int fd, bool wait, bool shared = false) {
    return ::flock(fd, (shared ? LOCK_SH : LOCK_EX) | (wait ? 0 : LOCK_NB));
}
inline void* map(int fd, std::size_t bytes, Map mode) {
    const int prot  = mode == Map::ReadWriteShared ? PROT_READ | PROT_WRITE : PROT_READ;
    const int flags = mode == Map::ReadShared    ? MAP_SHARED | MAP_NORESERVE
                      : mode == Map::ReadPrivate ? MAP_PRIVATE
                                                 : MAP_SHARED;
    void* base = ::mmap(nullptr, bytes, prot, flags, fd, 0);
    return base == MAP_FAILED ? nullptr : base;
}
inline int unmap(void* base, std::size_t bytes) { return ::munmap(base, bytes); }
inline void advise(void* base, std::size_t bytes, Advice advice) {
    (void)::madvise(base, bytes, advice == Advice::WillNeed   ? MADV_WILLNEED
                                 : advice == Advice::DontNeed ? MADV_DONTNEED
                                                              : MADV_NORMAL);
}
inline void advise_file(int fd, std::int64_t offset, std::int64_t bytes, Advice advice) {
    (void)::posix_fadvise(fd, static_cast<off_t>(offset), static_cast<off_t>(bytes),
                          advice == Advice::WillNeed   ? POSIX_FADV_WILLNEED
                          : advice == Advice::DontNeed ? POSIX_FADV_DONTNEED
                                                       : POSIX_FADV_NORMAL);
}
// Starts writing the file's dirty pages back without waiting (sync_file_range).
inline void start_writeback(int fd) { (void)::sync_file_range(fd, 0, 0, SYNC_FILE_RANGE_WRITE); }
inline std::size_t page_size() { return static_cast<std::size_t>(::sysconf(_SC_PAGESIZE)); }
inline long long pid() { return static_cast<long long>(::getpid()); }
inline std::filesystem::path self_exe() {
    std::error_code ec;
    auto path = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::filesystem::path() : path;
}
inline bool gmtime(std::time_t t, std::tm& out) { return ::gmtime_r(&t, &out) != nullptr; }
inline bool localtime(std::time_t t, std::tm& out) { return ::localtime_r(&t, &out) != nullptr; }
inline bool stderr_is_terminal() { return ::isatty(STDERR_FILENO) == 1; }
// The terminal's width on stderr; 0 when it is not a terminal.
inline int terminal_columns() {
    winsize size{};
    return ::ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0 ? size.ws_col : 0;
}
inline void* load_library(const char* linux_name, const char* /*windows_name*/) {
    return ::dlopen(linux_name, RTLD_NOW | RTLD_LOCAL);
}
inline void* symbol(void* library, const char* name) { return ::dlsym(library, name); }
inline void free_library(void* library) { ::dlclose(library); }
inline std::string error_text(int error) { return std::strerror(error); }

#endif

// Frees the file's blocks in [offset, offset + bytes) and keeps its size (fallocate PUNCH_HOLE |
// KEEP_SIZE; on Windows the file turns sparse and the range is zeroed); -1 and errno.
int punch_hole(int fd, std::int64_t offset, std::int64_t bytes);
// A new file from `name_template`, whose six X before its last `suffix_length` characters become
// random (mkostemps): the descriptor (read-write), `name_template` holding the name; -1 and errno.
int create_unique(std::string& name_template, int suffix_length);
// `bytes` random bytes from the operating system; false when it cannot give them.
bool random_bytes(void* buffer, std::size_t bytes);
// Memory aligned to `alignment` (a power of two), freed only by aligned_free; nullptr when none.
void* aligned_alloc(std::size_t alignment, std::size_t bytes);
void aligned_free(void* data) noexcept;

// The user's home: $HOME (on Windows %USERPROFILE%, where huggingface_hub keeps its cache too);
// without HOME on POSIX the account's home directory; empty when neither is known.
std::filesystem::path home_dir();

// NInferno's own cache directory (the update check's state, the tuners' choices):
// %LOCALAPPDATA%\NInferno on Windows; $XDG_CACHE_HOME/ninferno, else ~/.cache/ninferno; empty
// when none is known.
std::filesystem::path cache_dir();

// An environment variable, "" when unset. On Windows it is read wide and returned as UTF-8, so a
// non-ASCII directory in HF_HOME or a cache override reaches the narrow (UTF-8) path calls intact.
std::string env(const char* name);

// Makes the directory's entries durable (a rename into it, a new file): fsync of the directory on
// POSIX; nothing on Windows, where NTFS journals the metadata itself.
void sync_directory(const std::filesystem::path& dir);

} // namespace ninferno::sys
