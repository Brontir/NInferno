// core/sys.h, the same checks on Linux and Windows: descriptors (offset I/O, size, truncate, sequential
// writes, sync, locks), mappings, unique names, random bytes, aligned memory, the 128-bit helpers of
// core/checked.h, and the process's facts. Files in the system temporary directory; no GPU.

#include "core/checked.h"
#include "core/command_line.h"
#include "core/sys.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#endif

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << std::endl;
        ++failures;
    }
}

namespace fs  = std::filesystem;
namespace sys = ninferno::sys;

void test_descriptors(const fs::path& dir) {
    const fs::path file = dir / "data.bin";
    int fd              = sys::open(file, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    check(fd >= 0, "create");
    check(sys::open(file, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0644) < 0 && errno == EEXIST,
          "O_EXCL refuses an existing file with EEXIST");
    const std::string text = "hello, positional world";
    check(sys::pwrite(fd, text.data(), text.size(), 4096) == static_cast<std::int64_t>(text.size()),
          "pwrite past the end");
    check(sys::file_size(fd) == 4096 + static_cast<std::int64_t>(text.size()), "size after pwrite");
    std::string back(text.size(), '\0');
    check(sys::pread(fd, back.data(), back.size(), 4096) == static_cast<std::int64_t>(back.size()) &&
              back == text,
          "pread what pwrite wrote");
    check(sys::pread(fd, back.data(), back.size(), 1 << 20) == 0, "pread past the end reads 0");
    check(sys::truncate(fd, 3ULL << 30) == 0 && sys::file_size(fd) == (3LL << 30),
          "truncate beyond 2 GiB (64-bit offsets)");
    check(sys::pwrite(fd, "x", 1, (3LL << 30) - 1) == 1, "pwrite at a 64-bit offset");
    check(sys::truncate(fd, 8192) == 0 && sys::file_size(fd) == 8192, "truncate down");
    check(sys::seek(fd, 0, SEEK_END) == 8192, "seek to the end");
    check(sys::write(fd, "tail", 4) == 4 && sys::file_size(fd) == 8196, "sequential write appends");
    check(sys::fsync(fd) == 0 && sys::fdatasync(fd) == 0, "sync");
    check(sys::lock(fd, false) == 0, "an exclusive lock");
    const int other = sys::open(file, O_RDONLY | O_CLOEXEC);
    check(other >= 0 && sys::lock(other, false) != 0 && errno == EWOULDBLOCK,
          "a second descriptor cannot take it (EWOULDBLOCK)");
    sys::close(other);
    sys::close(fd);
    check(sys::open(dir / "missing", O_RDONLY) < 0 && errno == ENOENT, "a missing file is ENOENT");

    // O_DIRECT: aligned blocks, read back through the cache-free descriptor.
    const fs::path direct_file = dir / "direct.bin";
    fd = sys::open(direct_file, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    std::vector<char> pattern(8192);
    for (std::size_t i = 0; i < pattern.size(); ++i) { pattern[i] = static_cast<char>(i * 7); }
    check(sys::pwrite(fd, pattern.data(), pattern.size(), 0) == 8192, "write the block file");
    sys::close(fd);
    const int direct = sys::open(direct_file, O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (direct >= 0) { // tmpfs has no O_DIRECT (EINVAL): nothing to check there
        auto* block = static_cast<char*>(sys::aligned_alloc(4096, 4096));
        check(sys::pread(direct, block, 4096, 4096) == 4096 &&
                  std::memcmp(block, pattern.data() + 4096, 4096) == 0,
              "an aligned direct read");
        sys::aligned_free(block);
        sys::close(direct);
    }
}

void test_mappings(const fs::path& dir) {
    const fs::path file = dir / "map.bin";
    int fd              = sys::open(file, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    check(sys::truncate(fd, 1 << 16) == 0, "size the mapped file");
    auto* writable = static_cast<char*>(sys::map(fd, 1 << 16, sys::Map::ReadWriteShared));
    check(writable != nullptr, "a writable shared mapping");
    if (writable) {
        std::memcpy(writable + 100, "mapped", 6);
        sys::advise(writable, 1 << 16, sys::Advice::DontNeed);
        check(sys::unmap(writable, 1 << 16) == 0, "unmap");
    }
    char back[6] = {};
    check(sys::pread(fd, back, 6, 100) == 6 && std::memcmp(back, "mapped", 6) == 0,
          "stores through the mapping reach the file");
    auto* shared = static_cast<const char*>(sys::map(fd, 1 << 16, sys::Map::ReadShared));
    auto* copy   = static_cast<char*>(sys::map(fd, 1 << 16, sys::Map::ReadPrivate));
    check(shared && copy && std::memcmp(shared + 100, "mapped", 6) == 0, "read mappings");
    if (copy) { sys::unmap(copy, 1 << 16); }
    // The converter's consume mode: holes punched through a second descriptor while the checkpoint is
    // still mapped; the file keeps its size and reads zeros there.
    const int punch = sys::open(file, O_WRONLY | O_CLOEXEC);
    check(punch >= 0 && sys::punch_hole(punch, 0, 1 << 15) == 0,
          "punch a hole in a mapped file: " + sys::error_text(errno));
    sys::close(punch);
    check(sys::file_size(fd) == (1 << 16), "the punched file keeps its size");
    char zeros[6] = {1, 1, 1, 1, 1, 1};
    check(sys::pread(fd, zeros, 6, 100) == 6 && std::memcmp(zeros, "\0\0\0\0\0\0", 6) == 0,
          "the hole reads zeros");
    if (shared) { sys::unmap(const_cast<char*>(shared), 1 << 16); }
    sys::close(fd);
    check(sys::page_size() >= 4096, "page size");
}

void test_names_and_memory(const fs::path& dir) {
    std::string name = (dir / ".out.XXXXXX.tmp").string();
    const int fd     = sys::create_unique(name, 4);
    check(fd >= 0 && name.find("XXXXXX") == std::string::npos && name.ends_with(".tmp") &&
              fs::exists(name),
          "create_unique fills the six X and keeps the suffix");
    sys::close(fd);
    unsigned char a[32] = {}, b[32] = {};
    check(sys::random_bytes(a, sizeof(a)) && sys::random_bytes(b, sizeof(b)) &&
              std::memcmp(a, b, sizeof(a)) != 0,
          "random bytes differ");
    void* aligned = sys::aligned_alloc(4096, 1 << 20);
    check(aligned != nullptr && reinterpret_cast<std::uintptr_t>(aligned) % 4096 == 0,
          "aligned allocation");
    sys::aligned_free(aligned);
    check(sys::pid() > 0, "pid");
    check(!sys::self_exe().empty() && fs::exists(sys::self_exe()), "the executable's own path");
    check(!sys::home_dir().empty(), "a home directory");
#ifndef _WIN32
    // A service started without HOME still has the account's home (where the account has a
    // passwd entry: a container run under an arbitrary uid has none).
    if (const char* home = std::getenv("HOME");
        home != nullptr && ::getpwuid(::getuid()) != nullptr) {
        const std::string saved = home;
        ::unsetenv("HOME");
        check(!sys::home_dir().empty(), "a home directory without HOME");
        ::setenv("HOME", saved.c_str(), 1);
    }
#endif
    std::tm tm{};
    check(sys::gmtime(0, tm) && tm.tm_year == 70 && tm.tm_hour == 0, "gmtime of the epoch");
    check(sys::error_text(ENOENT).size() > 3, "error text");
    sys::sync_directory(dir);
}

void test_wide_arithmetic() {
    using ninferno::mul_overflow;
    using ninferno::mul_wide;
    using ninferno::saturating_u64;
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    const auto w       = mul_wide(max, max); // (2^64-1)^2 = 2^128 - 2^65 + 1
    check(w.high == max - 1 && w.low == 1, "mul_wide of the largest words");
    check(mul_wide(1ULL << 32, 1ULL << 32).high == 1 && mul_wide(1ULL << 32, 1ULL << 32).low == 0,
          "mul_wide carries into the high word");
    check(saturating_u64({0, 5}, {0, 7}) == 12, "a sum that fits");
    check(saturating_u64({0, max}, {0, 1}) == max, "a sum that carries saturates");
    check(saturating_u64({1, 0}, {0, 0}) == max, "a high word saturates");
    std::uint64_t out = 0;
    check(!mul_overflow(std::uint64_t{3}, std::uint64_t{5}, &out) && out == 15, "mul_overflow fits");
    check(mul_overflow(max, std::uint64_t{2}, &out), "mul_overflow detects 64-bit overflow");
    std::int64_t signed_out = 0;
    check(!mul_overflow(std::int64_t{-3}, std::int64_t{5}, &signed_out) && signed_out == -15,
          "signed product");
    std::uint64_t unsigned_out = 0;
    check(mul_overflow(std::int64_t{-3}, std::int64_t{5}, &unsigned_out),
          "a negative product does not fit an unsigned result");
}

} // namespace

// Windows command-line quoting: what CommandLineToArgvW reads back is the argument itself.
void test_command_line_quoting() {
    using ninferno::windows_quoted_argument;
    const auto q = [](std::string_view arg) { return windows_quoted_argument(arg); };
    check(q("plain") == "plain", "a plain argument stays unquoted");
    check(q("") == "\"\"", "an empty argument is a quoted empty string");
    check(q("a b") == "\"a b\"", "a space quotes the argument");
    check(q(R"(C:\dir\)") == R"(C:\dir\)", "backslashes without a quote stay as they are");
    check(q(R"(C:\a dir\)") == R"("C:\a dir\\")",
          "a trailing backslash doubles before the closing quote");
    check(q(R"(a"b)") == R"("a\"b")", "a quote is escaped");
    // A backslash before a quote: two backslashes and the escaped quote (2N+1), never a quote
    // that closes the argument (an argument injection through a model name).
    check(q(R"(x\" --host 0.0.0.0 y)") == R"("x\\\" --host 0.0.0.0 y")",
          "a backslash and a quote stay inside the argument");
    check(windows_quoted_argument(std::wstring_view(L"a\\\"b")) == L"\"a\\\\\\\"b\"",
          "the wide form quotes the same way");
}

// --version is a request only as the first argument (a model named --version is not one).
void test_version_request() {
    char program[] = "ninferno";
    char version[] = "--version";
    char model[]   = "model.ninferno";
    char* alone[]  = {program, version};
    char* later[]  = {program, model, version};
    check(ninferno::version_requested(2, alone), "--version first is a version request");
    check(!ninferno::version_requested(3, later), "--version after a model is not one");
    check(!ninferno::version_requested(1, alone), "no arguments are not one");
}

int main() {
    const fs::path dir = fs::temp_directory_path() / ("ninferno_sys_test_" + std::to_string(sys::pid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    test_command_line_quoting();
    test_version_request();
    test_descriptors(dir);
    test_mappings(dir);
    test_names_and_memory(dir);
    test_wide_arithmetic();
    std::error_code ignored;
    fs::remove_all(dir, ignored);
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
