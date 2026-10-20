#pragma once

// The few POSIX calls the tests use, on Windows too: getpid, setenv/unsetenv, mkdtemp and an
// aligned allocation the matching free releases. On Linux these are the system's own.

#include <cstddef>
#include <cstdlib>

#ifdef _WIN32
#include <direct.h>
#include <malloc.h>
#include <process.h>

#include <cstring>
#include <random>

// getpid: <process.h> declares it under its POSIX name.

inline int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite && std::getenv(name) != nullptr) { return 0; }
    return _putenv_s(name, value) == 0 ? 0 : -1;
}

inline int unsetenv(const char* name) { return _putenv_s(name, "") == 0 ? 0 : -1; }

// The last six characters of `name_template` (XXXXXX) become random; the directory is created.
inline char* mkdtemp(char* name_template) {
    const std::size_t length = std::strlen(name_template);
    if (length < 6 || std::strcmp(name_template + length - 6, "XXXXXX") != 0) { return nullptr; }
    static const char letters[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    std::random_device device;
    for (int attempt = 0; attempt < 100; ++attempt) {
        for (std::size_t i = length - 6; i < length; ++i) {
            name_template[i] = letters[device() % (sizeof(letters) - 1)];
        }
        if (_mkdir(name_template) == 0) { return name_template; }
    }
    return nullptr;
}
#else
#include <unistd.h>
#endif

namespace ninferno::test {

// std::aligned_alloc / std::free, which MSVC lacks; release with aligned_free.
inline void* aligned_alloc(std::size_t alignment, std::size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(bytes, alignment);
#else
    return std::aligned_alloc(alignment, bytes);
#endif
}

inline void aligned_free(void* pointer) {
#ifdef _WIN32
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}

} // namespace ninferno::test
