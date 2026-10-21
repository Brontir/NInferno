#pragma once

// A fresh directory of a test's own under the system's temporary directory. A fixed name (or one
// from an unseeded std::rand) is shared by every run on the machine: a run by another user leaves
// it behind unwritable, and two runs at once write over each other's files.

#include <cstdint>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

// <temp>/<stem>-<16 random hex digits>, newly created (never one that already existed: creation
// is the claim, retried under another name). Throws when none can be created.
[[nodiscard]] inline std::filesystem::path unique_temp_dir(std::string_view stem) {
    std::random_device entropy;
    std::mt19937_64 next((std::uint64_t{entropy()} << 32) ^ entropy());
    const std::filesystem::path base = std::filesystem::temp_directory_path();
    static constexpr char kHex[]     = "0123456789abcdef";
    for (int attempt = 0; attempt < 100; ++attempt) {
        std::string name(stem);
        name += '-';
        for (std::uint64_t bits = next(), i = 0; i < 16; ++i, bits >>= 4) {
            name += kHex[bits & 0xF];
        }
        const std::filesystem::path dir = base / name;
        std::error_code error;
        if (std::filesystem::create_directory(dir, error)) { return dir; }
    }
    throw std::runtime_error("unique_temp_dir: no new directory under " + base.string());
}

// A unique_temp_dir removed with everything under it when the scope ends, however it ends.
class TempDir {
public:
    explicit TempDir(std::string_view stem) : path_(unique_temp_dir(stem)) {}

    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    TempDir(const TempDir&)            = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};
