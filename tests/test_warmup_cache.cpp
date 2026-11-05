// The kernel warm-up's record (src/runtime/engine/warmup_cache.h): a written record reads back,
// another key or a hand-edited key is a miss, a file of another schema or a broken one is a miss,
// two loops agree within max(32 MiB, 2 %) on the same stack limit, and NINFERNO_WARMUP_CACHE=0
// neither reads nor writes. CPU.
#include "runtime/engine/warmup_cache.h"

#include "support/posix.h"
#include "support/temp_dir.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using ninferno::runtime::WarmUpRecord;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

} // namespace

int main() {
    const auto root = unique_temp_dir("ninferno-warm-up-cache");
    // cache_dir() reads LOCALAPPDATA first on Windows: both point at the test's own directory,
    // never at the user's cache.
    ::setenv("XDG_CACHE_HOME", root.string().c_str(), 1);
#ifdef _WIN32
    ::setenv("LOCALAPPDATA", root.string().c_str(), 1);
#endif
    ::unsetenv("NINFERNO_WARMUP_CACHE");
    namespace rt = ninferno::runtime;

    const WarmUpRecord written{.stack_limit_bytes  = 1024,
                               .loop_bytes         = std::size_t{300} << 20,
                               .stack_limit_source = "a prefill of 512 tokens",
                               .confirmed          = true};
    rt::write_warm_up_record("key a\n", written);
    const auto read = rt::read_warm_up_record("key a\n");
    check(read.has_value(), "a written record reads back");
    if (read) {
        check(read->stack_limit_bytes == 1024 && read->loop_bytes == written.loop_bytes &&
                  read->stack_limit_source == written.stack_limit_source && read->confirmed,
              "every field reads back");
    }
    check(!rt::read_warm_up_record("key b\n"), "another key is a miss");
    for (const auto& entry : std::filesystem::directory_iterator(rt::warm_up_cache_dir())) {
        check(entry.path().extension() == ".json", "no partial file is left behind");
    }

    // The file of key a, rewritten with another key text inside: a collision or an edit.
    std::filesystem::path file;
    for (const auto& entry : std::filesystem::directory_iterator(rt::warm_up_cache_dir())) {
        file = entry.path();
    }
    {
        std::ifstream in(file);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const auto at = text.find("key a");
        text.replace(at, 5, "key c");
        std::ofstream(file) << text;
    }
    check(!rt::read_warm_up_record("key a\n"), "a record whose key text differs is a miss");
    std::ofstream(file) << "{\"schema\": \"ninferno-warm-up/1\"}";
    check(!rt::read_warm_up_record("key a\n"), "another schema is a miss");
    std::ofstream(file) << "{ broken";
    check(!rt::read_warm_up_record("key a\n"), "a broken file is a miss");

    WarmUpRecord other = written;
    other.loop_bytes += std::size_t{20} << 20;
    check(rt::warm_up_records_agree(written, other), "20 MiB apart agree");
    other.loop_bytes = written.loop_bytes + (std::size_t{40} << 20);
    check(!rt::warm_up_records_agree(written, other), "40 MiB apart of 300 MiB do not");
    WarmUpRecord large = written;
    large.loop_bytes   = std::size_t{4} << 30;
    other              = large;
    other.loop_bytes += std::size_t{60} << 20;
    check(rt::warm_up_records_agree(large, other), "60 MiB of 4 GiB agree (2 %)");
    other = written;
    other.stack_limit_bytes = 2048;
    check(!rt::warm_up_records_agree(written, other), "another stack limit does not agree");

    ::setenv("NINFERNO_WARMUP_CACHE", "0", 1);
    check(!rt::warm_up_cache_enabled(), "0 turns the cache off");
    rt::write_warm_up_record("key d\n", written);
    ::unsetenv("NINFERNO_WARMUP_CACHE");
    check(!rt::read_warm_up_record("key d\n"), "nothing was written while off");
    rt::write_warm_up_record("key d\n", written);
    ::setenv("NINFERNO_WARMUP_CACHE", "off", 1);
    check(!rt::read_warm_up_record("key d\n"), "nothing is read while off");

    std::error_code error;
    std::filesystem::remove_all(root, error);
    if (failures != 0) { return 1; }
    std::cout << "warm-up cache: ok\n";
    return 0;
}
