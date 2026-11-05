#pragma once

// What the kernel warm-up's width loop measured (model_instance.cpp), kept across starts. The loop
// runs one prefill of every kernel class width up to the chunk (warmup_widths.h) for two lasting
// effects only: the Device memory the kernels' code takes once loaded (lazy module loading) and
// the per-thread stack limit, which the driver raises to the largest frame launched. Both depend
// on what runs, not on the free memory: the same binary, GPU, driver, model, plan and route data
// launch the same kernels. A later start with the same inputs sets the stack limit it ended at,
// reserves the bytes it took, and skips the loop (on an RTX 4090, Gemma 4 12B: a 22.9 s start
// takes 14.1 s; the first request then loads the kernels' code, a few ms). The decode-graph probe
// still runs, and the plan is still checked on the memory free then.
//
// The bytes are a difference of the device's free memory, which another process on the GPU can
// move while the loop runs. So one loop writes a candidate only: a record is used once a second
// loop with the same inputs measured the same bytes (warm_up_records_agree), and a loop that does
// not agree replaces the candidate.
//
// The key is the SHA-256 of a text naming every such input (warm_up_key_text in
// model_instance.cpp); one file per key under cache_dir()/warm-up. NINFERNO_WARMUP_CACHE=0 (off,
// no, false) neither reads nor writes it. A record that cannot be read is a miss.

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace ninferno::runtime {

struct WarmUpRecord {
    std::size_t stack_limit_bytes = 0; // the stack limit after the width loop
    // The Device memory the width requests took and kept beyond the small probe's (the probe is
    // built either way): their kernels' code and the stack's local memory for larger frames.
    std::size_t loop_bytes        = 0;
    std::string stack_limit_source; // the step that raised the stack limit last
    bool confirmed = false;         // a second loop measured the same
};

// False when NINFERNO_WARMUP_CACHE says 0, off, no or false.
[[nodiscard]] bool warm_up_cache_enabled();

// The record of `key_text` (its SHA-256 names the file), confirmed or a candidate; nullopt when
// there is none, it cannot be read, or the cache is off.
[[nodiscard]] std::optional<WarmUpRecord> read_warm_up_record(std::string_view key_text);

// Two loops measured the same: the same stack limit, and bytes within max(32 MiB, 2 %).
[[nodiscard]] bool warm_up_records_agree(const WarmUpRecord& a, const WarmUpRecord& b);

// Writes the record of `key_text`, best effort: a cache that cannot be written is skipped.
void write_warm_up_record(std::string_view key_text, const WarmUpRecord& record);

// The directory the records live in (cache_dir()/warm-up; empty without a cache directory).
[[nodiscard]] std::filesystem::path warm_up_cache_dir();

} // namespace ninferno::runtime
