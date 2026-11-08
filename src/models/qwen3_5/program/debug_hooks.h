#pragma once

#include <atomic>
#include <cstdint>

namespace ninferno::models::qwen3_5::debug {

// Test hook (tests/models/qwen3_5/test_engine_prefix_real.cpp): the next N materializations of a
// truncated endpoint are cancelled right after their source was prepared, the window between the
// source preparation and the start where an abort must leave the source whole. Zero in every
// other run; the Engine's cancellation flag is polled by another thread and cannot hit it.
inline std::atomic<std::uint32_t> abort_truncated_after_source_preparation{0};

[[nodiscard]] inline bool take_abort_truncated_after_source_preparation() noexcept {
    std::uint32_t remaining = abort_truncated_after_source_preparation.load();
    while (remaining != 0) {
        if (abort_truncated_after_source_preparation.compare_exchange_weak(remaining,
                                                                           remaining - 1U)) {
            return true;
        }
    }
    return false;
}

} // namespace ninferno::models::qwen3_5::debug
