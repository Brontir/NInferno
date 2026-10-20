#pragma once

#include "core/tensor.h"
#include "ninferno/types.h" // kMaximumConcurrency

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include <cuda_runtime.h>

namespace ninferno::models::qwen3_5::detail {

// Per-row flags of a round whose final hidden held a NaN or an infinity. The kernels write mapped
// host words, so reading them costs nothing beyond the round's own synchronization. A flagged
// row's request fails alone (RequestErrorKind::NumericalFailure); the other rows go on.
class NumericStatus {
public:
    enum class Launch : std::size_t { Speculative = 0, Plain = 1, Prefill = 2 };

    NumericStatus();
    ~NumericStatus();
    NumericStatus(const NumericStatus&)            = delete;
    NumericStatus& operator=(const NumericStatus&) = delete;

    // Host side, before a round's launches; the device must not be writing the words.
    void clear() noexcept;
    // Flags row r of `launch` when values[:, c < valid_columns[r], r] is not finite.
    void check(Launch launch, const Tensor& values, const std::int32_t* valid_columns,
               cudaStream_t stream);
    // After the round is synchronized.
    [[nodiscard]] bool faulted(Launch launch, std::size_t row) const noexcept;

private:
    [[nodiscard]] std::size_t offset(Launch launch) const noexcept {
        return static_cast<std::size_t>(launch) * kMaximumConcurrency;
    }

    std::uint32_t* host_   = nullptr;
    std::uint32_t* device_ = nullptr;
    // NINFERNO_DEBUG_NAN_FILE: when the named file exists, the next check removes it and writes a
    // NaN into its first row first (tests of the failure path).
    std::string inject_file_;
};

} // namespace ninferno::models::qwen3_5::detail
