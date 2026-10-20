#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"

#include <cstdint>

namespace ninferno::models::qwen3_5 {

inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
// Qwen 3.5's MTP layer: its decode batch runs at most five drafts (text MTP forward width 6).
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;
// Its forward over a round's verified columns runs a copy round's too (up to 15 copied drafts).
inline constexpr std::uint32_t kMaximumMtpVerifyWidth    = 16;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 15;

} // namespace ninferno::models::qwen3_5

namespace ninferno::models::qwen3_5::detail {
using ContractAccess = RuntimeContractAccess;

[[nodiscard]] inline std::uint32_t backend_frontier_at(SpeculativeBackend backend,
                                                       std::uint32_t main_frontier) noexcept {
    if (backend == SpeculativeBackend::Mtp) { return main_frontier == 0 ? 0U : main_frontier - 1U; }
    return backend == SpeculativeBackend::DFlash ? main_frontier : 0U;
}

} // namespace ninferno::models::qwen3_5::detail
