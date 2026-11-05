#pragma once

// The generic path's admission of ggml's matrix and vector kernels (generic_gemm.cpp), which its
// calls of those kernels (generic_ggml.cpp) share.

#include "ops/generic/generic_gemm.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninferno::ops::detail::generic {

// The token band of the ggml matrix kernel's route: the verify widths from kMmqMinTokens
// (generic_gemm.h) to the W4A8 tiles' (17 tokens and up), which ran the A16 register MMA.
// The dense tuner times it there against the register MMA on the card and the model's own weight
// (generic_tune.cpp): no fixed rule holds across chips and formats. Against the heuristic's tile,
// serve C4: RTX 5090 ISTA IQ3_S +10 % at 16 tokens; Muse Q4_K_M -7..14 %, RTX 4080 SUPER ISTA
// -12 % at 8; RTX 5080 IQ3_S even. Below the band it loses to R8 (5090 C1 -11 %).
inline constexpr std::int32_t kMmqMaxTokens = 16;

// A prefill call on ggml's matrix kernel (mmq_prefill_wins) runs this many tokens at a time: its
// FP32 plane (a 27B SwiGLU's gate and up rows: 2 x 17408 x 512 x 4 bytes = 71 MB) instead of the
// chunk's, and 512 columns of 128 give every SM of a 16 GB card whole tiles (no stream-k fixup).
inline constexpr std::int32_t kMmqPrefillSlice = 512;

// What a call of ggml's kernels leaves for the arena's 256-byte alignment of its first buffer: the
// capacity bound counts it too, or a workspace sized to the bound ran the call A16 (a vector call
// of 200 rows x 8 tokens: 11520 bytes needed and given).
inline constexpr std::size_t kGgmlArenaSlack = 256;

// The ggml block formats ggml's matrix kernel reads (ggml_mmq.h) in the layout NInferno stores
// them: Q4_K and Q5_K blocks unchanged, the others as planes.
std::optional<ggml_mmq::Type> mmq_type(const Weight& w);

bool mmq_admits(const Weight& w, LinearPolicy policy);

bool mmq_prefill_wins(const Problem& p, const Weight& w);

// ggml's vector kernel: dp4a (sm_61 and up, below every device NInferno runs on), K whole blocks.
bool vec_admits(const Weight& w, LinearPolicy policy);

bool vec_call(const Weight& w, LinearPolicy policy, std::int32_t tokens);

std::size_t vec_workspace_bytes(std::int32_t k, std::int32_t tokens);

// The planes of ggml weight rows from `row` of the call's parent.
ggml_mmq::Planes ggml_rows_at(const GemmArgs& args, QType qtype, int bpr, int row);

// Whether a call of `rows` output rows and T tokens takes the ggml matrix kernel.
bool mmq_call(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens);

std::size_t mmq_workspace_bytes(std::int32_t k, std::int32_t plane_rows, std::int32_t tokens);

} // namespace ninferno::ops::detail::generic
