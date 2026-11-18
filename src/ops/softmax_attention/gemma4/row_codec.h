#pragma once

// The codec of Gemma4KvLayer's one-byte rows (fp8_row.cuh), for host and device code alike: one
// tree builds one architecture (NINFERNO_BUILD_ARCH), so the planes the host allocates and checks
// hold the codes its kernels write and read.
//   - E4M3FN (DType::FP8_E4M3FN) on sm_89 and newer, where the conversion is in hardware, and on
//     Volta;
//   - INT8 (DType::I8) on sm_75 to sm_88 (#619 steps 9-10): there cuda_fp8.h widens E4M3 in
//     software, while an int8 code widens with one I2F and one FMUL.
// Every reader and writer takes the build's codec, the small-row decode of volta_decode.cu
// included (Turing runs it too: gemma4_volta_decode_takes).

#include "core/dtype.h"

namespace ninferno::ops {

#if defined(NINFERNO_BUILD_ARCH)
inline constexpr bool kGemma4RowInt8 = NINFERNO_BUILD_ARCH >= 75 && NINFERNO_BUILD_ARCH < 89;
#else
inline constexpr bool kGemma4RowInt8 = false;
#endif

// Whether the release engine a card of compute capability `cc` runs (one tree per architecture,
// apps/dispatch) writes int8 row codes: sm_75 to sm_88, the INT8 trees above. A card property for
// the planner, which may run in a tree other than the card's (a host plan for another board);
// the tree's own codec is the same rule at its architecture (the static_assert).
[[nodiscard]] constexpr bool gemma4_rows_int8_on(int cc) noexcept { return cc >= 75 && cc < 89; }
#if defined(NINFERNO_BUILD_ARCH)
static_assert(kGemma4RowInt8 == gemma4_rows_int8_on(NINFERNO_BUILD_ARCH),
              "the build's row codec must be the card rule at its own architecture");
#endif

// The dtype of a one-byte row plane in this build.
inline constexpr DType kGemma4RowDtype = kGemma4RowInt8 ? DType::I8 : DType::FP8_E4M3FN;

// The largest code magnitude: scale = absmax(row) / kGemma4RowCodeMax.
inline constexpr float kGemma4RowCodeMax = kGemma4RowInt8 ? 127.0f : 448.0f;

} // namespace ninferno::ops
