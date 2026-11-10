#pragma once

// The helpers of the generic path's route selection that the launch code (generic_gemm.cpp) shares
// with the selector (generic_select.cpp): lifted out of anonymous namespaces so the selector can
// be its own translation unit.

#include "ops/generic/generic_gemm.h"
#include "ops/generic/r8_decode_table.h"

#include <cstdint>
#include <optional>

namespace ninferno::ops::detail::generic {

// The bytes per block of each plane: the code plane (the whole block where blocks are unchanged),
// the high and the scale plane (core/weight.h ggml_plane_bytes).
struct MmqPlaneBytes {
    int codes, high, scales;
};

// Whether the format has the I8 tiles' codec (format_traits qx_g64).
bool i8_format(QType qtype);

// Whether the weight's code and scale planes sit on 16-byte boundaries and its high plane (Q5 only;
// Q4 has none) on an 8-byte one, as the I8 tiles read them.
bool i8_planes_aligned(const Weight& w);

// Whether K and the padded row count are multiples of 512, the I8 tiles' extent.
bool i8_extent(const Weight& w);

// Whether the producer-warp W4A8 tile can read this weight: its format, and IQ3_XXS's sign plane
// on an 8-byte boundary.
bool w4a8_ws_admits(const Weight& w);

// The bytes per block of each plane of format `q`.
MmqPlaneBytes mmq_plane_bytes(QType q);

// Whether the producer-warp tile also replaces the classic one for format `q` on this device
// (sm_89 / sm_120, where a measured choice names it).
bool w4a8_ws_extra(QType q);

// The R8 decode table of the nearest measured board (null where none is near enough) with that
// board's distance to this device.
struct R8TableLookup {
    const R8DecodeTable* table = nullptr;
    double distance            = 0; // board_distance of its board to this device
};

// The current device's R8 decode table.
R8TableLookup r8_decode_table();

// The R8 schedule index the rules give problem `p` at `tokens` tokens, by format and device.
int rules_r8(const Problem& p, std::int32_t tokens);

// The route a pinned prefill family gives a call (17 tokens or more); nullopt where it cannot.
std::optional<Route> prefill_family_route(PrefillRoute f, const Problem& p, const Weight& w,
                                          std::int32_t tokens, bool prefill);

// The route a pinned decode family gives a call (1..16 tokens); nullopt where it cannot.
std::optional<Route> decode_family_route(DecodeRoute f, const Problem& p, const Weight& w,
                                         std::int32_t tokens, bool prefill);

} // namespace ninferno::ops::detail::generic
