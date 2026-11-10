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

bool i8_format(QType qtype);

bool i8_planes_aligned(const Weight& w);

bool i8_extent(const Weight& w);

bool w4a8_ws_admits(const Weight& w);

MmqPlaneBytes mmq_plane_bytes(QType q);

bool w4a8_ws_extra(QType q);

// The R8 decode table of the nearest measured board (null where none is near enough) with that
// board's distance to this device.
struct R8TableLookup {
    const R8DecodeTable* table = nullptr;
    double distance            = 0; // board_distance of its board to this device
};

// The current device's R8 decode table.
R8TableLookup r8_decode_table();

int rules_r8(const Problem& p, std::int32_t tokens);

std::optional<Route> prefill_family_route(PrefillRoute f, const Problem& p, const Weight& w,
                                          std::int32_t tokens, bool prefill);

std::optional<Route> decode_family_route(DecodeRoute f, const Problem& p, const Weight& w,
                                         std::int32_t tokens, bool prefill);



} // namespace ninferno::ops::detail::generic
