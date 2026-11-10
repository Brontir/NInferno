#pragma once

// The helpers of the generic path's route selection that the launch code (generic_gemm.cpp) shares
// with the selector (generic_select.cpp): lifted out of anonymous namespaces so the selector can
// live in ninferno_ops_host with no CUDA runtime (#620).

#include "ops/generic/generic_gemm.h"

#include <cstdint>
#include <optional>

namespace ninferno::ops::detail::generic {

struct MmqPlaneBytes {
    int codes, high, scales;
};

bool i8_format(QType qtype);

bool i8_planes_aligned(const Weight& w);

bool i8_extent(const Weight& w);

bool w4a8_ws_admits(const Weight& w);

bool r8_call(const Weight& w, LinearPolicy policy, std::int32_t tokens);

MmqPlaneBytes mmq_plane_bytes(QType q);

bool vec_rule(const Weight& w, LinearPolicy policy, std::int32_t tokens);

bool bf16_mma_native();

int rmma_index(int warps_m, int warps_k, int m_tiles, int n_tiles, int cluster = 1);

bool w4a8_ws_extra(QType q);

int rules_r8(const Problem& p, std::int32_t tokens);

std::optional<Route> volta_hmma_route(const Problem& problem, const Weight& weight,
                                      std::int32_t tokens);

std::optional<Route> prefill_family_route(PrefillRoute f, const Problem& p, const Weight& w,
                                          std::int32_t tokens, bool prefill);

std::optional<Route> decode_family_route(DecodeRoute f, const Problem& p, const Weight& w,
                                         std::int32_t tokens, bool prefill);

} // namespace ninferno::ops::detail::generic
