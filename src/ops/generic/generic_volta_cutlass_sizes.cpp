// The sizes of Volta's CUTLASS prefill (generic_volta_cutlass.cu): the weight rows of a decode
// chunk and the workspace a chunk takes. They decide whether the route runs a call
// (volta_cutlass_call_runs, generic_select.cpp) and what the call reserves, so they are host code,
// compiled with the selector (ninferno_ops_host_objects, #620 step 13): a host-only build of an
// sm_70 tree with CUTLASS answers as the device build does. NINFERNO_VOLTA_CUTLASS is set on this
// file where it is set on the .cu (src/ops/generic/sources.cmake); without it the route is
// compiled out and every size is 0.
#include "ops/generic/generic_volta_cutlass_sizes.h"

#include "core/layout.h"
#include "ops/generic/generic_gemm_internal.h"

#include <algorithm>
#include <cstdint>

namespace ninferno::ops::detail::generic {

#ifdef NINFERNO_VOLTA_CUTLASS
namespace {

// Weight rows decoded per chunk (gate and up rows together under SwiGLU): as many as 140 MiB of
// FP16 holds at the call's K (4096 rows at K = 17408, 13824 at K = 5120). A fixed 4096 rows split
// the 27B gate_up (34816 x 5120) into nine GEMMs of 3.2 CTA waves each on the V100 (160 resident
// 128 x 128 tiles); one chunk of a third of it runs 10.8 waves (ninfer-v100-sm70 runs the whole
// parent).
constexpr std::size_t kChunkWeightBytes    = std::size_t{4096} * 17408 * 2;
constexpr std::int32_t kMinChunkWeightRows = 256;

// One FP16 value (the decoded weight and the cast activation) is two bytes.
using Half = std::uint16_t;

std::int32_t chunk_weight_rows(std::int32_t k) {
    const std::size_t rows = kChunkWeightBytes / (static_cast<std::size_t>(k) * sizeof(Half));
    return std::max(kMinChunkWeightRows,
                    static_cast<std::int32_t>(std::min<std::size_t>(rows, 1 << 20) / 128 * 128));
}

} // namespace

std::int32_t volta_cutlass_round_up(std::int32_t v, std::int32_t m) { return (v + m - 1) / m * m; }

std::int32_t volta_cutlass_needed_rows(std::int32_t n, bool swiglu) {
    return swiglu ? 2 * volta_cutlass_round_up(n, 8) : volta_cutlass_round_up(n, 8);
}

VoltaCutlassLayout volta_cutlass_layout(std::int32_t k, std::int32_t tokens,
                                        std::int32_t weight_rows) {
    WorkspaceLayoutBuilder b;
    VoltaCutlassLayout l;
    l.x = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(k) * sizeof(Half);
    l.w = static_cast<std::size_t>(weight_rows) * static_cast<std::size_t>(k) * sizeof(Half);
    l.c = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(weight_rows) * sizeof(float);
    (void)b.alloc_bytes(l.x, 256);
    (void)b.alloc_bytes(l.w, 256);
    (void)b.alloc_bytes(l.c, 256);
    l.total = b.peak_bytes(1);
    return l;
}

std::int32_t volta_cutlass_fitting_rows(std::int32_t k, std::int32_t tokens, std::int32_t needed,
                                        std::size_t free_bytes) {
    std::int32_t weight_rows = std::min(needed, chunk_weight_rows(k));
    while (volta_cutlass_layout(k, tokens, weight_rows).total > free_bytes) {
        if (weight_rows <= kMinChunkWeightRows) return 0;
        weight_rows = std::max(kMinChunkWeightRows, volta_cutlass_round_up(weight_rows / 2, 16));
    }
    return weight_rows;
}

bool volta_cutlass_compiled() noexcept { return true; }

std::size_t volta_cutlass_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens,
                                          bool swiglu) {
    // n: the problem's rows; a SwiGLU call decodes its gate and up rows together, the others
    // their rows alone (launch_volta_cutlass's first chunk).
    const std::int32_t rows = volta_cutlass_needed_rows(n, swiglu);
    return volta_cutlass_layout(k, tokens, std::min(rows, chunk_weight_rows(k))).total;
}

std::size_t volta_cutlass_chunk_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens,
                                      bool swiglu, std::size_t free_bytes) {
    if ((k % 8) != 0 || tokens <= 0 || n <= 0) return 0;
    const std::int32_t needed = volta_cutlass_needed_rows(n, swiglu);
    const std::int32_t rows   = volta_cutlass_fitting_rows(k, tokens, needed, free_bytes);
    return rows == 0 ? 0 : volta_cutlass_layout(k, tokens, rows).total;
}

#else

bool volta_cutlass_compiled() noexcept { return false; }

std::size_t volta_cutlass_workspace_bytes(std::int32_t, std::int32_t, std::int32_t, bool) {
    return 0;
}

std::size_t volta_cutlass_chunk_bytes(std::int32_t, std::int32_t, std::int32_t, bool, std::size_t) {
    return 0;
}

#endif

} // namespace ninferno::ops::detail::generic
