#pragma once

#include <cstddef>
#include <cstdint>

// The chunk arithmetic of Volta's CUTLASS prefill, shared by the host sizes
// (generic_volta_cutlass_sizes.cpp: volta_cutlass_workspace_bytes, volta_cutlass_chunk_bytes) and
// the launch (generic_volta_cutlass.cu), so one rule sizes the gate, the reservation and the
// chunk the launch takes. Defined only where the route is compiled (NINFERNO_VOLTA_CUTLASS).
namespace ninferno::ops::detail::generic {

// The workspace of one chunk: the FP16 activation [T, K], the chunk's FP16 weight rows and its
// FP32 product, each 256-byte aligned.
struct VoltaCutlassLayout {
    std::size_t x = 0, w = 0, c = 0, total = 0;
};

std::int32_t volta_cutlass_round_up(std::int32_t v, std::int32_t m);
// The weight rows a call decodes in all: its rows padded to eight, both halves under SwiGLU.
std::int32_t volta_cutlass_needed_rows(std::int32_t n, bool swiglu);
VoltaCutlassLayout volta_cutlass_layout(std::int32_t k, std::int32_t tokens,
                                        std::int32_t weight_rows);
// The chunk's weight rows: as many as the call has (`needed`), at most the 140 MiB chunk at this
// K, halved while `free_bytes` cannot hold it; 0 when not even the smallest chunk fits.
std::int32_t volta_cutlass_fitting_rows(std::int32_t k, std::int32_t tokens, std::int32_t needed,
                                        std::size_t free_bytes);

} // namespace ninferno::ops::detail::generic
