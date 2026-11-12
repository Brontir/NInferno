#pragma once

// The forced launches of tools/bench/mmq_launch_microbench.h for one block format: the production
// launch template (detail::launch) with a Launch built here from ggml's own config of the forced
// width instead of select_launch's choice. Included by one unit a format, which defines its
// FormatOps (NINFERNO_MMQ_BENCH_FORMAT). BENCH ONLY.

#include "mmq_launch_microbench.h"
#include "ops/generic/ggml_mmq/ggml_mmq_internal.cuh"

#include <stdexcept>

namespace ninferno::bench::mmq_launch::impl {

namespace detail = ::ninferno::ops::ggml_mmq::detail;

template <ggml_type type>
TileConfig config(int j) {
    const detail::DeviceFacts& device = detail::device_facts();
    TileConfig t;
    const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(type, j, false, device.cc);
    if (c.type == GGML_TYPE_COUNT) return t;
    t.shared = mmq_get_nbytes_shared(c, device.cc);
    if (t.shared > device.shared_per_block_optin) return t;
    t.ok       = true;
    t.rows     = c.I;
    t.threads  = c.nthreads;
    t.k_vram   = c.K_vram;
    t.stream_k = c.stream_k;
    return t;
}

template <ggml_type type>
detail::Launch launch_of(const Forced& f) {
    const TileConfig c = config<type>(f.j);
    if (!c.ok) throw std::invalid_argument("mmq_launch_microbench: no tile of this width here");
    if (f.rows % c.rows != 0 || f.k % c.k_vram != 0 || f.blocks <= 0) {
        throw std::invalid_argument("mmq_launch_microbench: unsupported geometry");
    }
    detail::Launch l;
    l.J             = f.j;
    l.I             = c.rows;
    l.threads       = c.threads;
    l.shared        = c.shared;
    l.k_vram        = c.k_vram;
    l.stream_k      = c.stream_k;
    l.row_tiles     = f.rows / c.rows;
    l.column_tiles  = (f.columns + f.j - 1) / f.j;
    const int tiles = l.row_tiles * l.column_tiles;
    // A config without stream-k launches a tile a CTA whatever the CTA count asked: a split grid
    // there would time another launch than the one it names, so it is refused, not run.
    if (!l.stream_k && f.blocks != tiles) {
        throw std::invalid_argument("mmq_launch_microbench: a split launch on a tile without "
                                    "stream-k");
    }
    l.blocks = f.blocks;
    l.fixup  = l.stream_k && tiles % l.blocks != 0;
    return l;
}

template <ggml_type type, int J>
void fixup_launch(const detail::Launch& l, const Forced& f, float* out, void* fixup,
                  cudaStream_t stream) {
    constexpr int qk      = ggml_cuda_type_traits<type>::qk;
    const uint3 blocks_fd = init_fastdiv_values(f.k / qk);
    const uint3 one_fd    = init_fastdiv_values(1);
    const uint3 ntx_fd    = init_fastdiv_values(l.column_tiles);
    const dim3 block_dims(WARP_SIZE, l.threads / WARP_SIZE, 1);
    const dim3 fixup_blocks(l.blocks, l.I / WARP_SIZE, 1);
    const dim3 fixup_dims(block_dims.x, block_dims.y / 2, 1);
    mul_mat_q_stream_k_fixup<type, J, false><<<fixup_blocks, fixup_dims, 0, stream>>>(
        nullptr, nullptr, out, static_cast<float*>(fixup), blocks_fd, f.rows, f.columns, f.rows,
        one_fd, 0, one_fd, 0, ntx_fd);
    detail::check(cudaGetLastError(), "bench fixup launch");
}

// The widths detail::launch is instantiated at (core/mmq_columns.h kTilesWork).
#define NINFERNO_MMQ_BENCH_WIDTHS(CASE) CASE(16) CASE(32) CASE(64) CASE(80) CASE(96) CASE(128)

template <ggml_type type>
void product(const ggml_mmq::Planes& w, int blocks_per_row, const Forced& f, const void* activation,
             float* out, void* fixup, cudaStream_t stream) {
    const detail::Launch l = launch_of<type>(f);
    const auto* x          = static_cast<const char*>(w.codes);
    const ggml_planes planes{static_cast<const char*>(w.high), static_cast<const char*>(w.scales)};
    const auto* y = static_cast<const int*>(activation);
    auto* fx      = static_cast<float*>(fixup);
    switch (l.J) {
#define NINFERNO_MMQ_BENCH_CASE(J)                                                                 \
    case J:                                                                                        \
        return detail::launch<type, J>(l, x, planes, blocks_per_row, f.rows, f.k, y, f.columns,    \
                                       out, f.rows, fx, stream);
        NINFERNO_MMQ_BENCH_WIDTHS(NINFERNO_MMQ_BENCH_CASE)
#undef NINFERNO_MMQ_BENCH_CASE
    default:
        break;
    }
    throw std::invalid_argument("mmq_launch_microbench: unexpected column tile");
}

template <ggml_type type>
void fixup_only(const Forced& f, float* out, void* fixup, cudaStream_t stream) {
    const detail::Launch l = launch_of<type>(f);
    if (!l.fixup) throw std::invalid_argument("mmq_launch_microbench: this launch has no fixup");
    switch (l.J) {
#define NINFERNO_MMQ_BENCH_CASE(J)                                                                 \
    case J:                                                                                        \
        return fixup_launch<type, J>(l, f, out, fixup, stream);
        NINFERNO_MMQ_BENCH_WIDTHS(NINFERNO_MMQ_BENCH_CASE)
#undef NINFERNO_MMQ_BENCH_CASE
    default:
        break;
    }
    throw std::invalid_argument("mmq_launch_microbench: unexpected column tile");
}

template <ggml_type type>
int resident_ctas(int j) {
    const TileConfig c = config<type>(j);
    if (!c.ok) return 0;
    int n              = 0;
    cudaError_t status = cudaErrorInvalidValue;
    switch (j) {
#define NINFERNO_MMQ_BENCH_CASE(J)                                                                 \
    case J:                                                                                        \
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, mul_mat_q<type, J, false>,      \
                                                               c.threads, c.shared);               \
        break;
        NINFERNO_MMQ_BENCH_WIDTHS(NINFERNO_MMQ_BENCH_CASE)
#undef NINFERNO_MMQ_BENCH_CASE
    default:
        break;
    }
    detail::check(status, "bench occupancy");
    return n;
}

#undef NINFERNO_MMQ_BENCH_WIDTHS

template <ggml_type type>
constexpr FormatOps ops(const char* name, ggml_mmq::Type t) {
    return FormatOps{
        name, t, &config<type>, &product<type>, &fixup_only<type>, &resident_ctas<type>};
}

} // namespace ninferno::bench::mmq_launch::impl
