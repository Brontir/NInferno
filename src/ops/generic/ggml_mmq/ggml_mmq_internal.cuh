#pragma once

// Host side of ggml's matrix kernel (mmq.cuh mul_mat_q) without ggml's backend context: the caller
// owns the activation, the FP32 output and the stream-k fixup plane. After ninfer-all's bridge
// (its src/ops/linear/gguf/ggml_bridge_mmq.cuh, Apache-2.0), reduced to the dense product.

#include "ggml_mmq.h"

#include "core/cuda_error.h"
#include "core/per_device.h"
#include "core/pdl.cuh"

// The tile kernel and its stream-k fixup launch as programmatic dependents: each waits for the
// kernel before (the activation quantizer, the tile kernel) and releases the next at once, which
// waits for it in turn (the fixup, the epilogue). Released early, nothing reads their output
// before they complete; the epilogue releases nothing, so a following call's quantizer cannot
// overwrite this call's activation while it is read.
#define NINFERNO_MMQ_PDL_ENTRY()                                                                   \
    do {                                                                                           \
        ::ninferno::pdl::wait_for_dependencies();                                                  \
        ::ninferno::pdl::trigger_dependents();                                                     \
    } while (0)

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ninferno::ops::ggml_mmq::detail {

struct DeviceFacts {
    int cc                             = 0; // ggml's: 100 * major + 10 * minor
    int sm_count                       = 0;
    std::size_t shared_per_block_optin = 0;
};

[[nodiscard]] const DeviceFacts& device_facts();

// A CudaError, so a device fault that first surfaces at one of these launches is still answered
// as the device's failure (HTTP 503) rather than an internal error.
inline void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw CudaError(status,
                        cuda_error_message((std::string("ggml mmq ") + what).c_str(), status));
    }
}

[[nodiscard]] inline ggml_type to_ggml(Type type) {
    switch (type) {
    case Type::Q4_K: return GGML_TYPE_Q4_K;
    case Type::Q5_K: return GGML_TYPE_Q5_K;
    case Type::Q6_K: return GGML_TYPE_Q6_K;
    case Type::IQ4_XS: return GGML_TYPE_IQ4_XS;
    case Type::IQ3_XXS: return GGML_TYPE_IQ3_XXS;
    case Type::IQ3_S: return GGML_TYPE_IQ3_S;
    case Type::Q4_0: return GGML_TYPE_Q4_0;
    case Type::Q4_1: return GGML_TYPE_Q4_1;
    case Type::Q5_0: return GGML_TYPE_Q5_0;
    case Type::Q5_1: return GGML_TYPE_Q5_1;
    case Type::IQ4_NL: return GGML_TYPE_IQ4_NL;
    case Type::Q2_K: return GGML_TYPE_Q2_K;
    case Type::Q3_K: return GGML_TYPE_Q3_K;
    case Type::IQ2_XXS: return GGML_TYPE_IQ2_XXS;
    case Type::IQ2_XS: return GGML_TYPE_IQ2_XS;
    case Type::IQ2_S: return GGML_TYPE_IQ2_S;
    case Type::IQ1_S: return GGML_TYPE_IQ1_S;
    case Type::IQ1_M: return GGML_TYPE_IQ1_M;
    }
    throw std::invalid_argument("ggml mmq: unknown block format");
}

// Column tiles instantiated; a product takes the fewest column tiles they allow.
inline constexpr int kColumnTiles[] = {16, 32, 64, 128};

struct Launch {
    int J              = 0;
    int I              = 0;
    int threads        = 0;
    std::size_t shared = 0;
    int row_tiles      = 0;
    int column_tiles   = 0;
    bool stream_k      = false;
    int blocks         = 0;
    bool fixup         = false;
};

template <ggml_type type>
Launch select_launch(int rows, int columns, Split split, int k_splits = 1) {
    const DeviceFacts& device = device_facts();
    Launch best;
    int best_tiles = INT_MAX;
    for (const int J : kColumnTiles) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, false, device.cc);
        if (config.type == GGML_TYPE_COUNT) continue;
        const std::size_t shared = mmq_get_nbytes_shared(config, device.cc);
        if (shared > device.shared_per_block_optin) continue;
        const int tiles = (columns + J - 1) / J;
        if (tiles < best_tiles) {
            best_tiles    = tiles;
            best.J        = J;
            best.I        = config.I;
            best.threads  = config.nthreads;
            best.shared   = shared;
            best.stream_k = config.stream_k;
        }
        if (tiles == 1) break;
    }
    if (best.J == 0) throw std::invalid_argument("ggml mmq: no tile fits this device");
    best.row_tiles    = (rows + best.I - 1) / best.I;
    best.column_tiles = (columns + best.J - 1) / best.J;
    const int tiles   = best.row_tiles * best.column_tiles;
    if (split == Split::Tiles && best.stream_k) {
        // The stream-k kernel with k_splits CTAs per tile: the grid's (tile, K block) range cut
        // into tiles * k_splits equal parts, each CTA one 1/k_splits of one tile's K (k_splits
        // divides K's 256-value steps, so no part straddles a tile or a step). The CTA of the
        // last part writes its sums, the fixup adds the others' in descending order: one order
        // per element for a given k_splits, whatever the tile count. One part: a tiled grid.
        best.blocks = tiles * k_splits;
        best.fixup  = k_splits > 1;
    } else if (split == Split::Tiles) {
        // A config compiled without stream-k launches the 2D grid: every CTA one tile's whole K,
        // which is token-invariant as is; K splits do not apply and need no fixup.
        best.blocks = tiles;
    } else if (best.stream_k) {
        const int waves      = (tiles + device.sm_count - 1) / device.sm_count;
        const int efficiency = 100 * tiles / (device.sm_count * waves);
        best.blocks          = efficiency >= 90 ? tiles : device.sm_count;
        best.fixup           = tiles % best.blocks != 0;
    } else {
        best.blocks = tiles;
    }
    return best;
}

template <ggml_type type>
std::size_t fixup_bytes_impl(int rows, int columns, Split split, int k_splits) {
    const Launch l = select_launch<type>(rows, columns, split, k_splits);
    return l.fixup ? std::size_t(l.blocks) * l.J * l.I * sizeof(float) : 0;
}

template <ggml_type type, int J>
void launch(const Launch& l, const char* codes, const ggml_planes& planes, int stride_row_blocks,
            int rows, int k, const int* activation, int columns, float* out,
            std::int64_t out_column_stride, float* fixup, cudaStream_t stream) {
    static PerDevice<bool> raised;
    allow_dynamic_smem(raised, mul_mat_q<type, J, false>, l.shared);
    constexpr int qk      = ggml_cuda_type_traits<type>::qk;
    const uint3 blocks_fd = init_fastdiv_values(k / qk);
    const uint3 one_fd    = init_fastdiv_values(1);
    const uint3 ntx_fd    = init_fastdiv_values(l.column_tiles);
    const dim3 block_dims(WARP_SIZE, l.threads / WARP_SIZE, 1);
    const int stride = static_cast<int>(out_column_stride);
    if (!l.stream_k) {
        check(::ninferno::pdl::launch_dependent(
                  {dim3(l.row_tiles, l.column_tiles, 1), block_dims, l.shared, stream},
                  mul_mat_q<type, J, false>, codes, activation, nullptr, nullptr, out, nullptr,
                  nullptr, blocks_fd, rows, columns, stride_row_blocks, columns, stride, one_fd,
                  one_fd, 0, 0, 0, one_fd, one_fd, 0, 0, 0, ntx_fd, planes),
              "launch");
        return;
    }
    if (l.fixup && fixup == nullptr) throw std::invalid_argument("ggml mmq: needs a fixup plane");
    check(::ninferno::pdl::launch_dependent(
              {dim3(l.blocks, 1, 1), block_dims, l.shared, stream}, mul_mat_q<type, J, false>,
              codes, activation, nullptr, nullptr, out, l.fixup ? fixup : nullptr, nullptr,
              blocks_fd, rows, columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0, 0,
              one_fd, one_fd, 0, 0, 0, ntx_fd, planes),
          "launch");
    if (!l.fixup) return;
    const dim3 fixup_blocks(l.blocks, l.I / WARP_SIZE, 1);
    const dim3 fixup_dims(block_dims.x, block_dims.y / 2, 1);
    check(::ninferno::pdl::launch_dependent({fixup_blocks, fixup_dims, 0, stream},
                                          mul_mat_q_stream_k_fixup<type, J, false>, nullptr,
                                          nullptr, out, fixup, blocks_fd, rows, columns, stride,
                                          one_fd, 0, one_fd, 0, ntx_fd),
          "fixup launch");
}

template <ggml_type type, int J>
int occupancy(const Launch& l) {
    static PerDevice<bool> raised;
    allow_dynamic_smem(raised, mul_mat_q<type, J, false>, l.shared);
    int per_sm = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, mul_mat_q<type, J, false>,
                                                        l.threads, l.shared),
          "occupancy");
    return per_sm;
}

template <ggml_type type>
int tile_slots_impl() {
    static PerDevice<int> slots;
    return slots.get([](int) {
        const Launch l = select_launch<type>(kRowMultiple, 128, Split::Tiles);
        int per_sm     = 0;
        switch (l.J) {
        case 16:
            per_sm = occupancy<type, 16>(l);
            break;
        case 32:
            per_sm = occupancy<type, 32>(l);
            break;
        case 64:
            per_sm = occupancy<type, 64>(l);
            break;
        case 128:
            per_sm = occupancy<type, 128>(l);
            break;
        default:
            throw std::invalid_argument("ggml mmq: unexpected column tile");
        }
        return std::max(per_sm, 1) * device_facts().sm_count;
    });
}

template <ggml_type type>
void product_impl(const Planes& w, int blocks_per_row, int rows, int k, const void* activation,
                  int columns, float* out, std::int64_t out_column_stride, void* fixup,
                  cudaStream_t stream, Split split, int k_splits) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    if (k % 256 != 0 || rows % kRowMultiple != 0 || blocks_per_row < k / qk) {
        throw std::invalid_argument("ggml mmq: unsupported geometry");
    }
    // Every format here steps K by MMQ_ITER_K (256) values (the FP4 formats, which step 512, do
    // not take this kernel).
    if (k_splits < 1 || (k / MMQ_ITER_K) % k_splits != 0 ||
        (split == Split::StreamK && k_splits != 1)) {
        throw std::invalid_argument("ggml mmq: K splits must divide K's 256-value steps");
    }
    const Launch l = select_launch<type>(rows, columns, split, k_splits);
    const auto* x  = static_cast<const char*>(w.codes);
    const ggml_planes planes{static_cast<const char*>(w.high), static_cast<const char*>(w.scales)};
    const auto* y = static_cast<const int*>(activation);
    auto* f       = static_cast<float*>(fixup);
    switch (l.J) {
    case 16:
        return launch<type, 16>(l, x, planes, blocks_per_row, rows, k, y, columns, out,
                                out_column_stride, f, stream);
    case 32:
        return launch<type, 32>(l, x, planes, blocks_per_row, rows, k, y, columns, out,
                                out_column_stride, f, stream);
    case 64:
        return launch<type, 64>(l, x, planes, blocks_per_row, rows, k, y, columns, out,
                                out_column_stride, f, stream);
    case 128:
        return launch<type, 128>(l, x, planes, blocks_per_row, rows, k, y, columns, out,
                                 out_column_stride, f, stream);
    default: break;
    }
    throw std::invalid_argument("ggml mmq: unexpected column tile");
}

} // namespace ninferno::ops::ggml_mmq::detail

#define NINFERNO_GGML_MMQ_INSTANCE(TYPE)                                                           \
    template std::size_t ninferno::ops::ggml_mmq::detail::fixup_bytes_impl<TYPE>(                  \
        int, int, ninferno::ops::ggml_mmq::Split, int);                                            \
    template int ninferno::ops::ggml_mmq::detail::tile_slots_impl<TYPE>();                         \
    template void ninferno::ops::ggml_mmq::detail::product_impl<TYPE>(                             \
        const ninferno::ops::ggml_mmq::Planes&, int, int, int, const void*, int, float*,           \
        std::int64_t, void*, cudaStream_t, ninferno::ops::ggml_mmq::Split, int)
