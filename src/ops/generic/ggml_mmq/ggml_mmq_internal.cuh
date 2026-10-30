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
    int k_vram         = 0; // the config's K_vram: a stream-k block's unit of work along k
};

template <ggml_type type>
Launch select_launch(int rows, int columns, int k) {
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
            best.k_vram   = config.K_vram;
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
    if (best.stream_k) {
        const int waves      = (tiles + device.sm_count - 1) / device.sm_count;
        const int efficiency = 100 * tiles / (device.sm_count * waves);
        // A block beyond the call's work units (an output tile times its K_vram steps) has nothing
        // to do and lengthens the fixup's scan (llama.cpp #30553).
        const int work = tiles * std::max(1, (k + best.k_vram - 1) / best.k_vram);
        (void)efficiency;
        (void)work;
        best.blocks    = tiles; // exp: whole waves, no stream-k (A/B)
        best.fixup           = tiles % best.blocks != 0;
    } else {
        best.blocks = tiles;
    }
    return best;
}

template <ggml_type type>
std::size_t fixup_bytes_impl(int rows, int columns, int k) {
    const Launch l = select_launch<type>(rows, columns, k);
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

template <ggml_type type>
void product_impl(const Planes& w, int blocks_per_row, int rows, int k, const void* activation,
                  int columns, float* out, std::int64_t out_column_stride, void* fixup,
                  cudaStream_t stream) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    if (k % 256 != 0 || rows % kRowMultiple != 0 || blocks_per_row < k / qk) {
        throw std::invalid_argument("ggml mmq: unsupported geometry");
    }
    const Launch l = select_launch<type>(rows, columns, k);
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
    template std::size_t ninferno::ops::ggml_mmq::detail::fixup_bytes_impl<TYPE>(int, int, int);        \
    template void ninferno::ops::ggml_mmq::detail::product_impl<TYPE>(                             \
        const ninferno::ops::ggml_mmq::Planes&, int, int, int, const void*, int, float*,           \
        std::int64_t, void*, cudaStream_t)
