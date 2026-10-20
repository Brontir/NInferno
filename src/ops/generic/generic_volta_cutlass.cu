// Volta prefill on CUTLASS's sm_70 tensor-core GEMM. The tiled kernel (generic_tiled.cuh) decodes
// its weight tile in every token-block CTA: a 2048-token chunk decoded each weight 16 times, and
// a 27B Q4 prefill ran at 86 tok/s on a V100, where decoding the weight once and multiplying FP16
// on CUTLASS ran at 1023. Here a call decodes its weight rows once, in row
// chunks, to FP16 (decode_element: every codec of the scalar route), casts the activation to FP16
// (as the tiled kernel's tiles do), multiplies to FP32 on cutlass::gemm::device::Gemm (Sm70,
// 128x128x32, m8n8k4) and rounds once through the route's own epilogue (store_output,
// generic_gate). Built when the tree compiles for sm_70 alone (CMakeLists.txt fetches CUTLASS's
// headers).
#include "core/device.h"
#include "core/layout.h"
#include "ops/generic/generic_gemm.cuh"
#include "ops/generic/generic_gemm_internal.h"

#ifdef NINFERNO_VOLTA_CUTLASS
#    include "cutlass/cutlass.h"
#    include "cutlass/epilogue/thread/linear_combination.h"
#    include "cutlass/gemm/device/gemm.h"
#endif

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninferno::ops::detail::generic {

#ifdef NINFERNO_VOLTA_CUTLASS
namespace {

// FP32 output into the workspace for the route's own epilogue kernel, or BF16 output straight into
// the call's output (D = acc + beta * C, C = D: beta 1 adds the residual, one FP32 add rounded
// once, as store_output does).
template <class ElementOut>
using GemmFor = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,    // A: the activation, [T, K]
    cutlass::half_t, cutlass::layout::ColumnMajor, // B: the decoded weight rows, K contiguous
    ElementOut, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOut, 128 / cutlass::sizeof_bits<ElementOut>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

// Weight rows decoded per chunk (gate and up rows together under SwiGLU): as many as 140 MiB of
// FP16 holds at the call's K (4096 rows at K = 17408, 13824 at K = 5120). A fixed 4096 rows split
// the 27B gate_up (34816 x 5120) into nine GEMMs of 3.2 CTA waves each on the V100 (160 resident
// 128 x 128 tiles); one chunk of a third of it runs 10.8 waves (ninfer-v100-sm70 runs the whole
// parent).
constexpr std::size_t kChunkWeightBytes    = std::size_t{4096} * 17408 * 2;
constexpr std::int32_t kMinChunkWeightRows = 256;

std::int32_t round_up(std::int32_t v, std::int32_t m) { return (v + m - 1) / m * m; }

std::int32_t chunk_weight_rows(std::int32_t k) {
    const std::size_t rows = kChunkWeightBytes / (static_cast<std::size_t>(k) * 2);
    return std::max(kMinChunkWeightRows,
                    static_cast<std::int32_t>(std::min<std::size_t>(rows, 1 << 20) / 128 * 128));
}

struct Layout {
    std::size_t x = 0, w = 0, c = 0, total = 0;
};

Layout layout_for(std::int32_t k, std::int32_t tokens, std::int32_t weight_rows) {
    WorkspaceLayoutBuilder b;
    Layout l;
    l.x = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(k) * sizeof(__half);
    l.w = static_cast<std::size_t>(weight_rows) * static_cast<std::size_t>(k) * sizeof(__half);
    l.c = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(weight_rows) * sizeof(float);
    (void)b.alloc_bytes(l.x, 256);
    (void)b.alloc_bytes(l.w, 256);
    (void)b.alloc_bytes(l.c, 256);
    l.total = b.peak_bytes(1);
    return l;
}

// The activation in FP16, [T, K] contiguous; eight values per thread (ldx and x are 16-byte
// aligned on this route: run_route sends any other call to the scalar route).
__global__ void cast_x_kernel(GemmArgs a, __half* __restrict__ out) {
    const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    const std::int64_t total = static_cast<std::int64_t>(a.tokens) * a.k;
    if (i >= total) return;
    const auto token = static_cast<int>(i / a.k);
    const auto k     = static_cast<int>(i % a.k);
    const uint4 raw  = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(a.x) +
                                                       static_cast<std::int64_t>(token) * a.ldx + k);
    const auto* in   = reinterpret_cast<const __nv_bfloat162*>(&raw);
    uint4 packed;
    auto* o = reinterpret_cast<__half2*>(&packed);
#    pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 v = __bfloat1622float2(in[j]);
        o[j] = __floats2half2_rn(bf16_value_in_f16_range(v.x), bf16_value_in_f16_range(v.y));
    }
    *reinterpret_cast<uint4*>(out + i) = packed;
}

// The codecs whose value is a code times one scale per row (FP8 with a BF16 or FP32 row scale,
// NVFP4's block scales times its row scale). The decoded copy holds the code side, exact in FP16
// (E4M3 codes up to 448, E2M1 times an E4M3 block scale up to 2688, at most five mantissa bits),
// and the epilogue multiplies the row's scale in FP32: a row scale past the f16 range (the FP8
// SwiGLU test's 2^20) stays exact, where a decoded value would saturate at 65504.
__host__ __device__ constexpr bool row_scaled(WeightCodec codec) {
    return codec == WeightCodec::Fp8 || codec == WeightCodec::Fp8RowF32 ||
           codec == WeightCodec::Nvfp4;
}

template <WeightCodec Codec>
__device__ __forceinline__ float decode_unscaled(const GemmArgs& a, int row, int k) {
    if constexpr (Codec == WeightCodec::Fp8 || Codec == WeightCodec::Fp8RowF32) {
        __nv_fp8_e4m3 v;
        v.__x = a.codes[static_cast<std::int64_t>(row) * a.k + k];
        return static_cast<float>(v);
    } else if constexpr (Codec == WeightCodec::Nvfp4) {
        const float2 values = __half22float2(
            e2m1x2_to_half2(a.codes[static_cast<std::int64_t>(row) * (a.k / 2) + k / 2]));
        __nv_fp8_e4m3 s;
        s.__x = a.scales[nvfp4_scale_offset(a, row, k / 16)];
        return ((k & 1) ? values.y : values.x) * static_cast<float>(s);
    } else {
        return decode_element(a, Codec, row, k);
    }
}

// The scale decode_unscaled leaves out (1 for the other codecs).
__device__ __forceinline__ float row_scale_of(const GemmArgs& a, WeightCodec codec, int row) {
    switch (codec) {
    case WeightCodec::Fp8:
        return Fp8Decoder::row_scale(a, row);
    case WeightCodec::Fp8RowF32:
        return Fp8RowF32Decoder::row_scale(a, row);
    case WeightCodec::Nvfp4:
        return Nvfp4Decoder::row_scale(a, row);
    default:
        return 1.0F;
    }
}

// Weight row r of the chunk (local rows [l0, l0 + count)): r < count is local row l0 + r (the
// gate under SwiGLU); with SwiGLU, r in [pad, pad + count) is that row's up row. Rows past them
// are zero (the GEMM's N is a whole multiple of 8).
// The codec is a template argument: decode_element's switch folds away and the eight elements of
// one thread share their row's and group's scale loads.
template <bool SwiGlu, WeightCodec Codec>
__global__ void decode_rows_kernel(GemmArgs a, int l0, int count, int pad, int weight_rows,
                                   __half* __restrict__ out) {
    const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    const std::int64_t total = static_cast<std::int64_t>(weight_rows) * a.k;
    if (i >= total) return;
    const auto r = static_cast<int>(i / a.k);
    const auto k = static_cast<int>(i % a.k);
    int parent   = -1;
    if (r < count) {
        parent = a.row_begin + l0 + r;
    } else if (SwiGlu && r >= pad && r - pad < count) {
        parent = a.row_begin + l0 + (r - pad) + a.up_offset;
    }
    uint4 packed;
    auto* o = reinterpret_cast<__half*>(&packed);
#    pragma unroll
    for (int j = 0; j < 8; ++j) {
        // Saturated: a weight past the f16 range (group scales no real model has) rounds to
        // 65504, not to an infinity (the tiled route shifts such a row instead).
        o[j] = __float2half_rn(
            parent >= 0 ? bf16_value_in_f16_range(decode_unscaled<Codec>(a, parent, k + j)) : 0.0F);
    }
    *reinterpret_cast<uint4*>(out + i) = packed;
}

template <bool SwiGlu, std::size_t... I>
void launch_decode_rows(WeightCodec codec, std::index_sequence<I...>, int blocks, int threads,
                        cudaStream_t stream, const GemmArgs& a, int l0, int count, int pad,
                        int weight_rows, __half* out) {
    const bool launched =
        ((codec == static_cast<WeightCodec>(I)
              ? (decode_rows_kernel<SwiGlu, static_cast<WeightCodec>(I)>
                 <<<blocks, threads, 0, stream>>>(a, l0, count, pad, weight_rows, out),
                 true)
              : false) ||
         ...);
    if (!launched) throw std::invalid_argument("generic Volta CUTLASS: unknown weight codec");
}

constexpr std::size_t kCodecs = static_cast<std::size_t>(WeightCodec::Q3G64) + 1;

template <Epilogue E>
__global__ void epilogue_kernel(GemmArgs a, WeightCodec codec, const float* __restrict__ c, int ldc,
                                int l0, int count, int pad) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::int64_t>(a.tokens) * count) return;
    const auto token = static_cast<int>(i / count);
    const auto j     = static_cast<int>(i % count);
    const float* row = c + static_cast<std::int64_t>(token) * ldc;
    const int parent = a.row_begin + l0 + j;
    float value      = row[j] * row_scale_of(a, codec, parent);
    if constexpr (E == Epilogue::SwiGlu) {
        value = generic_gate(a, value) * (row[pad + j] * row_scale_of(a, codec, parent + a.up_offset));
    }
    store_output<E>(a, l0 + j, token, value);
}

template <class ElementOut>
bool run_gemm(const __half* x, const __half* w, ElementOut* c, int ldc, float beta, int tokens,
              int n, int k, cudaStream_t stream) {
    using Gemm = GemmFor<ElementOut>;
    Gemm gemm;
    const typename Gemm::Arguments args{{tokens, n, k},
                                        {reinterpret_cast<const cutlass::half_t*>(x), k},
                                        {reinterpret_cast<const cutlass::half_t*>(w), k},
                                        {c, ldc},
                                        {c, ldc},
                                        {1.0F, beta},
                                        1};
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    if (Gemm::get_workspace_size(args) != 0) return false;
    if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return false;
    return gemm(stream) == cutlass::Status::kSuccess;
}

int blocks_for(std::int64_t items, int threads) {
    return static_cast<int>((items + threads - 1) / threads);
}

} // namespace

bool volta_cutlass_compiled() noexcept { return true; }

std::size_t volta_cutlass_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens,
                                          bool swiglu) {
    // n: the problem's rows; a SwiGLU call decodes its gate and up rows together, the others
    // their rows alone (launch_volta_cutlass's first chunk).
    const std::int32_t rows = swiglu ? 2 * round_up(n, 8) : round_up(n, 8);
    return layout_for(k, tokens, std::min(rows, chunk_weight_rows(k))).total;
}

bool launch_volta_cutlass(WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    if ((args.k % 8) != 0 || args.tokens <= 0 || args.rows <= 0) return false;
    const bool swiglu = epilogue == Epilogue::SwiGlu;
    // The chunk: as many weight rows as the call has (both halves under SwiGLU), at most
    // chunk_weight_rows(K), halved while the caller's workspace cannot hold it.
    const std::int32_t needed    = swiglu ? 2 * round_up(args.rows, 8) : round_up(args.rows, 8);
    std::int32_t weight_rows     = std::min(needed, chunk_weight_rows(args.k));
    const std::size_t free_bytes = workspace.capacity() - workspace.used();
    while (layout_for(args.k, args.tokens, weight_rows).total > free_bytes) {
        if (weight_rows <= kMinChunkWeightRows) return false;
        weight_rows = std::max(kMinChunkWeightRows, round_up(weight_rows / 2, 16));
    }
    const std::int32_t step = swiglu ? weight_rows / 2 : weight_rows;
    // Store and AddResidual into one BF16 output write it from the GEMM's epilogue: every chunk's
    // rows a multiple of eight (no padded columns), 16-byte aligned rows, no ReLU².
    auto* out_bf16    = static_cast<cutlass::bfloat16_t*>(args.out[0]);
    const bool direct = !swiglu && !row_scaled(codec) && args.segments == 1 &&
                        (epilogue == Epilogue::AddResidual || args.relu2 == 0) &&
                        args.rows % 8 == 0 && step % 8 == 0 && args.ld[0] % 8 == 0 &&
                        reinterpret_cast<std::uintptr_t>(out_bf16) % 16 == 0;

    auto scope     = workspace.scope();
    const Layout l = layout_for(args.k, args.tokens, weight_rows);
    auto* x16      = static_cast<__half*>(workspace.alloc_bytes(l.x).data);
    auto* w16      = static_cast<__half*>(workspace.alloc_bytes(l.w).data);
    auto* c32      = static_cast<float*>(workspace.alloc_bytes(l.c).data);

    constexpr int kThreads = 256;
    cast_x_kernel<<<blocks_for(static_cast<std::int64_t>(args.tokens) * args.k / 8, kThreads),
                    kThreads, 0, stream>>>(args, x16);
    CUDA_CHECK(cudaGetLastError());
    for (std::int32_t l0 = 0; l0 < args.rows; l0 += step) {
        const std::int32_t count = std::min(step, args.rows - l0);
        const std::int32_t pad   = round_up(count, 8);
        const std::int32_t n     = swiglu ? 2 * pad : pad;
        const std::int64_t items = static_cast<std::int64_t>(n) * args.k / 8;
        if (swiglu) {
            launch_decode_rows<true>(codec, std::make_index_sequence<kCodecs>{},
                                     blocks_for(items, kThreads), kThreads, stream, args, l0,
                                     count, pad, n, w16);
        } else {
            launch_decode_rows<false>(codec, std::make_index_sequence<kCodecs>{},
                                      blocks_for(items, kThreads), kThreads, stream, args, l0,
                                      count, pad, n, w16);
        }
        CUDA_CHECK(cudaGetLastError());
        const bool ran =
            direct ? run_gemm(x16, w16, out_bf16 + l0, args.ld[0],
                              epilogue == Epilogue::AddResidual ? 1.0F : 0.0F, args.tokens, n,
                              args.k, stream)
                   : run_gemm(x16, w16, c32, n, 0.0F, args.tokens, n, args.k, stream);
        if (!ran) {
            // Only the first chunk can refuse (every chunk has the same K and alignment); its
            // output rows are untouched, so the caller's route takes the call.
            if (l0 == 0) return false;
            throw std::runtime_error("generic Volta CUTLASS: GEMM refused a later row chunk");
        }
        if (direct) continue;
        const std::int64_t outputs = static_cast<std::int64_t>(args.tokens) * count;
        const int blocks           = blocks_for(outputs, kThreads);
        switch (epilogue) {
        case Epilogue::Store:
            epilogue_kernel<Epilogue::Store>
                <<<blocks, kThreads, 0, stream>>>(args, codec, c32, n, l0, count, pad);
            break;
        case Epilogue::AddResidual:
            epilogue_kernel<Epilogue::AddResidual>
                <<<blocks, kThreads, 0, stream>>>(args, codec, c32, n, l0, count, pad);
            break;
        case Epilogue::SwiGlu:
            epilogue_kernel<Epilogue::SwiGlu>
                <<<blocks, kThreads, 0, stream>>>(args, codec, c32, n, l0, count, pad);
            break;
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

#else

bool volta_cutlass_compiled() noexcept { return false; }

std::size_t volta_cutlass_workspace_bytes(std::int32_t, std::int32_t, std::int32_t, bool) {
    return 0;
}

bool launch_volta_cutlass(WeightCodec, Epilogue, const GemmArgs&, WorkspaceArena&, cudaStream_t) {
    return false;
}

#endif

} // namespace ninferno::ops::detail::generic
