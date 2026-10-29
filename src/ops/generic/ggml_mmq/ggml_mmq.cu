// After ninfer-all's ggml bridge (its src/ops/linear/gguf/ggml_bridge.cu; Apache-2.0, see NOTICE);
// changed here as the git history records.

// ggml_mmq.h: device facts, the activation quantizer (ggml's q8_1 tiles, read from BF16) and the
// per-format dispatch. The products themselves are instantiated one format per unit.

#include "ggml_mmq_internal.cuh"

#include "core/device_class.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

// ggml's GGML_ABORT/GGML_ASSERT land here; only the vendored headers' own invariants reach it.
extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fprintf(stderr, "ggml abort at %s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::abort();
}

#define NINFERNO_GGML_MMQ_EXTERN(TYPE)                                                             \
    extern template std::size_t ninferno::ops::ggml_mmq::detail::fixup_bytes_impl<TYPE>(int, int, int); \
    extern template void ninferno::ops::ggml_mmq::detail::product_impl<TYPE>(                      \
        const ninferno::ops::ggml_mmq::Planes&, int, int, int, const void*, int, float*,           \
        std::int64_t, void*, cudaStream_t)
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_Q4_K);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_Q5_K);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_Q6_K);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_IQ4_XS);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_IQ3_XXS);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_IQ3_S);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_Q4_0);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_Q4_1);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_Q5_0);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_Q5_1);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_IQ4_NL);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_Q2_K);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_Q3_K);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_IQ2_XXS);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_IQ2_XS);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_IQ2_S);
NINFERNO_GGML_MMQ_EXTERN(GGML_TYPE_IQ1_S);
#undef NINFERNO_GGML_MMQ_EXTERN

namespace ninferno::ops::ggml_mmq {
namespace detail {

const DeviceFacts& device_facts() {
    static PerDevice<DeviceFacts> facts;
    return facts.get([](int device) {
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "device properties");
        DeviceFacts f;
        f.cc                     = 100 * properties.major + 10 * properties.minor;
        f.sm_count               = properties.multiProcessorCount;
        f.shared_per_block_optin = properties.sharedMemPerBlockOptin;
        return f;
    });
}

namespace {

constexpr int kQuantizeThreads = 128;

// ggml's activation slack: the kernel over-reads up to one widest column tile past the end.
constexpr std::size_t kActivationSlack = 128 * sizeof(block_q8_1_mmq);

// Token c of x (BF16, at x + c * ldx) -> ggml's q8_1 tiles in the scale/sum layout of the format.
// A block's sum is its codes' (d * sum q), not the raw values' as in ggml's quantizer: the formats
// that read it (Q4_0, Q4_1, Q5_1, Q4_K, Q5_K, Q2_K) subtract an offset times it from the codes' dot,
// two large terms whose difference is the product, so both must see the same activation or the
// quantization noise of the dot stays uncancelled. The W4A8 tiles' activation (W4a8Activation::sums)
// does the same. Qwen3.5-9B Q4_K_M under --activations fast (ninferno-perplexity --quick against
// exact, RTX 4090, MMQ prefill): KLD 0.0302 -> 0.0040, top-1 95.75 % -> 98.44 %, the speed the same
// (5.66k -> 5.67k tok/s); the W4A8 tiles on the RTX 5070 Ti 0.0039.
template <mmq_q8_1_ds_layout ds_layout>
__global__ void quantize_kernel(const __nv_bfloat16* __restrict__ x, int ldx,
                                block_q8_1_mmq* __restrict__ y, int k, int columns) {
    // A programmatic dependent of the kernel that wrote x, releasing the tile kernel (which waits
    // for it) once x is complete.
    ::ninferno::pdl::wait_for_dependencies();
    ::ninferno::pdl::trigger_dependents();
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const std::int64_t i0 = (std::int64_t(blockDim.x) * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= k) return;
    const int column            = blockIdx.x;
    const __nv_bfloat16* source = x + std::int64_t(column) * ldx;
    const uint2 words           = *reinterpret_cast<const uint2*>(source + i0);
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.y));
    const float4 xi = make_float4(lo.x, lo.y, hi.x, hi.y);

    const std::int64_t k_block = i0 / QK8_1_MMQ;
    const std::int64_t iqs     = i0 % QK8_1_MMQ;

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }
    // An all-zero group stores zero codes and a zero scale (ggml divides by amax).
    const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    char4 q;
    q.x           = static_cast<signed char>(roundf(xi.x * d_inv));
    q.y           = static_cast<signed char>(roundf(xi.y * d_inv));
    q.z           = static_cast<signed char>(roundf(xi.z * d_inv));
    q.w           = static_cast<signed char>(roundf(xi.w * d_inv));
    const float d = amax / 127.0f;
    float sum     = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        // The codes' sum is an integer below 2^24: exact in FP32 whatever the shuffle order.
        sum = static_cast<float>(q.x + q.y + q.z + q.w);
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
        sum *= d;
    }
    // The layouts that keep the scale or the sums in fp16 clamp them to its range: a block of
    // large activations (32 SwiGLU outputs of a few thousand) sums past 65504, and an infinite sum
    // turns the dot of every format that reads it into NaN, for the token's whole row and every
    // layer after it. Values inside the range are stored unchanged.
    constexpr float kHalfMax = 65504.0f;
    const float d_half       = fminf(d, kHalfMax);
    const float sum_half     = fminf(fmaxf(sum, -kHalfMax), kHalfMax);

    const std::int64_t ib = k_block * columns + column;
    reinterpret_cast<char4*>(y[ib].qs)[iqs / 4] = q;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs / 16] = sum_half;
            if (iqs % 64 == 0) y[ib].d2s6[iqs / 64] = d_half;
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs / 32] = make_half2(d_half, sum_half);
        } else {
            y[ib].d4[iqs / 32] = d;
        }
    }
}

} // namespace
} // namespace detail

bool device_supported() {
    // The device class the selection is for: a selector test's substitute (ScopedDeviceClass)
    // decides as that card would, not as the GPU the test happens to run on.
    try {
        // TU116 / TU117 run its int8 mma.sync without tensor cores (mma_slow): the vector kernel
        // takes their verify widths, as on Volta.
        const auto& d = device_class();
        return d.compute_capability() >= 75 && !d.features.mma_slow;
    } catch (const std::exception&) {
        return false;
    }
}

std::size_t activation_bytes(int k, int columns) {
    return std::size_t(columns) * (k / QK8_1_MMQ) * sizeof(block_q8_1_mmq) +
           detail::kActivationSlack;
}

int activation_layout(Type type) {
    return static_cast<int>(mmq_get_q8_1_ds_layout(detail::to_ggml(type)));
}

void quantize_activation(Type type, const __nv_bfloat16* x, int ldx, int k, int columns, void* out,
                         cudaStream_t stream) {
    if (k <= 0 || k % kKMultiple != 0 || columns <= 0 || ldx < k || ldx % 4 != 0) {
        throw std::invalid_argument("ggml mmq activation: K must be whole 256-value steps");
    }
    // A CTA quantizes 512 values; a K of an odd number of 256-value steps leaves the last CTA's
    // second half (two whole warps, so no shuffle runs short) to the kernel's i0 >= k exit.
    constexpr int kValuesPerBlock = 4 * detail::kQuantizeThreads;
    const dim3 blocks(columns, (k + kValuesPerBlock - 1) / kValuesPerBlock, 1);
    auto* y = static_cast<block_q8_1_mmq*>(out);
    switch (mmq_get_q8_1_ds_layout(detail::to_ggml(type))) {
    case MMQ_Q8_1_DS_LAYOUT_D4:
        detail::check(::ninferno::pdl::launch_dependent(
                          {blocks, dim3(detail::kQuantizeThreads), 0, stream},
                          detail::quantize_kernel<MMQ_Q8_1_DS_LAYOUT_D4>, x, ldx, y, k, columns),
                      "activation launch");
        break;
    case MMQ_Q8_1_DS_LAYOUT_DS4:
        detail::check(::ninferno::pdl::launch_dependent(
                          {blocks, dim3(detail::kQuantizeThreads), 0, stream},
                          detail::quantize_kernel<MMQ_Q8_1_DS_LAYOUT_DS4>, x, ldx, y, k, columns),
                      "activation launch");
        break;
    case MMQ_Q8_1_DS_LAYOUT_D2S6:
        detail::check(::ninferno::pdl::launch_dependent(
                          {blocks, dim3(detail::kQuantizeThreads), 0, stream},
                          detail::quantize_kernel<MMQ_Q8_1_DS_LAYOUT_D2S6>, x, ldx, y, k, columns),
                      "activation launch");
        break;
    }
}

#define NINFERNO_GGML_MMQ_CASES(CALL)                                                              \
    case Type::Q4_K:                                                                               \
        return CALL(GGML_TYPE_Q4_K);                                                               \
    case Type::Q5_K:                                                                               \
        return CALL(GGML_TYPE_Q5_K);                                                               \
    case Type::Q6_K:                                                                               \
        return CALL(GGML_TYPE_Q6_K);                                                               \
    case Type::IQ4_XS:                                                                             \
        return CALL(GGML_TYPE_IQ4_XS);                                                             \
    case Type::IQ3_XXS:                                                                            \
        return CALL(GGML_TYPE_IQ3_XXS);                                                            \
    case Type::IQ3_S:                                                                              \
        return CALL(GGML_TYPE_IQ3_S);                                                              \
    case Type::Q4_0:                                                                               \
        return CALL(GGML_TYPE_Q4_0);                                                               \
    case Type::Q4_1:                                                                               \
        return CALL(GGML_TYPE_Q4_1);                                                               \
    case Type::Q5_0:                                                                               \
        return CALL(GGML_TYPE_Q5_0);                                                               \
    case Type::Q5_1:                                                                               \
        return CALL(GGML_TYPE_Q5_1);                                                               \
    case Type::IQ4_NL:                                                                             \
        return CALL(GGML_TYPE_IQ4_NL);                                                             \
    case Type::Q2_K:                                                                               \
        return CALL(GGML_TYPE_Q2_K);                                                               \
    case Type::Q3_K:                                                                               \
        return CALL(GGML_TYPE_Q3_K);                                                               \
    case Type::IQ2_XXS:                                                                            \
        return CALL(GGML_TYPE_IQ2_XXS);                                                            \
    case Type::IQ2_XS:                                                                             \
        return CALL(GGML_TYPE_IQ2_XS);                                                             \
    case Type::IQ2_S:                                                                              \
        return CALL(GGML_TYPE_IQ2_S);                                                              \
    case Type::IQ1_S:                                                                              \
        return CALL(GGML_TYPE_IQ1_S);

std::size_t fixup_bound(int columns) {
    // A stream-k launch holds at most one 128-row tile of the call's column tile per SM.
    int j = 16;
    while (j < columns && j < 128) j *= 2;
    return std::size_t(detail::device_facts().sm_count) * j * 128 * sizeof(float);
}

void product(Type type, const Planes& weight, int blocks_per_row, int rows, int k,
             const void* activation, int columns, float* out, std::int64_t out_column_stride,
             void* fixup, cudaStream_t stream) {
#define NINFERNO_PRODUCT(T)                                                                        \
    detail::product_impl<T>(weight, blocks_per_row, rows, k, activation, columns, out,             \
                            out_column_stride, fixup, stream)
    switch (type) { NINFERNO_GGML_MMQ_CASES(NINFERNO_PRODUCT) }
#undef NINFERNO_PRODUCT
    throw std::invalid_argument("ggml mmq: unknown block format");
}

#undef NINFERNO_GGML_MMQ_CASES

} // namespace ninferno::ops::ggml_mmq
