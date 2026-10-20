// Generic W4A8: the activation quantizer and the codec dispatch.
#include "ops/generic/generic_w4a8.cuh"

#include "ops/generic/generic_gemm_internal.h"

#include <stdexcept>

namespace ninferno::ops::detail::generic {
namespace {

// Activation quantization: one thread per (token, 32-value block): int8 codes rint(x*127/amax),
// scale amax/127 (0 for an all-zero block), and the sum of the quantized block (scale times the
// codes' integer sum): the min term m*sum then completes sum_i (d*q_i + m)*x_q_i exactly, so the
// activation error enters as sum_i w_i*(x_q_i - x_i) whatever the min (as ggml's Q8_1 sum).
// Rows of `ld` values (ld >= k, whole blocks): the blocks past k are zero codes, scale and sum.
__global__ __launch_bounds__(256) void w4a8_quantize_kernel(const __nv_bfloat16* __restrict__ x,
                                                            int ldx, int k, int ld, int tokens,
                                                            std::int8_t* __restrict__ codes,
                                                            float* __restrict__ scales,
                                                            float* __restrict__ sums) {
    const int blocks = ld / 32;
    const std::int64_t task =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (task >= static_cast<std::int64_t>(tokens) * blocks) return;
    const int token = static_cast<int>(task / blocks);
    const int block = static_cast<int>(task - static_cast<std::int64_t>(token) * blocks);
    unsigned words[8] = {};
    float scale       = 0.0F;
    int qsum          = 0;
    if (block * 32 < k) {
        float v[32];
        w4a8_load_block(x + static_cast<std::int64_t>(token) * ldx + block * 32, v);
        w4a8_quantize_block(v, words, scale, qsum);
    }
    auto* dst =
        reinterpret_cast<uint4*>(codes + static_cast<std::int64_t>(token) * ld + block * 32);
    dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
    dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
    scales[task] = scale;
    sums[task]   = scale * static_cast<float>(qsum);
}

} // namespace

void launch_w4a8_quantize(const GemmArgs& args, std::int8_t* codes, float* scales, float* sums,
                          cudaStream_t stream, int ld) {
    if (ld <= 0) ld = args.k;
    const std::int64_t tasks = static_cast<std::int64_t>(args.tokens) * (ld / 32);
    if (tasks > 0) {
        w4a8_quantize_kernel<<<static_cast<unsigned>((tasks + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(args.x), args.ldx, args.k, ld, args.tokens, codes,
            scales, sums);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_w4a8(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                 std::int8_t* codes, float* scales, float* sums, bool token_major,
                 cudaStream_t stream) {
    launch_w4a8_quantize(args, codes, scales, sums, stream, w4a8_ld(args.k));
    const W4a8Activation act{codes, scales, sums, token_major ? 1 : 0};
    if (launch_w4a8_ggml32(schedule, codec, epilogue, args, act, stream)) return;
    if (launch_w4a8_superblock(schedule, codec, epilogue, args, act, stream)) return;
    if (launch_w4a8_codebook(schedule, codec, epilogue, args, act, stream)) return;
    throw std::invalid_argument("generic W4A8: the weight format has no W4A8 codec");
}

} // namespace ninferno::ops::detail::generic
