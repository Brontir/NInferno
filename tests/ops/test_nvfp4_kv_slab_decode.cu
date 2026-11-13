// The register decoders of the NVFP4 KV slab kernel (nvfp4/slab_codec.cuh, #849), bit for bit,
// on whatever this build issues (the cvt forms of sm_89 and sm_100+ or the bit forms below them):
// 1. every code byte at every nibble position times every legal scale (the nonnegative finite
//    E4M3 codes 0x00..0x7e) equals the cache codec's FP16 value (kv_cache_nvfp4_dequant_f16x16)
//    after the fold, signed zeros included, and a scale byte at each position of its word widens
//    to the codec's FP16 scale;
// 2. random code words and scales, through one m16n8k16 MMA: the folded slab fragments times an
//    FP16 B give the codec values' sums times exactly 2^-14 (or 1), the HMMA's FP16 subnormal
//    inputs included;
// 3. movmatrix_trans (ops/common/mma.cuh) transposes an 8x8 fragment.

#include "ops/common/mma.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/slab_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

using namespace ninferno::ops;
using namespace ninferno::ops::detail;

constexpr int kScales  = 0x7f; // codes 0x00..0x7e
constexpr int kMmaRuns = 4096;

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

struct Failures {
    unsigned decode;
    unsigned scale;
    unsigned mma;
    unsigned transpose;
    unsigned first_decode[4]; // byte, scale, nibble, slab bits
    unsigned first_mma[4];    // run, lane, slab bits, reference bits
};

__device__ std::uint32_t hash32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ float half_low(std::uint32_t bits) {
    return __low2float(*reinterpret_cast<const __half2*>(&bits));
}

__device__ float half_high(std::uint32_t bits) {
    return __high2float(*reinterpret_cast<const __half2*>(&bits));
}

// The codec's eight FP16 values of a code word, nibble order.
__device__ void reference_values(std::uint32_t word, std::uint8_t scale, float (&values)[8]) {
    alignas(8) std::uint8_t bytes[8] = {};
    std::memcpy(bytes, &word, 4);
    const auto decoded            = kv_cache_nvfp4_dequant_f16x16(bytes, scale);
    const std::uint32_t halves[4] = {
        static_cast<std::uint32_t>(decoded.lo.x), static_cast<std::uint32_t>(decoded.lo.y),
        static_cast<std::uint32_t>(decoded.lo.z), static_cast<std::uint32_t>(decoded.lo.w)};
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        values[2 * b]     = half_low(halves[b]);
        values[2 * b + 1] = half_high(halves[b]);
    }
}

template <int P>
__device__ std::uint32_t slab_scaled_pair(std::uint32_t word, std::uint32_t scale_word,
                                          std::uint32_t position) {
    return nvfp4_slab_scaled(nvfp4_slab_pair<P>(word),
                             nvfp4_slab_scale2(scale_word, nvfp4_slab_scale_selector(position)));
}

__device__ void check_word(std::uint32_t word, std::uint8_t scale, std::uint32_t position,
                           Failures* failures) {
    float reference[8];
    reference_values(word, scale, reference);
    const std::uint32_t scale_word = static_cast<std::uint32_t>(scale) << (8 * position);
    const std::uint32_t pairs[4]   = {
        slab_scaled_pair<0>(word, scale_word, position),
        slab_scaled_pair<1>(word, scale_word, position),
        slab_scaled_pair<2>(word, scale_word, position),
        slab_scaled_pair<3>(word, scale_word, position),
    };
    for (int p = 0; p < 4; ++p) {
        for (int h = 0; h < 2; ++h) {
            const float got  = (h ? half_high(pairs[p]) : half_low(pairs[p])) * nvfp4_slab_fold();
            const float want = reference[nvfp4_slab_nibble(p, h)];
            if (__float_as_uint(got) != __float_as_uint(want) &&
                atomicAdd(&failures->decode, 1U) == 0) {
                failures->first_decode[0] = word;
                failures->first_decode[1] = scale;
                failures->first_decode[2] = static_cast<unsigned>(nvfp4_slab_nibble(p, h));
                failures->first_decode[3] = pairs[p];
            }
        }
    }
    __nv_fp8_e4m3 code;
    code.__x                = scale;
    const float scale_value = __half2float(static_cast<__half>(code));
    const __half2 scale2    = nvfp4_slab_scale2(scale_word, nvfp4_slab_scale_selector(position));
    if (__float_as_uint(__low2float(scale2)) != __float_as_uint(scale_value) ||
        __float_as_uint(__high2float(scale2)) != __float_as_uint(scale_value))
        atomicAdd(&failures->scale, 1U);
}

// Block: one scale code; thread: one code byte, repeated over the word (every nibble value at
// every position), then a random word.
__global__ void decode_all(Failures* failures) {
    const std::uint8_t scale = static_cast<std::uint8_t>(blockIdx.x);
    const std::uint32_t byte = threadIdx.x;
    check_word(byte * 0x01010101U, scale, byte & 3U, failures);
    check_word(hash32(byte * 131U + blockIdx.x), scale, (byte >> 2) & 3U, failures);
}

__device__ std::uint32_t pack(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const std::uint32_t*>(&h);
}

// One warp per run: lane (g, t) holds random code words of keys g and g + 8 with random legal
// scales, and a random FP16 B fragment. The same A slots from the slab decoders (folded) and from
// the codec's values (unfolded) through one MMA each.
__global__ void mma_fold(Failures* failures) {
    const int lane               = threadIdx.x;
    const unsigned run           = blockIdx.x;
    const std::uint32_t seed     = run * 977U + lane * 31U;
    const std::uint32_t words[2] = {hash32(seed), hash32(seed + 1U)};
    // Scales cover the codes from the smallest subnormal up (the cache stores 2^-9 .. 448).
    const std::uint8_t scales[2] = {static_cast<std::uint8_t>(1U + hash32(seed + 2U) % 0x7eU),
                                    static_cast<std::uint8_t>(1U + hash32(seed + 3U) % 0x7eU)};
    std::uint32_t slab[4][2], codec[4][2];
    for (int k = 0; k < 2; ++k) {
        float reference[8];
        reference_values(words[k], scales[k], reference);
        const std::uint32_t scale_word = scales[k];
        slab[0][k]                     = slab_scaled_pair<0>(words[k], scale_word, 0);
        slab[1][k]                     = slab_scaled_pair<1>(words[k], scale_word, 0);
        slab[2][k]                     = slab_scaled_pair<2>(words[k], scale_word, 0);
        slab[3][k]                     = slab_scaled_pair<3>(words[k], scale_word, 0);
        for (int p = 0; p < 4; ++p)
            codec[p][k] =
                pack(reference[nvfp4_slab_nibble(p, 0)], reference[nvfp4_slab_nibble(p, 1)]);
    }
    const auto random_half = [&](std::uint32_t salt) {
        return (static_cast<float>(hash32(seed + salt) % 4097U) - 2048.0F) / 1024.0F;
    };
    const std::uint32_t b0 = pack(random_half(4U), random_half(5U));
    const std::uint32_t b1 = pack(random_half(6U), random_half(7U));
    float folded[4] = {}, unfolded[4] = {};
    mma_f16(folded[0], folded[1], folded[2], folded[3], slab[0][0], slab[0][1], slab[1][0],
            slab[1][1], b0, b1);
    mma_f16(unfolded[0], unfolded[1], unfolded[2], unfolded[3], codec[0][0], codec[0][1],
            codec[1][0], codec[1][1], b0, b1);
    for (int i = 0; i < 4; ++i) {
        const float got = folded[i] * nvfp4_slab_fold();
        if (__float_as_uint(got) != __float_as_uint(unfolded[i]) &&
            atomicAdd(&failures->mma, 1U) == 0) {
            failures->first_mma[0] = run;
            failures->first_mma[1] = static_cast<unsigned>(lane);
            failures->first_mma[2] = __float_as_uint(got);
            failures->first_mma[3] = __float_as_uint(unfolded[i]);
        }
    }
}

// M[r][c] = 8r + c + 1 as FP16: lane (g, t) holds M[g][2t..2t+1] and must receive M[2t][g],
// M[2t+1][g].
__global__ void transpose(Failures* failures) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    const std::uint32_t in  = pack(8.0F * g + 2 * t + 1, 8.0F * g + 2 * t + 2);
    const std::uint32_t out = movmatrix_trans(in);
    if (out != pack(8.0F * (2 * t) + g + 1, 8.0F * (2 * t + 1) + g + 1))
        atomicAdd(&failures->transpose, 1U);
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    // One explicit stream for the fill, the kernels and the read-back.
    cudaStream_t stream = nullptr;
    Failures* device    = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc(&device, sizeof(Failures)) != cudaSuccess ||
        cudaMemsetAsync(device, 0, sizeof(Failures), stream) != cudaSuccess)
        return 1;
    decode_all<<<kScales, 256, 0, stream>>>(device);
    mma_fold<<<kMmaRuns, 32, 0, stream>>>(device);
    transpose<<<1, 32, 0, stream>>>(device);
    Failures host{};
    if (cudaMemcpyAsync(&host, device, sizeof(Failures), cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        std::printf("FAIL: kernel or copy: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    cudaFree(device);
    cudaStreamDestroy(stream);
    if (host.decode != 0)
        std::printf("FAIL decode: %u values; first word 0x%08x scale 0x%02x nibble %u slab "
                    "0x%08x\n",
                    host.decode, host.first_decode[0], host.first_decode[1], host.first_decode[2],
                    host.first_decode[3]);
    if (host.scale != 0) std::printf("FAIL scale widening: %u\n", host.scale);
    if (host.mma != 0)
        std::printf("FAIL folded MMA: %u sums; first run %u lane %u got 0x%08x want 0x%08x\n",
                    host.mma, host.first_mma[0], host.first_mma[1], host.first_mma[2],
                    host.first_mma[3]);
    if (host.transpose != 0) std::printf("FAIL movmatrix: %u lanes\n", host.transpose);
    if (host.decode || host.scale || host.mma || host.transpose) return 1;
    std::printf("PASS: %d scales x 512 words, %d folded MMAs, the transpose\n", kScales, kMmaRuns);
    return 0;
}
