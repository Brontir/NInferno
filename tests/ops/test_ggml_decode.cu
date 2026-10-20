// ggml's block formats kept as planes on the device, bit for bit against ggml's dequantization:
// Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M,
// TQ1_0, TQ2_0, Q1_0 and Q2_0.
//
// Random GGUF blocks (every byte random but the binary16 d/m words, which are finite) are split
// into the engine's ggml_planes_v1 planes the way the converter keeps them (GgufView), then
// decoded twice on the device:
//   - by generic::decode_element (the scalar fallback and the embedding gather), value by value;
//   - by the register-MMA decoders (load, pair, sub_scale, sub_min) exactly as mma_step feeds the
//     tensor cores: every lane of a quad, every MMA of a K step, the BF16 operand times its
//     group's FP32 scale (plus the min term: d*q + m).
// Both must equal the converter's ggml::dequantize bit for bit, which ninferno_convert_ggml_quants
// and gguf_pyref.py hold bit-exact with gguf-py (and so with ggml's dequantize_row_*): the
// decoders hand the MMA exactly ggml's values, only the order of the dot product's sums differs.
// The IQ3, IQ2 and IQ1 grids are read from shared memory as the kernels stage them. IQ1_M keeps
// its binary16 d in the top nibbles of its four scale words; those are made finite too.

#include "convert/ggml_quants.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace generic = ninferno::ops::detail::generic;
namespace ggml    = ninferno::convert::ggml;
using ninferno::QType;

namespace {

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

struct Format {
    const char* name;
    std::uint32_t ggml_type;
    QType qtype;
    generic::WeightCodec codec;
    std::uint32_t block_bytes;
    // Where the planes sit in a GGUF block (the converter's table): codes, high, scale bytes,
    // and the binary16 words among the scale bytes.
    std::uint32_t code_begin, code_end, high_begin, high_end, scale_begin, scale_end, halves;
};

const Format kFormats[] = {
    {"q4_0", ggml::Q4_0, QType::GGML_Q4_0, generic::WeightCodec::GgmlQ40, 18, 2, 18, 0, 0, 0, 2, 1},
    {"q4_1", ggml::Q4_1, QType::GGML_Q4_1, generic::WeightCodec::GgmlQ41, 20, 4, 20, 0, 0, 0, 4, 2},
    {"q5_0", ggml::Q5_0, QType::GGML_Q5_0, generic::WeightCodec::GgmlQ50, 22, 6, 22, 2, 6, 0, 2, 1},
    {"q5_1", ggml::Q5_1, QType::GGML_Q5_1, generic::WeightCodec::GgmlQ51, 24, 8, 24, 4, 8, 0, 4, 2},
    {"iq4_nl", ggml::IQ4_NL, QType::GGML_IQ4_NL, generic::WeightCodec::GgmlIq4Nl, 18, 2, 18, 0, 0,
     0, 2, 1},
    {"iq4_xs", ggml::IQ4_XS, QType::GGML_IQ4_XS, generic::WeightCodec::GgmlIq4Xs, 136, 8, 136, 4,
     8, 0, 4, 1},
    {"iq3_xxs", ggml::IQ3_XXS, QType::GGML_IQ3_XXS, generic::WeightCodec::GgmlIq3Xxs, 98, 2, 66,
     66, 98, 0, 2, 1},
    {"iq3_s", ggml::IQ3_S, QType::GGML_IQ3_S, generic::WeightCodec::GgmlIq3S, 110, 2, 106, 106,
     110, 0, 2, 1},
    {"iq2_xxs", ggml::IQ2_XXS, QType::GGML_IQ2_XXS, generic::WeightCodec::GgmlIq2Xxs, 66, 2, 66, 0,
     0, 0, 2, 1},
    {"iq2_xs", ggml::IQ2_XS, QType::GGML_IQ2_XS, generic::WeightCodec::GgmlIq2Xs, 74, 2, 66, 66, 74,
     0, 2, 1},
    {"iq2_s", ggml::IQ2_S, QType::GGML_IQ2_S, generic::WeightCodec::GgmlIq2S, 82, 2, 74, 74, 82, 0,
     2, 1},
    {"iq1_s", ggml::IQ1_S, QType::GGML_IQ1_S, generic::WeightCodec::GgmlIq1S, 50, 2, 34, 34, 50, 0,
     2, 1},
    {"iq1_m", ggml::IQ1_M, QType::GGML_IQ1_M, generic::WeightCodec::GgmlIq1M, 56, 0, 32, 32, 48, 48,
     56, 0},
    {"tq1_0", ggml::TQ1_0, QType::GGML_TQ1_0, generic::WeightCodec::GgmlTq10, 54, 0, 52, 0, 0, 52,
     54, 1},
    {"tq2_0", ggml::TQ2_0, QType::GGML_TQ2_0, generic::WeightCodec::GgmlTq20, 66, 0, 64, 0, 0, 64,
     66, 1},
    {"q1_0", ggml::Q1_0, QType::GGML_Q1_0, generic::WeightCodec::GgmlQ10, 18, 2, 18, 0, 0, 0, 2, 1},
    {"q2_0", ggml::Q2_0, QType::GGML_Q2_0, generic::WeightCodec::GgmlQ20, 18, 2, 18, 0, 0, 0, 2, 1},
};

// Every element through decode_element.
__global__ void decode_elements(generic::GemmArgs a, generic::WeightCodec codec, int rows,
                                float* out) {
    const std::int64_t n = static_cast<std::int64_t>(rows) * a.k;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        out[i] = generic::decode_element(a, codec, static_cast<int>(i / a.k),
                                         static_cast<int>(i % a.k));
    }
}

__device__ float bf16_bits_float(unsigned bits) {
    return __uint_as_float((bits & 0xffffU) << 16);
}

// One code of a decoder's pair as a float, in the element the decoder builds: FP16 for an F16Of
// decoder (the sm_75 build, generic_gemm.cuh kF16Operands), BF16 otherwise.
template <class D>
__device__ float code_bits_float(unsigned bits) {
    if constexpr (generic::F16Of<D>::value) {
        return __half2float(__ushort_as_half(static_cast<unsigned short>(bits & 0xffffU)));
    } else {
        return bf16_bits_float(bits);
    }
}

// Every element through the register-MMA decoder: thread t is lane t % 4 of a quad and walks
// the (row, K step) pairs t / 4, t / 4 + blockDim / 4, ...; value v of MMA j of its run is
// step value 16j + 4*lane + v (the spread assignment the activations are loaded in).
template <class D>
__global__ void decode_runs(generic::GemmArgs a, int rows, float* out) {
    if constexpr (generic::TableOf<D>::value) {
        D::stage();
        __syncthreads();
    }
    constexpr int S = generic::SplitsOf<D>::value;
    const int quad  = static_cast<int>(threadIdx.x) & 3;
    const int steps = a.k / 128;
    for (int item = static_cast<int>(blockIdx.x * (blockDim.x / 4) + threadIdx.x / 4);
         item < rows * steps; item += static_cast<int>(gridDim.x * (blockDim.x / 4))) {
        const int row = item / steps;
        const int k0  = (item % steps) * 128;
        const auto r  = D::load(a, row, k0 + 32 * quad);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned lo = 0, hi = 0;
            D::pair(r, j, lo, hi);
            const int g           = j / (8 / S);
            const float scale     = D::sub_scale(r, g);
            const unsigned w[4]   = {lo, lo >> 16, hi, hi >> 16};
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                float value = __fmul_rn(code_bits_float<D>(w[v]), scale);
                if constexpr (generic::MinTermOf<D>::value) {
                    value = __fadd_rn(value, -D::sub_min(r, g));
                }
                out[static_cast<std::int64_t>(row) * a.k + k0 + 16 * j + 4 * quad + v] = value;
            }
        }
    }
}

void launch_runs(generic::WeightCodec codec, const generic::GemmArgs& a, int rows, float* out) {
    const dim3 grid(64), block(128);
    switch (codec) {
    case generic::WeightCodec::GgmlQ40:
        decode_runs<generic::GgmlQ40Decoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlQ41:
        decode_runs<generic::GgmlQ41Decoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlQ50:
        decode_runs<generic::GgmlQ50Decoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlQ51:
        decode_runs<generic::GgmlQ51Decoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlIq4Nl:
        decode_runs<generic::GgmlIq4NlDecoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlIq4Xs:
        decode_runs<generic::GgmlIq4XsDecoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlIq3Xxs:
        decode_runs<generic::GgmlIq3XxsDecoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlIq3S:
        decode_runs<generic::GgmlIq3SDecoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlIq2Xxs:
        decode_runs<generic::GgmlIq2XxsDecoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlIq2Xs:
        decode_runs<generic::GgmlIq2XsDecoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlIq2S:
        decode_runs<generic::GgmlIq2SDecoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlIq1S:
        decode_runs<generic::GgmlIq1SDecoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlIq1M:
        decode_runs<generic::GgmlIq1MDecoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlTq10:
        decode_runs<generic::GgmlTq10Decoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlTq20:
        decode_runs<generic::GgmlTq20Decoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlQ10:
        decode_runs<generic::GgmlQ10Decoder><<<grid, block>>>(a, rows, out);
        break;
    case generic::WeightCodec::GgmlQ20:
        decode_runs<generic::GgmlQ20Decoder><<<grid, block>>>(a, rows, out);
        break;
    default:
        throw std::runtime_error("no decoder");
    }
}

std::uint16_t finite_half(std::mt19937& rng) {
    // Sign, an exponent inside binary16's normal range, any mantissa.
    const auto exponent = static_cast<std::uint16_t>(1 + rng() % 30);
    return static_cast<std::uint16_t>(((rng() & 1U) << 15) | (exponent << 10) | (rng() & 0x3ffU));
}

struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { cudaMalloc(&p, bytes == 0 ? 1 : bytes); }
    ~Device() { cudaFree(p); }
};

int check_format(const Format& f, int rows, int k, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const auto values = ninferno::ggml_block_values(f.qtype);
    const auto plane  = ninferno::ggml_plane_bytes(f.qtype);
    const int blocks  = k / static_cast<int>(values);
    const std::size_t nb = static_cast<std::size_t>(rows) * blocks;
    std::vector<std::uint8_t> gguf(nb * f.block_bytes);
    for (auto& b : gguf) b = static_cast<std::uint8_t>(rng());
    for (std::size_t b = 0; b < nb; ++b) {
        for (std::uint32_t h = 0; h < f.halves; ++h) {
            const auto word = finite_half(rng);
            std::memcpy(gguf.data() + b * f.block_bytes + f.scale_begin + 2 * h, &word, 2);
        }
        if (f.qtype == QType::GGML_IQ1_M) {
            // d's nibbles 0..3 top the scale words 0..3.
            const auto word = finite_half(rng);
            auto* sc        = gguf.data() + b * f.block_bytes + f.scale_begin;
            for (int i = 0; i < 4; ++i) {
                sc[2 * i + 1] = static_cast<std::uint8_t>((sc[2 * i + 1] & 0x0FU) |
                                                          (((word >> (4 * i)) & 0xFU) << 4));
            }
        }
    }
    // The planes, 256-byte aligned as ggml_planes_v1 lays them out.
    const auto align = [](std::size_t n) { return (n + 255) / 256 * 256; };
    const std::size_t code_bytes = nb * plane.code, high_bytes = nb * plane.high;
    const std::size_t high_offset = align(code_bytes), scale_offset = high_offset + align(high_bytes);
    std::vector<std::uint8_t> payload(scale_offset + nb * plane.scale);
    for (std::size_t b = 0; b < nb; ++b) {
        const auto* src = gguf.data() + b * f.block_bytes;
        std::memcpy(payload.data() + b * plane.code, src + f.code_begin, f.code_end - f.code_begin);
        std::memcpy(payload.data() + high_offset + b * plane.high, src + f.high_begin,
                    f.high_end - f.high_begin);
        std::memcpy(payload.data() + scale_offset + b * plane.scale, src + f.scale_begin,
                    f.scale_end - f.scale_begin);
    }
    std::vector<float> expect(static_cast<std::size_t>(rows) * k);
    ggml::dequantize(f.ggml_type, reinterpret_cast<const std::byte*>(gguf.data()), expect.size(),
                     expect.data());

    Device weights(payload.size()), out(expect.size() * sizeof(float));
    cudaMemcpy(weights.p, payload.data(), payload.size(), cudaMemcpyHostToDevice);
    generic::GemmArgs a;
    a.k              = k;
    a.codes          = static_cast<const std::uint8_t*>(weights.p);
    a.high           = static_cast<const std::uint8_t*>(weights.p) + high_offset;
    a.scales         = static_cast<const std::uint8_t*>(weights.p) + scale_offset;
    a.groups_per_row = blocks;

    int failures = 0;
    std::vector<float> got(expect.size());
    for (const bool runs : {false, true}) {
        cudaMemset(out.p, 0xff, expect.size() * sizeof(float));
        if (runs) {
            launch_runs(f.codec, a, rows, static_cast<float*>(out.p));
        } else {
            decode_elements<<<256, 256>>>(a, f.codec, rows, static_cast<float*>(out.p));
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::cerr << f.name << ": kernel failed: " << cudaGetErrorString(cudaGetLastError()) << "\n";
            return 1;
        }
        cudaMemcpy(got.data(), out.p, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::size_t bad = 0, first = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (std::memcmp(&got[i], &expect[i], 4) != 0) {
                if (bad++ == 0) first = i;
            }
        }
        const char* path = runs ? "register-MMA decoder" : "decode_element";
        if (bad) {
            std::cerr << f.name << " " << path << ": " << bad << " of " << got.size()
                      << " values differ from ggml; first at row " << first / k << " column "
                      << first % k << ": " << got[first] << " vs " << expect[first] << "\n";
            ++failures;
        } else {
            std::cout << f.name << " " << path << ": " << got.size()
                      << " values bit-exact with ggml\n";
        }
    }
    return failures;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "CUDA unavailable; skipping ggml decode test\n";
        return 77;
    }
    int failures = 0;
    std::uint32_t seed = 20260926U;
    for (const Format& f : kFormats) {
        // K of whole 128-value steps (the register MMA's) and 256-value superblocks.
        failures += check_format(f, 37, 1024, seed++);
        failures += check_format(f, 5, 2816, seed++);
    }
    std::cout << (failures ? "FAILED" : "PASSED") << "\n";
    return failures ? 1 : 0;
}
