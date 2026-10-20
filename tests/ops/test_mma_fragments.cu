// The register-fragment MMAs and ldmatrix forms of ops/common/mma.cuh against a host product, on
// whatever this build issues for them: the sm_80 instructions, Turing's composed k8 shapes, or the
// sm_70 FMA / dp4a forms. Operands are small integers (and BF16 ones scaled by 2^-100 for the
// full-range form), so every sum is exact and the comparison is bit for bit. ldmatrix_x4 /
// ldmatrix_x2_t from shared memory must give the fragments the layout comments name.

#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using namespace ninferno::ops;

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Inputs (row-major): A16 16x16 and B16 16x8 as small integers, A8 16x32 and B8 32x8 int8, AT 16x8
// and BT 8x8 for tf32 (floats).
struct Inputs {
    float a16[16 * 16];
    float b16[16 * 8];
    std::int8_t a8[16 * 32];
    std::int8_t b8[32 * 8];
    float at[16 * 8];
    float bt[8 * 8];
};

// Per lane: 4 outputs of each product, plus the ldmatrix fragments.
enum Out { kF16, kBf16, kBf16Full, kBf16Prepared, kAcc16, kS8, kS8K16, kTf32, kOuts };

struct Results {
    float f[kOuts][32][4];
    int i[2][32][4];
    unsigned ldm[32][4];  // ldmatrix_x4 of A16 (as f16)
    unsigned frag[32][4]; // the same fragments built from global memory
    unsigned ldt[32][2];  // ldmatrix_x2_t of B16
    unsigned fragt[32][2];
};

__device__ unsigned pack_h(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

__device__ unsigned pack_b(float lo, float hi) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

__device__ unsigned pack_s8(const std::int8_t* p) {
    return static_cast<unsigned>(static_cast<std::uint8_t>(p[0])) |
           (static_cast<unsigned>(static_cast<std::uint8_t>(p[1])) << 8) |
           (static_cast<unsigned>(static_cast<std::uint8_t>(p[2])) << 16) |
           (static_cast<unsigned>(static_cast<std::uint8_t>(p[3])) << 24);
}

__global__ void fragments(const Inputs* in, Results* out) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const float* A  = in->a16;
    const float* B  = in->b16;
    const auto a_at = [&](int r, int c) { return A[r * 16 + c]; };
    const auto b_at = [&](int k, int n) { return B[k * 8 + n]; };

    // 16-bit fragments (scale applies to A, for the full-range form)
    const auto frag16 = [&](auto pack, float scale, unsigned (&a)[4], unsigned (&b)[2]) {
        a[0] = pack(scale * a_at(g, 2 * t), scale * a_at(g, 2 * t + 1));
        a[1] = pack(scale * a_at(g + 8, 2 * t), scale * a_at(g + 8, 2 * t + 1));
        a[2] = pack(scale * a_at(g, 2 * t + 8), scale * a_at(g, 2 * t + 9));
        a[3] = pack(scale * a_at(g + 8, 2 * t + 8), scale * a_at(g + 8, 2 * t + 9));
        b[0] = pack(b_at(2 * t, g), b_at(2 * t + 1, g));
        b[1] = pack(b_at(2 * t + 8, g), b_at(2 * t + 9, g));
    };
    unsigned a[4], b[2];
    float c[4];
    const auto reset = [&](float v) { c[0] = c[1] = c[2] = c[3] = v; };
    const auto store = [&](int which) {
        for (int j = 0; j < 4; ++j) out->f[which][lane][j] = c[j];
    };

    frag16(pack_h, 1.0F, a, b);
    for (int j = 0; j < 4; ++j) out->frag[lane][j] = a[j];
    reset(1.0F);
    mma_f16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b[0], b[1]);
    store(kF16);

    unsigned acc[2] = {pack_h(1.0F, 1.0F), pack_h(1.0F, 1.0F)};
    mma_f16_acc16(acc[0], acc[1], a[0], a[1], a[2], a[3], b[0], b[1]);
    {
        const float2 top = __half22float2(*reinterpret_cast<const __half2*>(&acc[0]));
        const float2 bot = __half22float2(*reinterpret_cast<const __half2*>(&acc[1]));
        c[0] = top.x, c[1] = top.y, c[2] = bot.x, c[3] = bot.y;
        store(kAcc16);
    }

    frag16(pack_b, 1.0F, a, b);
    reset(1.0F);
    mma_bf16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b[0], b[1]);
    store(kBf16);
    reset(1.0F);
    mma_bf16_prepared(c[0], c[1], c[2], c[3], mma_a_bf16(a[0]), mma_a_bf16(a[1]), mma_a_bf16(a[2]),
                      mma_a_bf16(a[3]), b[0], b[1]);
    store(kBf16Prepared);

    frag16(pack_b, 0x1p-100F, a, b);
    reset(0.0F);
    mma_bf16_full_range(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b[0], b[1]);
    store(kBf16Full);

    // s8 m16n8k32 and m16n8k16
    {
        const std::int8_t* A8 = in->a8;
        const std::int8_t* B8 = in->b8;
        std::int8_t col[32];
        const auto bq = [&](int k0, int n) {
            for (int j = 0; j < 4; ++j) col[j] = B8[(k0 + j) * 8 + n];
            return pack_s8(col);
        };
        const unsigned s0 = pack_s8(A8 + g * 32 + 4 * t), s1 = pack_s8(A8 + (g + 8) * 32 + 4 * t);
        const unsigned s2 = pack_s8(A8 + g * 32 + 4 * t + 16);
        const unsigned s3 = pack_s8(A8 + (g + 8) * 32 + 4 * t + 16);
        int ci[4]         = {5, 5, 5, 5};
        mma_s8(ci[0], ci[1], ci[2], ci[3], s0, s1, s2, s3, bq(4 * t, g), bq(4 * t + 16, g));
        for (int j = 0; j < 4; ++j) out->i[0][lane][j] = ci[j];
        int ck[4] = {5, 5, 5, 5};
        mma_s8_k16(ck[0], ck[1], ck[2], ck[3], s0, s1, bq(4 * t, g));
        for (int j = 0; j < 4; ++j) out->i[1][lane][j] = ck[j];
    }

    // tf32 m16n8k8
    {
        const float* AT = in->at;
        const float* BT = in->bt;
        reset(1.0F);
        mma_tf32(c[0], c[1], c[2], c[3], AT[g * 8 + t], AT[(g + 8) * 8 + t], AT[g * 8 + t + 4],
                 AT[(g + 8) * 8 + t + 4], BT[t * 8 + g], BT[(t + 4) * 8 + g]);
        store(kTf32);
    }

    // ldmatrix from shared memory: A16 row-major 16x16 f16, B16 row-major 16x8 f16
    __shared__ __align__(16) __half sa[16 * 16];
    __shared__ __align__(16) __half sb[16 * 8];
    for (int i = lane; i < 256; i += 32) sa[i] = __float2half(A[i]);
    for (int i = lane; i < 128; i += 32) sb[i] = __float2half(B[i]);
    __syncwarp();
    // matrix m = lane / 8: m0 rows 0-7 cols 0-7, m1 rows 8-15, m2 rows 0-7 cols 8-15, m3 rows 8-15
    const int row = (lane & 7) + ((lane >> 3) & 1) * 8, colh = (lane >> 4) * 8;
    const unsigned addr_a = static_cast<unsigned>(__cvta_generic_to_shared(&sa[row * 16 + colh]));
    unsigned l[4];
    ldmatrix_x4(l[0], l[1], l[2], l[3], addr_a);
    for (int j = 0; j < 4; ++j) out->ldm[lane][j] = l[j];
    const unsigned addr_b = static_cast<unsigned>(__cvta_generic_to_shared(&sb[(lane & 15) * 8]));
    unsigned lt[2];
    ldmatrix_x2_t(lt[0], lt[1], addr_b);
    out->ldt[lane][0]   = lt[0];
    out->ldt[lane][1]   = lt[1];
    out->fragt[lane][0] = pack_h(b_at(2 * t, g), b_at(2 * t + 1, g));
    out->fragt[lane][1] = pack_h(b_at(2 * t + 8, g), b_at(2 * t + 9, g));
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    Inputs in{};
    unsigned seed  = 12345;
    const auto rnd = [&seed](int lo, int hi) {
        seed = seed * 1664525U + 1013904223U;
        return lo + static_cast<int>((seed >> 8) % static_cast<unsigned>(hi - lo + 1));
    };
    for (float& v : in.a16) v = static_cast<float>(rnd(-3, 3));
    for (float& v : in.b16) v = static_cast<float>(rnd(-3, 3));
    for (auto& v : in.a8) v = static_cast<std::int8_t>(rnd(-128, 127));
    for (auto& v : in.b8) v = static_cast<std::int8_t>(rnd(-128, 127));
    for (float& v : in.at) v = static_cast<float>(rnd(-7, 7));
    for (float& v : in.bt) v = static_cast<float>(rnd(-7, 7));

    Inputs* d_in   = nullptr;
    Results* d_out = nullptr;
    if (cudaMalloc(&d_in, sizeof(Inputs)) != cudaSuccess ||
        cudaMalloc(&d_out, sizeof(Results)) != cudaSuccess)
        return 1;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 1;
    Results r{};
    if (cudaMemcpyAsync(d_in, &in, sizeof(Inputs), cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return 1;
    fragments<<<1, 32, 0, stream>>>(d_in, d_out);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(&r, d_out, sizeof(Results), cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        std::printf("FAIL: kernel or copy\n");
        return 1;
    }
    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_out);

    int failures   = 0;
    const auto chk = [&failures](const char* what, int lane, int j, double got, double want) {
        if (got != want && failures++ < 24)
            std::printf("FAIL %s lane %d c%d: %.9g, want %.9g\n", what, lane, j, got, want);
    };
    const char* kNames[kOuts] = {
        "mma_f16", "mma_bf16", "mma_bf16_full_range", "mma_bf16_prepared", "mma_f16_acc16", "",
        "",        "mma_tf32"};
    for (int lane = 0; lane < 32; ++lane) {
        const int g = lane >> 2, t = lane & 3;
        for (int j = 0; j < 4; ++j) {
            const int row = g + (j >= 2 ? 8 : 0), n = 2 * t + (j & 1);
            double s16 = 0, s8 = 0, s8k16 = 0, st = 0;
            for (int k = 0; k < 16; ++k) s16 += double(in.a16[row * 16 + k]) * in.b16[k * 8 + n];
            for (int k = 0; k < 32; ++k) s8 += double(in.a8[row * 32 + k]) * in.b8[k * 8 + n];
            for (int k = 0; k < 16; ++k) s8k16 += double(in.a8[row * 32 + k]) * in.b8[k * 8 + n];
            for (int k = 0; k < 8; ++k) st += double(in.at[row * 8 + k]) * in.bt[k * 8 + n];
            for (int o : {kF16, kBf16, kBf16Prepared, kAcc16})
                chk(kNames[o], lane, j, r.f[o][lane][j], 1.0 + s16);
            chk(kNames[kBf16Full], lane, j, r.f[kBf16Full][lane][j],
                double(std::ldexp(static_cast<float>(s16), -100)));
            chk(kNames[kTf32], lane, j, r.f[kTf32][lane][j], 1.0 + st);
            chk("mma_s8", lane, j, r.i[0][lane][j], 5.0 + s8);
            chk("mma_s8_k16", lane, j, r.i[1][lane][j], 5.0 + s8k16);
            chk("ldmatrix_x4", lane, j, r.ldm[lane][j], r.frag[lane][j]);
        }
        for (int j = 0; j < 2; ++j) chk("ldmatrix_x2_t", lane, j, r.ldt[lane][j], r.fragt[lane][j]);
    }
    if (failures != 0) {
        std::printf("FAIL: %d mismatches\n", failures);
        return 1;
    }
    std::printf("PASS: f16, f16 acc16, bf16 (plain, prepared, full range), s8 k32/k16, tf32, "
                "ldmatrix x4 / x2.trans\n");
    return 0;
}
