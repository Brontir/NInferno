// The 4-column runtime-K plain instance of ggml's vector kernel has a twin bounded to the registers
// of 3 columns (ggml_vec.cu). Two properties: the twin's bytes equal the plain instance's at shapes
// below and above the plain instance's slot count (a bound moves registers, never the sums), and
// the launch's choice is the documented one: the twin when it does not spill and saves a round of
// row groups, the plain instance otherwise, read from the launch's own description.
#include "ops/generic/ggml_mmq/ggml_mmq.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace ninferno::ops::ggml_mmq;

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL: " << what << '\n';
    }
}

void cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cout << "CUDA error in " << what << ": " << cudaGetErrorString(e) << '\n';
        std::exit(1);
    }
}

constexpr int kColumns   = 4;
constexpr int kQ4KBlock  = 144;
constexpr int kBlockVals = 256;

struct Problem {
    int rows;
    int k;
};

struct Device {
    void* weight     = nullptr;
    void* activation = nullptr;
    float* out       = nullptr;
};

Device make(const Problem& p, bool describe_only) {
    Device d;
    const int bpr            = p.k / kBlockVals;
    const std::size_t wbytes = std::size_t(p.rows) * bpr * kQ4KBlock;
    const std::size_t abytes = vec_activation_bytes(p.k, kColumns);
    cuda(cudaMalloc(&d.weight, describe_only ? 16 : wbytes), "weight");
    cuda(cudaMalloc(&d.activation, abytes), "activation");
    cuda(cudaMalloc(reinterpret_cast<void**>(&d.out), sizeof(float) * kColumns * p.rows), "out");
    if (describe_only) return d;
    std::mt19937 rng(7);
    std::vector<std::uint8_t> w(wbytes);
    for (auto& b : w) b = std::uint8_t(rng());
    for (std::size_t blk = 0; blk < wbytes / kQ4KBlock; ++blk) { // d, dmin: small finite halves
        w[blk * kQ4KBlock + 0] = 0x00;
        w[blk * kQ4KBlock + 1] = 0x2C;
        w[blk * kQ4KBlock + 2] = 0x00;
        w[blk * kQ4KBlock + 3] = 0x28;
    }
    cuda(cudaMemcpy(d.weight, w.data(), wbytes, cudaMemcpyHostToDevice), "weight copy");
    std::vector<std::uint16_t> x(std::size_t(kColumns) * p.k);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : x) {
        const float f = dist(rng);
        std::uint32_t u;
        std::memcpy(&u, &f, 4);
        v = std::uint16_t(u >> 16);
    }
    void* xd = nullptr;
    cuda(cudaMalloc(&xd, x.size() * 2), "x");
    cuda(cudaMemcpy(xd, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x copy");
    vec_quantize_activation(static_cast<const __nv_bfloat16*>(xd), p.k, p.k, kColumns, d.activation,
                            nullptr);
    cuda(cudaDeviceSynchronize(), "quantize");
    cudaFree(xd);
    return d;
}

void free_all(Device& d) {
    cudaFree(d.weight);
    cudaFree(d.activation);
    cudaFree(d.out);
}

void product(const Problem& p, const Device& d) {
    Planes w;
    w.codes = d.weight;
    VecOut o;
    o.epilogue = VecOut::Epilogue::StoreF32;
    o.f32      = d.out;
    o.f32_ld   = p.rows;
    vec_product(Type::Q4_K, w, w, p.k / kBlockVals, p.rows, p.k, d.activation, kColumns, o,
                nullptr);
}

std::vector<float> run(const Problem& p, Device& d, VecInstance which) {
    cuda(cudaMemset(d.out, 0xFF, sizeof(float) * kColumns * p.rows), "clear");
    {
        ScopedVecInstance force(which);
        product(p, d);
    }
    cuda(cudaDeviceSynchronize(), "product");
    std::vector<float> r(std::size_t(kColumns) * p.rows);
    cuda(cudaMemcpy(r.data(), d.out, r.size() * 4, cudaMemcpyDeviceToHost), "read");
    return r;
}

LaunchShape describe(const Problem& p, const Device& d, VecInstance which) {
    LaunchShape shape;
    ScopedDescribe scope(shape);
    ScopedVecInstance force(which);
    product(p, d);
    return shape;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0 || !vec_supported(Type::Q4_K)) {
        std::cout << "no CUDA device: skipped\n";
        return 0;
    }
    // Bytes: the two instances at shapes below, at and above the plain slot count of an 82-SM board
    // (480 groups: 410 plain slots) and a 128-SM one (1024 groups: 640 plain slots).
    for (const Problem p : {Problem{3840, 15360}, Problem{4096, 3840}, Problem{8192, 4096}}) {
        Device d         = make(p, false);
        const auto plain = run(p, d, VecInstance::Plain);
        const auto twin  = run(p, d, VecInstance::Twin);
        const auto autos = run(p, d, VecInstance::Auto);
        expect(std::memcmp(plain.data(), twin.data(), plain.size() * 4) == 0,
               "twin bytes equal plain bytes");
        expect(std::memcmp(plain.data(), autos.data(), plain.size() * 4) == 0,
               "auto bytes equal plain bytes");
        bool finite = true;
        for (float v : plain) finite = finite && v == v && v != 0.0f;
        expect(finite, "the product is a finite non-trivial result");
        free_all(d);
    }
    // Choice: read the slots of both instances off a launch with more groups than any board holds,
    // then check what Auto describes at each shape against the documented rule.
    const Problem huge{1 << 20, 256};
    Device dh                    = make(huge, true);
    const LaunchShape plain_huge = describe(huge, dh, VecInstance::Plain);
    const LaunchShape twin_huge  = describe(huge, dh, VecInstance::Twin);
    free_all(dh);
    const unsigned plain_slots = plain_huge.grid_x;
    // A twin that spilled has the plain instance's grid (and is never taken); otherwise its own.
    const bool twin_spills    = twin_huge.local_bytes != 0;
    const unsigned twin_slots = twin_spills ? plain_slots : twin_huge.grid_x;
    std::printf("plain: %d regs, %zu B local, %u slots; twin: %d regs, %zu B local, %u slots\n",
                plain_huge.registers, plain_huge.local_bytes, plain_slots, twin_huge.registers,
                twin_huge.local_bytes, twin_slots);
    expect(plain_huge.local_bytes == 0, "the plain instance does not spill");
    expect(twin_huge.registers <= plain_huge.registers, "the twin uses no more registers");
    for (const Problem p : {Problem{3840, 15360}, Problem{4096, 12288}, Problem{8192, 4096},
                            Problem{2048, 3840}, Problem{1 << 20, 256}}) {
        Device d              = make(p, true);
        const unsigned groups = unsigned((p.rows + 7) / 8);
        const auto rounds     = [&](unsigned n) { return (groups + n - 1) / n; };
        const bool want_twin  = !twin_spills && rounds(twin_slots) < rounds(plain_slots);
        const LaunchShape got = describe(p, d, VecInstance::Auto);
        const LaunchShape ref = describe(p, d, want_twin ? VecInstance::Twin : VecInstance::Plain);
        std::printf(
            "rows %d k %d: groups %u, plain rounds %u, twin rounds %u -> %s (grid %u, %d regs)\n",
            p.rows, p.k, groups, rounds(plain_slots), rounds(twin_slots),
            want_twin ? "twin" : "plain", got.grid_x, got.registers);
        expect(got.registers == ref.registers && got.grid_x == ref.grid_x &&
                   got.local_bytes == ref.local_bytes,
               "Auto launches the instance the rule names");
        if (want_twin) expect(got.local_bytes == 0, "a twin taken has no local memory");
        free_all(d);
    }
    if (failures == 0) std::cout << "ok\n";
    return failures == 0 ? 0 : 1;
}
