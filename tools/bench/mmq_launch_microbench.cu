// Microbench of one product of ggml's matrix kernel (Q4_K) for the launch law (DECISIONS 77, #524):
// the per-call time of rows x k x columns through ggml_mmq::product, the launch forced by the grid
// hook's environment (tools/bench/524_grid_hook.patch: NINFERNO_MMQ_J, NINFERNO_MMQ_BLOCKS,
// NINFERNO_MMQ_TILE_MODE, NINFERNO_MMQ_PRINT). Calls run back to back on one stream, so a call's
// time is its tile kernel, its fixup kernel (if any) and the launch gaps between them. BENCH ONLY:
// built by hand against a hooked build's archives (run_524_turing_grid.sh build), never by CMake:
//   nvcc -O2 -std=c++20 -arch=sm_75 -Isrc -Iinclude -I$B/src/core/generated_public \
//     tools/bench/mmq_launch_microbench.cu -o mmq_launch_microbench \
//     $B/src/ops/libninferno_ggml_mmq.a $B/src/core/libninferno_core.a \
//     $B/src/core/libninferno_core_host.a $B/libninferno_sys.a -lcuda -lnvidia-ml
//   (B = the hooked build directory; build its ninferno_ggml_mmq, ninferno_core and
//   ninferno_core_host targets first)
//   mmq_launch_microbench ROWS K COLUMNS [CALLS] [REPS]
// Prints the median and the least per-call microseconds over REPS batches of CALLS calls. The
// weights are random Q4_K blocks (d = dmin = 2^-8), the activation a normal BF16 one: the time does
// not depend on the values.
#include "ops/generic/ggml_mmq/ggml_mmq.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace ninferno::ops;

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_));                          \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s ROWS K COLUMNS [CALLS] [REPS]\n", argv[0]);
        return 2;
    }
    const int rows           = std::atoi(argv[1]);
    const int k              = std::atoi(argv[2]);
    const int cols           = std::atoi(argv[3]);
    const int calls          = argc > 4 ? std::atoi(argv[4]) : 200;
    const int reps           = argc > 5 ? std::atoi(argv[5]) : 7;
    const int bpr            = k / 256;
    const std::size_t wbytes = std::size_t(rows) * bpr * 144;
    std::vector<std::uint8_t> w(wbytes);
    std::mt19937 rng(1234);
    for (auto& b : w) b = std::uint8_t(rng());
    for (std::size_t blk = 0; blk < std::size_t(rows) * bpr; ++blk) {
        std::uint8_t* p = w.data() + blk * 144;
        p[0]            = 0x00;
        p[1]            = 0x1c; // d    = 2^-8 (fp16)
        p[2]            = 0x00;
        p[3]            = 0x1c; // dmin = 2^-8
    }
    std::vector<__nv_bfloat16> x(std::size_t(k) * cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = __float2bfloat16(nd(rng));
    void *dw, *dx, *dact, *dout, *dfix;
    CK(cudaMalloc(&dw, wbytes));
    CK(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dx, x.size() * sizeof(__nv_bfloat16)));
    CK(cudaMemcpy(dx, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dact, ggml_mmq::activation_bytes(k, cols)));
    CK(cudaMalloc(&dout, std::size_t(rows) * cols * sizeof(float)));
    const std::size_t fb = ggml_mmq::fixup_bound(cols);
    CK(cudaMalloc(&dfix, fb));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    ggml_mmq::quantize_activation(ggml_mmq::Type::Q4_K, static_cast<const __nv_bfloat16*>(dx), k, k,
                                  cols, dact, s);
    const ggml_mmq::Planes planes{dw, nullptr, nullptr};
    auto call = [&] {
        ggml_mmq::product(ggml_mmq::Type::Q4_K, planes, bpr, rows, k, dact, cols,
                          static_cast<float*>(dout), rows, dfix, s);
    };
    for (int i = 0; i < 20; ++i) call();
    CK(cudaStreamSynchronize(s));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<double> us;
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(a, s));
        for (int i = 0; i < calls; ++i) call();
        CK(cudaEventRecord(b, s));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        us.push_back(1e3 * ms / calls);
    }
    CK(cudaGetLastError());
    std::sort(us.begin(), us.end());
    std::printf("rows %d k %d cols %d J %d: median %.2f us min %.2f us (reps %d x %d calls)\n",
                rows, k, cols, ggml_mmq::column_tile(ggml_mmq::Type::Q4_K, rows, cols, k),
                us[us.size() / 2], us[0], reps, calls);
    return 0;
}
