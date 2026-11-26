// cluster_probe: what a thread-block cluster costs the register-MMA K split on one card, the
// cluster facts of its architecture (#1228). The rmma cN schedules (generic_gemm.cuh, C 2 and 8,
// sm_90+) split K over the C CTAs of a cluster; rank 0 adds the C - 1 other ranks' partial tiles
// out of their shared memory between two cluster barriers. Three kinds of fact, each at C 2 and 8:
//   cluster_sync_ns_cC      one cluster.sync() of a cluster of C CTAs (256 threads each, the w1x8
//                           schedule's CTA), timed by %globaltimer over kSyncs back-to-back syncs;
//   cluster_rank_ns_cC_sS   rank 0 reading one other rank's partial as the epilogue does: one warp,
//                           S x 4 floats a lane (S weight sets: 1 store / add, 2 SwiGLU; 512 B a
//                           set), the ranks walked by a `#pragma unroll 1` loop and added into the
//                           warp's accumulators; a round = two syncs + C - 1 such reads, so the
//                           fact is (round - 2 x sync) / (C - 1);
//   cluster_slots_cC        the CTAs a launch of C-CTA clusters holds at once: the driver's
//                           cudaOccupancyMaxActiveClusters x C for a 256-thread CTA whose plain
//                           launch holds kPlainPerSm CTAs an SM (dynamic shared memory sized so).
//                           A cluster lives in one GPC, and a cluster launch was seen to hold 2
//                           CTAs an SM whatever its threads (#1228, RTX 5090 / 5070 Ti), fewer than
//                           a plain one. Self-check: a spin grid of 2 x slots cluster CTAs takes
//                           two rounds of its spin (within kRoundShare), a plain one of the same
//                           CTAs fewer.
// Each sync and rank time is the median over the clusters of one launch (kClusters clusters at
// once), the fact the median over kRepeats launches. Thread-block clusters are sm_90+: below it
// the probe measures nothing and its self-check fails (an architecture without clusters has no
// cluster section).
//
// Build (CUDA 13.x), as `isa_probe.py run --section cluster` does:
//   nvcc -O3 -std=c++17 -arch=sm_<arch> -o cluster_probe tools/bench/isa_probe/cluster_probe.cu
// Run (a few seconds; the GPU otherwise idle): `fact <name> <value> <unit>` lines; exit 0 with
// `selfcheck ok`, 3 when a result fails the self-check (its reasons printed), 2 on a CUDA error.

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));        \
            std::exit(2);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

namespace cg = cooperative_groups;

constexpr int kThreads       = 256;  // the w1x8 schedule's CTA: eight warps
constexpr int kClusters      = 16;   // clusters of one timed launch
constexpr int kSyncs         = 2000; // syncs a timed sync launch
constexpr int kRounds        = 500;  // reduce rounds a timed reduce launch
constexpr int kRepeats       = 7;
constexpr int kPlainPerSm    = 4; // the slots CTA: a plain launch holds four an SM
constexpr long long kSpinNs  = 20000;
constexpr double kRoundShare = 0.15;
constexpr int kSizes[]       = {2, 8}; // the product's cluster sizes (generic_gemm_launch.cuh)
constexpr long long kWarmNs  = 200'000'000; // the hold before each cluster size's timings

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// The hold: FFMAs on every SM until `ns` after each CTA started, so the timings that follow run
// at the working clock (launch_probe.cu's).
__global__ void hold(long long ns, float* sink) {
    const unsigned long long end = gtime() + ns;
    float x                      = threadIdx.x * 1e-3f;
    do {
#pragma unroll
        for (int i = 0; i < 256; ++i) x = fmaf(x, 0.999f, 0.001f);
    } while (gtime() < end);
    if (x == 12345.f) sink[0] = x;
}

// The syncs; `mhz` gets rank 0's SM clock over them (clock64 / %globaltimer), for the note.
__global__ void syncs(int n, unsigned long long* out, double* mhz) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    const unsigned long long t0 = gtime();
    const long long c0          = clock64();
    for (int i = 0; i < n; ++i) cl.sync();
    const long long c1          = clock64();
    const unsigned long long t1 = gtime();
    if (threadIdx.x == 0 && cl.block_rank() == 0) {
        out[blockIdx.x / cl.num_blocks()] = t1 - t0;
        mhz[blockIdx.x / cl.num_blocks()] = 1e3 * static_cast<double>(c1 - c0) / (t1 - t0);
    }
#endif
}

// The epilogue's reduce (generic_gemm.cuh, C > 1): every rank writes its partial
// [S][4][32] floats, rank 0's first warp adds the other ranks' between two syncs.
template <int S>
__global__ void reduce(int n, unsigned long long* out, float* sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __shared__ float part[S][4][32];
    cg::cluster_group cl = cg::this_cluster();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float acc[S][4];
#    pragma unroll
    for (int s = 0; s < S; ++s)
#    pragma unroll
        for (int e = 0; e < 4; ++e) acc[s][e] = 1.0f + s + e + cl.block_rank();
    if (warp == 0)
#    pragma unroll
        for (int s = 0; s < S; ++s)
#    pragma unroll
            for (int e = 0; e < 4; ++e) part[s][e][lane] = acc[s][e];
    cl.sync();
    const unsigned long long t0 = gtime();
    for (int it = 0; it < n; ++it) {
        cl.sync();
        if (cl.block_rank() == 0 && warp == 0) {
#    pragma unroll 1
            for (unsigned r = 1; r < cl.num_blocks(); ++r) {
                const auto& other = *cl.map_shared_rank(&part, r);
#    pragma unroll
                for (int s = 0; s < S; ++s)
#    pragma unroll
                    for (int e = 0; e < 4; ++e) acc[s][e] += other[s][e][lane];
            }
        }
        cl.sync();
    }
    const unsigned long long t1 = gtime();
    if (threadIdx.x == 0 && cl.block_rank() == 0) out[blockIdx.x / cl.num_blocks()] = t1 - t0;
    float x = 0.f;
#    pragma unroll
    for (int s = 0; s < S; ++s)
#    pragma unroll
        for (int e = 0; e < 4; ++e) x += acc[s][e];
    if (x == 1234.5f) sink[0] = x;
#endif
}

__global__ void slots_kernel(float* p) {
    extern __shared__ float s[];
    s[threadIdx.x] = threadIdx.x;
    if (p) p[blockIdx.x] = s[threadIdx.x];
}

__global__ void spin(long long ns, float* sink) {
    extern __shared__ float s[];
    const unsigned long long t0 = gtime();
    unsigned long long t;
    do { t = gtime(); } while (static_cast<long long>(t - t0) < ns);
    s[threadIdx.x] = static_cast<float>(t & 7);
    if (sink && s[threadIdx.x] == 9.f) sink[0] = 1.f;
}

double median(std::vector<double> d) {
    std::sort(d.begin(), d.end());
    return d[d.size() / 2];
}

std::vector<std::string> g_failures;

void check(bool ok, const std::string& what) {
    if (!ok) g_failures.push_back(what);
}

cudaLaunchConfig_t config(int C, int ctas, int threads, size_t smem, cudaLaunchAttribute* a) {
    a[0].id                = cudaLaunchAttributeClusterDimension;
    a[0].val.clusterDim.x  = C;
    a[0].val.clusterDim.y  = 1;
    a[0].val.clusterDim.z  = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim            = dim3(ctas);
    cfg.blockDim           = dim3(threads);
    cfg.dynamicSmemBytes   = smem;
    cfg.attrs              = C > 0 ? a : nullptr;
    cfg.numAttrs           = C > 0 ? 1 : 0;
    return cfg;
}

// One cluster.sync() of C-CTA clusters in ns: the median over the clusters of one launch of
// kSyncs syncs, the median over kRepeats launches (the first one dropped); `mhz` the SM clock
// rank 0 saw over them, the median likewise.
double sync_ns(int C, unsigned long long* d, double* dm, double* mhz) {
    cudaLaunchAttribute a[1];
    cudaLaunchConfig_t cfg = config(C, kClusters * C, kThreads, 0, a);
    std::vector<double> meds, clocks;
    std::vector<unsigned long long> h(kClusters);
    std::vector<double> hm(kClusters);
    for (int r = 0; r <= kRepeats; ++r) {
        CHECK(cudaLaunchKernelEx(&cfg, syncs, kSyncs, d, dm));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h.data(), d, kClusters * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hm.data(), dm, kClusters * sizeof(double), cudaMemcpyDeviceToHost));
        if (r == 0) continue; // the first launch warms the module
        meds.push_back(median(std::vector<double>(h.begin(), h.end())));
        clocks.push_back(median(hm));
    }
    *mhz = median(clocks);
    return median(meds) / kSyncs;
}

template <int S>
double rank_ns(int C, double sync_ns, unsigned long long* d, float* sink) {
    cudaLaunchAttribute a[1];
    cudaLaunchConfig_t cfg = config(C, kClusters * C, kThreads, 0, a);
    std::vector<double> meds;
    std::vector<unsigned long long> h(kClusters);
    for (int r = 0; r <= kRepeats; ++r) {
        CHECK(cudaLaunchKernelEx(&cfg, reduce<S>, kRounds, d, sink));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h.data(), d, kClusters * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
        if (r == 0) continue;
        std::vector<double> v(h.begin(), h.end());
        meds.push_back(median(v));
    }
    const double round = median(meds) / kRounds;
    std::printf(
        "note cluster_rank_ns_c%d_s%d: a round %.1f ns (two syncs %.1f ns, %d ranks read)\n", C, S,
        round, 2 * sync_ns, C - 1);
    return (round - 2.0 * sync_ns) / (C - 1);
}

// The wall time of one launch of `ctas` spin CTAs (plain when C is 0), in rounds of the spin.
double rounds(int C, int ctas, size_t smem, float* sink) {
    cudaLaunchAttribute a[1];
    cudaLaunchConfig_t cfg = config(C, ctas, kThreads, smem, a);
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    float best = 1e30f;
    for (int r = 0; r < 5; ++r) {
        CHECK(cudaEventRecord(e0));
        CHECK(cudaLaunchKernelEx(&cfg, spin, kSpinNs, sink));
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best = std::min(best, ms);
    }
    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    return best * 1e6 / kSpinNs;
}

} // namespace

int main() {
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    int driver = 0, runtime = 0;
    CHECK(cudaDriverGetVersion(&driver));
    CHECK(cudaRuntimeGetVersion(&runtime));
    const int cc = p.major * 10 + p.minor, sms = p.multiProcessorCount;
    char pci[32] = {};
    CHECK(cudaDeviceGetPCIBusId(pci, sizeof pci, 0));
    std::printf("probe cluster_probe 1\n");
    std::printf("card %s\n", p.name);
    std::printf("cc %d\n", cc);
    std::printf("sms %d\n", sms);
    std::printf("driver %d\n", driver);
    std::printf("runtime %d\n", runtime);
    std::printf("nvcc %d.%d.%d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__,
                __CUDACC_VER_BUILD__);
    std::printf("note device 0 is the GPU at PCI %s\n", pci);
    if (cc < 90) {
        check(false, "no thread-block clusters below sm_90: no cluster facts");
    } else {
        std::printf("note timer: %%globaltimer; a sync or rank fact is the median over %d launches "
                    "of the median over %d clusters of %d-thread CTAs (%d syncs, %d reduce rounds "
                    "a launch)\n",
                    kRepeats, kClusters, kThreads, kSyncs, kRounds);
        unsigned long long* d = nullptr;
        float* sink           = nullptr;
        CHECK(cudaMalloc(&d, kClusters * sizeof(unsigned long long)));
        double* dm = nullptr;
        CHECK(cudaMalloc(&dm, kClusters * sizeof(double)));
        CHECK(cudaMalloc(&sink, 4096 * sizeof(float)));
        CHECK(cudaFuncSetAttribute(syncs, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        CHECK(cudaFuncSetAttribute(reduce<1>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        CHECK(cudaFuncSetAttribute(reduce<2>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        CHECK(
            cudaFuncSetAttribute(slots_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        CHECK(cudaFuncSetAttribute(spin, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        const int smem = static_cast<int>(p.sharedMemPerMultiprocessor) / kPlainPerSm - 1024;
        CHECK(
            cudaFuncSetAttribute(slots_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        CHECK(cudaFuncSetAttribute(spin, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        int plain = 0;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&plain, slots_kernel, kThreads, smem));
        check(plain == kPlainPerSm, "the slots CTA's plain launch holds " + std::to_string(plain) +
                                        " CTAs an SM, not " + std::to_string(kPlainPerSm));
        std::printf("note cluster_slots: a %d-thread CTA of %d B dynamic shared memory, %d an SM "
                    "on a plain launch (%d)\n",
                    kThreads, smem, plain, plain * sms);
        for (const int C : kSizes) {
            hold<<<sms * kPlainPerSm, kThreads>>>(kWarmNs, sink);
            CHECK(cudaGetLastError());
            double mhz        = 0.0;
            const double sync = sync_ns(C, d, dm, &mhz);
            std::printf("note cluster_sync_ns_c%d: the SM clock over the syncs %.0f MHz (clock64 / "
                        "%%globaltimer, rank 0), after a %lld ms hold\n",
                        C, mhz, kWarmNs / 1000000);
            std::printf("fact cluster_sync_ns_c%d %.3f ns\n", C, sync);
            const double r1 = rank_ns<1>(C, sync, d, sink), r2 = rank_ns<2>(C, sync, d, sink);
            check(r1 > 0 && r2 > 0, "a rank read of C " + std::to_string(C) + " is not positive");
            std::printf("fact cluster_rank_ns_c%d_s1 %.3f ns\n", C, r1);
            std::printf("fact cluster_rank_ns_c%d_s2 %.3f ns\n", C, r2);
            cudaLaunchAttribute a[1];
            cudaLaunchConfig_t cfg = config(C, C * 64, kThreads, smem, a);
            int clusters           = 0;
            CHECK(cudaOccupancyMaxActiveClusters(&clusters, slots_kernel, &cfg));
            const int slots = clusters * C;
            check(slots > 0, "no cluster of " + std::to_string(C) + " fits");
            std::printf("fact cluster_slots_c%d %d ctas\n", C, slots);
            if (slots > 0) {
                const double rc = rounds(C, 2 * slots, smem, sink),
                             rp = rounds(0, 2 * slots, smem, sink);
                std::printf("note cluster_slots_c%d: %d CTAs take %.2f rounds of the spin as C-CTA "
                            "clusters, %.2f as a plain launch\n",
                            C, 2 * slots, rc, rp);
                check(std::fabs(rc - 2.0) <= 2.0 * kRoundShare,
                      "cluster_slots_c" + std::to_string(C) + ": 2 x slots CTAs ran " +
                          std::to_string(rc) + " rounds, not 2");
                check(rp < rc, "cluster_slots_c" + std::to_string(C) +
                                   ": the plain launch is no faster than the cluster one");
            }
        }
        CHECK(cudaFree(d));
        CHECK(cudaFree(dm));
        CHECK(cudaFree(sink));
    }
    if (g_failures.empty()) {
        std::printf("selfcheck ok\n");
    } else {
        for (const std::string& f : g_failures) std::printf("selfcheck FAIL %s\n", f.c_str());
    }
    std::printf("end\n");
    return g_failures.empty() ? 0 : 3;
}
