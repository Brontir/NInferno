// isa_probe: the ISA facts of one card that the register-MMA price's per-warp timeline needs
// (#613 PR-2): tensor pipe cycles per MMA at the SM and the dependent MMA latency, the INT32 and
// FP32 pipes' cycles per warp instruction, and the dependent load latencies (L1 hit, L2 hit, DRAM,
// shared memory) alone and under 2, 4 and 8 chasing warps per SM. Every number is in SM clocks
// (clock64), so it does not depend on the clock the card ran at.
//
// Build (CUDA 13.x; sm_70 needs CUDA 12.8's nvcc):
//   nvcc -O3 -std=c++17 -arch=sm_<arch> -o isa_probe tools/bench/isa_probe/isa_probe.cu
// Run (about 20 s, 600 MB of device memory; the GPU should be otherwise idle):
//   ./isa_probe            # `fact <name> <value> <unit>` lines, parsed by isa_probe.py
//
// Method. Throughput: SMs x 4 resident CTAs of 4 warps each run N iterations of U independent
// instructions; a CTA's clock64 span covers the work of every CTA resident with it, so cycles per
// instruction per SM = span / (instructions of the SM); the per-partition figure is 4x that.
// Latency: one warp's lane 0 follows a dependent chain (acc = mma(acc) / idx = buf[idx]); cycles
// per link include the chain's own address arithmetic (2-3 clocks, the same for every level).
// Loaded latency: W warps per SM each chase their own chain through the same level.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

constexpr int kWarpsPerCta  = 4;
constexpr int kCtasPerSm    = 4;
constexpr int kThroughputN  = 4096; // iterations of the throughput loops
constexpr int kUnroll       = 8;    // independent streams per thread
constexpr int kLatencyLinks = 4096; // links of a dependent chain

struct Span {
    long long begin, end;
};

// ---- Tensor pipe ----
// One MMA of the architecture's shape on packed 1.0 operands; `acc` holds the accumulators.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#    define ISA_PROBE_SHAPE "hmma_16816_f32_bf16"
#    define ISA_PROBE_FLOPS 4096

__device__ __forceinline__ void mma_once(float (&c)[4], unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#    define ISA_PROBE_SHAPE "hmma_1688_f32_f16"
#    define ISA_PROBE_FLOPS 2048

__device__ __forceinline__ void mma_once(float (&c)[4], unsigned a0, unsigned a1, unsigned,
                                         unsigned, unsigned b0, unsigned) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}
#else
#    define ISA_PROBE_SHAPE "hmma_884_f32_f16"
#    define ISA_PROBE_FLOPS 512

__device__ __forceinline__ void mma_once(float (&c)[4], unsigned a0, unsigned a1, unsigned,
                                         unsigned, unsigned b0, unsigned b1) {
    // m8n8k4 keeps 8 accumulators; the probe chains the first four and parks the rest.
    float d4 = 0.f, d5 = 0.f, d6 = 0.f, d7 = 0.f;
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, "
                 "{%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(d4), "+f"(d5), "+f"(d6),
                   "+f"(d7)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
#endif

__global__ void mma_throughput(Span* spans, float* sink, unsigned a, unsigned b) {
    float acc[kUnroll][4];
    for (int u = 0; u < kUnroll; ++u)
        for (int e = 0; e < 4; ++e) acc[u][e] = 0.f;
    __syncthreads();
    const long long t0 = clock64();
    for (int i = 0; i < kThroughputN; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) mma_once(acc[u], a, a, a, a, b, b);
    }
    __syncthreads();
    const long long t1 = clock64();
    float s            = 0.f;
    for (int u = 0; u < kUnroll; ++u) s += acc[u][0] + acc[u][1] + acc[u][2] + acc[u][3];
    if (threadIdx.x == 0) spans[blockIdx.x] = {t0, t1};
    if (s == 12345.f) sink[0] = s; // keeps the chain alive, never true
}

__global__ void mma_latency(Span* spans, float* sink, unsigned a, unsigned b) {
    float acc[4]       = {0.f, 0.f, 0.f, 0.f};
    const long long t0 = clock64();
    for (int i = 0; i < kLatencyLinks; ++i) mma_once(acc, a, a, a, a, b, b);
    const long long t1 = clock64();
    if (threadIdx.x == 0) spans[blockIdx.x] = {t0, t1};
    if (acc[0] + acc[1] + acc[2] + acc[3] == 12345.f) sink[0] = acc[0];
}

// ---- INT32 and FP32 pipes ----
__global__ void fp32_throughput(Span* spans, float* sink, float m, float c) {
    float x[kUnroll];
    for (int u = 0; u < kUnroll; ++u) x[u] = threadIdx.x * 1e-3f + u;
    __syncthreads();
    const long long t0 = clock64();
    for (int i = 0; i < kThroughputN; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) x[u] = fmaf(x[u], m, c);
    }
    __syncthreads();
    const long long t1 = clock64();
    float s            = 0.f;
    for (int u = 0; u < kUnroll; ++u) s += x[u];
    if (threadIdx.x == 0) spans[blockIdx.x] = {t0, t1};
    if (s == 12345.f) sink[0] = s;
}

__global__ void int32_throughput(Span* spans, float* sink, unsigned m, unsigned c) {
    unsigned x[kUnroll];
    for (int u = 0; u < kUnroll; ++u) x[u] = threadIdx.x + u;
    __syncthreads();
    const long long t0 = clock64();
    for (int i = 0; i < kThroughputN; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) x[u] = x[u] * m + c; // IMAD
    }
    __syncthreads();
    const long long t1 = clock64();
    unsigned s         = 0;
    for (int u = 0; u < kUnroll; ++u) s += x[u];
    if (threadIdx.x == 0) spans[blockIdx.x] = {t0, t1};
    if (s == 12345u) sink[0] = 1.f;
}

// ---- Memory latencies ----
// `chains` warps per CTA (one CTA per SM) each follow their own chain `buf[idx]` from `start +
// warp * offset`; lane 0 chases, the warp's other lanes idle. Two passes: the first warms the
// level, the second is timed.
__global__ void chase(const unsigned* __restrict__ buf, unsigned offset, Span* spans,
                      unsigned* sink) {
    const int warp = threadIdx.x / 32;
    if (threadIdx.x % 32 != 0) return;
    unsigned idx = warp * offset;
    for (int i = 0; i < kLatencyLinks; ++i) idx = buf[idx];
    const long long t0 = clock64();
    for (int i = 0; i < kLatencyLinks; ++i) idx = buf[idx];
    const long long t1 = clock64();
    if (warp == 0) spans[blockIdx.x] = {t0, t1};
    if (idx == 0xffffffffu) sink[0] = idx;
}

__global__ void chase_shared(Span* spans, unsigned* sink) {
    __shared__ unsigned s[4096];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) s[i] = (i + 33u) & 4095u;
    __syncthreads();
    if (threadIdx.x != 0) return;
    unsigned idx       = 0;
    const long long t0 = clock64();
    for (int i = 0; i < kLatencyLinks; ++i) idx = s[idx];
    const long long t1 = clock64();
    spans[blockIdx.x]  = {t0, t1};
    if (idx == 0xffffffffu) sink[0] = idx;
}

// buf[i] = (i + stride) mod n: a chain whose links are `stride` elements apart.
__global__ void fill_chain(unsigned* buf, unsigned n, unsigned stride) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = (i + stride) % n;
}

double median_span(const std::vector<Span>& spans) {
    std::vector<double> d;
    for (const Span& s : spans) d.push_back(static_cast<double>(s.end - s.begin));
    std::sort(d.begin(), d.end());
    return d[d.size() / 2];
}

struct Probe {
    int sms         = 0;
    Span* spans     = nullptr;
    float* fsink    = nullptr;
    unsigned* usink = nullptr;
    std::vector<Span> host;

    std::vector<Span>& collect(int ctas) {
        CHECK(cudaDeviceSynchronize());
        host.resize(ctas);
        CHECK(cudaMemcpy(host.data(), spans, ctas * sizeof(Span), cudaMemcpyDeviceToHost));
        return host;
    }

    // Cycles per instruction per SM of a throughput kernel, every SM holding kCtasPerSm CTAs.
    template <typename Kernel, typename... Args>
    double per_sm(Kernel kernel, Args... args) {
        const int ctas = sms * kCtasPerSm;
        kernel<<<ctas, 32 * kWarpsPerCta>>>(spans, fsink, args...); // warm
        kernel<<<ctas, 32 * kWarpsPerCta>>>(spans, fsink, args...);
        const double span = median_span(collect(ctas));
        return span / (static_cast<double>(kThroughputN) * kUnroll * kWarpsPerCta * kCtasPerSm);
    }
};

void fact(const char* name, double value, const char* unit) {
    std::printf("fact %s %.3f %s\n", name, value, unit);
}

} // namespace

int main() {
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    int driver = 0, runtime = 0;
    CHECK(cudaDriverGetVersion(&driver));
    CHECK(cudaRuntimeGetVersion(&runtime));
    std::printf("probe isa_probe 1\n");
    std::printf("card %s\n", p.name);
    std::printf("cc %d%d\n", p.major, p.minor);
    std::printf("sms %d\n", p.multiProcessorCount);
    std::printf("l2_bytes %zu\n", static_cast<size_t>(p.l2CacheSize));
    std::printf("driver %d\n", driver);
    std::printf("runtime %d\n", runtime);
    std::printf("nvcc %d.%d.%d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__,
                __CUDACC_VER_BUILD__);

    Probe pr;
    pr.sms = p.multiProcessorCount;
    CHECK(cudaMalloc(&pr.spans, pr.sms * kCtasPerSm * sizeof(Span)));
    CHECK(cudaMalloc(&pr.fsink, sizeof(float)));
    CHECK(cudaMalloc(&pr.usink, sizeof(unsigned)));

    // Tensor pipe: cycles per MMA at the SM (every partition busy) and the dependent latency.
    const unsigned ones_bf16 = 0x3f803f80u, ones_f16 = 0x3c003c00u;
    const unsigned ones = (p.major >= 8) ? ones_bf16 : ones_f16;
    const double mma_sm = pr.per_sm(mma_throughput, ones, ones);
    std::printf("shape %s %d\n", ISA_PROBE_SHAPE, ISA_PROBE_FLOPS);
    fact("mma_cycles_per_sm", mma_sm, "clk");
    fact("mma_cycles_per_partition", mma_sm * 4.0, "clk");
    fact("tensor_flops_per_sm_clock", ISA_PROBE_FLOPS / mma_sm, "flop");
    mma_latency<<<pr.sms, 32>>>(pr.spans, pr.fsink, ones, ones);
    mma_latency<<<pr.sms, 32>>>(pr.spans, pr.fsink, ones, ones);
    fact("mma_dependent_latency", median_span(pr.collect(pr.sms)) / kLatencyLinks, "clk");

    // INT32 / FP32 pipes: cycles per warp instruction per partition.
    fact("fp32_cycles_per_warp_instruction", pr.per_sm(fp32_throughput, 0.999f, 0.001f) * 4.0,
         "clk");
    fact("int32_cycles_per_warp_instruction", pr.per_sm(int32_throughput, 3u, 1u) * 4.0, "clk");

    // Memory latencies. L1: an 8 KB chain, L2: 1 MB, DRAM: a 512 MB chain whose links are 64 KB +
    // 128 B apart (every link a new line and a new DRAM page; 4096 links span 256 MB, past every
    // L2). Loaded: 1, 2, 4, 8 warps per SM on separate chains of the same level.
    struct Level {
        const char* name;
        size_t bytes;
        unsigned stride; // elements
    } levels[] = {{"l1", 8u << 10, 33u},
                  {"l2", 1u << 20, 1057u},
                  {"dram", 512u << 20, (64u << 10) / 4 + 32u}};

    unsigned* buf = nullptr;
    CHECK(cudaMalloc(&buf, 512u << 20));
    for (const Level& lv : levels) {
        const unsigned n = static_cast<unsigned>(lv.bytes / 4);
        fill_chain<<<(n + 255) / 256, 256>>>(buf, n, lv.stride);
        for (int warps : {1, 2, 4, 8}) {
            // The warps' chains start a quarter of the level apart (an odd multiple of the stride
            // keeps them disjoint for the 4096 timed links on every level).
            const unsigned offset = (n / 4 / lv.stride) * lv.stride + lv.stride;
            chase<<<pr.sms, 32 * warps>>>(buf, offset, pr.spans, pr.usink);
            const double clk = median_span(pr.collect(pr.sms)) / kLatencyLinks;
            char name[64];
            std::snprintf(name, sizeof name, "%s_latency_w%d", lv.name, warps);
            fact(name, clk, "clk");
        }
    }
    chase_shared<<<pr.sms, 128>>>(pr.spans, pr.usink);
    chase_shared<<<pr.sms, 128>>>(pr.spans, pr.usink);
    fact("shared_latency", median_span(pr.collect(pr.sms)) / kLatencyLinks, "clk");
    std::printf("end\n");
    return 0;
}
