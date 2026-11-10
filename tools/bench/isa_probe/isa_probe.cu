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
// Exit 0 with `selfcheck ok`; 3 when a result fails the self-check (its reasons printed), 2 on a
// CUDA error.
//
// Method. Throughput: SMs x 4 CTAs of 4 warps each run N iterations of U independent
// instructions; every CTA records its SM (%smid) and its clock64 span, and per SM the span of all
// its CTAs over the instructions those CTAs issued is the SM's cycles per instruction (the
// hardware does not spread the CTAs evenly, so the CTA count per SM is read, not assumed); the
// median over SMs is reported, with the per-partition figure 4x that. Latency: one warp's lane 0
// follows a dependent chain (acc = mma(acc) / idx = buf[idx]); cycles per link include the chain's
// own address arithmetic (2-3 clocks, the same for every level). Loaded latency: W warps per SM
// each chase a chain of their own through the same level, the chains disjoint in lines (each
// starts n / 8 elements after the previous, a power of two no stride divides).

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
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

constexpr int kWarpsPerCta  = 4;
constexpr int kCtasPerSm    = 4;
constexpr int kThroughputN  = 4096; // iterations of the throughput loops
constexpr int kUnroll       = 8;    // independent streams per thread
constexpr int kLatencyLinks = 4096; // links of a dependent chain

struct Span {
    long long begin, end;
    int sm;
};

__device__ __forceinline__ int smid() {
    int id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

// ---- Tensor pipe ----
// One MMA of the architecture's shape on packed 1.0 operands; `acc` holds the accumulators. The
// shape code is read back from the device (shape_code), never decided on the host, where
// __CUDA_ARCH__ is not defined.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#    define ISA_PROBE_SHAPE_CODE 2

__device__ __forceinline__ void mma_once(float (&c)[4], unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#    define ISA_PROBE_SHAPE_CODE 1

__device__ __forceinline__ void mma_once(float (&c)[4], unsigned a0, unsigned a1, unsigned,
                                         unsigned, unsigned b0, unsigned) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}
#else
#    define ISA_PROBE_SHAPE_CODE 0

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

__global__ void shape_code(int* out) {
#ifdef __CUDA_ARCH__
    *out = ISA_PROBE_SHAPE_CODE;
#endif
}

struct Shape {
    const char* name;
    int flops;
};

constexpr Shape kShapes[] = {
    {"hmma_884_f32_f16", 512}, {"hmma_1688_f32_f16", 2048}, {"hmma_16816_f32_bf16", 4096}};

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
    if (threadIdx.x == 0) spans[blockIdx.x] = {t0, t1, smid()};
    if (s == 12345.f) sink[0] = s; // keeps the chain alive, never true
}

__global__ void mma_latency(Span* spans, float* sink, unsigned a, unsigned b) {
    float acc[4]       = {0.f, 0.f, 0.f, 0.f};
    const long long t0 = clock64();
    for (int i = 0; i < kLatencyLinks; ++i) mma_once(acc, a, a, a, a, b, b);
    const long long t1 = clock64();
    if (threadIdx.x == 0) spans[blockIdx.x] = {t0, t1, smid()};
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
    if (threadIdx.x == 0) spans[blockIdx.x] = {t0, t1, smid()};
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
    if (threadIdx.x == 0) spans[blockIdx.x] = {t0, t1, smid()};
    if (s == 12345u) sink[0] = 1.f;
}

// ---- Memory latencies ----
// `warps` warps per CTA (one CTA per SM) each follow their own chain `buf[idx]` from `warp x
// offset`; lane 0 chases, the warp's other lanes idle. Two passes: the first warms the level, the
// second is timed. Every index read is a value of buf, below n by construction; the start
// `warp x offset` is below n for offset = n / 8 and at most 8 warps.
__global__ void chase(const unsigned* __restrict__ buf, unsigned offset, Span* spans,
                      unsigned* sink) {
    const int warp = threadIdx.x / 32;
    if (threadIdx.x % 32 != 0) return;
    unsigned idx = warp * offset;
    for (int i = 0; i < kLatencyLinks; ++i) idx = buf[idx];
    const long long t0 = clock64();
    for (int i = 0; i < kLatencyLinks; ++i) idx = buf[idx];
    const long long t1 = clock64();
    if (warp == 0) spans[blockIdx.x] = {t0, t1, smid()};
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
    spans[blockIdx.x]  = {t0, t1, smid()};
    if (idx == 0xffffffffu) sink[0] = idx;
}

// buf[i] = (i + stride) mod n: a chain whose links are `stride` elements apart.
__global__ void fill_chain(unsigned* buf, unsigned n, unsigned stride) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = (i + stride) % n;
}

double median(std::vector<double> d) {
    std::sort(d.begin(), d.end());
    return d[d.size() / 2];
}

double median_span(const std::vector<Span>& spans) {
    std::vector<double> d;
    for (const Span& s : spans) d.push_back(static_cast<double>(s.end - s.begin));
    return median(d);
}

struct Probe {
    int sms          = 0;
    Span* spans      = nullptr;
    float* fsink     = nullptr;
    unsigned* usink  = nullptr;
    double ctas_seen = 0; // the median CTA count per SM the last throughput kernel saw
    std::vector<Span> host;

    std::vector<Span>& collect(int ctas) {
        CHECK(cudaDeviceSynchronize());
        host.resize(ctas);
        CHECK(cudaMemcpy(host.data(), spans, ctas * sizeof(Span), cudaMemcpyDeviceToHost));
        return host;
    }

    // Cycles per instruction per SM of a throughput kernel: per SM, the span of all its CTAs over
    // the instructions they issued; the median over SMs.
    template <typename Kernel, typename... Args>
    double per_sm(Kernel kernel, Args... args) {
        const int ctas = sms * kCtasPerSm;
        kernel<<<ctas, 32 * kWarpsPerCta>>>(spans, fsink, args...); // warm
        kernel<<<ctas, 32 * kWarpsPerCta>>>(spans, fsink, args...);
        std::map<int, std::vector<Span>> by_sm;
        for (const Span& s : collect(ctas)) by_sm[s.sm].push_back(s);
        std::vector<double> cycles, counts;
        for (const auto& [sm, list] : by_sm) {
            long long lo = list[0].begin, hi = list[0].end;
            for (const Span& s : list) {
                lo = std::min(lo, s.begin);
                hi = std::max(hi, s.end);
            }
            const double work =
                static_cast<double>(kThroughputN) * kUnroll * kWarpsPerCta * list.size();
            cycles.push_back(static_cast<double>(hi - lo) / work);
            counts.push_back(static_cast<double>(list.size()));
        }
        ctas_seen = median(counts);
        return median(cycles);
    }
};

std::vector<std::string> g_failures;

void fact(const char* name, double value, const char* unit) {
    std::printf("fact %s %.3f %s\n", name, value, unit);
}

void check(bool ok, const std::string& what) {
    if (!ok) g_failures.push_back(what);
}

} // namespace

int main() {
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    int driver = 0, runtime = 0;
    CHECK(cudaDriverGetVersion(&driver));
    CHECK(cudaRuntimeGetVersion(&runtime));
    const int cc = p.major * 10 + p.minor;
    std::printf("probe isa_probe 2\n");
    std::printf("card %s\n", p.name);
    std::printf("cc %d\n", cc);
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

    // The MMA shape the device code took, checked against the card.
    int* code_dev = nullptr;
    int code      = -1;
    CHECK(cudaMalloc(&code_dev, sizeof(int)));
    shape_code<<<1, 1>>>(code_dev);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&code, code_dev, sizeof(int), cudaMemcpyDeviceToHost));
    check(code >= 0 && code < 3, "shape code not read from the device");
    const Shape shape = kShapes[std::clamp(code, 0, 2)];
    std::printf("shape %s %d\n", shape.name, shape.flops);
    const int expected_code = cc >= 80 ? 2 : cc >= 75 ? 1 : 0;
    check(code == expected_code, std::string("shape ") + shape.name + " is not the card's (cc " +
                                     std::to_string(cc) + ")");

    // Tensor pipe: cycles per MMA at the SM (every partition busy) and the dependent latency.
    const unsigned ones = (code == 2) ? 0x3f803f80u /* bf16 1.0 */ : 0x3c003c00u /* f16 1.0 */;
    const double mma_sm = pr.per_sm(mma_throughput, ones, ones);
    fact("resident_ctas_per_sm", pr.ctas_seen, "count");
    fact("mma_cycles_per_sm", mma_sm, "clk");
    fact("mma_cycles_per_partition", mma_sm * 4.0, "clk");
    const double tensor_rate = shape.flops / mma_sm;
    fact("tensor_flops_per_sm_clock", tensor_rate, "flop");
    // No SM the probe knows issues a dense MMA above 2048 flop a clock (the GA100 / GH100 class)
    // or below one m8n8k4 in 32 clocks.
    check(tensor_rate <= 2048.0 && tensor_rate >= 16.0, "tensor rate out of range");
    mma_latency<<<pr.sms, 32>>>(pr.spans, pr.fsink, ones, ones);
    mma_latency<<<pr.sms, 32>>>(pr.spans, pr.fsink, ones, ones);
    const double mma_lat = median_span(pr.collect(pr.sms)) / kLatencyLinks;
    fact("mma_dependent_latency", mma_lat, "clk");
    check(mma_lat >= mma_sm * 4.0 * 0.5, "dependent MMA latency under half the issue interval");

    // INT32 / FP32 pipes: cycles per warp instruction per partition. The floors are the lane
    // counts: 32 FP32 lanes a partition from Ampere (1 clock), 16 before (2); 16 INT32 lanes (2).
    const double fp32 = pr.per_sm(fp32_throughput, 0.999f, 0.001f) * 4.0;
    const double i32  = pr.per_sm(int32_throughput, 3u, 1u) * 4.0;
    fact("fp32_cycles_per_warp_instruction", fp32, "clk");
    fact("int32_cycles_per_warp_instruction", i32, "clk");
    const double fp32_floor = cc >= 80 ? 1.0 : 2.0;
    check(fp32 >= 0.97 * fp32_floor, "FP32 cycles per warp instruction under the lane floor");
    check(i32 >= 0.97 * 2.0, "INT32 cycles per warp instruction under the lane floor");

    // Memory latencies. L1: an 8 KB chain, L2: 1 MB, DRAM: a 512 MB chain whose links are 64 KB +
    // 128 B apart (every link a new line and a new DRAM page; 4096 links span 256 MB, past every
    // L2). Loaded: 1, 2, 4, 8 warps per SM, each on its own chain of the same level starting n / 8
    // elements after the previous: a power of two no stride divides, so no two chains share a
    // line within the timed links, and every start is below n.
    struct Level {
        const char* name;
        size_t bytes;
        unsigned stride; // elements
    } levels[] = {{"l1", 8u << 10, 33u},
                  {"l2", 1u << 20, 1057u},
                  {"dram", 512u << 20, (64u << 10) / 4 + 32u}};

    unsigned* buf = nullptr;
    CHECK(cudaMalloc(&buf, 512u << 20));
    std::map<std::string, double> lat;
    for (const Level& lv : levels) {
        const unsigned n = static_cast<unsigned>(lv.bytes / 4);
        fill_chain<<<(n + 255) / 256, 256>>>(buf, n, lv.stride);
        CHECK(cudaDeviceSynchronize());
        for (int warps : {1, 2, 4, 8}) {
            chase<<<pr.sms, 32 * warps>>>(buf, n / 8, pr.spans, pr.usink);
            const double clk = median_span(pr.collect(pr.sms)) / kLatencyLinks;
            char name[64];
            std::snprintf(name, sizeof name, "%s_latency_w%d", lv.name, warps);
            fact(name, clk, "clk");
            lat[name] = clk;
        }
    }
    chase_shared<<<pr.sms, 128>>>(pr.spans, pr.usink);
    chase_shared<<<pr.sms, 128>>>(pr.spans, pr.usink);
    fact("shared_latency", median_span(pr.collect(pr.sms)) / kLatencyLinks, "clk");
    // The levels must separate: an L2 hit is further than an L1 hit, DRAM further than the L2 at
    // every warp count (chains that alias into L2-resident lines would read as the L2), and more
    // chasing warps never make a warp's latency shorter.
    check(lat["l2_latency_w1"] > 2.0 * lat["l1_latency_w1"], "L2 latency not above 2x L1");
    for (int w : {1, 2, 4, 8}) {
        const std::string ws = "_latency_w" + std::to_string(w);
        check(lat["dram" + ws] > 1.2 * lat["l2" + ws],
              "DRAM latency not above 1.2x L2 at w" + std::to_string(w));
        for (const char* lv : {"l1", "l2", "dram"})
            check(lat[lv + ws] >= 0.95 * lat[std::string(lv) + "_latency_w1"],
                  std::string(lv) + " latency falls under the lone warp's at w" +
                      std::to_string(w));
    }
    if (g_failures.empty()) {
        std::printf("selfcheck ok\n");
    } else {
        for (const std::string& f : g_failures) std::printf("selfcheck FAIL %s\n", f.c_str());
    }
    std::printf("end\n");
    return g_failures.empty() ? 0 : 3;
}
