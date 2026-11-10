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
// each chase a pseudo-random cycle through a sub-buffer of their own (no two warps share a line
// or an order; the L2 and DRAM loads bypass the L1), each warp timing its own links.

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
    long long begin, end; // clock64: SM clocks
    int sm;
    long long ns_begin = 0, ns_end = 0; // %globaltimer: wall nanoseconds (the chases only)
};

__device__ __forceinline__ long long globaltimer_ns() {
    long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

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
// Every chasing warp owns a sub-buffer of `m` elements (a power of two) holding one cycle through
// all of them in pseudo-random order (a full-period LCG, fill_cycle: the line of the next link is
// unrelated to this one's), so no two warps' chains share a line and no warp's path is another's
// shifted. One CTA per SM; lane 0 of each of its `warps` warps chases its own chain, the other
// lanes idle. The first `kLatencyLinks` links warm the level (the lines of a resident level, the
// TLB of the DRAM one); the timed links continue the cycle from there, so on the DRAM level they
// are lines no warp has touched. `Cg` loads bypass the L1 (ld.global.cg: cached in L2 only), so an
// L2 or DRAM figure cannot be an L1 hit; the L1 level uses the default path. Each warp records its
// own span; the host takes the median over warps and SMs.
template <bool Cg>
__device__ __forceinline__ unsigned load_link(const unsigned* p) {
    unsigned v;
    if constexpr (Cg) {
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
    } else {
        // The L1-cached path spelled out: a plain `*p` on a const __restrict__ pointer may take
        // the read-only (texture) path, whose hit is longer than the L1's.
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p));
    }
    return v;
}

template <bool Cg>
__global__ void chase(const unsigned* __restrict__ buf, unsigned m, Span* spans, unsigned* sink) {
    const int warp = threadIdx.x / 32;
    if (threadIdx.x % 32 != 0) return;
    const unsigned* mine = buf + static_cast<size_t>(warp) * m;
    unsigned idx         = 0;
    for (int i = 0; i < kLatencyLinks; ++i) idx = load_link<Cg>(mine + idx);
    const long long n0 = globaltimer_ns();
    const long long t0 = clock64();
    for (int i = 0; i < kLatencyLinks; ++i) idx = load_link<Cg>(mine + idx);
    const long long t1           = clock64();
    const long long n1           = globaltimer_ns();
    spans[blockIdx.x * 8 + warp] = {t0, t1, smid(), n0, n1};
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

// Eight sub-buffers of `m` elements (m a power of two), each one cycle through all its elements:
// buf[w * m + i] = (a_w * i + c_w) mod m with a_w = 1 mod 4 and c_w odd is a full-period LCG step,
// a permutation of a single cycle (Hull-Dobell), whose successive links land on unrelated lines.
// The multipliers differ per sub-buffer so no two cycles share their order.
__global__ void fill_cycle(unsigned* buf, unsigned m) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(m) * 8) return;
    const unsigned w = static_cast<unsigned>(i / m), x = static_cast<unsigned>(i % m);
    const unsigned a = 1664525u + 4u * 1000003u * w; // 1 mod 4 for every w
    const unsigned c = 1013904223u + 2u * 7919u * w; // odd for every w
    buf[i]           = (a * x + c) & (m - 1u);
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
    std::printf("probe isa_probe 4\n");
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
    // Spans: one per CTA of the throughput kernels (4 an SM), one per chasing warp (8 an SM).
    CHECK(cudaMalloc(&pr.spans, pr.sms * 8 * sizeof(Span)));
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

    // Memory latencies: per chasing warp a sub-buffer holding one pseudo-random cycle (fill_cycle).
    // L1: 4 KB a warp (32 KB at 8 warps, inside every L1), default loads. L2: 128 KB a warp (1 MB
    // at 8 warps, inside every L2), loads that bypass the L1. DRAM: 128 MB a warp (1 GB at 8
    // warps; a warp's cycle alone outstrips the largest L2), loads that bypass the L1, the timed
    // links on lines no warp has touched. Loaded: 1, 2, 4, 8 warps per SM, each on its own chain.
    struct Level {
        const char* name;
        unsigned m; // elements a warp
        bool cg;
    } levels[] = {{"l1", 1u << 10, false}, {"l2", 1u << 15, true}, {"dram", 1u << 25, true}};

    unsigned* buf = nullptr;
    CHECK(cudaMalloc(&buf, (size_t{1} << 25) * 8 * sizeof(unsigned)));
    std::map<std::string, double> lat;
    std::vector<double> mhz; // the SM clock seen by each L2 / DRAM chase

    // The spans of the chasing warps: entry CTA x 8 + warp, the first `warps` of every CTA.
    struct Chase {
        double clk, ns, mhz; // per link in SM clocks and wall ns; the SM clock the chase ran at
    };

    const auto chase_clocks = [&](int warps) {
        std::vector<double> clk, ns;
        for (const Span& s : pr.collect(pr.sms * 8)) {
            const int entry = static_cast<int>(&s - pr.host.data());
            if (entry % 8 >= warps) continue;
            clk.push_back(static_cast<double>(s.end - s.begin));
            ns.push_back(static_cast<double>(s.ns_end - s.ns_begin));
        }
        const double c = median(clk), n = median(ns);
        return Chase{c / kLatencyLinks, n / kLatencyLinks, n > 0 ? c / n * 1e3 : 0.0};
    };
    for (const Level& lv : levels) {
        const size_t total = static_cast<size_t>(lv.m) * 8;
        fill_cycle<<<static_cast<unsigned>((total + 255) / 256), 256>>>(buf, lv.m);
        CHECK(cudaDeviceSynchronize());
        for (int warps : {1, 2, 4, 8}) {
            if (lv.cg) {
                chase<true><<<pr.sms, 32 * warps>>>(buf, lv.m, pr.spans, pr.usink);
            } else {
                chase<false><<<pr.sms, 32 * warps>>>(buf, lv.m, pr.spans, pr.usink);
            }
            const Chase c = chase_clocks(warps);
            char name[64];
            std::snprintf(name, sizeof name, "%s_latency_w%d", lv.name, warps);
            fact(name, c.clk, "clk");
            lat[name] = c.clk;
            if (lv.cg) {
                // Beyond the SM the latency is wall time: the L2 and DRAM figures are reported in
                // ns as well, with the SM clock the chase ran at (clock64 over globaltimer), so a
                // clock that moved between chases shows instead of passing for a latency change.
                std::snprintf(name, sizeof name, "%s_latency_ns_w%d", lv.name, warps);
                fact(name, c.ns, "ns");
                lat[name] = c.ns;
                mhz.push_back(c.mhz);
            }
        }
    }
    fact("sm_clock_mhz_chase", median(mhz), "mhz");
    check(mhz.size() == 8 && *std::max_element(mhz.begin(), mhz.end()) <=
                                 1.05 * *std::min_element(mhz.begin(), mhz.end()),
          "the SM clock moved by over 5 % between the L2 / DRAM chases (lock the clocks: "
          "nvidia-smi -lgc)");
    chase_shared<<<pr.sms, 128>>>(pr.spans, pr.usink);
    chase_shared<<<pr.sms, 128>>>(pr.spans, pr.usink);
    fact("shared_latency", median_span(pr.collect(pr.sms)) / kLatencyLinks, "clk");
    // The levels must separate: an L2 hit is further than an L1 hit, DRAM further than the L2 at
    // every warp count (chains that alias into L2-resident lines would read as the L2), and more
    // chasing warps never make a warp's latency shorter (in SM clocks for the L1, in wall ns for
    // the L2 and DRAM).
    check(lat["l2_latency_w1"] > 2.0 * lat["l1_latency_w1"], "L2 latency not above 2x L1");
    for (int w : {1, 2, 4, 8}) {
        const std::string ws = "_latency_w" + std::to_string(w);
        const std::string ns = "_latency_ns_w" + std::to_string(w);
        check(lat["dram" + ns] > 1.2 * lat["l2" + ns],
              "DRAM latency not above 1.2x L2 at w" + std::to_string(w));
        check(lat["l1" + ws] >= 0.95 * lat["l1_latency_w1"],
              "l1 latency falls under the lone warp's at w" + std::to_string(w));
        for (const char* lv : {"l2", "dram"})
            check(lat[lv + ns] >= 0.95 * lat[std::string(lv) + "_latency_ns_w1"],
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
