// isa_probe: the ISA facts of one card that the register-MMA price's per-warp timeline needs
// (#613 PR-2): tensor pipe cycles per MMA at the SM and the dependent MMA latency, the INT32 and
// FP32 pipes' cycles per warp instruction, and the dependent load latencies (L1 hit, L2 hit, DRAM,
// shared memory) alone and under 2, 4 and 8 chasing warps per SM. What happens inside the SM is in
// SM clocks (clock64) and does not depend on the clock the card ran at; the L2 and DRAM latencies
// are wall time beyond the SM, so they are reported in ns (%globaltimer) beside their clocks, with
// the SM clock the chase ran at.
//
// Build (CUDA 13.x; sm_70 needs CUDA 12.8's nvcc):
//   nvcc -O3 -std=c++17 -arch=sm_<arch> -o isa_probe tools/bench/isa_probe/isa_probe.cu
// Run (about 20 s; 2 GiB of device memory for the load-latency buffer, beside the CUDA context;
// the GPU should be otherwise idle):
//   ./isa_probe            # `fact <name> <value> <unit>` lines, parsed by isa_probe.py
// Exit 0 with `selfcheck ok`; 3 when a result fails the self-check (its reasons printed), 2 on a
// CUDA error.
//
// Method. Throughput: SMs x 4 CTAs of 4 warps each run N iterations of U independent
// instructions; every CTA records its SM (%smid) and its clock64 span, and per SM the span of all
// its CTAs over the instructions those CTAs issued is the SM's cycles per instruction (the
// hardware does not spread the CTAs evenly, so the CTA count per SM is read, not assumed); the
// median over SMs is reported, with the per-partition figure 4x that; the same loop with one CTA
// of 4, 8 and 16 warps per SM gives the tensor figure at fewer warps (probe 14). Latency: one
// warp's lane 0 follows a dependent chain (acc = mma(acc) / p = *p / idx = s[idx]); a link of a
// load chain is the load alone: a global node holds the address of the next (probe 14; the rows
// of earlier probes carry the index-to-address ops of their chain in every l1 / l2 / DRAM link),
// and the shared chain's index is the load's own scaled operand. Loaded latency: W warps per SM
// each chase a pseudo-random cycle of their own (the L2 and DRAM loads bypass the L1), each warp
// timing its own links; on the DRAM levels a link is a 128-byte line of its own (four 32-byte
// sectors) that no other link of the chip shares, whoever chases it (probe 14). A latency launch
// asks for one CTA per SM; a CTA that shared its SM with another (read from %smid) is left out,
// so every latency is at the stated warp count (probe 13), and so is a chase whose SM did not
// hold exactly one clock holder (probe 14).

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));        \
            std::exit(2);                                                                          \
        }                                                                                          \
    } while (0)
// After every kernel launch: one the driver refused is an error of the run, not a number.
#define LAUNCHED() CHECK(cudaGetLastError())

namespace {

constexpr int kWarpsPerCta  = 4;
constexpr int kCtasPerSm    = 4;
constexpr int kThroughputN  = 4096; // iterations of the throughput loops
constexpr int kUnroll       = 8;    // independent streams per thread
constexpr int kLatencyLinks = 4096; // links of a dependent chain
// Links of a DRAM chase, warm-up and timed alike on both DRAM levels: about 150 us of wall time
// a chase, against a %globaltimer that steps by about 1 us.
constexpr int kDramLinks = 512;

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
// ISA_PROBE_ACC is the accumulator count of the shape (8 for Volta's m8n8k4, 4 otherwise), all of
// them live in the chain so no instruction but the MMA sits in the measured loop.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#    define ISA_PROBE_SHAPE_CODE 2
#    define ISA_PROBE_ACC        4

__device__ __forceinline__ void mma_once(float (&c)[4], unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#    define ISA_PROBE_SHAPE_CODE 1
#    define ISA_PROBE_ACC        4

__device__ __forceinline__ void mma_once(float (&c)[4], unsigned a0, unsigned a1, unsigned,
                                         unsigned, unsigned b0, unsigned) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}
#else
#    define ISA_PROBE_SHAPE_CODE 0
#    define ISA_PROBE_ACC        8

// Volta's native shape (sm_70 has no m16n8k16: ptxas would lower it to several of these), all
// eight accumulators chained.
__device__ __forceinline__ void mma_once(float (&c)[8], unsigned a0, unsigned a1, unsigned,
                                         unsigned, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, "
                 "{%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
                   "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
#endif
constexpr int kAcc = ISA_PROBE_ACC;

__global__ void shape_code(int* out) {
#ifdef __CUDA_ARCH__
    *out = ISA_PROBE_SHAPE_CODE;
#endif
}

struct Shape {
    const char* name;
    int flops;
};

// Flops per warp instruction: 2 x m x n x k, and on Volta four of them, since one
// mma.sync.m8n8k4 performs four independent 8x8x4 products, one per quad pair (PTX ISA, "the
// warp performs four matrix multiply-accumulates"): 4 x 2 x 8 x 8 x 4 = 2048.
constexpr Shape kShapes[] = {
    {"hmma_884_f32_f16", 2048}, {"hmma_1688_f32_f16", 2048}, {"hmma_16816_f32_bf16", 4096}};

__global__ void mma_throughput(Span* spans, float* sink, unsigned a, unsigned b) {
    float acc[kUnroll][kAcc];
    for (int u = 0; u < kUnroll; ++u)
        for (int e = 0; e < kAcc; ++e) acc[u][e] = 0.f;
    __syncthreads();
    const long long t0 = clock64();
    for (int i = 0; i < kThroughputN; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) mma_once(acc[u], a, a, a, a, b, b);
    }
    __syncthreads();
    const long long t1 = clock64();
    float s            = 0.f;
    for (int u = 0; u < kUnroll; ++u)
        for (int e = 0; e < kAcc; ++e) s += acc[u][e];
    if (threadIdx.x == 0) spans[blockIdx.x] = {t0, t1, smid()};
    if (s == 12345.f) sink[0] = s; // keeps the chain alive, never true
}

__global__ void mma_latency(Span* spans, float* sink, unsigned a, unsigned b) {
    float acc[kAcc];
    for (int e = 0; e < kAcc; ++e) acc[e] = 0.f;
    const long long t0 = clock64();
    for (int i = 0; i < kLatencyLinks; ++i) mma_once(acc, a, a, a, a, b, b);
    const long long t1 = clock64();
    if (threadIdx.x == 0) spans[blockIdx.x] = {t0, t1, smid()};
    float s = 0.f;
    for (int e = 0; e < kAcc; ++e) s += acc[e];
    if (s == 12345.f) sink[0] = s;
}

// A warp of FFMAs on every SM beside the chasing warp, so the card's clock does not sink to its
// idle step during the latency chases (a lone warp waiting on memory does not lift it); it stops
// when the host raises the mapped flag (read volatile, system scope, every 1024 FFMAs) and in any
// case `deadline_ns` after it started, so it can never hold a card. One warp of FFMA on one
// partition contends with the chase for at most an issue slot a link. Nothing in the latency
// section synchronises the device as a whole (that would wait on this kernel): the chases and
// their copies run and are waited for on the legacy stream alone, which a non-blocking stream
// does not join.
// The holder's words in mapped host memory: the host's stop flag, a flag a holder raises when it
// leaves on its deadline (the chases after that ran without it: the run is refused), and from
// kHolderSms on the SM every holder CTA runs on (-1 until it started), so a chase counts only
// where exactly one holder sat beside it.
constexpr int kHolderStop = 0, kHolderExpired = 1, kHolderSms = 2;
// The holder's deadline: several times the latency section's few seconds.
constexpr long long kHolderSeconds = 60;

__global__ void spinner(volatile int* words, long long deadline_ns, float* sink) {
    float x                  = threadIdx.x * 1e-3f;
    const long long deadline = globaltimer_ns() + deadline_ns;
    if (threadIdx.x == 0) words[kHolderSms + blockIdx.x] = smid();
    while (true) {
#pragma unroll
        for (int i = 0; i < 1024; ++i) x = fmaf(x, 0.999f, 0.001f);
        __threadfence_system();
        if (words[kHolderStop] != 0) break;
        if (globaltimer_ns() > deadline) {
            if (threadIdx.x == 0) words[kHolderExpired] = 1;
            __threadfence_system();
            break;
        }
    }
    if (x == 12345.f) sink[0] = x;
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

// ---- Dependent latencies of the ALU ops the decode chains use ----
// One warp per SM runs a chain of kLatencyLinks dependent ops of one kind (each written in PTX
// so ptxas keeps the op: IMAD, IADD3, LOP3, PRMT, SHF, HFMA2, FFMA, SHFL, and the conversion pairs
// I2F+F2I and F2F f32->f16->f32, reported per op); cycles per link are the op's issue-to-use
// latency. lat_iadd is the add ptxas makes of a dependent doubling, not IADD3 alone: in the sm_89
// SASS the chain alternates IADD3 and IMAD.IADD, so the figure is the mean of the two forms. Each
// is held against the published range of the Volta..Blackwell SM (ALU 2-16 clk, SHFL 10-60) by
// the self-check.
enum class Op { IMAD, IADD, LOP3, PRMT, SHF, HFMA2, FFMA, SHFL, CVT_I2F_F2I, CVT_F32_F16 };

template <Op op>
__device__ __forceinline__ void dep_step(unsigned& x, unsigned k, unsigned m) {
    if constexpr (op == Op::IMAD) {
        asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(k), "r"(m));
    } else if constexpr (op == Op::IADD) {
        // x + x: two dependent adds of a loop-invariant k would pair into one IADD3 (x + k + k)
        // and read as half the latency; a doubling cannot pair (x + x + x is not 4x).
        asm volatile("add.u32 %0, %0, %0;" : "+r"(x));
    } else if constexpr (op == Op::LOP3) {
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(x) : "r"(k), "r"(m));
    } else if constexpr (op == Op::PRMT) {
        asm volatile("prmt.b32 %0, %0, %1, 0x3210;" : "+r"(x) : "r"(k));
    } else if constexpr (op == Op::SHF) {
        asm volatile("shf.l.wrap.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(k), "r"(m));
    } else if constexpr (op == Op::HFMA2) {
        asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(k), "r"(m));
    } else if constexpr (op == Op::FFMA) {
        float f = __uint_as_float(x);
        asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                     : "+f"(f)
                     : "f"(__uint_as_float(k)), "f"(__uint_as_float(m)));
        x = __float_as_uint(f);
    } else if constexpr (op == Op::SHFL) {
        // The source lane is the chained value itself (the 0x1f clamp takes it modulo 32): a
        // constant lane let ptxas fold the whole chain away (probe 11 read 0 clk). From x = lane
        // + 1 the lanes' values settle within five links (the hop doubles each link, 32 hops are
        // none); the dependence of every SHFL on the one before stays, which is what is timed.
        asm volatile("shfl.sync.idx.b32 %0, %0, %0, 0x1f, 0xffffffff;" : "+r"(x));
    } else if constexpr (op == Op::CVT_I2F_F2I) {
        float f;
        asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"(x));
        asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(x) : "f"(f));
    } else {
        unsigned short h;
        float f = __uint_as_float(x);
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
        x = __float_as_uint(f);
    }
}

template <Op op>
__global__ void dep_latency(Span* spans, unsigned* sink, unsigned k, unsigned m) {
    unsigned x = threadIdx.x + 1u;
    for (int i = 0; i < 64; ++i) dep_step<op>(x, k, m); // warm the pipe
    const long long t0 = clock64();
    for (int i = 0; i < kLatencyLinks; ++i) dep_step<op>(x, k, m);
    const long long t1 = clock64();
    if (threadIdx.x == 0) spans[blockIdx.x] = {t0, t1, smid()};
    if (x == 0xffffffffu) sink[0] = x;
}

// ---- Memory latencies ----
// A cycle is `m` nodes (a power of two) visited in pseudo-random order (fill_cycle: the node of
// the next link is unrelated to this one's); a node is one 8-byte word holding the address of the
// next node, so a link is the load and nothing else (probe 14; until then a node held an index,
// and its add to the base was three dependent ops in the SASS: IADD3, IMAD.X, LEA.HI.X). Nodes
// are `stride` words apart, so with a stride of 16 words every node is a 128-byte line of its
// own (four 32-byte sectors) and two links never read the same line. One CTA per SM; lane 0 of
// each of its `warps` warps chases its own chain, the other lanes idle. The first `links` links
// warm the level (the lines of a resident level, the pages of the TLB-warm one); the timed links
// continue the cycle from there, so on the DRAM levels they are lines nothing has read. `Cg`
// loads bypass the L1 (ld.global.cg: cached in L2 only), so an L2 or DRAM figure cannot be an L1
// hit; the L1 level uses the L1-cached path. Each warp records its own span; the host takes the
// median over warps and SMs.
// A bijection of the `bits`-bit indices (bits even): a four-round balanced Feistel network whose
// round function is a multiplicative hash, so the images of neighbouring indices sit anywhere in
// the sub-buffer. Seeded per sub-buffer (cycle_seed).
__device__ __forceinline__ unsigned feistel(unsigned x, unsigned bits, unsigned seed) {
    const unsigned half = bits / 2, mask = (1u << half) - 1u;
    unsigned lo = x & mask, hi = x >> half;
    for (unsigned r = 0; r < 4; ++r) {
        unsigned f = (hi ^ (seed + r * 0x9E3779B9u)) * 0x85EBCA6Bu;
        f ^= f >> 13;
        f *= 0xC2B2AE35u;
        f ^= f >> 16;
        const unsigned nlo = hi, nhi = (lo ^ f) & mask;
        lo = nlo;
        hi = nhi;
    }
    return (hi << half) | lo;
}

__device__ __forceinline__ unsigned cycle_seed(unsigned sub) {
    return 0x1234567u + 0x9E3779B9u * sub;
}

using Node = unsigned long long; // a node: the address of the next one

template <bool Cg>
__device__ __forceinline__ const Node* load_link(const Node* p) {
    Node v;
    if constexpr (Cg) {
        asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p));
    } else {
        // The L1-cached path spelled out: a plain `*p` on a const __restrict__ pointer may take
        // the read-only (texture) path, whose hit is longer than the L1's.
        asm volatile("ld.global.ca.u64 %0, [%1];" : "=l"(v) : "l"(p));
    }
    return reinterpret_cast<const Node*>(v);
}

// Whose links a chasing warp walks.
enum class Own {
    // l1, l2: 8 cycles, one per warp index, shared by the CTAs; CTA b starts at position b x m /
    // grid. The cycles are shorter than the chase, so the lines are met again: a line another
    // warp or SM pulled into the level is what a hit of it is.
    Shared,
    // dram_tlbwarm: every (CTA, warp) pair has a cycle of its own in a range of its own, small
    // enough for the warm-up to have walked its pages.
    Buffer,
    // dram: one cycle over the whole buffer for the chip, every (CTA, warp) pair on a segment of
    // positions of its own, its lines anywhere in the buffer.
    Segment,
};

// The cycle runs through the positions x = 0, 1, 2, ... (fill_cycle), so segments are plain index
// ranges, and the `run`-th chase of a level (one per warp count) begins 2 x links further on
// than the one before: on the DRAM levels, where a pair's range holds its four chases (the host
// checks it), no chase reads a line an earlier link of any chase read.
template <bool Cg>
__global__ void chase(const Node* buf, unsigned bits, unsigned stride, Own own, int links,
                      unsigned run, Span* spans, unsigned* sink) {
    const int warp = threadIdx.x / 32;
    if (threadIdx.x % 32 != 0) return;
    const unsigned m     = 1u << bits;
    const unsigned pair  = blockIdx.x * 8 + warp;
    const unsigned sub   = own == Own::Shared ? warp : own == Own::Buffer ? pair : 0u;
    const Node* mine     = buf + static_cast<size_t>(sub) * m * stride;
    const unsigned first = own == Own::Shared    ? blockIdx.x * (m / gridDim.x)
                           : own == Own::Segment ? pair * (m / (gridDim.x * 8u))
                                                 : 0u;
    const unsigned x0    = first + run * 2u * links;
    const Node* p =
        mine + static_cast<size_t>(feistel(x0 & (m - 1u), bits, cycle_seed(sub))) * stride;
    for (int i = 0; i < links; ++i) p = load_link<Cg>(p);
    const long long n0 = globaltimer_ns();
    const long long t0 = clock64();
    for (int i = 0; i < links; ++i) p = load_link<Cg>(p);
    const long long t1           = clock64();
    const long long n1           = globaltimer_ns();
    spans[blockIdx.x * 8 + warp] = {t0, t1, smid(), n0, n1};
    if (p == nullptr) sink[0] = 1u; // keeps the chain alive, never true
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

// `count` cycles of 2^bits nodes, `stride` words a node, each cycle through all its nodes in
// a pseudo-random order: the successor cycle conjugated by the Feistel bijection, seeded per
// cycle so no two share their order. (An LCG cycle was tried first: its own low bits repeat with
// short periods and its orbit order is not an index order, so segments of it could neither be
// kept apart from a DRAM row nor from one another.)
__global__ void fill_cycle(Node* buf, unsigned bits, unsigned stride, unsigned count) {
    const unsigned m = 1u << bits;
    const size_t i   = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(m) * count) return;
    const unsigned w = static_cast<unsigned>(i / m), x = static_cast<unsigned>(i % m);
    const unsigned seed = cycle_seed(w);
    Node* mine          = buf + static_cast<size_t>(w) * m * stride;
    // The successor cycle x -> x + 1 mod m, conjugated by the bijection: node P(x) holds the
    // address of node P(x + 1). Position k of the cycle is P(k), so a segment of the cycle is a
    // plain range of k, and the addresses of successive links are unrelated.
    mine[static_cast<size_t>(feistel(x, bits, seed)) * stride] = reinterpret_cast<Node>(
        mine + static_cast<size_t>(feistel((x + 1u) & (m - 1u), bits, seed)) * stride);
}

// One word written per 32-byte sector of a scratch range twice the L2's size, after a DRAM
// level's cycles are filled: the nodes the fill wrote last leave the L2, so a timed first read is
// not the fill's residue. (That the L2 gives the older lines up is the cache's ordinary
// behaviour, not a fact read here: isa_probe_ncu.sh counts the L2 hits of a DRAM chase.)
__global__ void evict(Node* scratch, size_t sectors) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < sectors) scratch[i * 4] = i;
}

double median(std::vector<double> d) {
    std::sort(d.begin(), d.end());
    return d[d.size() / 2];
}

std::vector<std::string> g_failures;

void fact(const char* name, double value, const char* unit) {
    std::printf("fact %s %.3f %s\n", name, value, unit);
}

void check(bool ok, const std::string& what) {
    if (!ok) g_failures.push_back(what);
}

struct Probe {
    int sms         = 0;
    Span* spans     = nullptr;
    float* fsink    = nullptr;
    unsigned* usink = nullptr;
    std::vector<Span> host;

    std::vector<Span>& collect(int ctas) {
        // The legacy stream alone: a device-wide wait would hang on the clock holder.
        CHECK(cudaStreamSynchronize(nullptr));
        host.resize(ctas);
        CHECK(cudaMemcpyAsync(host.data(), spans, ctas * sizeof(Span), cudaMemcpyDeviceToHost,
                              nullptr));
        CHECK(cudaStreamSynchronize(nullptr));
        return host;
    }

    // Cycles per instruction per SM of a throughput kernel launched as `ctas_per_sm` CTAs of
    // `warps` warps an SM: per SM, the span of all its CTAs over the instructions they issued;
    // the median over SMs. The hardware does not spread the CTAs evenly, so the count per SM is
    // read, not assumed. `saturated`: every SM counts whatever it held (more or fewer CTAs of a
    // loop that fills every partition issue at the same rate); otherwise only the SMs that held
    // exactly the CTAs asked for, all at once (their spans overlap), so the figure is at the
    // stated warp count, and fewer than half of them fails the self-check. What the SMs held is
    // a note of the run: a property of the launch, not a fact of the card.
    template <typename Kernel, typename... Args>
    double per_sm(const char* what, bool saturated, int ctas_per_sm, int warps, Kernel kernel,
                  Args... args) {
        const int ctas = sms * ctas_per_sm;
        kernel<<<ctas, 32 * warps>>>(spans, fsink, args...); // warm
        LAUNCHED();
        kernel<<<ctas, 32 * warps>>>(spans, fsink, args...);
        LAUNCHED();
        std::map<int, std::vector<Span>> by_sm;
        for (const Span& s : collect(ctas)) by_sm[s.sm].push_back(s);
        std::vector<double> cycles, every;
        for (const auto& [sm, list] : by_sm) {
            long long lo = list[0].begin, hi = list[0].end; // the span of them all
            long long last_begin = list[0].begin, first_end = list[0].end;
            for (const Span& s : list) {
                lo         = std::min(lo, s.begin);
                hi         = std::max(hi, s.end);
                last_begin = std::max(last_begin, s.begin);
                first_end  = std::min(first_end, s.end);
            }
            const double work = static_cast<double>(kThroughputN) * kUnroll * warps * list.size();
            every.push_back(static_cast<double>(hi - lo) / work);
            if (static_cast<int>(list.size()) == ctas_per_sm && last_begin < first_end)
                cycles.push_back(every.back());
        }
        std::printf("note %s: %d of %d SMs ran %d CTA(s) of %d warps at once%s\n", what,
                    static_cast<int>(cycles.size()), sms, ctas_per_sm, warps,
                    saturated ? "; every SM counts" : ", the others are left out");
        if (saturated) return median(every);
        check(2 * static_cast<int>(cycles.size()) >= sms,
              std::string(what) + ": fewer than half the SMs ran the CTAs asked for at once");
        // With none the check above has failed the run; the median still needs spans.
        return median(cycles.empty() ? every : cycles);
    }
};

// The CTAs of a latency launch (one CTA per SM asked for; `per_cta` span entries each, the first
// always written) that had their SM to themselves. The hardware does not spread CTAs evenly: two
// on one SM run beside twice the stated warps, so their spans are left out of the latency and
// the count is noted. Fewer than half the CTAs alone fails the self-check. `holders`, for the
// launches that run beside the clock holder: the holder CTAs on every SM; a CTA whose SM holds
// none or two of them ran at another clock or beside another load, and is left out as well.
std::vector<char> alone(const std::vector<Span>& spans, int per_cta, const std::string& what,
                        const std::map<int, int>* holders = nullptr) {
    const int ctas = static_cast<int>(spans.size()) / per_cta;
    std::map<int, int> on_sm;
    for (int c = 0; c < ctas; ++c) ++on_sm[spans[c * per_cta].sm];
    std::vector<char> keep(ctas);
    int kept = 0;
    for (int c = 0; c < ctas; ++c) {
        const int sm = spans[c * per_cta].sm;
        keep[c]      = on_sm[sm] == 1;
        if (holders != nullptr) {
            const auto h = holders->find(sm);
            keep[c]      = keep[c] && h != holders->end() && h->second == 1;
        }
        kept += keep[c];
    }
    if (kept < ctas)
        std::printf("note %s: %d of %d CTAs had an SM to themselves%s, the others are left out\n",
                    what.c_str(), kept, ctas, holders != nullptr ? " and one clock holder" : "");
    check(2 * kept >= ctas, what + ": fewer than half the CTAs had an SM to themselves");
    // With none alone the check above has failed the run; the median still needs spans.
    if (kept == 0) std::fill(keep.begin(), keep.end(), char{1});
    return keep;
}

// The median span of a one-warp latency launch over the CTAs alone on their SM.
double latency_span(const std::vector<Span>& spans, const std::string& what) {
    const std::vector<char> keep = alone(spans, 1, what);
    std::vector<double> d;
    for (size_t c = 0; c < spans.size(); ++c)
        if (keep[c]) d.push_back(static_cast<double>(spans[c].end - spans[c].begin));
    return median(d);
}

} // namespace

int main() {
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    int driver = 0, runtime = 0;
    CHECK(cudaDriverGetVersion(&driver));
    CHECK(cudaRuntimeGetVersion(&runtime));
    const int cc = p.major * 10 + p.minor;
    std::printf("probe isa_probe 14\n");
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
    LAUNCHED();
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
    const double mma_sm =
        pr.per_sm("mma_cycles_per_sm", true, kCtasPerSm, kWarpsPerCta, mma_throughput, ones, ones);
    fact("mma_cycles_per_sm", mma_sm, "clk");
    fact("mma_cycles_per_partition", mma_sm * 4.0, "clk");
    const double tensor_rate = shape.flops / mma_sm;
    fact("tensor_flops_per_sm_clock", tensor_rate, "flop");
    // No SM the probe knows issues a dense MMA above 2048 flop a clock (the GA100 / GH100 class)
    // or below one m8n8k4 in 32 clocks.
    check(tensor_rate <= 2048.0 && tensor_rate >= 16.0, "tensor rate out of range");
    // The same loop at 4, 8 and 16 warps an SM, one CTA each (1, 2 and 4 warps a scheduler where
    // the SM deals a CTA's warps round its four, which is not read here): whether the SM's rate
    // above needs every warp, or one warp's independent MMAs already fill a partition's pipe.
    for (int warps : {4, 8, 16}) {
        char name[64];
        std::snprintf(name, sizeof name, "mma_cycles_per_sm_w%d", warps);
        fact(name, pr.per_sm(name, false, 1, warps, mma_throughput, ones, ones), "clk");
    }
    mma_latency<<<pr.sms, 32>>>(pr.spans, pr.fsink, ones, ones);
    LAUNCHED();
    mma_latency<<<pr.sms, 32>>>(pr.spans, pr.fsink, ones, ones);
    LAUNCHED();
    const double mma_lat =
        latency_span(pr.collect(pr.sms), "mma_dependent_latency") / kLatencyLinks;
    fact("mma_dependent_latency", mma_lat, "clk");
    check(mma_lat >= mma_sm * 4.0 * 0.5, "dependent MMA latency under half the issue interval");

    // INT32 / FP32 pipes: cycles per warp instruction per partition. The floors are the lane
    // counts: 32 FP32 lanes a partition from Ampere (1 clock), 16 before (2); 16 INT32 lanes (2).
    const double fp32 = pr.per_sm("fp32_cycles_per_warp_instruction", true, kCtasPerSm,
                                  kWarpsPerCta, fp32_throughput, 0.999f, 0.001f) *
                        4.0;
    const double i32 = pr.per_sm("int32_cycles_per_warp_instruction", true, kCtasPerSm,
                                 kWarpsPerCta, int32_throughput, 3u, 1u) *
                       4.0;
    fact("fp32_cycles_per_warp_instruction", fp32, "clk");
    fact("int32_cycles_per_warp_instruction", i32, "clk");
    const double fp32_floor = cc >= 80 ? 1.0 : 2.0;
    check(fp32 >= 0.97 * fp32_floor, "FP32 cycles per warp instruction under the lane floor");
    check(i32 >= 0.97 * 2.0, "INT32 cycles per warp instruction under the lane floor");

    // Dependent latencies of the decode chain's ops, one warp per SM; the conversion pairs count
    // two ops a link and are reported per op.
    const auto dep = [&](auto launch, const char* name, double ops_per_link, double lo, double hi) {
        launch();
        LAUNCHED();
        launch();
        LAUNCHED();
        const double clk_per_op =
            latency_span(pr.collect(pr.sms), name) / kLatencyLinks / ops_per_link;
        fact(name, clk_per_op, "clk");
        check(clk_per_op >= lo && clk_per_op <= hi,
              std::string(name) + " outside its published range");
    };
#define ISA_PROBE_DEP(OP, NAME, OPS, LO, HI)                                                       \
    dep([&] { dep_latency<Op::OP><<<pr.sms, 32>>>(pr.spans, pr.usink, 3u, 0x1234567u); }, NAME,    \
        OPS, LO, HI)
    ISA_PROBE_DEP(IMAD, "lat_imad", 1.0, 2.0, 16.0);
    ISA_PROBE_DEP(IADD, "lat_iadd", 1.0, 2.0, 16.0);
    ISA_PROBE_DEP(LOP3, "lat_lop3", 1.0, 2.0, 16.0);
    ISA_PROBE_DEP(PRMT, "lat_prmt", 1.0, 2.0, 16.0);
    ISA_PROBE_DEP(SHF, "lat_shf", 1.0, 2.0, 16.0);
    ISA_PROBE_DEP(HFMA2, "lat_hfma2", 1.0, 2.0, 16.0);
    ISA_PROBE_DEP(FFMA, "lat_ffma", 1.0, 2.0, 16.0);
    ISA_PROBE_DEP(SHFL, "lat_shfl", 1.0, 10.0, 60.0);
    ISA_PROBE_DEP(CVT_I2F_F2I, "lat_cvt_i2f_f2i", 2.0, 2.0, 32.0);
    ISA_PROBE_DEP(CVT_F32_F16, "lat_cvt_f32_f16", 2.0, 2.0, 32.0);
#undef ISA_PROBE_DEP

    // Memory latencies: W = 1, 2, 4, 8 chasing warps per SM, every warp on a pseudo-random cycle
    // (fill_cycle). The cycle sizes are 2^bits nodes of 8 bytes, bits even for the Feistel halves.
    // L1: 2 KB a warp (16 KB at 8 warps, inside every L1), L1-cached loads. L2: 32 KB a warp
    // (256 KB at 8 warps, inside every L2), loads that bypass the L1; its warm-up links walk the
    // whole cycle. These two share their 8 cycles between the SMs: a line another SM pulled into
    // the L2 is what an L2 hit is.
    // DRAM: loads that bypass the L1, a node a 128-byte line of its own, the timed links on lines
    // nothing has read: no two links of the chip are on one line, and the lines the fill wrote
    // last are pushed out of the L2 before the chases (evict). dram_tlbwarm gives every (SM,
    // warp) pair its own 512 KB range (a quarter of a 2 MB page, at most 8 pages an SM at 8
    // warps), whose pages the warm-up links have walked; dram runs every pair on its own segment
    // of one cycle over the whole 2 GiB buffer (1024 pages of 2 MB, a link on any of them). That
    // the first level's page walks are cached and the second's are not is the levels' intent,
    // not a reading: the probe does not see the TLB, only that dram is not the shorter of the two
    // (checked below). Both time kDramLinks links after as many warm-up links, and a pair's four
    // chases (one per warp count) never meet a line again.
    struct Level {
        const char* name;
        unsigned bits;
        unsigned stride; // 8-byte words from a node to the next
        bool cg;
        Own own;
        int links;
    } levels[] = {{"l1", 8, 1, false, Own::Shared, kLatencyLinks},
                  {"l2", 12, 1, true, Own::Shared, kLatencyLinks},
                  {"dram_tlbwarm", 12, 16, true, Own::Buffer, kDramLinks},
                  {"dram", 24, 16, true, Own::Segment, kDramLinks}};

    const auto cycles_of = [&](const Level& lv) {
        return lv.own == Own::Shared ? 8u : lv.own == Own::Buffer ? pr.sms * 8u : 1u;
    };

    Node* buf           = nullptr;
    size_t words_of_buf = 0;
    for (const Level& lv : levels)
        words_of_buf = std::max(words_of_buf, (size_t{1} << lv.bits) * lv.stride * cycles_of(lv));
    CHECK(cudaMalloc(&buf, words_of_buf * sizeof(Node)));
    // The scratch range evict writes: twice the L2, one word per 32-byte sector.
    Node* scratch              = nullptr;
    const size_t evict_sectors = 2 * static_cast<size_t>(p.l2CacheSize) / 32;
    CHECK(cudaMalloc(&scratch, evict_sectors * 32));
    std::map<std::string, double> lat;
    std::vector<double> mhz; // the SM clock seen by each L2 / DRAM chase
    // The clock holder: one warp of FFMA per SM asked for on a second stream for the whole
    // latency section, stopped through mapped host words; every holder CTA writes the SM it
    // landed on, so the chases know which SMs ran beside exactly one.
    volatile int* words = nullptr;
    int* words_dev      = nullptr;
    CHECK(cudaHostAlloc(reinterpret_cast<void**>(const_cast<int**>(&words)),
                        (kHolderSms + pr.sms) * sizeof(int), cudaHostAllocMapped));
    words[kHolderStop]    = 0;
    words[kHolderExpired] = 0;
    for (int i = 0; i < pr.sms; ++i) words[kHolderSms + i] = -1;
    CHECK(
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&words_dev), const_cast<int*>(words), 0));
    cudaStream_t side = nullptr;
    CHECK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
    spinner<<<pr.sms, 32, 0, side>>>(words_dev, kHolderSeconds * 1000000000LL, pr.fsink);
    LAUNCHED();
    // Wait (5 s at most) until every holder CTA has named its SM.
    std::map<int, int> holders;
    {
        const auto started = [&] {
            for (int i = 0; i < pr.sms; ++i)
                if (words[kHolderSms + i] < 0) return false;
            return true;
        };
        const auto t0 = std::chrono::steady_clock::now();
        while (!started() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        check(started(), "the clock holder did not start on every CTA within 5 s");
        for (int i = 0; i < pr.sms; ++i) {
            const int sm = words[kHolderSms + i];
            if (sm >= 0) ++holders[sm];
        }
    }

    // The spans of the chasing warps: entry CTA x 8 + warp, the first `warps` of every CTA.
    struct Chase {
        double clk, ns, mhz; // per link in SM clocks and wall ns; the SM clock the chase ran at
    };

    // Only the CTAs alone on their SM, beside one clock holder, count: on a shared SM a warp
    // chases beside twice the stated warps, which is another level's latency.
    const auto chase_clocks = [&](int warps, int links, const std::string& what) {
        std::vector<double> clk, ns;
        const std::vector<Span>& spans = pr.collect(pr.sms * 8);
        const std::vector<char> keep   = alone(spans, 8, what, &holders);
        for (size_t entry = 0; entry < spans.size(); ++entry) {
            if (!keep[entry / 8] || static_cast<int>(entry % 8) >= warps) continue;
            const Span& s = spans[entry];
            clk.push_back(static_cast<double>(s.end - s.begin));
            ns.push_back(static_cast<double>(s.ns_end - s.ns_begin));
        }
        const double c = median(clk), n = median(ns);
        return Chase{c / links, n / links, n > 0 ? c / n * 1e3 : 0.0};
    };
    for (const Level& lv : levels) {
        const unsigned m     = 1u << lv.bits;
        const unsigned count = cycles_of(lv);
        const size_t total   = static_cast<size_t>(m) * count;
        // A pair's range of the cycle holds its four chases, warm-up and timed links each.
        const unsigned range = lv.own == Own::Buffer    ? m
                               : lv.own == Own::Segment ? m / (pr.sms * 8u)
                                                        : 0u;
        if (lv.own != Own::Shared)
            check(range >= 4u * 2u * lv.links,
                  std::string(lv.name) +
                      ": a warp's range of the cycle is shorter than its chases");
        fill_cycle<<<static_cast<unsigned>((total + 255) / 256), 256>>>(buf, lv.bits, lv.stride,
                                                                        count);
        LAUNCHED();
        if (lv.own != Own::Shared) {
            evict<<<static_cast<unsigned>((evict_sectors + 255) / 256), 256>>>(scratch,
                                                                               evict_sectors);
            LAUNCHED();
        }
        CHECK(cudaStreamSynchronize(nullptr));
        unsigned run = 0;
        for (int warps : {1, 2, 4, 8}) {
            if (lv.cg) {
                chase<true><<<pr.sms, 32 * warps>>>(buf, lv.bits, lv.stride, lv.own, lv.links, run,
                                                    pr.spans, pr.usink);
            } else {
                chase<false><<<pr.sms, 32 * warps>>>(buf, lv.bits, lv.stride, lv.own, lv.links, run,
                                                     pr.spans, pr.usink);
            }
            LAUNCHED();
            ++run;
            char name[64];
            std::snprintf(name, sizeof name, "%s_latency_w%d", lv.name, warps);
            const Chase c = chase_clocks(warps, lv.links, name);
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
    words[kHolderStop] = 1;
    CHECK(cudaStreamSynchronize(side));
    CHECK(cudaStreamDestroy(side));
    // A holder that left at its deadline left the later chases without their clock.
    check(words[kHolderExpired] == 0, "the clock holder's deadline passed before the chases ended");
    fact("sm_clock_mhz_chase", median(mhz), "mhz");
    // Each chase's ns come from its own window (clock64 against globaltimer within it), so a clock
    // that moved between chases leaves the ns right and only the clk figures of those levels at
    // the clock they ran at: reported, not refused.
    if (mhz.size() == 12) {
        const double lo = *std::min_element(mhz.begin(), mhz.end());
        const double hi = *std::max_element(mhz.begin(), mhz.end());
        if (hi > 1.05 * lo)
            std::printf("note the SM clock ranged %.0f..%.0f MHz over the L2 / DRAM chases; their "
                        "clk figures are at those clocks, the ns stand\n",
                        lo, hi);
    }
    chase_shared<<<pr.sms, 128>>>(pr.spans, pr.usink);
    LAUNCHED();
    chase_shared<<<pr.sms, 128>>>(pr.spans, pr.usink);
    LAUNCHED();
    fact("shared_latency", latency_span(pr.collect(pr.sms), "shared_latency") / kLatencyLinks,
         "clk");
    // The levels must separate: an L2 hit is further than an L1 hit, DRAM further than the L2 at
    // every warp count (chains that alias into L2-resident lines would read as the L2), the DRAM
    // level whose pages are anywhere is not shorter than the one whose pages are walked (3 % for
    // the timer's step; shorter would mean its links met lines the L2 held), and more chasing
    // warps never make a warp's latency shorter (in SM clocks for the L1, in wall ns for the L2
    // and DRAM).
    check(lat["l2_latency_w1"] > 2.0 * lat["l1_latency_w1"], "L2 latency not above 2x L1");
    for (int w : {1, 2, 4, 8}) {
        const std::string ws = "_latency_w" + std::to_string(w);
        const std::string ns = "_latency_ns_w" + std::to_string(w);
        for (const char* lv : {"dram_tlbwarm", "dram"})
            check(lat[lv + ns] > 1.2 * lat["l2" + ns],
                  std::string(lv) + " latency not above 1.2x L2 at w" + std::to_string(w));
        check(lat["dram" + ns] >= 0.97 * lat["dram_tlbwarm" + ns],
              "dram latency under the TLB-warm level's at w" + std::to_string(w));
        check(lat["l1" + ws] >= 0.95 * lat["l1_latency_w1"],
              "l1 latency falls under the lone warp's at w" + std::to_string(w));
        for (const char* lv : {"l2", "dram_tlbwarm", "dram"})
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
