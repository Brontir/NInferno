// isa_probe: the ISA facts of one card that the register-MMA price's per-warp timeline needs
// (#613 PR-2): tensor pipe cycles per MMA at the SM and the dependent MMA latency, the INT32 and
// FP32 pipes' cycles per warp instruction, and the dependent load latencies (L1 hit, L2 hit, DRAM,
// shared memory) alone and under 2, 4 and 8 chasing warps per SM. What a pipe or the L1 does is in
// SM clocks (clock64). A load beyond the L1 is reported both in SM clocks and in wall ns
// (%globaltimer), with the SM clock the chase ran at, and measured twice (probe 15): on a chip
// that runs nothing else, and beside one warp of FFMAs on every SM (the `_held` facts). The card
// clocks an SM whose only warp waits on memory differently from a working one, so which part of
// a latency follows the clock and which is wall time is read from the pair, not assumed.
//
// Build (CUDA 13.x; sm_70 needs CUDA 12.8's nvcc):
//   nvcc -O3 -std=c++17 -arch=sm_<arch> -o isa_probe tools/bench/isa_probe/isa_probe.cu
// Run (10 to 20 s; 2 GiB of device memory for the load-latency buffer, beside the CUDA context;
// the GPU should be otherwise idle; one kernel runs for several seconds, which a display
// watchdog would end, so not on the card that drives a desktop):
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
// and a shared node holds the shared-window address of the next, so a shared link is a bare
// `LDS R, [R]` on every architecture (probe 17; until then the node held an index, which sm_75 and
// sm_86 scale inside the load, `[R.X4]`, but sm_70 scales with a shift before every LDS, so the
// sm_70 figure of probes up to 16 is LDS plus a shift). Loaded latency: W warps per SM
// each chase a pseudo-random cycle of their own (the L2 and DRAM loads bypass the L1), each warp
// timing its own links; on the DRAM levels a link is a 128-byte line of its own (four 32-byte
// sectors) that no other link of the chip shares, whoever chases it (probe 14). A latency launch
// asks for one CTA per SM; a CTA that shared its SM with another (read from %smid) is left out,
// so every latency is at the stated warp count (probe 13), and so is a `_held` chase whose SM did
// not hold exactly one FFMA warp. The `dram_lanes` level is the DRAM chase with all 32 lanes of
// every warp on chains of their own: 32 x W lines in flight an SM, the warp's link as long as
// its slowest line (probe 15), which is what the memory system's throughput is read from.
// `dram_lanes_half` is that chase on half the SMs (probe 16): with the same lines in flight per
// SM and half as many on the chip, the pair says whether a link waits on its own SM's lines or
// on the chip's.
//
// The module is loaded eagerly (CUDA_MODULE_LOADING=EAGER, set by main before the first CUDA
// call). Under the default lazy loading the first launch of a kernel waits until nothing else
// runs in the context, so the first launch beside the FFMA kernel waited for that kernel to
// leave: every chase of probes 12 to 14 ran on a chip with nothing else on it (probe 14 refused
// such a run, 12 and 13 could not tell), which is the lone measurement of this version.

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
// a chase, against a %globaltimer that steps by about 1 us. The ns facts carry that step: a
// one-lane DRAM figure is a whole number of 1 us steps over 512 links (the probe-16 rows read in
// steps of about 2 ns), a dram_lanes figure over 256 links (about 4 ns), well inside the 3 % and
// 5 % margins of the checks below.
constexpr int kDramLinks = 512;
// The 32-lane DRAM chase: a lane's segment of the 2^24-line cycle is 2^24 / (SMs x 256) positions
// (385 on a 170-SM card), which holds these.
constexpr int kLanesWarm = 64, kLanesLinks = 256;

struct Span {
    long long begin, end; // clock64: SM clocks
    int sm;
    long long ns_begin = 0, ns_end = 0; // %globaltimer: wall ns (the chases and the MMA loop)
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
// The MMA facts (cycles per MMA, the dependent latency) are per PTX mma.sync of this shape. ptxas
// lowers one mma.sync.m8n8k4 on sm_70 to four SASS instructions (HMMA.884.F32.F32.STEP0..3, one
// per 8x8x4 product), and one mma.sync elsewhere to one HMMA: how many SASS MMAs one mma.sync is
// on the build is read from its SASS by `isa_probe.py run` (mma_unit below) and kept on the row
// as `sass_per_mma`, so a price per SASS MMA is the fact over that count.
constexpr Shape kShapes[] = {
    {"hmma_884_f32_f16", 2048}, {"hmma_1688_f32_f16", 2048}, {"hmma_16816_f32_bf16", 4096}};

} // namespace

// Exactly one mma.sync of the shape and nothing else of the tensor pipe: never launched, built only
// so that `isa_probe.py run` can count the SASS MMA instructions of one mma.sync in this kernel
// (cuobjdump -sass), whatever the unrolling of the timed loops. Outside the anonymous namespace:
// a kernel of external linkage is kept in the cubin although nothing launches it.
__global__ void isa_probe_mma_unit(float* out, unsigned a, unsigned b) {
    float acc[kAcc];
    for (int e = 0; e < kAcc; ++e) acc[e] = out[e];
    mma_once(acc, a, a, a, a, b, b);
    for (int e = 0; e < kAcc; ++e) out[e] = acc[e];
}

namespace {

__global__ void mma_throughput(Span* spans, float* sink, unsigned a, unsigned b) {
    float acc[kUnroll][kAcc];
    for (int u = 0; u < kUnroll; ++u)
        for (int e = 0; e < kAcc; ++e) acc[u][e] = 0.f;
    __syncthreads();
    const long long n0 = globaltimer_ns();
    const long long t0 = clock64();
    for (int i = 0; i < kThroughputN; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) mma_once(acc[u], a, a, a, a, b, b);
    }
    __syncthreads();
    const long long t1 = clock64();
    const long long n1 = globaltimer_ns();
    float s            = 0.f;
    for (int u = 0; u < kUnroll; ++u)
        for (int e = 0; e < kAcc; ++e) s += acc[u][e];
    if (threadIdx.x == 0) spans[blockIdx.x] = {t0, t1, smid(), n0, n1};
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

// The FFMA warp of the `_held` chases: one warp of dependent FFMAs asked for on every SM, on a
// second stream, so each chasing warp shares its SM with a working warp and the card clocks the
// SM as one at work. Its loop touches no memory; it leaves `deadline_ns` after it started and not
// before. (Probe 14's loop polled a host flag: once the kernel really ran beside the chases, that
// poll cost them tens of clocks a link.) The host checks after the last held chase that the
// kernel is still running (cudaStreamQuery), so a deadline too short for the chases refuses the
// run. The one store, before the loop, names the SM each CTA landed on (mapped host words, -1
// until then): a chase counts only where exactly one of these warps sat beside it. Nothing in
// the held section synchronises the device as a whole (that would wait on this kernel): the
// chases and their copies run and are waited for on the legacy stream alone, which a
// non-blocking stream does not join.
__global__ void spinner(volatile int* sm_of_cta, long long deadline_ns, float* sink) {
    float x                  = threadIdx.x * 1e-3f;
    const long long deadline = globaltimer_ns() + deadline_ns;
    if (threadIdx.x == 0) {
        sm_of_cta[blockIdx.x] = smid();
        __threadfence_system();
    }
    do {
#pragma unroll
        for (int i = 0; i < 1024; ++i) x = fmaf(x, 0.999f, 0.001f);
    } while (globaltimer_ns() < deadline);
    if (x == 12345.f) sink[0] = x;
}

// ---- INT32 and FP32 pipes ----
// The span holds the loop's own instructions beside the FFMAs: counter, compare, branch, and on
// sm_86 / sm_89 a reload of the multiplier and a predicated exit call, 3 to 5 per iteration in the
// SASS of nvcc 12.8 / 13.2. Probe 17 unrolls the loop 32 times, 256 FFMAs an iteration against 128
// before, when the sm_86 / sm_120a figure sat about 5 % above the one-clock floor of 32 FP32
// lanes (the INT ops share a datapath with half of those lanes there). What is left of the loop
// is in the fact, not taken out.
__global__ void fp32_throughput(Span* spans, float* sink, float m, float c) {
    float x[kUnroll];
    for (int u = 0; u < kUnroll; ++u) x[u] = threadIdx.x * 1e-3f + u;
    __syncthreads();
    const long long t0 = clock64();
#pragma unroll 32
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
// so ptxas keeps the op: IMAD, the add chain, LOP3, PRMT, SHF, HFMA2, FFMA, SHFL, and each of the
// four conversions alone); cycles per link are the op's issue-to-use latency. lat_add_chain is
// the add ptxas makes of a dependent doubling, not IADD3 alone: in the sm_70, sm_75, sm_86 and
// sm_89 SASS (nvcc 12.8 for sm_70, 13.2 for the others) the chain alternates IADD3 and IMAD.IADD
// (half each), so the figure is the mean of the two forms; on sm_120a it is IADD alone (probes up
// to 16 named it lat_iadd). A conversion link is the one conversion and a
// reinterpretation of its result as the next link's source, which costs no SASS instruction (the
// register is read as the other type, or one half of it is selected by the consumer's operand):
// int -> float, float -> int, f32 -> f16 and f16 -> f32 each time their own pipe (probes up to 16
// timed the pairs I2F + F2I and f32 -> f16 -> f32 and reported the mean of two ops on different
// pipes). Each is held against the published range of the Volta..Blackwell SM (ALU 2-16 clk,
// conversions 2-32, SHFL 10-60) by the self-check.
enum class Op { IMAD, ADD, LOP3, PRMT, SHF, HFMA2, FFMA, SHFL, I2F, F2I, F32_TO_F16, F16_TO_F32 };

template <Op op>
__device__ __forceinline__ void dep_step(unsigned& x, unsigned k, unsigned m) {
    if constexpr (op == Op::IMAD) {
        asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(k), "r"(m));
    } else if constexpr (op == Op::ADD) {
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
    } else if constexpr (op == Op::I2F) {
        // The float's bits are the next link's integer.
        float f;
        asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"(x));
        x = __float_as_uint(f);
    } else if constexpr (op == Op::F2I) {
        // The integer's bits are the next link's float.
        asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(x) : "f"(__uint_as_float(x)));
    } else if constexpr (op == Op::F32_TO_F16) {
        // The half's 16 bits, zero-extended, are the next link's float (the conversion writes the
        // whole register). ptxas makes this F2F.F16.F32 on sm_70 to sm_120a; the packing form
        // F2FP.PACK_AB, which sm_80 and later use where the half feeds a half op (the pair of
        // probes up to 16 on sm_86+), is another instruction and not timed here.
        unsigned short h;
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(__uint_as_float(x)));
        x = h;
    } else {
        // The low half of the float's bits is the next link's half (selected by the operand).
        float f;
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(static_cast<unsigned short>(x)));
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
    // Keeps the chain alive. A constant the chain's result provably cannot equal would not:
    // f32 -> f16 leaves 16 bits, and against 0xffffffff ptxas dropped that whole chain. `m` is a
    // launch argument, unknown to the compiler; a store on a chance match is harmless.
    if (x == m) sink[0] = x;
}

// ---- Memory latencies ----
// A cycle is `m` nodes (a power of two) visited in pseudo-random order (fill_cycle: the node of
// the next link is unrelated to this one's); a node is one 8-byte word holding the address of the
// next node, so a link is the load and nothing else (probe 14; until then a node held an index,
// and its add to the base was three dependent ops in the SASS: IADD3, IMAD.X, LEA.HI.X). Nodes
// are `stride` words apart, so with a stride of 16 words every node is a 128-byte line of its
// own (four 32-byte sectors) and two links never read the same line. One CTA per SM; lane 0 of
// each of its `warps` warps chases its own chain, the other lanes idle (dram_lanes: every lane a
// chain of its own, see below). The first `warm` links warm the level (the lines of a resident
// level, the pages of the TLB-warm one); the timed links continue the cycle from there, so on the
// DRAM levels they are lines nothing has read. `Cg` loads bypass the L1 (ld.global.cg: cached in L2
// only), so an L2 or DRAM figure cannot be an L1 hit; the L1 level uses the L1-cached path. Each
// warp records its own span; the host takes the median over warps and SMs.
//
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
// ranges, and the `run`-th chase of a level (one per warp count) begins warm + links further on
// than the one before: on the DRAM levels, where a chain's range holds its chases (the host
// checks it), no chase reads a line an earlier link of any chase read.
// `lanes` is 1 (lane 0 of every warp chases, the others return) or 32 with Own::Segment (every
// lane of every warp on a segment of its own: the warp's 32 loads of a link go out together and
// the next link waits for the last of them, so the SM has 32 x warps lines in flight); lane 0
// times the warp.
template <bool Cg>
__global__ void chase(const Node* buf, unsigned bits, unsigned stride, Own own, int warm, int links,
                      unsigned lanes, unsigned run, Span* spans, unsigned* sink) {
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane >= lanes) return;
    const unsigned m     = 1u << bits;
    const unsigned pair  = blockIdx.x * 8u + warp;
    const unsigned sub   = own == Own::Shared ? warp : own == Own::Buffer ? pair : 0u;
    const Node* mine     = buf + static_cast<size_t>(sub) * m * stride;
    const unsigned first = own == Own::Shared ? blockIdx.x * (m / gridDim.x)
                           : own == Own::Segment
                               ? (pair * lanes + lane) * (m / (gridDim.x * 8u * lanes))
                               : 0u;
    const unsigned x0    = first + run * static_cast<unsigned>(warm + links);
    const Node* p =
        mine + static_cast<size_t>(feistel(x0 & (m - 1u), bits, cycle_seed(sub))) * stride;
    for (int i = 0; i < warm; ++i) p = load_link<Cg>(p);
    const long long n0 = globaltimer_ns();
    const long long t0 = clock64();
    for (int i = 0; i < links; ++i) p = load_link<Cg>(p);
    const long long t1 = clock64();
    const long long n1 = globaltimer_ns();
    if (lane == 0) spans[pair] = {t0, t1, smid(), n0, n1};
    if (p == nullptr) sink[0] = 1u; // keeps the chain alive, never true
}

// A node of the shared chain holds the shared-window address (the byte offset in the shared state
// space) of the next node, and the load takes that register as its address: a link is
// `LDS R, [R]` and nothing else on every architecture (probe 17; with an index in the node sm_70
// put a shift before every LDS).
__global__ void chase_shared(Span* spans, unsigned* sink) {
    __shared__ unsigned s[4096];
    const unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(s));
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) s[i] = base + ((i + 33u) & 4095u) * 4u;
    __syncthreads();
    if (threadIdx.x != 0) return;
    unsigned a         = base;
    const long long t0 = clock64();
    for (int i = 0; i < kLatencyLinks; ++i)
        asm volatile("ld.shared.u32 %0, [%0];" : "+r"(a) : : "memory");
    const long long t1 = clock64();
    spans[blockIdx.x]  = {t0, t1, smid()};
    if (a == 0xffffffffu) sink[0] = a;
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
// `_held` launches: the FFMA CTAs on every SM; a CTA whose SM holds none or two of them ran
// beside another load, and is left out as well. Beside the FFMA kernel the hardware places the
// chasing CTAs less evenly (18 to 54 of 80 kept on an 80-SM card, by level, warp count and run;
// in the probe-16 rows only the 2-warp held launches lost CTAs, on four cards of five and none
// on the RTX 3090, which is read, not explained), and a kept CTA is at the stated load however many
// are kept: there the median needs 8 of them.
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
                    what.c_str(), kept, ctas, holders != nullptr ? " and one FFMA warp" : "");
    if (holders != nullptr)
        check(kept >= 8, what + ": fewer than 8 CTAs had an SM to themselves and one FFMA warp");
    else
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
    // Before the first CUDA call: every kernel of the module is loaded when the context is made,
    // so no launch waits for the FFMA kernel to leave (see the head of the file).
#ifdef _WIN32
    _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
#endif
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    int driver = 0, runtime = 0;
    CHECK(cudaDriverGetVersion(&driver));
    CHECK(cudaRuntimeGetVersion(&runtime));
    const int cc = p.major * 10 + p.minor;
    // The memory clock the card is rated at (kHz; the attribute, which CUDA 13 keeps where it
    // dropped the cudaDeviceProp field). The clock the memory ran at during the run is a P-state
    // matter the device API does not give: `isa_probe.py run` samples it with nvidia-smi.
    int mem_khz = 0;
    CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0));
    char pci[32] = {};
    CHECK(cudaDeviceGetPCIBusId(pci, sizeof pci, 0));
    std::printf("probe isa_probe 17\n");
    std::printf("card %s\n", p.name);
    std::printf("cc %d\n", cc);
    std::printf("sms %d\n", p.multiProcessorCount);
    std::printf("l2_bytes %zu\n", static_cast<size_t>(p.l2CacheSize));
    std::printf("mem_clock_khz %d\n", mem_khz);
    std::printf("driver %d\n", driver);
    std::printf("runtime %d\n", runtime);
    std::printf("nvcc %d.%d.%d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__,
                __CUDACC_VER_BUILD__);
    // Which GPU of the machine device 0 is, for `isa_probe.py run` to read that GPU's process
    // list and clocks (nvidia-smi numbers the GPUs its own way).
    std::printf("note device 0 is the GPU at PCI %s\n", pci);

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
    // The SM clock under tensor load: the saturated MMA loop launched back to back for a second,
    // every CTA's clock64 span over its %globaltimer span. The fact is the median over the last
    // quarter of the launches; the note says where the clock was at the first launch (a card
    // that boosts a nearly idle chip steps down once every SM works), and a clock still moving by
    // more than 3 % over that last quarter is refused.
    {
        std::vector<double> seen;
        const auto t0 = std::chrono::steady_clock::now();
        do {
            mma_throughput<<<pr.sms * kCtasPerSm, 32 * kWarpsPerCta>>>(pr.spans, pr.fsink, ones,
                                                                       ones);
            LAUNCHED();
            std::vector<double> m;
            for (const Span& s : pr.collect(pr.sms * kCtasPerSm))
                if (s.ns_end > s.ns_begin)
                    m.push_back(static_cast<double>(s.end - s.begin) * 1e3 /
                                static_cast<double>(s.ns_end - s.ns_begin));
            if (!m.empty()) seen.push_back(median(m));
        } while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(1));
        check(seen.size() >= 8, "the SM clock under the MMA loop was not read");
        if (seen.size() >= 8) {
            const std::vector<double> tail(seen.end() - seen.size() / 4, seen.end());
            const double lo = *std::min_element(tail.begin(), tail.end());
            const double hi = *std::max_element(tail.begin(), tail.end());
            fact("sm_clock_mhz_mma", median(tail), "mhz");
            std::printf("note the SM clock under the MMA loop: %.0f MHz at the first of %zu "
                        "launches, %.0f..%.0f over the last quarter\n",
                        seen.front(), seen.size(), lo, hi);
            check(hi <= 1.03 * lo, "the SM clock under the MMA loop was still moving");
        }
    }

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

    // Dependent latencies of the decode chain's ops, one warp per SM, one op a link.
    const auto dep = [&](auto launch, const char* name, double lo, double hi) {
        launch();
        LAUNCHED();
        launch();
        LAUNCHED();
        const double clk_per_op = latency_span(pr.collect(pr.sms), name) / kLatencyLinks;
        fact(name, clk_per_op, "clk");
        check(clk_per_op >= lo && clk_per_op <= hi,
              std::string(name) + " outside its published range");
    };
#define ISA_PROBE_DEP(OP, NAME, LO, HI)                                                            \
    dep([&] { dep_latency<Op::OP><<<pr.sms, 32>>>(pr.spans, pr.usink, 3u, 0x1234567u); }, NAME,    \
        LO, HI)
    ISA_PROBE_DEP(IMAD, "lat_imad", 2.0, 16.0);
    ISA_PROBE_DEP(ADD, "lat_add_chain", 2.0, 16.0);
    ISA_PROBE_DEP(LOP3, "lat_lop3", 2.0, 16.0);
    ISA_PROBE_DEP(PRMT, "lat_prmt", 2.0, 16.0);
    ISA_PROBE_DEP(SHF, "lat_shf", 2.0, 16.0);
    ISA_PROBE_DEP(HFMA2, "lat_hfma2", 2.0, 16.0);
    ISA_PROBE_DEP(FFMA, "lat_ffma", 2.0, 16.0);
    ISA_PROBE_DEP(SHFL, "lat_shfl", 10.0, 60.0);
    ISA_PROBE_DEP(I2F, "lat_cvt_i2f", 2.0, 32.0);
    ISA_PROBE_DEP(F2I, "lat_cvt_f2i", 2.0, 32.0);
    ISA_PROBE_DEP(F32_TO_F16, "lat_cvt_f32_to_f16", 2.0, 32.0);
    ISA_PROBE_DEP(F16_TO_F32, "lat_cvt_f16_to_f32", 2.0, 32.0);
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
    // dram_lanes is dram with every lane of every warp on a segment of its own (32 x W lines in
    // flight an SM, the warp's link as long as its slowest line): the segments are too short for
    // four chases, so every warp count starts from the segments' first positions after the L2 is
    // pushed out again (evict; the nodes are only read, so nothing is filled twice), and a link
    // found in the L2 would show as a latency under dram's at the same warp count (checked
    // below). dram_lanes_half is the same chase launched as half as many CTAs (one each on half
    // the SMs, the segments the first half of dram_lanes'): 32 x W lines in flight on each of
    // those SMs as before, half as many on the chip.
    struct Level {
        const char* name;
        unsigned bits;
        unsigned stride; // 8-byte words from a node to the next
        bool cg;
        Own own;
        int warm, links; // warm-up links, then the timed ones
        unsigned lanes;  // chasing lanes of a warp
        bool fresh;      // every warp count from the range's start, the L2 pushed out before it
        int sm_div;      // the chase runs one CTA on each of SMs / sm_div SMs
    } levels[] = {
        {"l1", 8, 1, false, Own::Shared, kLatencyLinks, kLatencyLinks, 1, false, 1},
        {"l2", 12, 1, true, Own::Shared, kLatencyLinks, kLatencyLinks, 1, false, 1},
        {"dram_tlbwarm", 12, 16, true, Own::Buffer, kDramLinks, kDramLinks, 1, false, 1},
        {"dram", 24, 16, true, Own::Segment, kDramLinks, kDramLinks, 1, false, 1},
        {"dram_lanes", 24, 16, true, Own::Segment, kLanesWarm, kLanesLinks, 32, true, 1},
        {"dram_lanes_half", 24, 16, true, Own::Segment, kLanesWarm, kLanesLinks, 32, true, 2}};

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
    // The FFMA kernel of the held chases (spinner) runs on a second stream and names the SM of
    // each of its CTAs in mapped host words.
    volatile int* sm_of_cta = nullptr;
    int* sm_of_cta_dev      = nullptr;
    CHECK(cudaHostAlloc(reinterpret_cast<void**>(const_cast<int**>(&sm_of_cta)),
                        pr.sms * sizeof(int), cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&sm_of_cta_dev),
                                   const_cast<int*>(sm_of_cta), 0));
    cudaStream_t side = nullptr;
    CHECK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));

    // The spans of the chasing warps: entry CTA x 8 + warp, the first `warps` of every CTA.
    struct Chase {
        double clk, ns, mhz; // per link in SM clocks and wall ns; the SM clock the chase ran at
    };

    // Only the CTAs alone on their SM (in the held chases: beside one FFMA warp) count: on a
    // shared SM a warp chases beside twice the stated warps, which is another level's latency.
    const auto chase_clocks = [&](int ctas, int warps, int links, const std::string& what,
                                  const std::map<int, int>* holders) {
        std::vector<double> clk, ns;
        const std::vector<Span>& spans = pr.collect(ctas * 8);
        const std::vector<char> keep   = alone(spans, 8, what, holders);
        for (size_t entry = 0; entry < spans.size(); ++entry) {
            if (!keep[entry / 8] || static_cast<int>(entry % 8) >= warps) continue;
            const Span& s = spans[entry];
            clk.push_back(static_cast<double>(s.end - s.begin));
            ns.push_back(static_cast<double>(s.ns_end - s.ns_begin));
        }
        const double c = median(clk), n = median(ns);
        return Chase{c / links, n / links, n > 0 ? c / n * 1e3 : 0.0};
    };
    const auto push_out_of_l2 = [&] {
        evict<<<static_cast<unsigned>((evict_sectors + 255) / 256), 256>>>(scratch, evict_sectors);
        LAUNCHED();
    };
    // Every level twice: on a chip that runs nothing else (the facts named by the level alone),
    // then beside the FFMA kernel (`<level>_held_...`), filled and pushed out of the L2 alike.
    double lone_seconds = 0.0;
    for (const bool held : {false, true}) {
        const std::string tag = held ? "_held" : "";
        const auto phase_t0   = std::chrono::steady_clock::now();
        std::map<int, int> holders; // FFMA CTAs on every SM
        if (held) {
            for (int i = 0; i < pr.sms; ++i) sm_of_cta[i] = -1;
            // The FFMA kernel's time: three times what the same chases took alone (they run at
            // a lower clock now) and two seconds. It is checked below to have outlasted them.
            const auto deadline_ns = static_cast<long long>((3.0 * lone_seconds + 2.0) * 1e9);
            spinner<<<pr.sms, 32, 0, side>>>(sm_of_cta_dev, deadline_ns, pr.fsink);
            LAUNCHED();
            // Wait (5 s at most) until every CTA of it has named its SM.
            const auto started = [&] {
                for (int i = 0; i < pr.sms; ++i)
                    if (sm_of_cta[i] < 0) return false;
                return true;
            };
            while (!started() &&
                   std::chrono::steady_clock::now() - phase_t0 < std::chrono::seconds(5))
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            check(started(), "the FFMA kernel did not start on every CTA within 5 s");
            for (int i = 0; i < pr.sms; ++i) {
                const int sm = sm_of_cta[i];
                if (sm >= 0) ++holders[sm];
            }
        }
        std::vector<double> mhz; // the SM clock seen by each L2 / DRAM chase of the phase
        for (const Level& lv : levels) {
            const unsigned m     = 1u << lv.bits;
            const unsigned count = cycles_of(lv);
            const size_t total   = static_cast<size_t>(m) * count;
            // A chain's range of the cycle holds its chases (four, or one of a level that starts
            // every warp count afresh), warm-up and timed links each.
            const unsigned range = lv.own == Own::Buffer    ? m
                                   : lv.own == Own::Segment ? m / (pr.sms * 8u * lv.lanes)
                                                            : 0u;
            if (lv.own != Own::Shared)
                check(range >= (lv.fresh ? 1u : 4u) * static_cast<unsigned>(lv.warm + lv.links),
                      std::string(lv.name) +
                          ": a chain's range of the cycle is shorter than its chases");
            check(lv.lanes == 1 || lv.own == Own::Segment,
                  std::string(lv.name) + ": 32 lanes chase segments only");
            fill_cycle<<<static_cast<unsigned>((total + 255) / 256), 256>>>(buf, lv.bits, lv.stride,
                                                                            count);
            LAUNCHED();
            if (lv.own != Own::Shared && !lv.fresh) push_out_of_l2();
            CHECK(cudaStreamSynchronize(nullptr));
            unsigned run   = 0;
            const int ctas = pr.sms / lv.sm_div;
            for (int warps : {1, 2, 4, 8}) {
                if (lv.fresh) push_out_of_l2();
                if (lv.cg) {
                    chase<true><<<ctas, 32 * warps>>>(buf, lv.bits, lv.stride, lv.own, lv.warm,
                                                      lv.links, lv.lanes, run, pr.spans, pr.usink);
                } else {
                    chase<false><<<ctas, 32 * warps>>>(buf, lv.bits, lv.stride, lv.own, lv.warm,
                                                       lv.links, lv.lanes, run, pr.spans, pr.usink);
                }
                LAUNCHED();
                if (!lv.fresh) ++run;
                const std::string name = lv.name + tag + "_latency_w" + std::to_string(warps);
                const Chase c =
                    chase_clocks(ctas, warps, lv.links, name, held ? &holders : nullptr);
                fact(name.c_str(), c.clk, "clk");
                lat[name] = c.clk;
                if (lv.cg) {
                    // Beyond the L1 the figures are in wall ns as well, with the SM clock the
                    // chase ran at (clock64 over globaltimer), so a clock that moved between
                    // chases shows instead of passing for a latency change.
                    const std::string ns = lv.name + tag + "_latency_ns_w" + std::to_string(warps);
                    fact(ns.c_str(), c.ns, "ns");
                    lat[ns] = c.ns;
                    mhz.push_back(c.mhz);
                }
            }
        }
        if (held) {
            // Still running after the last held chase was read back: every held chase ran beside
            // it. (cudaStreamQuery's not-ready answer is cleared so no later launch reads it as
            // its own error.) Then it is waited for.
            const cudaError_t running = cudaStreamQuery(side);
            (void)cudaGetLastError();
            check(running == cudaErrorNotReady,
                  "the FFMA kernel had left before the held chases ended");
            CHECK(cudaStreamSynchronize(side));
        } else {
            lone_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - phase_t0).count();
        }
        fact(("sm_clock_mhz_chase" + tag).c_str(), median(mhz), "mhz");
        // Each chase's ns come from its own window (clock64 against globaltimer within it), so a
        // clock that moved between chases leaves the ns right and only the clk figures of those
        // levels at the clock they ran at: reported, not refused.
        const double lo = *std::min_element(mhz.begin(), mhz.end());
        const double hi = *std::max_element(mhz.begin(), mhz.end());
        if (hi > 1.05 * lo)
            std::printf("note the SM clock ranged %.0f..%.0f MHz over the %sL2 / DRAM chases; "
                        "their clk figures are at those clocks, the ns stand\n",
                        lo, hi, held ? "held " : "");
    }
    CHECK(cudaStreamDestroy(side));
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
    // and DRAM). A warp of 32 chasing lanes waits for its slowest line, so its link is not shorter
    // than the one-lane warp's at the same warp count. All of it holds beside the FFMA kernel as
    // on the chip alone.
    for (const std::string tag : {"", "_held"}) {
        const auto at = [&](const std::string& level, const std::string& unit_w) {
            return lat[level + tag + unit_w];
        };
        check(at("l2", "_latency_w1") > 2.0 * at("l1", "_latency_w1"),
              "L2 latency not above 2x L1" + tag);
        for (int w : {1, 2, 4, 8}) {
            const std::string ws = "_latency_w" + std::to_string(w);
            const std::string ns = "_latency_ns_w" + std::to_string(w);
            const std::string wt = " at w" + std::to_string(w) + tag;
            for (const char* lv : {"dram_tlbwarm", "dram", "dram_lanes", "dram_lanes_half"})
                check(at(lv, ns) > 1.2 * at("l2", ns),
                      std::string(lv) + " latency not above 1.2x L2" + wt);
            check(at("dram", ns) >= 0.97 * at("dram_tlbwarm", ns),
                  "dram latency under the TLB-warm level's" + wt);
            for (const char* lv : {"dram_lanes", "dram_lanes_half"})
                check(at(lv, ns) >= 0.95 * at("dram", ns),
                      std::string(lv) + " latency under the one-lane chase's" + wt);
            check(at("l1", ws) >= 0.95 * at("l1", "_latency_w1"),
                  "l1 latency falls under the lone warp's" + wt);
            for (const char* lv : {"l2", "dram_tlbwarm", "dram", "dram_lanes", "dram_lanes_half"})
                check(at(lv, ns) >= 0.95 * at(lv, "_latency_ns_w1"),
                      std::string(lv) + " latency falls under the lone warp's" + wt);
        }
    }
    if (g_failures.empty()) {
        std::printf("selfcheck ok\n");
    } else {
        for (const std::string& f : g_failures) std::printf("selfcheck FAIL %s\n", f.c_str());
    }
    std::printf("end\n");
    return g_failures.empty() ? 0 : 3;
}
