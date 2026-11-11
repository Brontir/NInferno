// warp_inflight: the DRAM rate one SM reaches with W resident warps that stream 16-byte loads, the
// fact behind the per-SM bound R_sm(W) = min(P, W x b_w / L0) of the decode price (#690, #735):
// b_w the bytes a resident warp keeps in flight, P the saturated per-SM rate, L0 the unloaded
// DRAM latency (isa_probe's dram_tlbwarm_latency_ns_w1). This probe measures rate(W); b_w is read
// as rate(W) x L0 / W where the rate is still linear in W, P as the rate where it stops growing.
//
// Build (CUDA 13.x; sm_70 needs CUDA 12.8's nvcc):
//   nvcc -O3 -std=c++17 -arch=sm_<arch> -o warp_inflight tools/bench/warp_inflight/warp_inflight.cu
// Run (5 to 15 s; 1 GiB of device memory; the GPU otherwise idle; no kernel runs longer than a few
// ms, so a card that drives a desktop takes it):
//   ./warp_inflight        # `row ...` lines, parsed by warp_inflight.py
// Exit 0 with `selfcheck ok`; 3 when a measurement fails the self-check (its reasons printed), 2 on
// a CUDA error.
//
// Method. Every lane of a measured warp loads 16 bytes with ld.global.cs.v4.u32, the load of the
// decode kernels (generic_gemm.cuh ld_stream_v4): one warp-wide load is 512 contiguous bytes, and
// each warp streams a range of its own that no other warp and no earlier measurement touched, in a
// 1 GiB buffer (at least 4 x L2 beside the largest measurement), so every load misses the L2.
// What a warp keeps in flight is set by data dependence, not left to the compiler: the addresses
// of a fetch add (acc & zero), with acc the XOR of the data the warp consumed and zero a kernel
// argument that is 0, so a fetch cannot issue before the loads it depends on have returned.
//   chainF: a fetch is F loads a lane (F x 512 bytes a warp), consumed before the next fetch
//           issues: exactly F loads a lane in flight.
//   ringF:  the decode kernels' double-buffered ring (rmma_kernel's and r8_kernel's `fetch` /
//           `compute` on two slots): the next slot's F loads issue before this slot's are
//           consumed, so up to 2F loads a lane are in flight (2F x 512 bytes a warp).
// Per fetch the decode kernels issue, per lane, kSets x MT x 2 runs (rmma_kernel: one run is a
// 16-byte code load on most formats) and kSets x kRows x kStride 16-byte blocks (r8_kernel): F = 2
// for a one-row-set Store call of the m1 register-MMA schedules and of R8 with two rows a warp,
// F = 1 for R8 with one row, F = 4 under SwiGLU (two row sets). The ring therefore keeps 2, 4 or 8
// loads a lane in flight in its latency-bound limit, where the step's compute is short against L0.
//
// Two placements of the W warps:
//   free:   k CTAs of W warps (W <= 32), no shared memory and the L1-largest carve-out; the CTA
//           scheduler spreads them, and a run whose k CTAs did not land on k distinct SMs (%smid)
//           is refused. Which k SMs is the scheduler's choice: at k > 1 they may share a TPC or a
//           GPC, unlike the placed SMs, so the two modes are compared at k = 1 only.
//   placed: c = 1 or 2 CTAs of W / c warps on every SM (W up to the SM's resident warps), sized by
//           dynamic shared memory so that exactly c fit an SM (the occupancy API says so, and the
//           CTAs' %smid show it), with the shared-largest carve-out; only the CTAs on k chosen SMs
//           (spread over the SM ids a discovery launch saw) stream, the others return at once.
// The two agree at W <= 32 unless the carve-out (the L1's size) bounds what a warp keeps in
// flight; free is the decode kernels' case without shared memory, placed reaches W > 32.
//
// Timing. Each warp reads %globaltimer and clock64 around its loop; an SM's rate is its warps'
// bytes over the span from its first warp's start to its last warp's end (so a straggler lowers
// it, never raises it), in bytes per ns (GB/s); a measurement is the median over its k SMs, and a
// row the median of three repetitions. ns_fetch is the median over warps of a warp's span over its
// fetches: a chain's loaded latency of one fetch, half of it for a ring (two fetches overlap);
// sm_mhz is clock64 over %globaltimer. A span under 100 us is refused (the timer steps by ~1 us).
// A placed (variant, W) whose registers fit fewer than c CTAs an SM has no row and says so in a
// note; every other point of the grid has one.
//
// Clocks (probe 2). The measured part is bracketed by a `span_ns <begin> <end>` line in host wall
// time (system_clock, ns since the epoch): warp_inflight.py keeps the nvidia-smi clock samples
// inside it and refuses a run whose memory clock or P-state moved there.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
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
#define LAUNCHED() CHECK(cudaGetLastError())

namespace {

constexpr int kLoadBytes = 16;              // one lane's load
constexpr int kWarpLoad  = 32 * kLoadBytes; // one warp-wide load: 512 contiguous bytes
constexpr size_t kBuffer = size_t{1} << 30;
// The bytes of one measurement (all its warps) and of one warp: the largest measurement stays
// far from the buffer's size, so no measurement reads a line an earlier one may have left in L2.
constexpr size_t kRunBudget = size_t{256} << 20;
constexpr size_t kWarpMax   = size_t{4} << 20; // a ring4 warp alone still spans ~300 us
constexpr int kReps         = 3;
constexpr int kMaxSmId      = 1024; // %smid is below %nsmid, which is below this on every chip

struct Rec {
    long long ns_begin, ns_end, clk_begin, clk_end;
    int sm; // -1: the CTA's SM was not chosen, the warp did nothing
    int cta;
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

// The decode kernels' weight load (generic_gemm.cuh ld_stream_v4): streaming, 16 bytes a lane.
__device__ __forceinline__ uint4 ld_stream_v4(const unsigned char* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

template <int F>
__device__ __forceinline__ void fetch(uint4 (&v)[F], const unsigned char* p) {
#pragma unroll
    for (int f = 0; f < F; ++f) v[f] = ld_stream_v4(p + f * kWarpLoad);
}

template <int F>
__device__ __forceinline__ unsigned consume(const uint4 (&v)[F], unsigned acc) {
#pragma unroll
    for (int f = 0; f < F; ++f) acc ^= v[f].x ^ v[f].y ^ v[f].z ^ v[f].w;
    return acc;
}

// One measured warp streams `fetches` fetches of F warp-wide loads from its own range (ranked by
// an atomic over the working warps, so the ranges are dense whichever CTAs work). `active`, when
// given, holds a flag per SM id: a CTA on an SM without it returns at once.
template <int F, bool Ring>
__global__ void stream(const unsigned char* buf, unsigned long long base,
                       unsigned long long per_warp, int fetches, const int* active, unsigned zero,
                       unsigned* rank, Rec* recs, unsigned* sink) {
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int sm = smid();
    Rec* rec     = recs + blockIdx.x * (blockDim.x / 32) + warp;
    if (active != nullptr && active[sm] == 0) {
        if (lane == 0) *rec = Rec{0, 0, 0, 0, -1, static_cast<int>(blockIdx.x)};
        return;
    }
    unsigned r = 0;
    if (lane == 0) r = atomicAdd(rank, 1u);
    r                                  = __shfl_sync(0xffffffffu, r, 0);
    const unsigned char* p             = buf + base + r * per_warp + lane * kLoadBytes;
    constexpr unsigned long long kStep = static_cast<unsigned long long>(F) * kWarpLoad;
    unsigned long long off             = 0;
    unsigned acc                       = 0;
    __syncwarp();
    const long long n0 = globaltimer_ns();
    const long long c0 = clock64();
    if constexpr (!Ring) {
#pragma unroll 1
        for (int i = 0; i < fetches; ++i) {
            uint4 v[F];
            fetch<F>(v, p + off);
            acc = consume<F>(v, acc);
            off += kStep + (acc & zero);
        }
    } else {
        // rmma_kernel's loop: fetch(Slot1, next); compute(Slot0); fetch(Slot0, next);
        // compute(Slot1). `fetches` is even.
        uint4 a[F], b[F];
        fetch<F>(a, p + off);
        off += kStep;
#pragma unroll 1
        for (int i = 0; i < fetches; i += 2) {
            fetch<F>(b, p + off + (acc & zero));
            off += kStep;
            acc = consume<F>(a, acc);
            if (i + 2 < fetches) {
                fetch<F>(a, p + off + (acc & zero));
                off += kStep;
            }
            acc = consume<F>(b, acc);
        }
    }
    const long long c1 = clock64();
    const long long n1 = globaltimer_ns();
    if (lane == 0) *rec = Rec{n0, n1, c0, c1, sm, static_cast<int>(blockIdx.x)};
    if (acc == (zero ^ 0x9E3779B9u) && lane == 0 && r == 0xffffffffu) sink[0] = acc; // never true
}

using Kernel = void (*)(const unsigned char*, unsigned long long, unsigned long long, int,
                        const int*, unsigned, unsigned*, Rec*, unsigned*);

struct Variant {
    const char* name;
    int f;
    bool ring;
    Kernel kernel;

    int inflight() const { return ring ? 2 * f : f; } // loads a lane keeps in flight, at most
};

const Variant kVariants[] = {
    {"chain1", 1, false, stream<1, false>}, {"chain2", 2, false, stream<2, false>},
    {"chain4", 4, false, stream<4, false>}, {"ring1", 1, true, stream<1, true>},
    {"ring2", 2, true, stream<2, true>},    {"ring4", 4, true, stream<4, true>},
};

std::vector<std::string> g_failures;

void check(bool ok, const std::string& what) {
    if (!ok) g_failures.push_back(what);
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct Result {
    bool ok     = false;
    double gbps = 0, gbps_lo = 0, gbps_hi = 0, ns_fetch = 0, sm_mhz = 0, bytes_per_sm = 0;
};

struct Probe {
    int sms = 0, max_warps = 0;
    unsigned char* buf = nullptr;
    size_t cursor      = 0;
    int* active        = nullptr; // device, kMaxSmId flags
    unsigned* rank     = nullptr;
    unsigned* sink     = nullptr;
    Rec* recs          = nullptr;
    size_t rec_cap     = 0;
    std::vector<int> sm_ids;       // every SM id a discovery launch saw, sorted
    int shared_for[3] = {0, 0, 0}; // dynamic shared bytes that fit exactly c CTAs an SM

    // Where the next measurement's ranges start: after the last one's, from 0 again when it would
    // pass the buffer's end (kBuffer is at least 2 x kRunBudget beyond L2, see main).
    size_t take(size_t bytes) {
        if (cursor + bytes > kBuffer) cursor = 0;
        const size_t at = cursor;
        cursor += bytes;
        return at;
    }

    // One launch: `ctas` CTAs of `warps_per_cta` warps, `k` SMs expected to stream with `c` CTAs
    // each. Returns the per-SM rate and the warp figures, or ok = false with the reason recorded.
    Result launch(const Variant& v, int k, int c, int ctas, int warps_per_cta, int shared,
                  const int* active, const std::string& what) {
        const int working_warps = k * c * warps_per_cta;
        size_t per_warp         = std::min(kWarpMax, kRunBudget / working_warps);
        const size_t quantum    = static_cast<size_t>(2) * v.f * kWarpLoad; // even fetch count
        per_warp                = per_warp / quantum * quantum;
        const int fetches = static_cast<int>(per_warp / (static_cast<size_t>(v.f) * kWarpLoad));
        const size_t base = take(per_warp * working_warps);
        CHECK(cudaMemset(rank, 0, sizeof(unsigned)));
        v.kernel<<<ctas, 32 * warps_per_cta, shared>>>(buf, base, per_warp, fetches, active, 0u,
                                                       rank, recs, sink);
        LAUNCHED();
        CHECK(cudaDeviceSynchronize());
        const size_t n = static_cast<size_t>(ctas) * warps_per_cta;
        std::vector<Rec> h(n);
        CHECK(cudaMemcpy(h.data(), recs, n * sizeof(Rec), cudaMemcpyDeviceToHost));
        Result out;
        // Per SM: its CTAs, its warps' spans.
        std::map<int, std::set<int>> ctas_on;
        std::map<int, std::vector<const Rec*>> warps_on;
        for (const Rec& r : h) {
            if (r.sm < 0) continue;
            ctas_on[r.sm].insert(r.cta);
            warps_on[r.sm].push_back(&r);
        }
        bool placed_ok = static_cast<int>(warps_on.size()) == k;
        for (const auto& [sm, set] : ctas_on)
            placed_ok = placed_ok && static_cast<int>(set.size()) == c;
        if (!placed_ok) {
            check(false, what + ": the working CTAs were not " + std::to_string(c) +
                             " on each of " + std::to_string(k) + " SMs (" +
                             std::to_string(warps_on.size()) + " SMs worked)");
            return out;
        }
        std::vector<double> rates, fetch_ns, mhz;
        for (const auto& [sm, list] : warps_on) {
            long long lo = list[0]->ns_begin, hi = list[0]->ns_end;
            long long last_begin = list[0]->ns_begin, first_end = list[0]->ns_end;
            for (const Rec* r : list) {
                lo                = std::min(lo, r->ns_begin);
                hi                = std::max(hi, r->ns_end);
                last_begin        = std::max(last_begin, r->ns_begin);
                first_end         = std::min(first_end, r->ns_end);
                const double span = static_cast<double>(r->ns_end - r->ns_begin);
                fetch_ns.push_back(span / fetches);
                if (span > 0)
                    mhz.push_back(1e3 * static_cast<double>(r->clk_end - r->clk_begin) / span);
            }
            // The SM's warps ran at once: every one began before any ended.
            if (last_begin >= first_end) {
                check(false, what + ": the warps on SM " + std::to_string(sm) + " did not overlap");
                return out;
            }
            const double span = static_cast<double>(hi - lo);
            if (span < 1e5) { // 100 us: far above the ~1 us %globaltimer step
                check(false,
                      what + ": a span of " + std::to_string(span) + " ns is too short to time");
                return out;
            }
            rates.push_back(static_cast<double>(per_warp) * list.size() / span);
        }
        out.ok           = true;
        out.gbps         = median(rates);
        out.gbps_lo      = *std::min_element(rates.begin(), rates.end());
        out.gbps_hi      = *std::max_element(rates.begin(), rates.end());
        out.ns_fetch     = median(fetch_ns);
        out.sm_mhz       = median(mhz);
        out.bytes_per_sm = static_cast<double>(per_warp) * c * warps_per_cta;
        return out;
    }

    // A row: the median of kReps launches by rate (the other figures of that launch).
    void row(const Variant& v, const char* mode, int k, int w, int c, int ctas, int shared,
             const int* active) {
        const std::string what =
            std::string(mode) + " k" + std::to_string(k) + " " + v.name + " w" + std::to_string(w);
        std::vector<Result> reps;
        for (int i = 0; i < kReps; ++i) {
            const Result r = launch(v, k, c, ctas, w / c, shared, active, what);
            if (!r.ok) return;
            reps.push_back(r);
        }
        std::sort(reps.begin(), reps.end(),
                  [](const Result& a, const Result& b) { return a.gbps < b.gbps; });
        const Result& r = reps[reps.size() / 2];
        std::printf("row %s %d %s %d %d %.4f %.4f %.4f %.1f %.0f %.3f\n", mode, k, v.name,
                    v.inflight(), w, r.gbps, r.gbps_lo, r.gbps_hi, r.ns_fetch, r.sm_mhz,
                    r.bytes_per_sm / (1 << 20));
    }
};

__global__ void discover(int* sm_of_cta) {
    if (threadIdx.x == 0) sm_of_cta[blockIdx.x] = smid();
}

} // namespace

int main() {
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    int driver = 0, runtime = 0, mem_khz = 0, threads_sm = 0, smem_sm = 0, smem_optin = 0,
        smem_reserved = 0;
    CHECK(cudaDriverGetVersion(&driver));
    CHECK(cudaRuntimeGetVersion(&runtime));
    CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0));
    CHECK(cudaDeviceGetAttribute(&threads_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0));
    CHECK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CHECK(cudaDeviceGetAttribute(&smem_reserved, cudaDevAttrReservedSharedMemoryPerBlock, 0));
    const int cc = p.major * 10 + p.minor;
    char pci[32] = {};
    CHECK(cudaDeviceGetPCIBusId(pci, sizeof pci, 0));
    std::printf("probe warp_inflight 2\n");
    std::printf("card %s\n", p.name);
    std::printf("cc %d\n", cc);
    std::printf("sms %d\n", p.multiProcessorCount);
    std::printf("l2_bytes %zu\n", static_cast<size_t>(p.l2CacheSize));
    std::printf("mem_clock_khz %d\n", mem_khz);
    std::printf("max_warps_per_sm %d\n", threads_sm / 32);
    std::printf("smem_per_sm %d\n", smem_sm);
    std::printf("driver %d\n", driver);
    std::printf("runtime %d\n", runtime);
    std::printf("nvcc %d.%d.%d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__,
                __CUDACC_VER_BUILD__);
    std::printf("note device 0 is the GPU at PCI %s\n", pci);
    std::printf("columns mode k variant inflight w gbps gbps_lo gbps_hi ns_fetch sm_mhz mib\n");

    Probe pr;
    pr.sms       = p.multiProcessorCount;
    pr.max_warps = threads_sm / 32;
    check(kBuffer >= 4 * static_cast<size_t>(p.l2CacheSize) + 2 * kRunBudget,
          "the buffer is not 4 x L2 beside two measurements");
    CHECK(cudaMalloc(&pr.buf, kBuffer));
    CHECK(cudaMemset(pr.buf, 0x5a, kBuffer));
    CHECK(cudaMalloc(&pr.active, kMaxSmId * sizeof(int)));
    CHECK(cudaMalloc(&pr.rank, sizeof(unsigned)));
    CHECK(cudaMalloc(&pr.sink, sizeof(unsigned)));
    pr.rec_cap = static_cast<size_t>(2) * pr.sms * 32;
    CHECK(cudaMalloc(&pr.recs, pr.rec_cap * sizeof(Rec)));

    // Dynamic shared memory that fits exactly c CTAs an SM (c = 1, 2), with the shared-largest
    // carve-out; the occupancy API confirms it for every variant.
    for (int c = 1; c <= 2; ++c) {
        int s            = smem_sm / c - smem_reserved;
        s                = std::min(s, smem_optin) / 256 * 256;
        pr.shared_for[c] = s;
        check((c + 1) * (s + smem_reserved) > smem_sm,
              "the placed shared size lets more than " + std::to_string(c) + " CTAs fit an SM");
    }
    for (const Variant& v : kVariants) {
        CHECK(cudaFuncSetAttribute(v.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   pr.shared_for[1]));
    }

    // The SM ids: one CTA an SM (the c = 1 shared size), each records its %smid.
    {
        int* d = nullptr;
        CHECK(cudaMalloc(&d, pr.sms * sizeof(int)));
        CHECK(cudaFuncSetAttribute(discover, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   pr.shared_for[1]));
        discover<<<pr.sms, 32, pr.shared_for[1]>>>(d);
        LAUNCHED();
        std::vector<int> h(pr.sms);
        CHECK(cudaMemcpy(h.data(), d, pr.sms * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d));
        std::set<int> ids(h.begin(), h.end());
        pr.sm_ids.assign(ids.begin(), ids.end());
        check(static_cast<int>(pr.sm_ids.size()) == pr.sms,
              "the discovery launch saw " + std::to_string(pr.sm_ids.size()) + " SM ids, not " +
                  std::to_string(pr.sms));
        check(pr.sm_ids.back() < kMaxSmId, "an SM id past the probe's table");
    }

    // Warm-up: every SM streams for about 0.3 s, so the clocks and the memory's P-state have risen
    // before the first measurement (the measured launches then follow back to back).
    {
        const Variant& v = kVariants[2];
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0));
        CHECK(cudaEventCreate(&e1));
        CHECK(cudaEventRecord(e0));
        float ms = 0.0F;
        for (int i = 0; i < 2000 && ms < 300.0F; ++i) {
            CHECK(cudaMemset(pr.rank, 0, sizeof(unsigned)));
            const size_t per_warp = kRunBudget / (static_cast<size_t>(pr.sms) * 32);
            const size_t quantum  = static_cast<size_t>(2) * v.f * kWarpLoad;
            const size_t pw       = per_warp / quantum * quantum;
            v.kernel<<<pr.sms, 1024>>>(pr.buf, pr.take(pw * pr.sms * 32), pw,
                                       static_cast<int>(pw / (v.f * kWarpLoad)), nullptr, 0u,
                                       pr.rank, pr.recs, pr.sink);
            LAUNCHED();
            CHECK(cudaEventRecord(e1));
            CHECK(cudaEventSynchronize(e1));
            CHECK(cudaEventElapsedTime(&ms, e0, e1));
        }
        std::printf("note warm-up: %.0f ms of every SM streaming\n", ms);
    }

    const int w_list[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    const int k_list[] = {1, 2, 4, 8};
    std::printf("note free k > 1: the CTA scheduler chose the SMs (they may share a TPC or a "
                "GPC); compare free and placed at k = 1\n");
    const auto wall_ns = [] {
        return static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::system_clock::now().time_since_epoch())
                                          .count());
    };
    const long long span_begin = wall_ns();

    // free: k CTAs of W warps, no shared memory, the L1-largest carve-out.
    for (const Variant& v : kVariants) {
        CHECK(cudaFuncSetAttribute(v.kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                   cudaSharedmemCarveoutMaxL1));
    }
    for (const int k : k_list) {
        if (k > pr.sms) continue;
        for (const Variant& v : kVariants)
            for (const int w : w_list) {
                if (w > 32 || w > pr.max_warps) continue;
                pr.row(v, "free", k, w, 1, k, 0, nullptr);
            }
    }

    // placed: c CTAs of W / c warps on every SM, the CTAs on k chosen SMs streaming.
    for (const Variant& v : kVariants) {
        CHECK(cudaFuncSetAttribute(v.kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                   cudaSharedmemCarveoutMaxShared));
    }
    std::vector<int> flags(kMaxSmId);
    for (const int k : k_list) {
        if (k > pr.sms) continue;
        // k SM ids spread over the ids the discovery launch saw.
        std::fill(flags.begin(), flags.end(), 0);
        for (int i = 0; i < k; ++i)
            flags[pr.sm_ids[static_cast<size_t>(i) * pr.sm_ids.size() / k]] = 1;
        CHECK(cudaMemcpy(pr.active, flags.data(), kMaxSmId * sizeof(int), cudaMemcpyHostToDevice));
        for (const Variant& v : kVariants)
            for (const int w : w_list) {
                if (w > pr.max_warps) continue;
                const int c = w > 32 ? 2 : 1;
                if (w % c != 0) continue;
                int fit = 0;
                CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, v.kernel, 32 * (w / c),
                                                                    pr.shared_for[c]));
                // More than c CTAs is the shared size wrong; fewer is the variant's registers (the
                // ring4 loop's 32 data registers a lane leave no room for 48 or 64 warps): no row.
                check(fit <= c, std::string("placed ") + v.name + " w" + std::to_string(w) +
                                    ": the occupancy API fits " + std::to_string(fit) +
                                    " CTAs an SM, more than " + std::to_string(c));
                if (fit != c) {
                    if (k == 1)
                        std::printf("note placed %s w%d: %d of %d CTAs fit an SM (registers), no "
                                    "row\n",
                                    v.name, w, fit, c);
                    continue;
                }
                pr.row(v, "placed", k, w, c, c * pr.sms, pr.shared_for[c], pr.active);
            }
    }
    std::printf("span_ns %lld %lld\n", span_begin, wall_ns());

    if (g_failures.empty()) {
        std::printf("selfcheck ok\n");
    } else {
        for (const std::string& f : g_failures) std::printf("selfcheck FAIL %s\n", f.c_str());
    }
    std::printf("end\n");
    return g_failures.empty() ? 0 : 3;
}
