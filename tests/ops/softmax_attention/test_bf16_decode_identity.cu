// The BF16 KV width-1 decode (the GroupedDecode instance, #856) bit for bit against the instance it
// replaced. Both compile into this binary: one D warp at width 1 is the former kernel (Q in
// registers, both MMA rows kept), and the shipped schedule splits D over two warps and keeps Q in
// shared memory (kBf16KvLowRowsOnly). The same decode runs through both, for every listed
// geometry, B = 1 and B = 8, append and cached, with and without the valid-column mask, at
// contexts on both sides of the split and page boundaries. The output, every split's partials
// (O, the log2 maximum, the sum; unwritten entries included) and, for append, the cache must
// match byte for byte. --self compares each schedule with a second run of itself instead (the
// kernels' determinism).
//
// On an sm_70 build and card, Volta's key-major kernel (#877) runs too. It is not the MMA
// emulation, so against the shipped schedule it is held to sums, not bits: the appended cache and
// the set of written partials byte for byte, the output within one BF16 step and the partials
// within the FP16 rounding of P (near_outputs). --self holds it to its own bits.

#include "ops/softmax_attention/common/causal_geometry.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/template_launch.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::ops;
using namespace ninferno::ops::detail;

using Shipped = Bf16KvInstanceTraits<Bf16KvInstance::GroupedDecode>::Schedule;
// The GroupedDecode schedule before #856: one D warp.
using Former = Bf16KvGroupedMmaSchedule<Shipped::kQueryRows, Shipped::kKeyRows, Shipped::kWarpsKV,
                                        Shipped::kMinBlocks, Shipped::kFixedWidth, 1>;
static_assert(Shipped::kWarpsD == 2 && Former::kWarpsD == 1);

// Volta's key-major kernel (#877) in a schedule's place.
struct VoltaKeyMajor {};

constexpr int D = kCausalHeadDim;

// Every copy, fill and launch goes through this one stream, then a synchronize: a non-blocking
// stream does not wait for cudaMemcpy/cudaMemset on the legacy default stream (a pageable
// cudaMemcpy may return before its DMA lands), so the kernels would race the cache restore and the
// sentinel fill.
cudaStream_t g_stream = nullptr;
bool g_self           = false;
bool g_volta          = false; // an sm_70 build on an sm_70 card

void check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(status));
    std::exit(1);
}

template <class T>
struct DeviceArray {
    T* data          = nullptr;
    std::size_t size = 0;

    explicit DeviceArray(std::size_t n) : size(n) {
        check(cudaMalloc(&data, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    }

    ~DeviceArray() { cudaFree(data); }

    DeviceArray(const DeviceArray&)            = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    void upload(const std::vector<T>& host) {
        check(
            cudaMemcpyAsync(data, host.data(), size * sizeof(T), cudaMemcpyHostToDevice, g_stream),
            "upload");
        check(cudaStreamSynchronize(g_stream), "upload");
    }

    void fill(unsigned char byte) {
        check(cudaMemsetAsync(data, byte, size * sizeof(T), g_stream), "memset");
        check(cudaStreamSynchronize(g_stream), "memset");
    }

    std::vector<T> download() const {
        std::vector<T> host(size);
        check(
            cudaMemcpyAsync(host.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost, g_stream),
            "download");
        check(cudaStreamSynchronize(g_stream), "download");
        return host;
    }
};

struct Random {
    std::uint64_t state;

    float next(float lo, float hi) {
        state            = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const float unit = static_cast<float>(state >> 40) / static_cast<float>(1ULL << 24);
        return lo + (hi - lo) * unit;
    }
};

std::uint16_t bf16_bits(float v) {
    const __nv_bfloat16 b = __float2bfloat16(v);
    std::uint16_t bits;
    std::memcpy(&bits, &b, sizeof(bits));
    return bits;
}

std::uint16_t f16_bits(float v) {
    const __half h = __float2half(v);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

std::vector<std::uint16_t> bf16_values(std::size_t n, std::uint64_t seed, float amplitude) {
    Random random{seed};
    std::vector<std::uint16_t> bits(n);
    for (auto& b : bits) b = bf16_bits(random.next(-amplitude, amplitude));
    return bits;
}

std::vector<std::uint16_t> f16_values(std::size_t n, std::uint64_t seed) {
    Random random{seed};
    std::vector<std::uint16_t> bits(n);
    for (auto& b : bits) b = f16_bits(random.next(-1.0f, 1.0f));
    return bits;
}

// One decode call: a context (keys seen, the new token's included) per batch row.
struct Case {
    std::vector<int> contexts;
};

// The device inputs of one geometry's calls: a page pool both runs read (and, for append, the
// cache restored before every run), the page tables, Q, the new K/V, the positions.
template <class G>
struct Inputs {
    int batch, pages_per_row, pool_pages;
    std::vector<std::uint16_t> k_host, v_host;
    DeviceArray<std::uint16_t> k, v, q, new_k, new_v;
    DeviceArray<std::int32_t> tables, rows, positions, valid;

    static int pages_for(int keys) { return (keys + kPagedKVPageSize - 1) / kPagedKVPageSize; }

    Inputs(const Case& c, std::uint64_t seed, bool empty_row)
        : batch(static_cast<int>(c.contexts.size())),
          pages_per_row(pages_for(*std::max_element(c.contexts.begin(), c.contexts.end()))),
          pool_pages(batch * pages_per_row),
          k(std::size_t(pool_pages) * kPagedKVPageSize * G::KVHeads * D), v(k.size),
          q(std::size_t(batch) * G::QHeads * D), new_k(std::size_t(batch) * G::KVHeads * D),
          new_v(new_k.size), tables(std::size_t(batch) * pages_per_row), rows(batch),
          positions(batch), valid(batch) {
        k_host = bf16_values(k.size, seed, 1.0f);
        v_host = f16_values(v.size, seed + 1);
        q.upload(bf16_values(q.size, seed + 2, 1.0f));
        new_k.upload(bf16_values(new_k.size, seed + 3, 1.0f));
        new_v.upload(bf16_values(new_v.size, seed + 4, 1.0f));
        // Every row's pages scattered over the pool (a reversed interleave), the rest of its table
        // on page 0.
        std::vector<std::int32_t> table(tables.size, 0), row(batch), position(batch), live(batch);
        for (int b = 0; b < batch; ++b) {
            for (int p = 0; p < pages_for(c.contexts[b]); ++p)
                table[std::size_t(b) * pages_per_row + p] = pool_pages - 1 - (p * batch + b);
            row[b]      = b;
            position[b] = c.contexts[b] - 1;
            live[b]     = empty_row && b == batch / 2 ? 0 : 1;
        }
        tables.upload(table);
        rows.upload(row);
        positions.upload(position);
        valid.upload(live);
    }
};

struct Outputs {
    std::vector<std::uint16_t> out, k, v;
    std::vector<float> acc, maximum, sum;
};

template <class G, class S, bool MultiBatch, bool Masked, bool Append>
Outputs run(Inputs<G>& in, const Bf16KvCausalPlan& plan) {
    const cudaStream_t stream = g_stream;
    in.k.upload(in.k_host);
    in.v.upload(in.v_host);
    const int capacity = plan.partition.capacity;
    DeviceArray<float> acc(std::size_t(in.batch) * capacity * G::QHeads * D),
        maximum(std::size_t(in.batch) * capacity * G::QHeads),
        sum(std::size_t(in.batch) * capacity * G::QHeads);
    DeviceArray<std::uint16_t> out(std::size_t(in.batch) * G::QHeads * D);
    acc.fill(0xa5);
    maximum.fill(0xa5);
    sum.fill(0xa5);
    out.fill(0xa5);
    CausalAttentionOperands p{reinterpret_cast<const __nv_bfloat16*>(in.q.data),
                              in.positions.data,
                              reinterpret_cast<__nv_bfloat16*>(out.data),
                              0.0625f,
                              1,
                              in.batch,
                              static_cast<int>(plan.envelope.max_visible_keys),
                              D,
                              G::QHeads,
                              G::KVHeads};
    const CausalPartialView partials{acc.data, maximum.data, sum.data};
    Bf16KvCacheView<Append> cache{
        reinterpret_cast<typename Bf16KvCacheView<Append>::Key*>(in.k.data),
        reinterpret_cast<typename Bf16KvCacheView<Append>::Value*>(in.v.data),
        in.tables.data,
        Masked ? in.valid.data : nullptr,
        in.rows.data,
        in.pages_per_row,
        D,
        G::KVHeads};
    const auto decode = [&](auto input) {
        if constexpr (std::is_same_v<S, VoltaKeyMajor>) {
            launch_bf16_kv_key_major_volta<G, MultiBatch, Masked, Append>(
                p, cache, input, plan.partition, partials, stream);
        } else {
            launch_bf16_kv_grouped_mma<G, S, MultiBatch, Masked, Append>(
                p, cache, input, plan.partition, partials, stream);
        }
    };
    if constexpr (Append) {
        decode(CausalAppendInput{reinterpret_cast<const __nv_bfloat16*>(in.new_k.data),
                                 reinterpret_cast<const __nv_bfloat16*>(in.new_v.data)});
    } else {
        decode(CausalCachedInput{});
    }
    launch_bf16_kv_merge<G, Bf16KvMergeSchedule<256>, MultiBatch, Masked, Append>(
        p, cache, plan.partition, partials, stream);
    check(cudaStreamSynchronize(stream), "decode");
    Outputs result{out.download(), {}, {}, acc.download(), maximum.download(), sum.download()};
    if constexpr (Append) {
        result.k = in.k.download();
        result.v = in.v.download();
    }
    return result;
}

template <class T>
int compare(const std::string& label, const char* what, const std::vector<T>& a,
            const std::vector<T>& b) {
    if (a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0)
        return 0;
    std::size_t i = 0;
    while (i < std::min(a.size(), b.size()) && std::memcmp(&a[i], &b[i], sizeof(T)) == 0) ++i;
    std::fprintf(stderr, "FAIL %s: %s differs at %zu of %zu\n", label.c_str(), what, i, a.size());
    return 1;
}

float bf16_value(std::uint16_t bits) {
    const std::uint32_t wide = std::uint32_t(bits) << 16;
    float v;
    std::memcpy(&v, &wide, sizeof(v));
    return v;
}

// The fill pattern of an entry no kernel wrote.
template <class T>
bool unwritten(const T& x) {
    T fill;
    std::memset(&fill, 0xa5, sizeof(fill));
    return std::memcmp(&x, &fill, sizeof(T)) == 0;
}

// a and b within tolerance(i, x, y) at every entry, and written at the same entries.
template <class T, class Value, class Tolerance>
int near(const std::string& label, const char* what, const std::vector<T>& a,
         const std::vector<T>& b, Value value, Tolerance tolerance) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (unwritten(a[i]) != unwritten(b[i])) {
            std::fprintf(stderr, "FAIL %s: %s written by one kernel only at %zu\n", label.c_str(),
                         what, i);
            return 1;
        }
        if (unwritten(a[i])) continue;
        const double x = value(a[i]), y = value(b[i]);
        if (x == y || std::fabs(x - y) <= tolerance(i, x, y)) continue;
        std::fprintf(stderr, "FAIL %s: %s at %zu: %.9g against %.9g\n", label.c_str(), what, i, x,
                     y);
        return 1;
    }
    return 0;
}

// Volta's kernel (b) against the shipped schedule (a). The scores differ in FP32 summation order
// only (~1e-6 relative), so a probability can round to the neighbouring FP16 value (2^-11
// relative): a numerator within 2^-9 of its row's largest magnitude, the output within one BF16
// step (2^-8 relative) plus 2^-9 of its row's largest, the log2 maximum within 1e-5 and the sum
// within 2^-10 relative.
int near_outputs(const std::string& label, const Outputs& a, const Outputs& b) {
    const auto same = [](float x) { return double(x); };
    std::vector<double> row_scale(a.acc.size() / D, 0.0);
    for (std::size_t i = 0; i < a.acc.size(); ++i)
        if (!unwritten(a.acc[i]))
            row_scale[i / D] = std::max(row_scale[i / D], double(std::fabs(a.acc[i])));
    std::vector<double> out_scale(a.out.size() / D, 0.0);
    for (std::size_t i = 0; i < a.out.size(); ++i)
        if (!unwritten(a.out[i]))
            out_scale[i / D] = std::max(out_scale[i / D], double(std::fabs(bf16_value(a.out[i]))));
    return near(label, "output", a.out, b.out, bf16_value,
                [&](std::size_t i, double x, double y) {
                    return std::ldexp(std::max(std::fabs(x), std::fabs(y)), -8) +
                           std::ldexp(out_scale[i / D], -9);
                }) +
           near(label, "partial O", a.acc, b.acc, same,
                [&](std::size_t i, double, double) { return std::ldexp(row_scale[i / D], -9); }) +
           near(label, "partial maximum", a.maximum, b.maximum, same,
                [](std::size_t, double x, double) { return 1e-5 * std::max(1.0, std::fabs(x)); }) +
           near(label, "partial sum", a.sum, b.sum, same,
                [](std::size_t, double x, double) { return std::ldexp(std::fabs(x), -10); }) +
           compare(label, "cache K", a.k, b.k) + compare(label, "cache V", a.v, b.v);
}

// 1 when the two runs differ in any compared byte.
template <class G, class A, class B, bool MultiBatch, bool Masked, bool Append>
int run_pair(Inputs<G>& in, const Bf16KvCausalPlan& plan, const std::string& label) {
    const Outputs a = run<G, A, MultiBatch, Masked, Append>(in, plan);
    const Outputs b = run<G, B, MultiBatch, Masked, Append>(in, plan);
    int differs     = compare(label, "output", a.out, b.out) +
                  compare(label, "partial O", a.acc, b.acc) +
                  compare(label, "partial maximum", a.maximum, b.maximum) +
                  compare(label, "partial sum", a.sum, b.sum);
    if constexpr (Append) {
        differs += compare(label, "cache K", a.k, b.k) + compare(label, "cache V", a.v, b.v);
    }
    return differs ? 1 : 0;
}

// The pairs that differ; pairs counts those run: former against shipped, or with --self each
// schedule against itself.
template <class G, bool MultiBatch, bool Masked, bool Append>
int run_both(Inputs<G>& in, const Bf16KvCausalPlan& plan, const std::string& label, int& pairs) {
    int differs = 0;
    if (g_volta) {
        pairs += 1;
        if (g_self) {
            differs += run_pair<G, VoltaKeyMajor, VoltaKeyMajor, MultiBatch, Masked, Append>(
                in, plan, label + " [volta]");
        } else {
            const Outputs a = run<G, Shipped, MultiBatch, Masked, Append>(in, plan);
            const Outputs b = run<G, VoltaKeyMajor, MultiBatch, Masked, Append>(in, plan);
            differs += near_outputs(label + " [volta]", a, b) ? 1 : 0;
        }
    }
    if (g_self) {
        pairs += 2;
        return differs +
               run_pair<G, Former, Former, MultiBatch, Masked, Append>(in, plan,
                                                                       label + " [former]") +
               run_pair<G, Shipped, Shipped, MultiBatch, Masked, Append>(in, plan,
                                                                         label + " [shipped]");
    }
    pairs += 1;
    return differs + run_pair<G, Former, Shipped, MultiBatch, Masked, Append>(in, plan, label);
}

template <class G>
int run_case(const Case& c, int sms, std::uint64_t seed, int& pairs) {
    const auto [lo, hi] = std::minmax_element(c.contexts.begin(), c.contexts.end());
    const CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(*lo),
                                                    static_cast<std::uint32_t>(*hi)};
    const int batch   = static_cast<int>(c.contexts.size());
    const auto plan   = make_bf16_kv_causal_plan(G::QHeads, G::KVHeads, 1, batch, envelope, sms);
    std::string label = std::to_string(G::QHeads) + "/" + std::to_string(G::KVHeads) + " B" +
                        std::to_string(batch) + " keys";
    for (int keys : c.contexts) label += " " + std::to_string(keys);
    label += " (" + std::to_string(plan.partition.capacity) + " splits)";
    if (plan.instance != Bf16KvInstance::GroupedDecode) {
        std::fprintf(stderr, "FAIL %s: the plan is not GroupedDecode\n", label.c_str());
        return 1;
    }
    int failures = 0;
    for (const bool empty_row : {false, true}) {
        if (empty_row && batch == 1) continue;
        Inputs<G> in(c, seed, empty_row);
        const auto both = [&]<bool MultiBatch>() {
            failures += run_both<G, MultiBatch, false, false>(in, plan, label + " cached", pairs);
            failures += run_both<G, MultiBatch, false, true>(in, plan, label + " append", pairs);
            failures +=
                run_both<G, MultiBatch, true, false>(in, plan, label + " cached masked", pairs);
            failures +=
                run_both<G, MultiBatch, true, true>(in, plan, label + " append masked", pairs);
        };
        if (empty_row) {
            // Only the masked runs differ: a valid-column count of 0 empties the middle row.
            failures +=
                run_both<G, true, true, false>(in, plan, label + " cached, a row empty", pairs);
            failures +=
                run_both<G, true, true, true>(in, plan, label + " append, a row empty", pairs);
        } else if (batch > 1) {
            both.template operator()<true>();
        } else {
            both.template operator()<false>();
        }
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--self") == 0) {
            g_self = true;
        } else {
            std::fprintf(stderr, "usage: %s [--self]\n", argv[0]);
            return 2;
        }
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    int sms = 0;
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), "SM count");
    int major = 0, minor = 0;
    check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), "major");
    check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), "minor");
    g_volta = NINFERNO_BUILD_ARCH == 70 && major == 7 && minor == 0;
    check(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking), "stream");
    // One row: page (64) and key-tile (32) edges, the split rounding near the targets, the long
    // target from 32768 keys. Eight rows: ragged contexts in one envelope, and equal ones.
    const std::vector<Case> cases = {
        {{1}},
        {{31}},
        {{32}},
        {{33}},
        {{64}},
        {{65}},
        {{2047}},
        {{2048}},
        {{2049}},
        {{8193}},
        {{21961}},
        {{32767}},
        {{32768}},
        {{40001}},
        {{1, 63, 64, 65, 2049, 4096, 8191, 12000}},
        {{7843, 7843, 7843, 7843, 7843, 7843, 7843, 7843}},
    };
    int failures = 0, pairs = 0;
    std::uint64_t seed = 856;
    for (const Case& c : cases) {
#define NINFERNO_IDENTITY_GEOMETRY(Q, KV)                                                          \
    failures += run_case<CausalGeometry<D, (Q), (KV)>>(c, sms, seed++, pairs);
        NINFERNO_CAUSAL_GEOMETRIES(NINFERNO_IDENTITY_GEOMETRY)
#undef NINFERNO_IDENTITY_GEOMETRY
    }
    cudaStreamDestroy(g_stream);
    const char* mode = g_self ? "self" : "former vs shipped";
    if (failures) {
        std::printf("FAIL (%s): %d of %d decode pairs differ\n", mode, failures, pairs);
        return 1;
    }
    std::printf("bf16 decode identity (%s%s): %d decode pairs identical on %d SMs\n", mode,
                g_volta ? "; Volta key-major within its bounds" : "", pairs, sms);
    return 0;
}
