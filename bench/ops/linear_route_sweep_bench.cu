// Route sweep for the pure Linear shape tables on the running GPU.
//
// The tables were tuned on one GPU class. For every token count T this bench times the route the
// table picks (the dispatcher's choice) and then every route the same table would pick at a larger
// width T' (ops::detail::RouteProbe), each at the real T, cold-cache, and checks each candidate's
// output against the dispatcher's. A candidate that beats the dispatcher's choice at T on this
// card is a crossover that moved: the input of a per-GPU-class route table.
//
//   ninferno_linear_route_sweep_bench --qtype q8 --n 12288 --k 2048
//   ninferno_linear_route_sweep_bench --qtype nvfp4 --policy a4 --n 14336 --k 5120 --quantized both
//
// Output: one CSV row per (T, candidate) on stdout, or into --csv-out.
//
// --random-weight fills the weight's code planes with pseudo-random codes (the scales stay as the
// format's bench helper writes them), so an output that equals the dispatcher's bit for bit means
// the candidate sums K in the same order: the evidence a route_as table entry needs
// (ops/common/shape_routes.h). --no-class-table drops the GPU class's route table first, so the
// dispatcher row is the shape table as tuned; without it the dispatcher row includes the table.

#include "core/device.h"
#include "core/weight.h"
#include "direct_bf16_weight.cuh"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/route_table.h"
#include "ninferno_bench_common.h"
#include "ops/common/shape_routes.h"
#include "quantized_weight.cuh"
#include "random_weight.h"
#include "ops/generic/route_table.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninferno;
using ninferno::ops::LinearPolicy;

namespace {

constexpr std::int32_t kDefaultTokens[] = {1,  2,   3,   4,   6,   8,    12,   16,   24,  32,  48,
                                           64, 96,  128, 192, 256, 384,  512,  768,  1024, 1536,
                                           2048};
constexpr std::int32_t kDefaultAs[]     = {1,  2,   3,   4,   6,   8,    12,   16,   24,   32,
                                           48, 64,  96,  128, 192, 256,  384,  512,  768,  1024,
                                           1280, 1536, 2048, 3072, 4096};
// A candidate whose output differs from the dispatcher's by more than this relative L2 error is
// reported and not ranked: a route may round differently, never compute something else.
constexpr double kRankableRelL2 = 3.0e-2;

struct Options {
    QType qtype         = QType::Q8_G32_FP16;
    LinearPolicy policy = LinearPolicy::A16Only;
    std::int32_t n      = 0;
    std::int32_t k      = 0;
    std::vector<std::int32_t> tokens{std::begin(kDefaultTokens), std::end(kDefaultTokens)};
    std::vector<std::int32_t> as{std::begin(kDefaultAs), std::end(kDefaultAs)};
    bool quantized_both = false;
    bool random_weight  = false;
    bool class_table    = true;
    int warmup          = 2;
    int repeat          = 10;
    std::string csv_out;
};

std::string lower(std::string_view text) {
    std::string out(text);
    for (char& c : out) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
    return out;
}

QType parse_qtype(std::string_view text) {
    const std::string v = lower(text);
    if (v == "q4") return QType::Q4_G64_FP16;
    if (v == "q5") return QType::Q5_G64_FP16;
    if (v == "q6") return QType::Q6_G64_FP16;
    if (v == "q8") return QType::Q8_G32_FP16;
    if (v == "bf16") return QType::BF16;
    if (v == "nvfp4") return QType::NVFP4;
    if (v == "fp8") return QType::FP8_E4M3FN_ROW_BF16;
    // Every other format by the route table's name (q4k, iq4xs, mxfp4, ...).
    for (int value = 0; value <= 0xff; ++value) {
        const auto q = static_cast<QType>(value);
        if (v == ops::detail::generic::qtype_name(q)) return q;
    }
    throw std::invalid_argument("unknown qtype " + std::string(text));
}

const char* qtype_name(QType q) {
    switch (q) {
    case QType::Q4_G64_FP16: return "Q4";
    case QType::Q5_G64_FP16: return "Q5";
    case QType::Q6_G64_FP16: return "Q6";
    case QType::Q8_G32_FP16: return "Q8";
    case QType::BF16: return "BF16";
    case QType::NVFP4: return "NVFP4";
    case QType::FP8_E4M3FN_ROW_BF16: return "FP8";
    default: return ops::detail::generic::qtype_name(q);
    }
}

LinearPolicy parse_policy(std::string_view text) {
    const std::string v = lower(text);
    if (v == "a16") return LinearPolicy::A16Only;
    if (v == "a8") return LinearPolicy::AllowA8;
    if (v == "a4") return LinearPolicy::AllowA4;
    throw std::invalid_argument("policy must be a16, a8 or a4");
}

const char* policy_name(LinearPolicy p) {
    return p == LinearPolicy::A16Only ? "A16" : p == LinearPolicy::AllowA8 ? "A8" : "A4";
}

std::vector<std::int32_t> parse_list(std::string_view text) {
    std::vector<std::int32_t> out;
    std::stringstream in{std::string(text)};
    std::string item;
    while (std::getline(in, item, ',')) {
        const long value = std::strtol(item.c_str(), nullptr, 10);
        if (value <= 0) throw std::invalid_argument("list entries must be positive");
        out.push_back(static_cast<std::int32_t>(value));
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

Options parse_args(int argc, char** argv) {
    Options o;
    bool have_n = false, have_k = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        const auto next = [&]() -> std::string_view {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(a));
            return argv[++i];
        };
        if (a == "--qtype") o.qtype = parse_qtype(next());
        else if (a == "--policy") o.policy = parse_policy(next());
        else if (a == "--n") { o.n = std::atoi(std::string(next()).c_str()); have_n = true; }
        else if (a == "--k") { o.k = std::atoi(std::string(next()).c_str()); have_k = true; }
        else if (a == "--t") o.tokens = parse_list(next());
        else if (a == "--as") o.as = parse_list(next());
        else if (a == "--quantized") o.quantized_both = next() == "both";
        else if (a == "--random-weight") o.random_weight = true;
        else if (a == "--no-class-table") o.class_table = false;
        else if (a == "--warmup") o.warmup = std::atoi(std::string(next()).c_str());
        else if (a == "--repeat") o.repeat = std::atoi(std::string(next()).c_str());
        else if (a == "--csv-out") o.csv_out = std::string(next());
        else throw std::invalid_argument("unknown option " + std::string(a));
    }
    if (!have_n || !have_k || o.n <= 0 || o.k <= 0) throw std::invalid_argument("--n and --k");
    if (o.repeat <= 0 || o.warmup < 0) throw std::invalid_argument("--repeat/--warmup");
    return o;
}

struct BenchWeight {
    DeviceBuffer storage;
    Weight weight{};
    // A ggml or other format's random parent (random_weight.h); its Weight names the parent by
    // address, so it lives on the heap.
    std::unique_ptr<bench::RandomWeight> generic;
};

// Pseudo-random bytes (a multiplicative hash of the index), masked: every code of the integer
// formats and of NVFP4 is a value; FP8 E4M3 clears bit 6 so no byte is NaN.
__global__ void randomize_bytes_kernel(std::uint8_t* bytes, std::uint64_t count, std::uint32_t seed,
                                       std::uint8_t mask) {
    const std::uint64_t stride = gridDim.x * static_cast<std::uint64_t>(blockDim.x);
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         i < count; i += stride) {
        std::uint64_t h = (i + seed) * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 29;
        bytes[i] = static_cast<std::uint8_t>(h >> 24) & mask;
    }
}

void randomize_bytes(void* base, std::uint64_t offset, std::uint64_t count, std::uint32_t seed,
                     std::uint8_t mask = 0xff) {
    if (count == 0) return;
    randomize_bytes_kernel<<<1024, 256>>>(static_cast<std::uint8_t*>(base) + offset, count, seed,
                                          mask);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

BenchWeight make_weight(QType qtype, std::int32_t n, std::int32_t k, bool random) {
    if (qtype == QType::BF16) {
        auto direct = bench::make_direct_bf16_weight(n, k); // already pseudo-random
        return {std::move(direct.storage), direct.weight, nullptr};
    }
    if (qtype == QType::NVFP4) {
        auto packed = bench::make_nvfp4_weight(n, k);
        if (random) randomize_bytes(packed.storage.p, 0, packed.low_bytes, 0x4e);
        return {std::move(packed.storage), packed.weight};
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16) {
        auto packed = bench::make_fp8_weight(n, k);
        if (random) randomize_bytes(packed.storage.p, 0, packed.low_bytes, 0xf8, 0xbf);
        return {std::move(packed.storage), packed.weight};
    }
    if (qtype != QType::Q4_G64_FP16 && qtype != QType::Q5_G64_FP16 && qtype != QType::Q6_G64_FP16 &&
        qtype != QType::Q8_G32_FP16) {
        BenchWeight out;
        out.generic = std::make_unique<bench::RandomWeight>(bench::random_weight(qtype, n, k, 501U));
        const auto& g = out.generic->parent.geometry;
        const WeightView view{g.shape, {WeightRegion{&out.generic->parent, 0, g.elements}}};
        out.generic->weight = native_weight(view, 4.0F);
        out.weight          = out.generic->weight;
        return out;
    }
    const auto padded_k = static_cast<std::int32_t>((static_cast<std::int64_t>(k) + 127) / 128 * 128);
    auto packed         = bench::make_row_split_weight(qtype, n, k, padded_k, 0x31U);
    if (random) {
        randomize_bytes(packed.storage.p, 0, packed.low_bytes, 0x10);
        randomize_bytes(packed.storage.p, packed.high_offset, packed.high_bytes, 0x11);
    }
    return {std::move(packed.storage), packed.weight};
}

__global__ void fill_activation_kernel(__nv_bfloat16* values, std::uint64_t count) {
    const std::uint64_t stride = gridDim.x * static_cast<std::uint64_t>(blockDim.x);
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         i < count; i += stride) {
        values[i] = __float2bfloat16(0.5F - static_cast<float>((i * 2654435761ULL) % 251ULL) / 250.0F);
    }
}

// With --random-weight the activations spread over thirteen octaves with both signs, so FP32
// partial sums round and a candidate that sums K in another order shows as a mismatch (the
// patterned fill above hides such schedules; tests/ops/linear/test_linear_gemma4_token_invariance
// found the same).
__global__ void fill_wide_activation_kernel(__nv_bfloat16* values, std::uint64_t count) {
    const std::uint64_t stride = gridDim.x * static_cast<std::uint64_t>(blockDim.x);
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         i < count; i += stride) {
        std::uint64_t h = (i + 0x5bd1e995ULL) * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 31;
        const float mantissa = 1.0F + static_cast<float>(h & 0xffffU) / 65536.0F;
        const int exponent   = static_cast<int>((h >> 16) % 13U) - 6;
        const float sign     = ((h >> 40) & 1U) != 0U ? -1.0F : 1.0F;
        values[i]            = __float2bfloat16(sign * ldexpf(mantissa, exponent));
    }
}

__global__ void rel_l2_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, std::uint64_t count,
                              double* sums) {
    double diff = 0.0, ref = 0.0;
    const std::uint64_t stride = gridDim.x * static_cast<std::uint64_t>(blockDim.x);
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         i < count; i += stride) {
        const double x = __bfloat162float(a[i]);
        const double y = __bfloat162float(b[i]);
        diff += (x - y) * (x - y);
        ref += y * y;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        diff += __shfl_down_sync(0xffffffffu, diff, offset);
        ref += __shfl_down_sync(0xffffffffu, ref, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&sums[0], diff);
        atomicAdd(&sums[1], ref);
    }
}

double rel_l2(const DeviceBuffer& a, const DeviceBuffer& b, std::uint64_t count, DeviceBuffer& sums,
              cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(sums.p, 0, 2 * sizeof(double), stream));
    rel_l2_kernel<<<1024, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(a.p),
                                           static_cast<const __nv_bfloat16*>(b.p), count,
                                           static_cast<double*>(sums.p));
    CUDA_CHECK(cudaGetLastError());
    double host[2] = {0.0, 0.0};
    CUDA_CHECK(cudaMemcpyAsync(host, sums.p, sizeof(host), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!(host[0] >= 0.0) || !std::isfinite(host[0])) return std::numeric_limits<double>::infinity();
    return host[1] > 0.0 ? std::sqrt(host[0] / host[1]) : std::sqrt(host[0]);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse_args(argc, argv);
        bench::prefill_routes_from_env();
        if (!o.class_table) ops::clear_route_table();
        DeviceContext ctx(0);
        const auto& device  = ctx.device_class();
        cudaStream_t stream = ctx.stream;

        const std::int32_t t_cap =
            std::max(o.tokens.back(), o.as.empty() ? o.tokens.back() : o.as.back());
        BenchWeight weight = make_weight(o.qtype, o.n, o.k, o.random_weight);
        const std::uint64_t x_elems   = static_cast<std::uint64_t>(o.k) * t_cap;
        const std::uint64_t out_elems = static_cast<std::uint64_t>(o.n) * t_cap;
        DeviceBuffer x(x_elems * 2), out(out_elems * 2), ref(out_elems * 2), sums(64);
        if (o.random_weight) {
            fill_wide_activation_kernel<<<1024, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(x.p),
                                                                  x_elems);
        } else {
            fill_activation_kernel<<<1024, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(x.p),
                                                             x_elems);
        }
        CUDA_CHECK(cudaGetLastError());
        // Size the workspace for every decision the sweep can force.
        std::size_t ws_bytes = ops::linear_workspace_capacity_bytes(o.qtype, o.n, o.k, o.policy, 1,
                                                                    t_cap);
        {
            const ops::detail::RouteProbe forced(0, 1);
            ws_bytes = std::max(ws_bytes, ops::linear_workspace_capacity_bytes(
                                              o.qtype, o.n, o.k, o.policy, 1, t_cap));
        }
        DeviceArena workspace(std::max<std::size_t>(ws_bytes, 256));
        const std::size_t flush_bytes = std::max<std::size_t>(std::size_t{256} << 20,
                                                              device.l2_bytes * 4);
        bench::L2FlushBuffer flush(flush_bytes);

        std::ofstream file;
        std::ostream* csv = &std::cout;
        if (!o.csv_out.empty()) {
            file.open(o.csv_out);
            if (!file) throw std::runtime_error("cannot open " + o.csv_out);
            csv = &file;
        }
        *csv << "gpu_class,gpu,qtype,policy,n,k,t,route_as,quantized,median_us,min_us,p95_us,"
                "rel_l2,status\n";
        std::fprintf(stderr, "# %s %s %s %s N=%d K=%d %s; %s\n", device.name.c_str(),
                     device.id().c_str(), qtype_name(o.qtype), policy_name(o.policy), o.n, o.k,
                     o.random_weight ? "random weight" : "constant weight",
                     ops::route_table_status().message.c_str());

        for (const std::int32_t t : o.tokens) {
            Tensor xt(x.p, DType::BF16, {o.k, t});
            Tensor ot(out.p, DType::BF16, {o.n, t});
            const auto launch = [&](cudaStream_t s) {
                ops::linear(xt, weight.weight, ot, o.policy, workspace, s);
            };
            // The dispatcher's own choice first: the reference output and the time to beat.
            launch(stream);
            CUDA_CHECK(cudaMemcpyAsync(ref.p, out.p, static_cast<std::size_t>(o.n) * t * 2,
                                       cudaMemcpyDeviceToDevice, stream));
            const auto base = bench::measure_cold_launch(launch, flush, stream, o.warmup, o.repeat);
            const auto row  = [&](std::int32_t as, int quantized, double med, double mn,
                                 double p95, double err, const std::string& status) {
                *csv << device.id() << ',' << '"' << device.name << '"' << ','
                     << qtype_name(o.qtype) << ',' << policy_name(o.policy) << ',' << o.n << ','
                     << o.k << ',' << t << ',' << as << ',' << quantized << ',' << med << ','
                     << mn << ',' << p95 << ',' << err << ',' << status << '\n';
            };
            row(0, -1, base.median_us, base.min_us, base.p95_us, 0.0, "dispatcher");
            double best = base.median_us;
            std::int32_t best_as = 0;
            int best_q           = -1;

            std::vector<std::pair<std::int32_t, int>> candidates;
            for (const std::int32_t as : o.as) {
                if (as > t) candidates.push_back({as, -1});
            }
            if (o.quantized_both && o.policy != LinearPolicy::A16Only) {
                candidates.push_back({0, 0});
                candidates.push_back({0, 1});
            }
            for (const auto& [as, quantized] : candidates) {
                const ops::detail::RouteProbe probe(as, static_cast<std::int8_t>(quantized));
                try {
                    launch(stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                } catch (const std::exception& e) {
                    std::string what = e.what();
                    std::replace(what.begin(), what.end(), ',', ';');
                    row(as, quantized, NAN, NAN, NAN, NAN, "rejected: " + what);
                    continue;
                }
                const double err = rel_l2(out, ref, static_cast<std::uint64_t>(o.n) * t, sums, stream);
                const auto timing =
                    bench::measure_cold_launch(launch, flush, stream, o.warmup, o.repeat);
                const bool rankable = err <= kRankableRelL2;
                row(as, quantized, timing.median_us, timing.min_us, timing.p95_us, err,
                    rankable ? "ok" : "mismatch");
                if (rankable && timing.median_us < best) {
                    best    = timing.median_us;
                    best_as = as;
                    best_q  = quantized;
                }
            }
            std::fprintf(stderr, "T=%5d dispatcher %9.2f us | best %9.2f us (as=%d q=%d) %+6.1f%%\n",
                         t, base.median_us, best, best_as, best_q,
                         (best / base.median_us - 1.0) * 100.0);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
