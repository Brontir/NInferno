// Cold-L2 operator benchmark of the Gemma 4 26B-A4B sparse MoE, route by route.
//
// Every arm is one route (or the public dispatch) captured as a CUDA graph. The
// arms of a token count run interleaved, pass after pass, in one series, and
// one arm is repeated under a second name as the null control: the spread
// between an arm and its twin is the noise floor of the series. Weights, router
// and inputs are random; a constant fill makes these kernels look faster.
//
// Usage: ninferno_gemma4_moe_bench [--tokens 1,2,8,...] [--passes 5] [--repeat 20] [--warmup 3]
//                                [--arms public,decode,small,a4e,a4t,a4n,a16] [--warm]
//                                [--calibrate] [--duplicate N]
//
// Arms: public (the dispatch), decode, small (small-T), a4e / a4t (A4 prefill with the per-expert
// or per-token activation divisor), a4n (A4 prefill, narrow row tile), a16 (A16 prefill), and the
// decode variants decode-sel / -order / -stage / -all (see gemma4_moe_internal.h). --duplicate N
// repeats every N-th token over the N-1 after it, which concentrates the routing the way
// correlated tokens do.

#include "core/weight.h"
#include "ninferno/ops/gemma4_moe.h"

#include "core/device.h"
#include "ninferno_bench_common.h"
#include "ops/gemma4_moe/gemma4_moe_internal.h"
#include "core/fnv.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninferno;
namespace g4 = ninferno::ops::detail::gemma4;

namespace {

constexpr int kHidden  = 2816;
constexpr int kExperts = 128;
constexpr int kTopK    = 8;
constexpr int kInter   = 704;
constexpr double kExpertBytes =
    (1408.0 * 2816.0 + 2816.0 * 704.0) * (0.5 + 1.0 / 16.0); // codes + block scales
constexpr double kRouterBytes = 128.0 * 2816.0 * 2.0;
constexpr double kExpertFlops = 2.0 * (1408.0 * 2816.0 + 2816.0 * 704.0);

__device__ std::uint64_t hash64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__global__ void fill_random_bytes(std::uint8_t* data, std::size_t words, std::uint64_t seed) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < words;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        reinterpret_cast<std::uint64_t*>(data)[i] = hash64(seed ^ (i * ninferno::fnv::kPrime));
    }
}

// Block scales between 32 and 448, never the e4m3 NaN.
__global__ void fill_scales(std::uint8_t* data, std::size_t bytes, std::uint64_t seed) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < bytes;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::uint64_t r = hash64(seed + i);
        const int exponent    = 12 + static_cast<int>(r & 3);
        int mantissa          = static_cast<int>((r >> 2) & 7);
        if (exponent == 15 && mantissa == 7) { mantissa = 6; }
        data[i] = static_cast<std::uint8_t>((exponent << 3) | mantissa);
    }
}

// Roughly normal BF16: the sum of four uniforms, scaled.
__global__ void fill_bf16(__nv_bfloat16* data, std::size_t count, std::uint64_t seed, float scale) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::uint64_t r = hash64(seed + i);
        float sum             = 0.0F;
        for (int part = 0; part < 4; ++part) {
            sum += static_cast<float>((r >> (16 * part)) & 0xffff) / 65535.0F - 0.5F;
        }
        data[i] = __float2bfloat16_rn(sum * 1.7320508F * scale);
    }
}

// The ceiling a kernel of a given size can reach: a plain streaming read of
// `bytes` contiguous bytes, sixteen a thread, summed so the loads cannot be
// dropped.
__global__ void stream_read_kernel(const uint4* __restrict__ data, std::size_t vectors,
                                   unsigned* __restrict__ sink) {
    unsigned acc = 0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
         i < vectors; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const uint4 v = __ldcs(data + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678U) { *sink = acc; }
}

struct Plane {
    DeviceBuffer payload;
    Weight weight{};
};

Plane make_plane(int rows, int k, std::uint64_t seed, float amax_low, float amax_high) {
    Plane plane;
    const std::size_t code_bytes     = static_cast<std::size_t>(rows) * k / 2;
    const std::size_t scale_offset   = (code_bytes + 255) / 256 * 256;
    const std::size_t scale_bytes    = static_cast<std::size_t>(rows) * k / 16;
    const std::size_t divisor_offset = scale_offset + scale_bytes;
    plane.payload                    = DeviceBuffer(divisor_offset + kExperts * sizeof(float));
    auto* base                       = static_cast<std::uint8_t*>(plane.payload.p);
    fill_random_bytes<<<4096, 256>>>(base, code_bytes / 8, seed);
    fill_scales<<<4096, 256>>>(base + scale_offset, scale_bytes, seed * 3 + 1);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> divisors(kExperts);
    for (int e = 0; e < kExperts; ++e) {
        const float amax = amax_low + (amax_high - amax_low) * std::fmod(e * 0.6180339887F, 1.0F);
        divisors[e]      = 2688.0F / amax;
    }
    plane.payload.copy_from_host(divisors.data(), divisors.size() * sizeof(float), divisor_offset);
    Weight& w       = plane.weight;
    w.payload       = base;
    w.payload_bytes = plane.payload.bytes;
    w.qtype         = QType::NVFP4;
    w.group_size    = 16;
    w.group         = 16;
    w.layout        = QuantLayout::BlockScaleK16M128x4;
    w.scale_dtype   = DType::FP8_E4M3FN;
    w.ndim          = 2;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.qdata                              = base;
    w.scales                             = base + scale_offset;
    w.weight_divisors                    = base + divisor_offset;
    w.weight_divisor_rows                = rows / kExperts;
    w.weight_scale_divisor               = divisors[0];
    w.input_scale_divisor                = 1.0F;
    return plane;
}

DeviceBuffer upload(const std::vector<float>& values) {
    DeviceBuffer buffer(values.size() * sizeof(float));
    buffer.copy_from_host(values.data(), buffer.bytes);
    return buffer;
}

std::vector<int> parse_ints(const std::string& text) {
    std::vector<int> out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) { out.push_back(std::stoi(item)); }
    return out;
}

std::vector<std::string> parse_names(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) { out.push_back(item); }
    return out;
}

struct Arm {
    std::string name;
    bool public_entry = false;
    g4::Route route   = g4::Route::Decode;
    g4::A4Input input = g4::A4Input::PerExpert;
    bool narrow       = false;
    g4::DecodeOptions decode{};
};

bool arm_from_name(const std::string& name, Arm& arm) {
    arm.name = name;
    if (name == "public") {
        arm.public_entry = true;
        return true;
    }
    if (name == "decode") {
        arm.route = g4::Route::Decode;
        return true;
    }
    // Decode variants: each change alone, then all three.
    if (name == "decode-sel" || name == "decode-order" || name == "decode-stage" ||
        name == "decode-all") {
        arm.route              = g4::Route::Decode;
        const bool all         = name == "decode-all";
        arm.decode.rank_select = all || name == "decode-sel";
        arm.decode.loads_first = all || name == "decode-order";
        arm.decode.staged_down = all || name == "decode-stage";
        return true;
    }
    if (name == "small") {
        arm.route = g4::Route::SmallT;
        return true;
    }
    if (name == "a4e") {
        arm.route = g4::Route::PrefillA4;
        arm.input = g4::A4Input::PerExpert;
        return true;
    }
    if (name == "a4t") {
        arm.route = g4::Route::PrefillA4;
        arm.input = g4::A4Input::PerToken;
        return true;
    }
    if (name == "a4n") {
        arm.route  = g4::Route::PrefillA4;
        arm.narrow = true;
        return true;
    }
    if (name == "a16") {
        arm.route = g4::Route::PrefillA16;
        return true;
    }
    return false;
}

bool arm_valid(const Arm& arm, int tokens) {
    if (arm.public_entry) { return true; }
    if (arm.route == g4::Route::Decode) { return tokens == 1; }
    if (arm.route == g4::Route::SmallT) { return tokens <= g4::kSmallTCapacity; }
    return true;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> token_counts{1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 512, 2048, 4096};
    std::vector<std::string> arm_names{"public", "decode", "small", "a4e", "a4t", "a16"};
    int passes = 5, repeat = 20, warmup = 3;
    bool warm      = false;
    bool calibrate = false;
    int duplicate  = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next             = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument("missing value for " + arg); }
            return argv[++i];
        };
        if (arg == "--tokens") {
            token_counts = parse_ints(next());
        } else if (arg == "--arms") {
            arm_names = parse_names(next());
        } else if (arg == "--passes") {
            passes = std::stoi(next());
        } else if (arg == "--repeat") {
            repeat = std::stoi(next());
        } else if (arg == "--warmup") {
            warmup = std::stoi(next());
        } else if (arg == "--warm") {
            warm = true;
        } else if (arg == "--calibrate") {
            calibrate = true;
        } else if (arg == "--duplicate") {
            duplicate = std::stoi(next());
            if (duplicate < 1) { throw std::invalid_argument("--duplicate takes N >= 1"); }
        } else {
            throw std::invalid_argument("unknown argument " + arg);
        }
    }

    Plane gate_up = make_plane(kExperts * 1408, kHidden, 0x51, 0.05F, 0.20F);
    Plane down    = make_plane(kExperts * kHidden, kInter, 0x52, 0.10F, 0.40F);
    DeviceBuffer router(static_cast<std::size_t>(kExperts) * kHidden * 2);
    fill_bf16<<<1024, 256>>>(static_cast<__nv_bfloat16*>(router.p),
                             static_cast<std::size_t>(kExperts) * kHidden, 0x53, 2.0F);
    std::vector<float> router_scale(kHidden), expert_scale(kExperts), gate_div(kExperts),
        down_div(kExperts);
    for (int h = 0; h < kHidden; ++h) { router_scale[h] = 0.5F + std::fmod(h * 0.37F, 1.0F); }
    for (int e = 0; e < kExperts; ++e) {
        expert_scale[e] = 0.5F + 1.5F * std::fmod(e * 0.41F, 1.0F);
        gate_div[e]     = 2688.0F / (8.0F * (1.0F + 0.5F * std::fmod(e * 0.75F, 1.0F)));
        down_div[e]     = 2688.0F / (40.0F * (1.0F + 0.5F * std::fmod(e * 0.75F, 1.0F)));
    }
    DeviceBuffer router_scale_d = upload(router_scale), expert_scale_d = upload(expert_scale),
                 gate_div_d = upload(gate_div), down_div_d = upload(down_div);
    ops::Gemma4MoeWeights weights;
    weights.router.payload       = router.p;
    weights.router.payload_bytes = router.bytes;
    weights.router.qtype         = QType::BF16;
    weights.router.qdata         = router.p;
    weights.router.layout        = QuantLayout::Contiguous;
    weights.router.ndim          = 2;
    weights.router.n = weights.router.shape[0] = weights.router.padded_shape[0] = kExperts;
    weights.router.k = weights.router.shape[1] = weights.router.padded_shape[1] = kHidden;
    weights.router_scale           = Tensor(router_scale_d.p, DType::FP32, {kHidden});
    weights.expert_scale           = Tensor(expert_scale_d.p, DType::FP32, {kExperts});
    weights.gate_up                = gate_up.weight;
    weights.down                   = down.weight;
    weights.gate_up_input_divisors = Tensor(gate_div_d.p, DType::FP32, {kExperts});
    weights.down_input_divisors    = Tensor(down_div_d.p, DType::FP32, {kExperts});

    const int max_tokens = *std::max_element(token_counts.begin(), token_counts.end());
    DeviceBuffer router_input(static_cast<std::size_t>(kHidden) * max_tokens * 2);
    DeviceBuffer expert_input(static_cast<std::size_t>(kHidden) * max_tokens * 2);
    fill_bf16<<<1024, 256>>>(static_cast<__nv_bfloat16*>(router_input.p),
                             static_cast<std::size_t>(kHidden) * max_tokens, 0x54, 1.0F);
    fill_bf16<<<1024, 256>>>(static_cast<__nv_bfloat16*>(expert_input.p),
                             static_cast<std::size_t>(kHidden) * max_tokens, 0x55, 1.0F);
    DeviceBuffer output(static_cast<std::size_t>(kHidden) * max_tokens * 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Concentrated routing: every `duplicate` consecutive tokens are one token repeated, so they
    // select the same experts (a stand-in for the correlated tokens of a verify pass).
    if (duplicate > 1) {
        const std::size_t column = static_cast<std::size_t>(kHidden) * 2;
        for (int t = 0; t < max_tokens; ++t) {
            if (t % duplicate == 0) { continue; }
            const std::size_t from = static_cast<std::size_t>(t - t % duplicate) * column;
            for (DeviceBuffer* buffer : {&router_input, &expert_input}) {
                auto* base = static_cast<std::uint8_t*>(buffer->p);
                CUDA_CHECK(
                    cudaMemcpy(base + t * column, base + from, column, cudaMemcpyDeviceToDevice));
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::printf("duplicate=%d\n", duplicate);

    bench::L2FlushBuffer flush(256ULL << 20);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::printf("cache=%s passes=%d repeat=%d warmup=%d\n", warm ? "warm" : "cold", passes, repeat,
                warmup);
    std::printf("%6s %-12s %10s %9s %9s %8s %9s %9s %s\n", "T", "arm", "median_us", "min_us",
                "max_us", "experts", "GB/s", "TFLOPS", "note");

    // Calibration: the streaming-read ceiling at the byte counts the decode
    // kernels move.
    if (calibrate) {
        DeviceBuffer sink(64);
        for (const double megabytes : {8.9, 17.8, 26.8, 64.0, 428.0}) {
            const std::size_t vectors = static_cast<std::size_t>(megabytes * 1.0e6 / 16.0);
            const auto* data          = reinterpret_cast<const uint4*>(gate_up.payload.p);
            for (const int blocks : {170 * 4, 170 * 8, 170 * 16}) {
                bench::TimedGraph graph;
                graph.capture(stream, [&](cudaStream_t s) {
                    stream_read_kernel<<<blocks, 256, 0, s>>>(data, vectors,
                                                              static_cast<unsigned*>(sink.p));
                });
                const bench::ColdTiming timing =
                    warm ? bench::measure_graph(graph, stream, warmup, repeat)
                         : bench::measure_cold_graph(graph, flush, stream, warmup, repeat);
                std::printf("CALIBRATION read %7.1f MB blocks=%5d  %8.2f us  %7.1f GB/s\n",
                            megabytes, blocks, timing.median_us,
                            megabytes * 1.0e6 / (timing.median_us * 1.0e-6) / 1.0e9);
            }
        }
    }

    for (const int tokens : token_counts) {
        std::vector<Arm> arms;
        for (const auto& name : arm_names) {
            Arm arm;
            if (!arm_from_name(name, arm)) { throw std::invalid_argument("unknown arm " + name); }
            if (arm_valid(arm, tokens)) { arms.push_back(arm); }
        }
        if (arms.empty()) { continue; }
        Arm null_arm = arms.front();
        null_arm.name += "-null";
        arms.push_back(null_arm);

        Tensor x_r(router_input.p, DType::BF16, {kHidden, tokens});
        Tensor x_e(expert_input.p, DType::BF16, {kHidden, tokens});
        Tensor out(output.p, DType::BF16, {kHidden, tokens});
        std::vector<std::unique_ptr<WorkspaceArena>> workspaces;
        std::vector<std::unique_ptr<bench::TimedGraph>> graphs;
        std::vector<int> experts_touched;
        for (const Arm& arm : arms) {
            const std::size_t bytes = arm.public_entry
                                          ? ops::gemma4_moe_workspace_bytes(tokens, tokens)
                                          : g4::route_workspace_bytes(arm.route, tokens);
            workspaces.push_back(std::make_unique<WorkspaceArena>(bytes));
            WorkspaceArena& ws = *workspaces.back();
            auto body          = [&](cudaStream_t s) {
                if (arm.public_entry) {
                    ops::gemma4_moe(x_r, x_e, weights, 1.0e-6F, out, ws, s);
                } else {
                    g4::gemma4_moe_with_route(x_r, x_e, weights, 1.0e-6F, out, ws, s, arm.route,
                                                       arm.input, arm.narrow, arm.decode);
                }
            };
            graphs.push_back(std::make_unique<bench::TimedGraph>());
            graphs.back()->capture(stream, body);
            graphs.back()->launch(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const int capacity = std::min(tokens, g4::kSliceMax);
            std::vector<int> ids(static_cast<std::size_t>(kTopK) * capacity);
            CUDA_CHECK(cudaMemcpy(ids.data(), ws.base(), ids.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            experts_touched.push_back(
                static_cast<int>(std::set<int>(ids.begin(), ids.end()).size()));
        }

        std::vector<std::vector<double>> samples(arms.size());
        for (int pass = 0; pass < passes; ++pass) {
            for (std::size_t a = 0; a < arms.size(); ++a) {
                const bench::ColdTiming timing =
                    warm ? bench::measure_graph(*graphs[a], stream, warmup, repeat)
                         : bench::measure_cold_graph(*graphs[a], flush, stream, warmup, repeat);
                samples[a].push_back(timing.median_us);
            }
        }
        const double twin = median(samples.front());
        for (std::size_t a = 0; a < arms.size(); ++a) {
            const double us = median(samples[a]);
            const double lo = *std::min_element(samples[a].begin(), samples[a].end());
            const double hi = *std::max_element(samples[a].begin(), samples[a].end());
            const double gbs =
                (experts_touched[a] * kExpertBytes + kRouterBytes) / (us * 1.0e-6) / 1.0e9;
            const double tflops = tokens * kTopK * kExpertFlops / (us * 1.0e-6) / 1.0e12;
            char note[64]       = "";
            if (a + 1 == arms.size()) {
                std::snprintf(note, sizeof(note), "null delta %+.2f%%", 100.0 * (us / twin - 1.0));
            }
            std::printf("%6d %-12s %10.2f %9.2f %9.2f %8d %9.1f %9.1f %s\n", tokens,
                        arms[a].name.c_str(), us, lo, hi, experts_touched[a], gbs, tflops, note);
        }
        std::fflush(stdout);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
