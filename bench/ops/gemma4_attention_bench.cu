// Public-Op benchmark for Gemma 4 attention (gemma4_attention). Every timed launch is a captured
// graph of public calls; the workspace comes from the public capacity query.
//
// Timing protocol. A graph holds N consecutive calls, each over its own layer cache, with N chosen
// so the layers' K/V together exceed L2 four times: every call streams its K/V from DRAM, as in a
// real decode step across layers, and the per-call time is the graph time over N. This also keeps
// the ~2 us granularity of event timing on this platform far below the resolution that matters.
// The card's clock wanders, so each configuration captures two identical graphs, A and B, and
// times them interleaved (A, B, A, B, ...). B is the null arm: B/A - 1 is the noise floor of the
// row. Inputs are random, never constant fills.
//
// Decode rows report K/V bytes of the call's union key range per second (the same range under
// both masks), and the useful FLOPs of the visible scores beside it: at head_dim 128 with
// T=16 the drafter's full layer carries 4 * 128 * Hq * T FLOPs per 4 KiB key, which puts the
// 31B drafter (Hq=64) at the tensor-core ridge rather than purely on the bandwidth roof. Prefill
// rows report useful FLOPs (only visible scores: 4 * D * Hq * sum of visible keys) per second.
//
// The *-draft-* geometries are the DFlash drafters' layers. They take scale 1/sqrt(128) and, on
// the full layer, `--mask block` (or both) times the bidirectional Block mask.

#include "ninferno/ops/gemma4_attention.h"

#include "core/device.h"
#include "ops/softmax_attention/gemma4/launch.h"
#include "ninferno_bench_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

using namespace ninferno;

namespace {

const double kDenseBf16TcTflops = bench::device_roofs().bf16_fp32acc_tflops;
const double kReadGBs           = bench::device_roofs().read_gbs;
constexpr std::size_t kColdBytes    = std::size_t{4} * (std::size_t{96} << 20); // 4x L2

struct Geometry {
    const char* name;
    int head_dim;
    int q_heads;
    int kv_heads;
    int window;
};

constexpr Geometry kGeometries[] = {
    {"26B-sliding", 256, 16, 8, 1024},
    {"26B-full", 512, 16, 2, 0},
    {"31B-sliding", 256, 32, 16, 1024},
    {"31B-full", 512, 32, 4, 0},
    {"26B-draft-sliding", 128, 32, 8, 2048},
    {"26B-draft-full", 128, 32, 8, 0},
    {"31B-draft-sliding", 128, 64, 8, 2048},
    {"31B-draft-full", 128, 64, 8, 0},
    // gpt-oss-20b's layers (the dense decoder routes them here; its sinks are not timed).
    {"oss-sliding", 64, 64, 8, 128},
    {"oss-full", 64, 64, 8, 0},
};

bool admits_block(const Geometry& g) { return g.head_dim == 128 && g.window == 0; }

float scale_of(const Geometry& g) {
    return g.head_dim == 128 || g.head_dim == 64 ? 1.0F / std::sqrt(static_cast<float>(g.head_dim))
                                                 : 1.0F;
}

const char* mask_name(ops::Gemma4AttentionMask mask) {
    return mask == ops::Gemma4AttentionMask::Block ? "block" : "causal";
}

struct Options {
    bool decode  = true;
    bool prefill = true;
    bool causal  = true;
    bool block   = false;
    std::string geometry;
    std::vector<int> decode_tokens{1};
    std::vector<int> contexts{1024, 4096, 16384, 32768};
    std::vector<int> prefill_tokens{512, 2048, 4096};
    std::vector<int> bases{0, 3000};
    int envelope = 0; // decode envelope max_visible_keys; 0 = the measured context
    int warmup   = 3;
    int rounds   = 15;
    bool fp8     = false; // FP8_E4M3FN planes with a scale per row (Gemma4KvLayer's FP8 storage)
};

bool g_fp8 = false;

[[noreturn]] void usage(const char* message) {
    std::fprintf(
        stderr,
        "error: %s\nusage: ninferno_gemma4_attention_bench [--mode decode|prefill|all] "
        "[--geometry 26B-sliding|26B-full|31B-sliding|31B-full|26B-draft-sliding|"
        "26B-draft-full|31B-draft-sliding|31B-draft-full|oss-sliding|oss-full] [--mask "
        "causal|block|both] "
        "[--decode-tokens 1,4] [--contexts 1024,...] [--prefill-tokens 512,...] "
        "[--bases 0,3000] [--slices auto|narrow|wide] [--kv bf16|fp8] [--envelope N] [--warmup N] [--rounds N]\n",
        message);
    std::exit(2);
}

std::vector<int> parse_list(const char* text, int minimum, int maximum) {
    std::vector<int> values;
    std::string_view rest(text);
    while (!rest.empty()) {
        const std::size_t comma = rest.find(',');
        const std::string item(rest.substr(0, comma));
        char* end         = nullptr;
        errno             = 0;
        const long parsed = std::strtol(item.c_str(), &end, 10);
        if (errno != 0 || end == item.c_str() || *end != '\0' || parsed < minimum ||
            parsed > maximum) {
            usage("invalid list value");
        }
        values.push_back(static_cast<int>(parsed));
        if (comma == std::string_view::npos) { break; }
        rest.remove_prefix(comma + 1);
    }
    return values;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&]() -> const char* {
            if (++i == argc) { usage("missing value"); }
            return argv[i];
        };
        if (arg == "--mode") {
            const std::string_view mode(next());
            options.decode  = mode == "decode" || mode == "all";
            options.prefill = mode == "prefill" || mode == "all";
            if (!options.decode && !options.prefill) { usage("--mode"); }
        } else if (arg == "--geometry") {
            options.geometry = next();
        } else if (arg == "--mask") {
            const std::string_view mask(next());
            options.causal = mask == "causal" || mask == "both";
            options.block  = mask == "block" || mask == "both";
            if (!options.causal && !options.block) { usage("--mask"); }
        } else if (arg == "--decode-tokens") {
            options.decode_tokens = parse_list(next(), 1, 16);
        } else if (arg == "--contexts") {
            options.contexts = parse_list(next(), 1, 1 << 20);
        } else if (arg == "--prefill-tokens") {
            options.prefill_tokens = parse_list(next(), 17, 1 << 16);
        } else if (arg == "--bases") {
            options.bases = parse_list(next(), 0, 1 << 20);
        } else if (arg == "--slices") {
            // The D=512 prefill tile: narrow (2x4 warps) or wide (8 warp slices), for A/B in one
            // run.
            const std::string_view slices(next());
            using ops::detail::Gemma4PrefillSlices;
            if (slices == "auto") {
                ops::detail::gemma4_attention_set_prefill_slices(Gemma4PrefillSlices::Auto);
            } else if (slices == "narrow") {
                ops::detail::gemma4_attention_set_prefill_slices(Gemma4PrefillSlices::Narrow);
            } else if (slices == "wide") {
                ops::detail::gemma4_attention_set_prefill_slices(Gemma4PrefillSlices::Wide);
            } else {
                usage("--slices");
            }
        } else if (arg == "--chain") {
            // The chained prefill of full layers: auto (on where its workspace fits), off, on.
            const std::string_view chain(next());
            using ops::detail::Gemma4PrefillChain;
            if (chain == "auto") {
                ops::detail::gemma4_attention_set_prefill_chain(Gemma4PrefillChain::Auto);
            } else if (chain == "off") {
                ops::detail::gemma4_attention_set_prefill_chain(Gemma4PrefillChain::Off);
            } else if (chain == "on") {
                ops::detail::gemma4_attention_set_prefill_chain(Gemma4PrefillChain::On);
            } else {
                usage("--chain");
            }
        } else if (arg == "--kv") {
            const std::string_view kv(next());
            if (kv != "bf16" && kv != "fp8") { usage("--kv"); }
            options.fp8 = kv == "fp8";
        } else if (arg == "--envelope") {
            options.envelope = parse_list(next(), 1, 1 << 20).front();
        } else if (arg == "--warmup") {
            options.warmup = parse_list(next(), 0, 100000).front();
        } else if (arg == "--rounds") {
            options.rounds = parse_list(next(), 1, 100000).front();
        } else {
            usage("unknown argument");
        }
    }
    return options;
}

__global__ void fill_random_bf16(__nv_bfloat16* data, std::size_t count, std::uint32_t seed,
                                 float amplitude) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B1U ^ seed;
    x ^= x >> 16;
    x *= 0x85EBCA6BU;
    x ^= x >> 13;
    x *= 0xC2B2AE35U;
    x ^= x >> 16;
    const float unit = static_cast<float>(x) * (2.0F / 4294967296.0F) - 1.0F;
    data[i]          = __float2bfloat16_rn(amplitude * unit);
}

// Random E4M3FN codes (never the NaN codes 0x7F / 0xFF) and positive row scales.
__global__ void fill_random_fp8(std::uint8_t* data, std::size_t count, std::uint32_t seed) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B1U ^ seed;
    x ^= x >> 16;
    x *= 0x85EBCA6BU;
    x ^= x >> 13;
    const std::uint32_t code = x & 0xFFU;
    data[i] = static_cast<std::uint8_t>((code & 0x7FU) == 0x7FU ? code ^ 0x01U : code);
}

__global__ void fill_random_scales(float* data, std::size_t count, std::uint32_t seed) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B1U ^ seed;
    x ^= x >> 15;
    x *= 0x2C1B3C6DU;
    x ^= x >> 12;
    data[i] = (0.5F + static_cast<float>(x) * (1.0F / 4294967296.0F)) / 448.0F;
}

DeviceBuffer random_fp8(std::size_t count, std::uint32_t seed) {
    DeviceBuffer buffer(count);
    fill_random_fp8<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        static_cast<std::uint8_t*>(buffer.p), count, seed);
    CUDA_CHECK(cudaGetLastError());
    return buffer;
}

DeviceBuffer random_scales(std::size_t count, std::uint32_t seed) {
    DeviceBuffer buffer(count * sizeof(float));
    fill_random_scales<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        static_cast<float*>(buffer.p), count, seed);
    CUDA_CHECK(cudaGetLastError());
    return buffer;
}

DeviceBuffer random_bf16(std::size_t count, std::uint32_t seed, float amplitude) {
    DeviceBuffer buffer(count * sizeof(__nv_bfloat16));
    fill_random_bf16<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        static_cast<__nv_bfloat16*>(buffer.p), count, seed, amplitude);
    CUDA_CHECK(cudaGetLastError());
    return buffer;
}

// N layers of one call site: fixed shapes, positions base..base+T-1, caches already populated.
class Layers {
public:
    Layers(const Geometry& g, int tokens, int base, std::uint32_t max_visible,
           ops::Gemma4AttentionMask mask = ops::Gemma4AttentionMask::Causal)
        : g_(g), tokens_(tokens), mask_(mask),
          capacity_(g.window > 0 ? g.window + tokens - 1 : static_cast<int>(max_visible)),
          plane_(static_cast<std::size_t>(g.head_dim) * capacity_ * g.kv_heads),
          // Small layers (the drafter at short context) need more than 64 to exceed L2.
          count_(static_cast<int>(
              std::clamp<std::size_t>((kColdBytes + 4 * plane_ - 1) / (4 * plane_), 2, 256))),
          envelope_{max_visible, tokens},
          q_(random_bf16(static_cast<std::size_t>(g.head_dim) * g.q_heads * tokens, 11,
                         6.0F / std::sqrt(static_cast<float>(g.head_dim)))),
          k_(g_fp8 ? random_fp8(plane_ * count_, 12) : random_bf16(plane_ * count_, 12, 1.0F)),
          v_(g_fp8 ? random_fp8(plane_ * count_, 13) : random_bf16(plane_ * count_, 13, 1.0F)),
          k_scale_(g_fp8 ? random_scales(plane_ / g.head_dim * count_, 14) : DeviceBuffer(4)),
          v_scale_(g_fp8 ? random_scales(plane_ / g.head_dim * count_, 15) : DeviceBuffer(4)),
          positions_(static_cast<std::size_t>(tokens) * sizeof(std::int32_t)),
          out_(static_cast<std::size_t>(g.head_dim) * g.q_heads * tokens * 2),
          workspace_(std::max<std::size_t>(1, ops::gemma4_attention_workspace_bytes(
                                                  g.head_dim, g.q_heads, g.kv_heads, envelope_))) {
        std::vector<std::int32_t> host(static_cast<std::size_t>(tokens));
        for (int t = 0; t < tokens; ++t) { host[static_cast<std::size_t>(t)] = base + t; }
        positions_.copy_from_host(host.data(), positions_.bytes);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // One public call per layer, in order.
    void launch(cudaStream_t stream) {
        const Tensor q(q_.p, DType::BF16, {g_.head_dim, g_.q_heads, tokens_});
        const Tensor positions(positions_.p, DType::I32, {tokens_});
        Tensor out(out_.p, DType::BF16, {g_.head_dim, g_.q_heads, tokens_});
        for (int layer = 0; layer < count_; ++layer) {
            ops::Gemma4KvLayer cache;
            cache.window = g_.window;
            if (g_fp8) {
                auto* k = static_cast<std::uint8_t*>(k_.p) + plane_ * layer;
                auto* v = static_cast<std::uint8_t*>(v_.p) + plane_ * layer;
                const std::size_t rows = plane_ / static_cast<std::size_t>(g_.head_dim);
                cache.k = Tensor(k, DType::FP8_E4M3FN, {g_.head_dim, capacity_, g_.kv_heads});
                cache.v = Tensor(v, DType::FP8_E4M3FN, {g_.head_dim, capacity_, g_.kv_heads});
                cache.k_scale = Tensor(static_cast<float*>(k_scale_.p) + rows * layer, DType::FP32,
                                       {capacity_, g_.kv_heads});
                cache.v_scale = Tensor(static_cast<float*>(v_scale_.p) + rows * layer, DType::FP32,
                                       {capacity_, g_.kv_heads});
            } else {
                auto* k = static_cast<__nv_bfloat16*>(k_.p) + plane_ * layer;
                auto* v = static_cast<__nv_bfloat16*>(v_.p) + plane_ * layer;
                cache.k = Tensor(k, DType::BF16, {g_.head_dim, capacity_, g_.kv_heads});
                cache.v = Tensor(v, DType::BF16, {g_.head_dim, capacity_, g_.kv_heads});
            }
            ops::gemma4_attention(q, positions, cache, scale_of(g_), envelope_, workspace_, out,
                                  stream, mask_);
        }
    }

    [[nodiscard]] int count() const noexcept { return count_; }

private:
    Geometry g_;
    int tokens_;
    ops::Gemma4AttentionMask mask_;
    int capacity_;
    std::size_t plane_;
    int count_;
    ops::Gemma4AttentionEnvelope envelope_;
    DeviceBuffer q_, k_, v_, k_scale_, v_scale_, positions_, out_;
    WorkspaceArena workspace_;
};

struct Timing {
    double us      = 0.0; // median per-call time of arm A
    double null_pc = 0.0; // median(B)/median(A) - 1, in percent
};

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

Timing measure(Layers& layers, const Options& options, cudaStream_t stream) {
    layers.launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bench::TimedGraph a;
    bench::TimedGraph b;
    a.capture(stream, [&](cudaStream_t s) { layers.launch(s); });
    b.capture(stream, [&](cudaStream_t s) { layers.launch(s); });
    for (int i = 0; i < options.warmup; ++i) {
        a.launch(stream);
        b.launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<double> sa;
    std::vector<double> sb;
    for (int round = 0; round < options.rounds; ++round) {
        sa.push_back(a.launch_timed(stream) / layers.count());
        sb.push_back(b.launch_timed(stream) / layers.count());
    }
    const double ma = median(sa);
    return {ma, (median(sb) / ma - 1.0) * 100.0};
}

bool selected(const Options& options, const Geometry& g) {
    return options.geometry.empty() || options.geometry == g.name;
}

void run_decode(const Options& options, cudaStream_t stream) {
    std::printf("decode (T <= 16, split-K); GB/s = K/V bytes of the union key range, from DRAM; "
                "TFLOP/s = visible scores only\n");
    for (const Geometry& g : kGeometries) {
        if (!selected(options, g)) { continue; }
        for (const ops::Gemma4AttentionMask mask :
             {ops::Gemma4AttentionMask::Causal, ops::Gemma4AttentionMask::Block}) {
            const bool block = mask == ops::Gemma4AttentionMask::Block;
            if (block ? !(options.block && admits_block(g)) : !options.causal) { continue; }
            for (const int tokens : options.decode_tokens) {
                for (const int context : options.contexts) {
                    if (context < tokens) { continue; }
                    const int base = context - tokens; // last query sees `context` keys
                    const auto max_visible =
                        static_cast<std::uint32_t>(std::max(options.envelope, context));
                    Layers layers(g, tokens, base, max_visible, mask);
                    const Timing timing = measure(layers, options, stream);
                    const int keys =
                        g.window > 0 ? std::min(context, g.window + tokens - 1) : context;
                    const double bytes =
                        2.0 * 2.0 * static_cast<double>(g.head_dim) * g.kv_heads * keys;
                    const double gbps = bytes / (timing.us * 1.0e-6) / 1.0e9;
                    double visible    = 0.0;
                    for (int t = 0; t < tokens; ++t) {
                        const int p = block ? context - 1 : base + t;
                        visible += g.window > 0 ? std::min(p + 1, g.window) : p + 1;
                    }
                    const double tflops =
                        4.0 * g.head_dim * g.q_heads * visible / (timing.us * 1.0e-6) / 1.0e12;
                    std::printf("  %-17s %-6s T=%2d ctx=%6d env=%6u keys=%6d layers=%3d  %8.2f us "
                                " %7.1f GB/s (%5.1f%% of %.0f, %5.1f%% of %.0f read)  %5.1f "
                                "TFLOP/s  null arm %+5.2f%%\n",
                                g.name, mask_name(mask), tokens, context, max_visible, keys,
                                layers.count(), timing.us, gbps, gbps / bench::device_roofs().dram_gbs * 100.0,
                                bench::device_roofs().dram_gbs, gbps / kReadGBs * 100.0, kReadGBs, tflops,
                                timing.null_pc);
                }
            }
        }
    }
}

void run_prefill(const Options& options, cudaStream_t stream) {
    std::printf("prefill (T > 16); TFLOP/s of useful work (visible scores only)\n");
    for (const Geometry& g : kGeometries) {
        if (!selected(options, g)) { continue; }
        for (const int tokens : options.prefill_tokens) {
            for (const int base : options.bases) {
                const auto max_visible = static_cast<std::uint32_t>(base + tokens);
                Layers layers(g, tokens, base, max_visible);
                const Timing timing = measure(layers, options, stream);
                double visible      = 0.0;
                for (int t = 0; t < tokens; ++t) {
                    const int p = base + t;
                    visible += g.window > 0 ? std::min(p + 1, g.window) : p + 1;
                }
                const double flops  = 4.0 * g.head_dim * g.q_heads * visible;
                const double tflops = flops / (timing.us * 1.0e-6) / 1.0e12;
                std::printf("  %-12s T=%5d base=%5d layers=%2d  %9.2f us  %6.1f TFLOP/s (%4.1f%% "
                            "of %.1f)  null arm %+5.2f%%\n",
                            g.name, tokens, base, layers.count(), timing.us, tflops,
                            tflops / kDenseBf16TcTflops * 100.0, kDenseBf16TcTflops,
                            timing.null_pc);
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse(argc, argv);
        g_fp8                 = options.fp8;
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (options.decode) { run_decode(options, stream); }
        if (options.prefill) { run_prefill(options, stream); }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninferno_gemma4_attention_bench: %s\n", error.what());
        return 1;
    }
}
