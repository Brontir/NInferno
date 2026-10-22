// ninferno_ggml_mmq_prefill_bench: ggml's matrix kernel at prefill widths in its two schedules,
// stream-k (the speed profile) and one CTA per tile (the prefill profile's token-invariant form,
// Route{GgmlMmq, kMmqTiles}), on a format-correct random weight with the L2 flushed before each
// sample; the median of --samples.
//
//   ninferno_ggml_mmq_prefill_bench --format q4k --epilogue swiglu --n 17408 --k 5120
//                                   [--tokens 512,1024,2048,4096] [--samples 9]
//
// --format is a qtype_name (q4k, q6k, iq3xxs, ...); --n is the output rows (SwiGLU: the gate
// rows; the weight holds gate and up). One line per width: both times and tiles / stream-k.
//
// --raw S1,S2,... times the kernel alone (quantizer and product, no epilogue; store only) for
// each K split S of the tiles at each slice of 128..512 tokens, against stream-k's 512: the
// data behind generic's mmq_k_splits and mmq_tiles_slice rules.
#include "core/arena.h"
#include "core/device.h"
#include "core/device_class.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include "ops/generic/route_table.h"
#include "random_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninferno;
namespace generic  = ninferno::ops::detail::generic;
namespace ggml_mmq = ninferno::ops::ggml_mmq;

QType parse_format(const std::string& s) {
    for (int v = 0; v <= 0xff; ++v) {
        const auto q = static_cast<QType>(v);
        if (s == generic::qtype_name(q)) return q;
    }
    throw std::invalid_argument("unknown format " + s);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> opt;
        for (int i = 1; i + 1 < argc; i += 2) opt[argv[i]] = argv[i + 1];
        if (!opt.count("--format") || !opt.count("--n") || !opt.count("--k")) {
            std::cerr << "usage: ninferno_ggml_mmq_prefill_bench --format F --n N --k K "
                         "[--epilogue store|swiglu] [--tokens T,T,...] [--samples S]\n";
            return 2;
        }
        const QType qtype    = parse_format(opt["--format"]);
        const std::int32_t n = std::stoi(opt["--n"]);
        const std::int32_t k = std::stoi(opt["--k"]);
        const bool swiglu    = opt.count("--epilogue") && opt["--epilogue"] == "swiglu";
        const int samples    = opt.count("--samples") ? std::stoi(opt["--samples"]) : 9;
        std::vector<std::int32_t> tokens{512, 1024, 2048, 4096};
        if (opt.count("--tokens")) {
            tokens.clear();
            std::stringstream list(opt["--tokens"]);
            for (std::string v; std::getline(list, v, ',');) tokens.push_back(std::stoi(v));
        }
        const std::int32_t t_max = *std::max_element(tokens.begin(), tokens.end());
        const std::int32_t rows  = swiglu ? 2 * n : n;

        const auto& device    = device_class();
        bench::RandomWeight w = bench::random_weight(qtype, rows, k, 7U + n + k);
        const generic::Route stream_k{generic::Family::GgmlMmq, generic::kMmqStreamK};
        const generic::Route tiles{generic::Family::GgmlMmq, generic::kMmqTiles};
        if (!generic::route_admits(tiles, w.weight, ops::LinearPolicy::AllowA8)) {
            std::cerr << "ggml_mmq does not take this weight on this device\n";
            return 1;
        }
        DeviceBuffer x(static_cast<std::size_t>(k) * t_max * 2);
        {
            std::vector<std::uint16_t> host(static_cast<std::size_t>(k) * t_max);
            std::mt19937 rng(11);
            std::normal_distribution<float> nd(0.0F, 1.0F);
            for (auto& v : host) {
                const float f = nd(rng);
                std::uint32_t bits;
                std::memcpy(&bits, &f, 4);
                v = static_cast<std::uint16_t>(bits >> 16);
            }
            x.copy_from_host(host.data(), host.size() * 2);
        }
        DeviceBuffer y(static_cast<std::size_t>(n) * t_max * 2);
        DeviceArena workspace(generic::ggml_mmq_workspace_bytes(k, rows, 1024) + 256);
        DeviceBuffer flush(std::max<std::size_t>(device.l2_bytes * 2, 1 << 20));
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        const auto measure = [&](generic::Route r, std::int32_t t) {
            Tensor tx(x.p, DType::BF16, {k, t});
            Tensor ty(y.p, DType::BF16, {n, t});
            generic::Call call;
            call.x               = &tx;
            call.weight          = &w.weight;
            call.rows            = n;
            call.up_offset       = swiglu ? n : 0;
            call.epilogue        = swiglu ? generic::Epilogue::SwiGlu : generic::Epilogue::Store;
            call.out[0]          = &ty;
            call.segment_rows[0] = n;
            generic::run_route(r, call, &workspace, nullptr);
            std::vector<float> ms;
            for (int s = 0; s < samples; ++s) {
                CUDA_CHECK(cudaMemsetAsync(flush.p, s & 0xff, flush.bytes, nullptr));
                CUDA_CHECK(cudaEventRecord(start, nullptr));
                generic::run_route(r, call, &workspace, nullptr);
                CUDA_CHECK(cudaEventRecord(stop, nullptr));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float v = 0;
                CUDA_CHECK(cudaEventElapsedTime(&v, start, stop));
                ms.push_back(v);
            }
            std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());
            return static_cast<double>(ms[ms.size() / 2]) * 1000.0;
        };
        if (opt.count("--raw")) {
            ggml_mmq::Type type;
            switch (qtype) {
            case QType::GGML_Q4_K:
                type = ggml_mmq::Type::Q4_K;
                break;
            case QType::GGML_Q5_K:
                type = ggml_mmq::Type::Q5_K;
                break;
            case QType::GGML_Q6_K:
                type = ggml_mmq::Type::Q6_K;
                break;
            case QType::GGML_IQ4_XS:
                type = ggml_mmq::Type::IQ4_XS;
                break;
            case QType::GGML_IQ3_XXS:
                type = ggml_mmq::Type::IQ3_XXS;
                break;
            default:
                throw std::invalid_argument("--raw: q4k q5k q6k iq4xs iq3xxs only");
            }
            std::vector<int> splits;
            std::stringstream list(opt["--raw"]);
            for (std::string v; std::getline(list, v, ',');) splits.push_back(std::stoi(v));
            const auto args = generic::weight_args(w.weight);
            const ggml_mmq::Planes planes{args.codes, args.high, args.scales};
            const int bpr = k / 256;
            DeviceBuffer act(ggml_mmq::activation_bytes(k, 512));
            DeviceBuffer plane(static_cast<std::size_t>(n) * 512 * sizeof(float));
            std::size_t fixup_bytes = ggml_mmq::fixup_bound(512);
            for (const int sp : splits) {
                fixup_bytes = std::max(fixup_bytes, ggml_mmq::tiles_fixup_bytes(type, n, 512, sp));
            }
            DeviceBuffer fixup(fixup_bytes);
            const auto run = [&](std::int32_t t, int slice, ggml_mmq::Split split, int sp) {
                for (std::int32_t t0 = 0; t0 < t; t0 += slice) {
                    const int part = std::min<std::int32_t>(slice, t - t0);
                    ggml_mmq::quantize_activation(
                        type, static_cast<const __nv_bfloat16*>(x.p) + std::size_t(t0) * k, k, k,
                        part, act.p, nullptr);
                    ggml_mmq::product(type, planes, bpr, n, k, act.p, part,
                                      static_cast<float*>(plane.p), n, fixup.p, nullptr, split, sp);
                }
            };
            const auto time = [&](auto&& body) {
                body();
                std::vector<float> ms;
                for (int s = 0; s < samples; ++s) {
                    CUDA_CHECK(cudaMemsetAsync(flush.p, s & 0xff, flush.bytes, nullptr));
                    CUDA_CHECK(cudaEventRecord(start, nullptr));
                    body();
                    CUDA_CHECK(cudaEventRecord(stop, nullptr));
                    CUDA_CHECK(cudaEventSynchronize(stop));
                    float v = 0;
                    CUDA_CHECK(cudaEventElapsedTime(&v, start, stop));
                    ms.push_back(v);
                }
                std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());
                return static_cast<double>(ms[ms.size() / 2]) * 1000.0;
            };
            std::printf("raw %s %dx%d slots %d\n", generic::qtype_name(qtype), n, k,
                        ggml_mmq::tile_slots(type));
            for (const std::int32_t t : tokens) {
                const double base = time([&] { run(t, 512, ggml_mmq::Split::StreamK, 1); });
                std::printf("T=%5d stream-k %8.1f us |", t, base);
                for (const int sp : splits) {
                    for (int c = 1; c <= 4; ++c) {
                        const double us =
                            time([&] { run(t, 128 * c, ggml_mmq::Split::Tiles, sp); });
                        std::printf(" s%d c%d %.3f", sp, c, us / base);
                    }
                    std::printf(" |");
                }
                std::printf("\n");
            }
            return 0;
        }
        std::printf("device %s, %s %s %dx%d, median of %d\n", device.name.c_str(),
                    generic::qtype_name(qtype), swiglu ? "swiglu" : "store", n, k, samples);
        for (const std::int32_t t : tokens) {
            const double a = measure(stream_k, t);
            const double b = measure(tiles, t);
            std::printf("T=%5d stream-k %9.1f us  tiles %9.1f us  tiles/stream-k %.3f\n", t, a, b,
                        b / a);
        }
        if (generic::ggml_mmq_fallback_count() != 0) {
            std::cerr << "ggml_mmq ran A16 (workspace short)\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
