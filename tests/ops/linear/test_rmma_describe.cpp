// The register-MMA kernel description of the selector (rmma_resources, generic_rmma_describe.cpp:
// the committed kernel facts, no CUDA) equals the launch's own (rmma_resources_by_launch: the real
// launch code run in describe mode), for every format, schedule, epilogue and width on every board
// that has facts (#620). The host description decides the kernel variant (add-sums, latency, wide)
// from a table of the decoders' traits (generic_rmma_decoders.h); a decoder whose traits change, or
// a rule changed in one place only, fails here with the call it differs on. No GPU is needed: both
// read the facts of a substituted board.
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/selector_facts.h"
#include "ninferno/ops/prefill_routes.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>

namespace {

using namespace ninferno;
using namespace ninferno::test;
namespace generic = ninferno::ops::detail::generic;

constexpr QType kFormats[] = {
    QType::BF16,
    QType::Q4_G64_FP16,
    QType::Q5_G64_FP16,
    QType::Q6_G64_FP16,
    QType::Q8_G32_FP16,
    QType::FP8_E4M3FN_ROW_BF16,
    QType::NVFP4,
    QType::MXFP4,
    QType::U4_G128_ZP_FP16,
    QType::GGML_Q4_K,
    QType::GGML_Q5_K,
    QType::GGML_Q6_K,
    QType::GGML_Q3_K,
    QType::GGML_Q2_K,
    QType::GGML_IQ4_XS,
    QType::GGML_IQ3_XXS,
    QType::GGML_IQ3_S,
    QType::GGML_IQ2_S,
    QType::GGML_Q4_0,
    QType::GGML_IQ4_NL,
    QType::FP8_E4M3FN_ROW_F32,
    QType::FP8_E4M3FN_B128_F32,
    QType::U4_G32_ZP_FP16,
    QType::U4_G64_ZP_FP16,
    QType::Q3_G128_FP16,
    QType::Q3_G64_FP16,
    QType::GGML_Q4_1,
    QType::GGML_Q5_0,
    QType::GGML_Q5_1,
    QType::GGML_IQ2_XXS,
    QType::GGML_IQ2_XS,
    QType::GGML_IQ1_S,
    QType::GGML_IQ1_M,
    QType::GGML_TQ1_0,
    QType::GGML_TQ2_0,
    QType::GGML_Q1_0,
    QType::GGML_Q2_0,
};

struct Shape {
    std::int32_t n, k;
};

// 4096 and 5120 are multiples of 128 (the quad U4Zp decoders), 2112 and 2880 are not (Gemma 4's and
// gpt-oss's hidden sizes).
constexpr Shape kShapes[]        = {{1024, 4096}, {6144, 5120}, {2048, 2112}, {2880, 2880}};
constexpr std::int32_t kTokens[] = {1, 4, 8, 16, 64, 512};

bool same(const generic::KernelResources& a, const generic::KernelResources& b) {
    return a.registers == b.registers && a.local_bytes == b.local_bytes &&
           a.static_smem == b.static_smem && a.dynamic_smem == b.dynamic_smem &&
           a.threads == b.threads && a.max_threads_per_block == b.max_threads_per_block &&
           a.ptx_version == b.ptx_version && a.binary_version == b.binary_version &&
           a.grid_x == b.grid_x && a.grid_y == b.grid_y && a.grid_z == b.grid_z &&
           a.latency_variant == b.latency_variant;
}

std::string text(const std::optional<generic::KernelResources>& r) {
    if (!r) return "none";
    return "regs " + std::to_string(r->registers) + " local " + std::to_string(r->local_bytes) +
           " static " + std::to_string(r->static_smem) + " dyn " + std::to_string(r->dynamic_smem) +
           " threads " + std::to_string(r->threads) + " grid " + std::to_string(r->grid_x) + "x" +
           std::to_string(r->grid_y) + "x" + std::to_string(r->grid_z) +
           (r->latency_variant ? " latency" : "");
}

} // namespace

int main() {
    int failures        = 0;
    long compared       = 0;
    long described      = 0;
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    for (const QType qtype : kFormats) {
        for (const Shape shape : kShapes) {
            quantized_weight::PatternedWeightOptions options;
            if (qtype == QType::NVFP4)
                options.weight_scale_divisor = options.input_scale_divisor = 1.0F;
            std::optional<quantized_weight::PackedWeight> host;
            try {
                host.emplace(
                    quantized_weight::make_patterned_weight(qtype, shape.n, shape.k, 7U, options));
            } catch (const std::invalid_argument&) { continue; }
            const Weight w = host->device_weight(payload);
            if (!generic::supports(w)) continue;
            for (const auto& board : generic::kFactsBoards) {
                const DeviceClass d = generic::board_device_class(board);
                const ScopedDeviceClass scoped(d);
                // The call's activation form (the lane-local and add-sums choices) both ways.
                for (const bool invariant : {false, true}) {
                    std::optional<ops::ScopedInvariantRoutes> routes;
                    if (invariant) routes.emplace();
                    for (const generic::Epilogue epilogue :
                         {generic::Epilogue::Store, generic::Epilogue::AddResidual,
                          generic::Epilogue::SwiGlu}) {
                        for (int schedule = 0; schedule < generic::rmma_schedule_count();
                             ++schedule) {
                            for (const std::int32_t tokens : kTokens) {
                                std::optional<generic::KernelResources> host_side, launched;
                                std::string host_error, launch_error;
                                try {
                                    host_side =
                                        generic::rmma_resources(schedule, w, epilogue, tokens);
                                } catch (const std::exception& e) { host_error = e.what(); }
                                try {
                                    launched = generic::rmma_resources_by_launch(schedule, w,
                                                                                 epilogue, tokens);
                                } catch (const std::exception& e) { launch_error = e.what(); }
                                ++compared;
                                if (host_side) ++described;
                                const bool equal = host_error.empty() == launch_error.empty() &&
                                                   host_side.has_value() == launched.has_value() &&
                                                   (!host_side || same(*host_side, *launched));
                                if (equal) continue;
                                if (++failures <= 20) {
                                    std::cerr << board.name << " " << generic::qtype_name(qtype)
                                              << " " << shape.n << "x" << shape.k << " "
                                              << generic::epilogue_name(epilogue) << " schedule "
                                              << schedule << " T=" << tokens
                                              << (invariant ? " invariant" : "")
                                              << ":
                                        host : "
                                              << (host_error.empty() ? text(host_side)
                                                                     : "threw " + host_error)
                                              << "
                                               launch : "
                                              << (launch_error.empty() ? text(launched)
                                                                       : "threw " + launch_error)
                                              << "
                                                 ";
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << compared << " calls compared, " << described << " described, " << failures
              << " differ\n";
    if (described == 0) {
        std::cerr << "nothing was described: the test reads no facts\n";
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
