// Route selection on GPU classes this machine does not have (code review 30.09, item 5). For each
// board below a substituted DeviceClass (core/device_class.h) stands in for the device while the
// generic path chooses routes for every format it reads, each policy, three epilogues, three shapes
// and decode, verify and prefill widths. Every choice must be one route_admits accepts and must
// need no feature the board lacks (route_requires); a route whose kernels this build lacks
// (W4A4 and the NVFP4 tiled routes outside 120a) is never chosen, whatever the board; and the
// route table in effect is the nearest measured board's of its instruction set (board_distance,
// route_table.cpp), the board's own where routes/ has it.
//
// The route table and several heuristics cache what they read of the device on first use, so one
// process runs one board: without arguments the test runs itself once per board.
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/units.h"
#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/route_table.h"
#include "ops/common/native_nvfp4.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_table.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::test;
namespace generic = ninferno::ops::detail::generic;
using ops::LinearPolicy;

struct TestBoard {
    const char* name;
    std::uint32_t pci_device_id; // 0: unknown to the id table, the chip follows from the SM count
    int major, minor, sms;
    std::size_t l2_mib, memory_gib;
    const char* table; // the class of the nearest route table in routes/ ("" when none is near)
};

constexpr TestBoard kBoards[] = {
    {"rtx5090", 0x2B85, 12, 0, 170, 96, 32, "gb202"},
    {"rtx5080", 0x2C02, 12, 0, 84, 64, 16, "gb203"},
    {"rtx4090", 0x2684, 8, 9, 128, 72, 24, "ad102"},
    {"rtx4070", 0x2786, 8, 9, 46, 36, 12, "ad106"}, // no ad104 table: the RTX 4060 Ti's, 0.5 away
    {"rtx3090", 0x2204, 8, 6, 82, 6, 24, "ga102"},
    {"rtx2080ti", 0x1E07, 7, 5, 68, 5, 11, "tu102"},
    {"unknown-sm120-48sm", 0, 12, 0, 48, 32, 12, "gb206"}, // the RTX 5060 Ti's, the nearest
};

constexpr QType kFormats[] = {
    QType::Q4_G64_FP16,
    QType::Q5_G64_FP16,
    QType::Q6_G64_FP16,
    QType::Q8_G32_FP16,
    QType::FP8_E4M3FN_ROW_BF16,
    QType::FP8_E4M3FN_ROW_F32,
    QType::FP8_E4M3FN_B128_F32,
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
    QType::GGML_IQ2_XXS,
    QType::GGML_IQ2_XS,
    QType::GGML_IQ2_S,
    QType::GGML_Q4_0,
    QType::GGML_Q5_0,
    QType::GGML_IQ4_NL,
};

struct Shape {
    std::int32_t n, k;
};

constexpr Shape kShapes[]        = {{1024, 2048}, {5120, 5120}, {1024, 12288}};
constexpr std::int32_t kTokens[] = {1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 64, 128, 512, 2048};

int run_board(const TestBoard& b) {
    const ScopedDeviceClass scoped(synthetic_device_class(
        b.pci_device_id, b.major, b.minor, b.sms, b.l2_mib * kMiB, b.memory_gib * 1024 * kMiB));
    const DeviceClass& d = device_class();
    int failures = 0, checked = 0;
    const auto fail = [&](const std::string& what) {
        if (++failures <= 20) std::cerr << b.name << " (" << d.id() << "): " << what << '\n';
    };
    // Weights are never read here: an aligned address stands in for the payload.
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    for (const QType qtype : kFormats) {
        for (const Shape shape : kShapes) {
            quantized_weight::PatternedWeightOptions options;
            if (qtype == QType::NVFP4)
                options.weight_scale_divisor = options.input_scale_divisor = 1.0F;
            const auto host =
                quantized_weight::make_patterned_weight(qtype, shape.n, shape.k, 7U, options);
            const Weight w = host.device_weight(payload);
            if (!generic::supports(w)) continue;
            for (const LinearPolicy policy :
                 {LinearPolicy::A16Only, LinearPolicy::AllowA8, LinearPolicy::AllowA4}) {
                if (policy == LinearPolicy::AllowA4 && qtype != QType::NVFP4) continue;
                for (const generic::Epilogue epilogue :
                     {generic::Epilogue::Store, generic::Epilogue::AddResidual,
                      generic::Epilogue::SwiGlu}) {
                    const std::int32_t rows =
                        epilogue == generic::Epilogue::SwiGlu ? shape.n / 2 : shape.n;
                    const generic::Problem problem{qtype, policy, rows, shape.k, epilogue};
                    for (const std::int32_t t : kTokens) {
                        for (const bool prefill : {false, true}) {
                            if (prefill && t < 17) continue;
                            const std::string where =
                                std::string(generic::qtype_name(qtype)) + " " +
                                std::to_string(rows) + "x" + std::to_string(shape.k) +
                                " T=" + std::to_string(t) + (prefill ? " prefill" : "") +
                                " policy=" + std::to_string(static_cast<int>(policy)) +
                                " epilogue=" + std::to_string(static_cast<int>(epilogue));
                            generic::Route route;
                            try {
                                route = generic::select_route(problem, w, t, prefill);
                            } catch (const std::exception& error) {
                                fail(where + ": select_route threw: " + error.what());
                                continue;
                            }
                            ++checked;
                            const std::string name = generic::route_name(route);
                            if (!generic::route_admits(route, w, policy)) {
                                fail(where + ": chose " + name + ", which route_admits refuses");
                            }
                            if (!has_features(d.features, generic::route_requires(route))) {
                                fail(where + ": chose " + name + ", which needs a feature " +
                                     "the board lacks");
                            }
                            if (!ops::detail::kNativeNvfp4 &&
                                (route.family == generic::Family::W4a4Mma ||
                                 route.family == generic::Family::Nvfp4TiledMma)) {
                                fail(where + ": chose " + name + ", absent from this build");
                            }
                        }
                    }
                }
            }
        }
    }
    const auto status     = ops::route_table_status();
    const bool want_table = std::string_view(b.table) != "";
    if (status.loaded != want_table) {
        fail(std::string("route table ") + (status.loaded ? "loaded" : "not loaded") + " (" +
             status.message + "), expected " + (want_table ? b.table : "none"));
    }
    if (status.loaded && status.gpu_class != b.table) {
        fail("route table class " + status.gpu_class + ", expected the nearest board's " + b.table);
    }
    if (checked < 1000) fail("only " + std::to_string(checked) + " choices checked");
    std::cout << b.name << " (" << d.id() << ", " << d.arch() << "): " << checked
              << " route choices, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--board") {
            for (const TestBoard& b : kBoards) {
                if (std::string_view(argv[2]) == b.name) return run_board(b);
            }
            std::cerr << "unknown board " << argv[2] << '\n';
            return 2;
        }
        int failed = 0;
        for (const TestBoard& b : kBoards) {
            const std::string command = std::string(argv[0]) + " --board " + b.name;
            if (std::system(command.c_str()) != 0) ++failed;
        }
        if (failed != 0) {
            std::cerr << "route classes: " << failed << " boards failed\n";
            return 1;
        }
        std::cout << "route classes: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "route classes: " << error.what() << '\n';
        return 1;
    }
}
