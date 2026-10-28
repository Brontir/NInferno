// The generic projection path's choices, frozen: for each board below (a substituted DeviceClass
// with the board's SMs, clocks, memory bus, L2 and per-SM limits, core/device_class.h) the route and
// the workspace it needs for every format, policy, epilogue, shape and width, under the speed,
// prefill and invariant profiles. The dump is compared with selector_golden/<board>.txt; a change
// of the selector shows there line by line, and a refactor that means none must leave every file
// as it was. Regenerate with --update (writes the files) after a change that means to move routes.
//
// Without arguments the test runs itself once per board, then every board again in one process:
// the selector's per-device caches keep a substituted board's answers apart by its content, so the
// two must agree. Each choice is also asked of the pure query (route_query.h); a difference is
// written into the dump.
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/units.h"
#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::test;
namespace generic = ninferno::ops::detail::generic;
using ops::LinearPolicy;

// Datasheet numbers as the CUDA device query reports them (boost SM clock, memory clock = data
// rate / 2, bus width).
struct GoldenBoard {
    const char* name;
    std::uint32_t pci_device_id;
    int major, minor, sms;
    std::size_t l2_mib, memory_gib;
    int sm_clock_mhz, memory_clock_mhz, bus_bits;
};

constexpr GoldenBoard kBoards[] = {
    {"rtx5090", 0x2B85, 12, 0, 170, 96, 32, 2407, 14001, 512},
    {"rtx5080", 0x2C02, 12, 0, 84, 64, 16, 2617, 15001, 256},
    {"rtx5070ti", 0x2C05, 12, 0, 70, 48, 16, 2452, 14001, 256},
    {"rtx5060ti", 0x2D04, 12, 0, 36, 32, 16, 2572, 14001, 128},
    {"rtx4090", 0x2684, 8, 9, 128, 72, 24, 2520, 10501, 384},
    {"rtx4060ti", 0x2805, 8, 9, 34, 32, 16, 2535, 9001, 128},
    {"l4", 0x27B8, 8, 9, 58, 48, 24, 2040, 6251, 192},
    {"rtx3090", 0x2204, 8, 6, 82, 6, 24, 1695, 9751, 384},
    {"a10", 0x2236, 8, 6, 72, 6, 24, 1695, 6251, 384},
    {"a100", 0x20B0, 8, 0, 108, 40, 40, 1410, 1215, 5120},
    {"rtx2080ti", 0x1E07, 7, 5, 68, 5, 11, 1545, 7000, 352},
    {"v100", 0x1DB5, 7, 0, 80, 6, 32, 1530, 877, 4096},
    // Not targets, but the rows where spellings of one generation fact used to disagree.
    {"h100-sxm", 0x2330, 9, 0, 132, 50, 80, 1980, 2619, 5120},
    {"b300", 0, 10, 3, 148, 126, 268, 2032, 3996, 8192},
    {"gb10-spark", 0, 12, 1, 48, 24, 128, 2418, 4266, 256},
    {"unknown-sm120-48sm", 0, 12, 0, 48, 32, 12, 2500, 14001, 192},
};

constexpr QType kFormats[] = {
    QType::BF16,           QType::Q4_G64_FP16,         QType::Q5_G64_FP16,
    QType::Q6_G64_FP16,    QType::Q8_G32_FP16,         QType::FP8_E4M3FN_ROW_BF16,
    QType::NVFP4,          QType::MXFP4,               QType::U4_G128_ZP_FP16,
    QType::GGML_Q4_K,      QType::GGML_Q5_K,           QType::GGML_Q6_K,
    QType::GGML_Q3_K,      QType::GGML_Q2_K,           QType::GGML_IQ4_XS,
    QType::GGML_IQ3_XXS,   QType::GGML_IQ3_S,          QType::GGML_IQ2_S,
    QType::GGML_Q4_0,      QType::GGML_IQ4_NL,
};

struct Shape {
    std::int32_t n, k;
};

constexpr Shape kShapes[] = {{256, 2048},  {1024, 4096},  {2048, 1024}, {4096, 4096},
                             {6144, 5120}, {17408, 5120}, {5120, 17408}, {28672, 8192}};
constexpr std::int32_t kTokens[] = {1,  2,  3,  4,  5,   8,   12,  16,  17,   24,  32,
                                    48, 64, 96, 128, 192, 256, 512, 1024, 2048};

enum class Profile { Speed, Prefill, Invariant };

const char* profile_name(Profile p) {
    return p == Profile::Speed ? "speed" : p == Profile::Prefill ? "prefill" : "invariant";
}

// The route set a build compiles decides what a board can pick, so each set keeps its own files:
// sm_70 (FMA-emulated MMA, 96 KiB), sm_75 (64 KiB: no producer-warp W4A8 tiles), the full builds
// (120a / 121a / 120f, with NVFP4 and the cluster K-split schedules) and every other build (base;
// sm_80's deeper staged BF16 tiles change nothing here, the goldens pack no BF16 weights).
constexpr const char* kBuildSet =
#if NINFERNO_BUILD_ARCH == 70
    "sm70";
#elif NINFERNO_BUILD_ARCH == 75
    "sm75";
#elif defined(NINFERNO_WITHOUT_NVFP4)
    "base";
#else
    "full";
#endif

std::filesystem::path golden_path(const GoldenBoard& b) {
    return std::filesystem::path(NINFERNO_SELECTOR_GOLDEN_DIR) / kBuildSet /
           (std::string(b.name) + ".txt");
}

// One board's choices, a line per (format, policy, epilogue, shape, profile) and run of widths
// with the same route and workspace.
std::string dump_board(const GoldenBoard& b) {
    DeviceClass d = synthetic_device_class(b.pci_device_id, b.major, b.minor, b.sms,
                                           b.l2_mib * kMiB, b.memory_gib * 1024 * kMiB);
    d.sm_clock_khz     = b.sm_clock_mhz * 1000;
    d.memory_clock_khz = b.memory_clock_mhz * 1000;
    d.memory_bus_bits  = b.bus_bits;
    d.facts            = machine_facts_of(d); // as the query derives them from these values
    const ScopedDeviceClass scoped(d);
    std::ostringstream out;
    out << "# " << b.name << " " << device_class().id() << " " << device_class().arch() << '\n';
    // Weights are never read here: an aligned address stands in for the payload.
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    for (const QType qtype : kFormats) {
        for (const Shape shape : kShapes) {
            quantized_weight::PatternedWeightOptions options;
            if (qtype == QType::NVFP4)
                options.weight_scale_divisor = options.input_scale_divisor = 1.0F;
            // A format the test packer cannot build is named in the file, not silently dropped.
            std::optional<quantized_weight::PackedWeight> host;
            try {
                host.emplace(
                    quantized_weight::make_patterned_weight(qtype, shape.n, shape.k, 7U, options));
            } catch (const std::invalid_argument& error) {
                out << generic::qtype_name(qtype) << " " << shape.n << "x" << shape.k
                    << " not packed: " << error.what() << '\n';
                continue;
            }
            const Weight w = host->device_weight(payload);
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
                    for (const Profile profile :
                         {Profile::Speed, Profile::Prefill, Profile::Invariant}) {
                        std::optional<ops::ScopedInvariantRoutes> invariant;
                        if (profile == Profile::Invariant) invariant.emplace();
                        std::string last;
                        std::int32_t first_t = 0, last_t = 0;
                        const auto flush = [&] {
                            if (last.empty()) return;
                            out << generic::qtype_name(qtype) << " p"
                                << static_cast<int>(policy) << " "
                                << generic::epilogue_name(epilogue) << " " << rows << "x"
                                << shape.k << " " << profile_name(profile) << " T=" << first_t;
                            if (last_t != first_t) out << ".." << last_t;
                            out << " " << last << '\n';
                        };
                        for (const std::int32_t t : kTokens) {
                            const bool prefill = profile == Profile::Prefill;
                            if (prefill && t < 17) continue;
                            std::string what;
                            try {
                                const generic::Route r =
                                    generic::select_route(problem, w, t, prefill);
                                what = generic::route_name(r) + " ws=" +
                                       std::to_string(generic::route_workspace_bytes(r, shape.k, t));
                                // The pure query (a weight described by format and shape, the
                                // board by its class) answers as the selector does; a difference
                                // shows in the dump and so fails the comparison.
                                const auto q = generic::route_shape(d, qtype, policy, epilogue,
                                                                    rows, shape.k, t, prefill);
                                if (!q || !(q->route == r)) {
                                    what += q ? " query=" + generic::route_name(q->route)
                                              : std::string(" query=none");
                                }
                            } catch (const std::exception& error) {
                                what = std::string("threw: ") + error.what();
                            }
                            if (what != last) {
                                flush();
                                last    = what;
                                first_t = t;
                            }
                            last_t = t;
                        }
                        flush();
                    }
                }
            }
        }
    }
    return out.str();
}

int run_board(const GoldenBoard& b, bool update) {
    const std::string now = dump_board(b);
    const auto path       = golden_path(b);
    if (update) {
        // The directory may not exist yet in a fresh checkout; a write that fails is a failure.
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path, std::ios::binary);
        file << now;
        if (!file.good()) {
            std::cerr << b.name << ": cannot write " << path.string() << '\n';
            return 1;
        }
        std::cout << b.name << ": wrote " << path.string() << '\n';
        return 0;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << b.name << ": no golden file " << path.string() << " (run with --update)\n";
        return 1;
    }
    std::stringstream was;
    was << in.rdbuf();
    if (was.str() == now) {
        std::cout << b.name << ": choices as the golden file\n";
        return 0;
    }
    // The first lines that differ, from both sides.
    std::istringstream a(was.str()), c(now);
    std::string la, lc;
    int shown = 0, line = 0;
    while (shown < 20) {
        const bool ga = static_cast<bool>(std::getline(a, la));
        const bool gc = static_cast<bool>(std::getline(c, lc));
        if (!ga && !gc) break;
        ++line;
        if (!ga) la = "<end>";
        if (!gc) lc = "<end>";
        if (la != lc) {
            std::cerr << b.name << ":" << line << "\n  golden: " << la << "\n  now:    " << lc << '\n';
            ++shown;
        }
    }
    std::cerr << b.name << ": choices differ from " << path.string() << '\n';
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool update = false;
        std::string board;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--update") update = true;
            if (arg == "--board" && i + 1 < argc) board = argv[++i];
        }
        if (!board.empty()) {
            for (const GoldenBoard& b : kBoards) {
                if (board == b.name) return run_board(b, update);
            }
            std::cerr << "unknown board " << board << '\n';
            return 2;
        }
        int failed = 0;
        for (const GoldenBoard& b : kBoards) {
            const std::string command = std::string(argv[0]) + (update ? " --update" : "") +
                                        " --board " + b.name;
            if (std::system(command.c_str()) != 0) ++failed;
        }
        // Every board again in this one process, one after the other: the selector's per-device
        // caches and route tables keep a substitute's answers apart by its content
        // (core/per_device.h), so a board answers as it does alone (route_query.h relies on it).
        if (!update) {
            for (const GoldenBoard& b : kBoards) {
                if (run_board(b, false) != 0) ++failed;
            }
        }
        if (failed != 0) {
            std::cerr << "selector golden: " << failed << " boards differ\n";
            return 1;
        }
        std::cout << "selector golden: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "selector golden: " << error.what() << '\n';
        return 1;
    }
}
