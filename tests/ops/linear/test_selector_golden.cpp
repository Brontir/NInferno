// The generic projection path's choices, frozen: for each board below (a substituted DeviceClass
// with the board's SMs, clocks, memory bus, L2 and per-SM limits, core/device_class.h) the route
// and the workspace it needs for every format, policy, epilogue, shape and width, under the speed,
// prefill and invariant profiles and in a round of one sequence. The dump is compared with
// selector_golden/<board>.txt; a change of the selector shows there line by line, and a refactor
// that means none must leave every file as it was. Regenerate with --update (writes the files)
// after a change that means to move routes.
//
// Without arguments the test runs itself once per board, then every board again in one process:
// the selector's per-device caches keep a substituted board's answers apart by its content, so the
// two must agree. Each choice is also asked of the pure query (route_query.h); a difference is
// written into the dump. A board the PCI-ID table knows is dumped again with its driver's clocks
// moved, and must choose as at its datasheet clocks. No GPU is needed: the register-MMA price
// reads the committed kernel-facts snapshots (ops/generic/kernel_facts.h), so every board's file
// is checked by every build set on any host.
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/generation.h"
#include "core/units.h"
#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/selector_facts.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::test;
namespace generic = ninferno::ops::detail::generic;
using ops::LinearPolicy;

using GoldenBoard             = generic::FactsBoard;
constexpr const auto& kBoards = generic::kFactsBoards;

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
    // The rest of the formats the generic path takes (rules of their own: the R8 IQ2 branches,
    // the producer-warp W4A8 extras, the ggml block formats' vector and MMQ gates).
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

constexpr Shape kShapes[] = {{256, 2048},  {1024, 4096},  {2048, 1024}, {4096, 4096},
                             {6144, 5120}, {17408, 5120}, {5120, 17408}, {28672, 8192}};
// 9: the first width of the verify band 9..16 (kMmqVerifyMinTokens), where the MMQ verify rule
// starts.
constexpr std::int32_t kTokens[] = {1,  2,  3,  4,  5,   8,   9,   12,  16,   17,  24,
                                    32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048};

// SpeedC1: the speed profile inside a round of one sequence (ops::ScopedRoundLanes(1): a C1 MTP
// or copy round), where the verify widths' ggml gate answers otherwise; T <= 16 only, and only
// the widths whose answer differs from the speed profile's are written.
enum class Profile { Speed, Prefill, Invariant, SpeedC1 };
constexpr const char* kAsSpeed = "=speed"; // a SpeedC1 width that answers as Speed

const char* profile_name(Profile p) {
    return p == Profile::Speed     ? "speed"
           : p == Profile::Prefill ? "prefill"
           : p == Profile::SpeedC1 ? "speed-c1"
                                   : "invariant";
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

// A board whose register-MMA price rests on the compiled kernels' registers and spill
// (rmma_terms_measured, generic_cost.cpp) reads them from the committed kernel-facts snapshot of
// the release engine the board runs (ops/generic/kernel_facts.h, #612), never from the host GPU's
// kernels: its golden is one file per board like every other's, reproducible on any build and any
// host, a CPU included. The snapshot itself is checked against each build's kernels by
// ninferno_kernel_facts_test; a kernel or toolchain change regenerates both, snapshot and goldens,
// in one reviewable change.

std::filesystem::path golden_dir() { return std::filesystem::path(NINFERNO_SELECTOR_GOLDEN_DIR); }

std::filesystem::path golden_file(const GoldenBoard& b) {
    return golden_dir() / kBuildSet / (std::string(b.name) + ".txt");
}

// One board's choices, a line per (format, policy, epilogue, shape, profile) and run of widths
// with the same route and workspace. The scales multiply the clocks the driver reports.
std::string dump_body(const GoldenBoard& b, double sm_scale, double memory_scale) {
    const DeviceClass d = generic::board_device_class(b, sm_scale, memory_scale);
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
                    std::vector<std::string> speed_at(std::size(kTokens));
                    for (const Profile profile :
                         {Profile::Speed, Profile::Prefill, Profile::Invariant, Profile::SpeedC1}) {
                        std::optional<ops::ScopedInvariantRoutes> invariant;
                        if (profile == Profile::Invariant) invariant.emplace();
                        std::optional<ops::ScopedRoundLanes> one_lane;
                        if (profile == Profile::SpeedC1) one_lane.emplace(1);
                        std::string last;
                        std::int32_t first_t = 0, last_t = 0;
                        const auto flush = [&] {
                            if (last.empty() || last == kAsSpeed) return;
                            out << generic::qtype_name(qtype) << " p"
                                << static_cast<int>(policy) << " "
                                << generic::epilogue_name(epilogue) << " " << rows << "x"
                                << shape.k << " " << profile_name(profile) << " T=" << first_t;
                            if (last_t != first_t) out << ".." << last_t;
                            out << " " << last << '\n';
                        };
                        for (std::size_t ti = 0; ti < std::size(kTokens); ++ti) {
                            const std::int32_t t = kTokens[ti];
                            const bool prefill = profile == Profile::Prefill;
                            if (prefill && t < 17) continue;
                            if (profile == Profile::SpeedC1 && t > 16) continue;
                            std::string what;
                            try {
                                const generic::Route r =
                                    generic::select_route(problem, w, t, prefill);
                                what = generic::route_name(r) + " ws=" +
                                       std::to_string(generic::route_workspace_bytes(r, shape.k, t));
                                // The pure query (a weight described by format and shape, the
                                // board by its class) answers as the selector does; a difference
                                // shows in the dump and so fails the comparison.
                                const auto q = generic::route_query(d, qtype, policy, epilogue,
                                                                    rows, shape.k, t, prefill);
                                if (!q || !(*q == r)) {
                                    what += q ? " query=" + generic::route_name(*q)
                                              : std::string(" query=none");
                                }
                                // The engine's call (contiguous, aligned) in the workspace the
                                // plan reserves for it (workspace_capacity_bytes) runs what was
                                // selected (#298): the planned scratch is what the launch needs,
                                // and nothing demotes it but a W4A8 tile the weight is not
                                // admitted to (the same bytes). The resolved query resolves as the
                                // launch does. Either difference shows in the dump.
                                const std::size_t own =
                                    generic::workspace_capacity_bytes(problem, &w, t, t);
                                const generic::ResolvedRoute ran = generic::resolve_route(
                                    r, w,
                                    generic::CallLayout{.ldx            = shape.k,
                                                        .tokens         = t,
                                                        .rows           = rows,
                                                        .epilogue       = epilogue,
                                                        .workspace_free = own});
                                // The pre-quantized R8 form reads its planes from that workspace.
                                if (ran.r8_prequantized &&
                                    generic::route_workspace_bytes(ran.executed, shape.k, t) > own) {
                                    what += " r8_planes_unplanned";
                                }
                                if (ran.reason != generic::Demotion::None &&
                                    ran.reason != generic::Demotion::W4a8Admission) {
                                    what += " runs=" + generic::route_name(ran.executed) + " (" +
                                            std::string(generic::demotion_name(ran.reason)) + ")";
                                }
                                const auto rq = generic::resolved_route_query(
                                    d, qtype, policy, epilogue, rows, shape.k, t, own, prefill);
                                if (!rq || !(rq->executed == ran.executed) ||
                                    rq->r8_prequantized != ran.r8_prequantized) {
                                    what += rq ? " resolved_query=" + generic::route_name(rq->executed)
                                               : std::string(" resolved_query=none");
                                }
                            } catch (const std::exception& error) {
                                what = std::string("threw: ") + error.what();
                            }
                            if (profile == Profile::Speed) speed_at[ti] = what;
                            if (profile == Profile::SpeedC1 && what == speed_at[ti])
                                what = kAsSpeed;
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

// A facts-priced board's dump must read kernel facts (the snapshot of its release engine): a
// dump that read none would be priced by threads alone and silently differ from the release.
std::string dump_board(const GoldenBoard& b, double sm_scale = 1.0, double memory_scale = 1.0) {
    if (!generic::board_prices_from_facts(b)) return dump_body(b, sm_scale, memory_scale);
    std::vector<generic::RmmaFactRead> reads;
    std::string body;
    {
        const generic::ScopedRmmaFactsSink sink(reads);
        body = dump_body(b, sm_scale, memory_scale);
    }
    if (reads.empty()) {
        throw std::runtime_error(std::string(b.name) +
                                 ": the dump read no kernel facts (no snapshot for sm_" +
                                 generic::kernel_facts::release_arch(b.major, b.minor) + ")");
    }
    return body;
}

int run_board(const GoldenBoard& b, bool update) {
    const std::string now            = dump_board(b);
    const std::filesystem::path path = golden_file(b);
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
    std::stringstream was_text;
    was_text << in.rdbuf();
    const std::string was_cmp  = was_text.str();
    const std::string& now_cmp = now;
    if (was_cmp == now_cmp) {
        std::cout << b.name << ": choices as the golden file\n";
        return 0;
    }
    // The first lines that differ, from both sides.
    std::istringstream a(was_cmp), c(now_cmp);
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

// A board the PCI-ID table knows is decided from its row (core/machine_facts.h), so the clocks
// its driver reports (a factory overclock, a vendor's lower boost, a rounding) must not move a
// choice. Every line after the header is compared, the pure query's included; a route that read
// the driver's clocks past the facts fails here even where no text scan sees the read. A memory
// clock more than half a grid step (2.2 %) below the row's is another board under the id (a
// slower memory) and is read on purpose; the variants stay inside that step.
int run_clock_invariance(const GoldenBoard& b) {
    if (b.pci_device_id == 0) return 0; // unknown ids are priced from the driver's values
    const auto body = [](const std::string& dump) {
        std::size_t at = 0; // past the header line
        while (at < dump.size() && dump[at] == '#') at = dump.find('\n', at) + 1;
        return dump.substr(at);
    };
    const std::string stock = body(dump_board(b));
    constexpr struct {
        double sm, memory;
    } kVariants[] = {{0.85, 1.0}, {1.0, 0.985}, {1.08, 1.07},
                     // A driver that reports no clock (MIG, some WSL drivers): the table's facts.
                     {0.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}};
    int failed = 0;
    for (const auto v : kVariants) {
        if (body(dump_board(b, v.sm, v.memory)) == stock) continue;
        std::cerr << b.name << ": choices move with the driver's clocks (SM x" << v.sm
                  << ", memory x" << v.memory << ")\n";
        ++failed;
    }
    if (failed == 0) std::cout << b.name << ": choices hold under the driver's clocks\n";
    return failed;
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
                if (board == b.name) {
                    const int golden = run_board(b, update);
                    return update ? golden : golden + run_clock_invariance(b);
                }
            }
            std::cerr << "unknown board " << board << '\n';
            return 2;
        }
        // One process per board, a few at a time (each describes kernels and holds a context).
        std::atomic<int> failed{0};
        std::atomic<std::size_t> next{0};
        const auto worker = [&] {
            for (std::size_t i; (i = next++) < std::size(kBoards);) {
                const std::string command = std::string(argv[0]) + (update ? " --update" : "") +
                                            " --board " + kBoards[i].name;
                if (std::system(command.c_str()) != 0) ++failed;
            }
        };
        std::vector<std::thread> workers;
        for (int i = 0; i < 6; ++i) workers.emplace_back(worker);
        for (auto& t : workers) t.join();
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
