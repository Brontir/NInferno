// The generic projection path's choices, frozen: for each board below (a substituted DeviceClass
// with the board's SMs, clocks, memory bus, L2 and per-SM limits, core/device_class.h) the route
// and the workspace it needs for every format, policy, epilogue, shape and width, under the speed,
// prefill and invariant profiles and in a round of one sequence. The dump is compared with
// selector_golden/<build set>/<board>/: header.txt (the board's header line) and one file per
// format, <format>.txt (#620 step 8: a change of one format's price moves that format's files
// only, so decision PRs touch fewer of the same files). Joined in the order of kFormats they are
// the board's whole dump. A change of the selector shows there line by line, and a refactor that
// means none must leave every file as it was. Regenerate with --update (writes the files, and
// removes the .txt files of the board the dump no longer writes) after a change that means to
// move routes.
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
#include "ops/route_pins.h"

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
#include <utility>
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
// the deeper staged BF16 tiles of sm_80, sm_90 and sm_100 change nothing here, the goldens pack no
// BF16 weights).
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

// A board's golden directory, and the name of its header file (no format is called "header").
std::filesystem::path board_dir(const GoldenBoard& b) { return golden_dir() / kBuildSet / b.name; }

constexpr std::string_view kHeaderFile = "header.txt";

// One board's dump: its header line, then each format's lines in the order of kFormats (a format
// with no line has no entry). Joined in that order they are the board's whole dump.
struct BoardDump {
    std::string header;
    std::vector<std::pair<std::string, std::string>> formats; // (format name, its lines)
};

// Every line of the dump after the header.
std::string decisions_text(const BoardDump& dump) {
    std::string text;
    for (const auto& [name, lines] : dump.formats) text += lines;
    return text;
}

// One board's choices, a line per (format, policy, epilogue, shape, profile) and run of widths
// with the same route and workspace. The scales multiply the clocks the driver reports.
BoardDump dump_body(const GoldenBoard& b, double sm_scale, double memory_scale) {
    const DeviceClass d = generic::board_device_class(b, sm_scale, memory_scale);
    const ScopedDeviceClass scoped(d);
    BoardDump dump;
    {
        std::ostringstream header;
        header << "# " << b.name << " " << device_class().id() << " " << device_class().arch()
               << '\n';
        dump.header = header.str();
    }
    // Weights are never read here: an aligned address stands in for the payload.
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    for (const QType qtype : kFormats) {
        std::ostringstream out; // this format's lines
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
        if (out.tellp() > 0) dump.formats.emplace_back(generic::qtype_name(qtype), out.str());
    }
    return dump;
}

// A facts-priced board's dump must read kernel facts (the snapshot of its release engine): a
// dump that read none would be priced by threads alone and silently differ from the release.
BoardDump dump_board(const GoldenBoard& b, double sm_scale = 1.0, double memory_scale = 1.0) {
    if (!generic::board_prices_from_facts(b)) return dump_body(b, sm_scale, memory_scale);
    std::vector<generic::RmmaFactRead> reads;
    BoardDump body;
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

// The files of a board's golden directory, (file name, text): the header, then one per format.
std::vector<std::pair<std::string, std::string>> golden_files(const BoardDump& dump) {
    std::vector<std::pair<std::string, std::string>> files{{std::string(kHeaderFile), dump.header}};
    for (const auto& [name, lines] : dump.formats) {
        if (name + ".txt" == kHeaderFile)
            throw std::logic_error("a format named like the golden header file: " + name);
        files.emplace_back(name + ".txt", lines);
    }
    return files;
}

// The .txt files on disk that a board's dump does not write: a format file whose format has no
// line any more, and the board's single file of the layout before #620 step 8.
std::vector<std::filesystem::path>
stale_files(const GoldenBoard& b, const std::vector<std::pair<std::string, std::string>>& files) {
    std::vector<std::filesystem::path> stale;
    const std::filesystem::path dir = board_dir(b);
    if (std::filesystem::is_directory(dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (entry.path().extension() != ".txt") continue;
            const std::string name = entry.path().filename().string();
            if (std::none_of(files.begin(), files.end(),
                             [&](const auto& file) { return file.first == name; }))
                stale.push_back(entry.path());
        }
    }
    const std::filesystem::path single = golden_dir() / kBuildSet / (std::string(b.name) + ".txt");
    if (std::filesystem::exists(single)) stale.push_back(single);
    std::sort(stale.begin(), stale.end());
    return stale;
}

int run_board(const GoldenBoard& b, bool update) {
    const auto files                = golden_files(dump_board(b));
    const std::filesystem::path dir = board_dir(b);
    const auto stale                = stale_files(b, files);
    if (update) {
        // The directory may not exist yet in a fresh checkout; a write that fails is a failure.
        std::filesystem::create_directories(dir);
        for (const auto& [name, text] : files) {
            std::ofstream file(dir / name, std::ios::binary);
            file << text;
            if (!file.good()) {
                std::cerr << b.name << ": cannot write " << (dir / name).string() << '\n';
                return 1;
            }
        }
        for (const auto& path : stale) std::filesystem::remove(path);
        std::cout << b.name << ": wrote " << files.size() << " files in " << dir.string() << '\n';
        return 0;
    }
    int differ = 0, shown = 0;
    for (const auto& [name, now] : files) {
        const std::filesystem::path path = dir / name;
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::cerr << b.name << ": no golden file " << path.string() << " (run with --update)\n";
            ++differ;
            continue;
        }
        std::stringstream was_text;
        was_text << in.rdbuf();
        const std::string was = was_text.str();
        if (was == now) continue;
        ++differ;
        // The first lines that differ, from both sides (at most 20 for the board).
        std::istringstream a(was), c(now);
        std::string la, lc;
        int line = 0;
        while (shown < 20) {
            const bool ga = static_cast<bool>(std::getline(a, la));
            const bool gc = static_cast<bool>(std::getline(c, lc));
            if (!ga && !gc) break;
            ++line;
            if (!ga) la = "<end>";
            if (!gc) lc = "<end>";
            if (la != lc) {
                std::cerr << b.name << "/" << name << ":" << line << "\n  golden: " << la
                          << "\n  now:    " << lc << '\n';
                ++shown;
            }
        }
        std::cerr << b.name << ": choices differ from " << path.string() << '\n';
    }
    for (const auto& path : stale) {
        std::cerr << b.name << ": " << path.string()
                  << " is not written by the dump (a format with no line now, or the single-file"
                     " layout; --update removes it)\n";
        ++differ;
    }
    if (differ != 0) return 1;
    std::cout << b.name << ": choices as the golden files\n";
    return 0;
}

// A board the PCI-ID table knows is decided from its row (core/machine_facts.h), so the clocks
// its driver reports (a factory overclock, a vendor's lower boost, a rounding) must not move a
// choice. Every line after the header is compared, the pure query's included; a route that read
// the driver's clocks past the facts fails here even where no text scan sees the read. A memory
// clock more than half a grid step (2.2 %) below the row's is another board under the id (a
// slower memory) and is read on purpose; the variants stay inside that step.
int run_clock_invariance(const GoldenBoard& b) {
    if (b.pci_device_id == 0) return 0; // unknown ids are priced from the driver's values
    const std::string stock = decisions_text(dump_board(b)); // past the header line
    constexpr struct {
        double sm, memory;
    } kVariants[] = {{0.85, 1.0}, {1.0, 0.985}, {1.08, 1.07},
                     // A driver that reports no clock (MIG, some WSL drivers): the table's facts.
                     {0.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}};
    int failed = 0;
    for (const auto v : kVariants) {
        if (decisions_text(dump_board(b, v.sm, v.memory)) == stock) continue;
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
        // The goldens are the legacy route policy's (#614 F1: --route-policy legacy, the release's
        // default), pinned here rather than assumed: a default that moved would fail first.
        if (ops::route_policy() != RoutePolicy::Legacy) {
            std::cerr << "selector golden: the default route policy is not legacy\n";
            return 1;
        }
        ops::set_route_policy(RoutePolicy::Legacy);
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
