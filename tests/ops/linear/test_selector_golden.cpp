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
// moved, and must choose as at its datasheet clocks.
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/generation.h"
#include "core/units.h"
#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
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
#include <tuple>
#include <vector>
#ifdef _WIN32
#    include <process.h>
#else
#    include <signal.h>
#    include <unistd.h>
#endif

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
// (rmma_terms_measured, generic_cost.cpp) is priced from what its build's compiler made of the
// kernels: nvcc 12.8 and 13.x allocate other registers for the same source (up to a third apart
// for one kernel, #556), the price turns them into resident CTAs per SM, and a tile choice can
// move with no change of source or spec. The choices are deterministic for one binary; the
// release's toolchain fixes them (tools/release/package.sh). Such a board's golden is therefore
// keyed by the digest of the facts the price read while the board was dumped: the header line
// "# facts <digest>"; selector_golden/facts_index.txt maps (board, digest) to the file holding the
// choices and records the nvcc each was made with. A toolchain with other facts and no entry fails
// and names its digest; --update rewrites the entry of its own nvcc.
bool facts_keyed(const GoldenBoard& b) { return generic::board_prices_from_facts(b); }

// facts_index.txt: which file holds the choices for which facts digest and nvcc (see the file).
struct IndexEntry {
    std::string board, digest, file, nvcc;
};

struct Index {
    std::vector<std::string> kept; // comment lines and the release line, as they stand
    std::string release;
    std::vector<IndexEntry> entries;
};

std::filesystem::path golden_dir() { return std::filesystem::path(NINFERNO_SELECTOR_GOLDEN_DIR); }

std::string board_key(const GoldenBoard& b) { return std::string(kBuildSet) + "/" + b.name; }

Index load_index() {
    Index index;
    std::ifstream in(golden_dir() / "facts_index.txt");
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') {
            index.kept.push_back(line);
        } else if (line.rfind("release ", 0) == 0) {
            index.kept.push_back(line);
            index.release = line.substr(8);
        } else {
            IndexEntry e;
            std::istringstream fields(line);
            if (fields >> e.board >> e.digest >> e.file >> e.nvcc) index.entries.push_back(e);
        }
    }
    return index;
}

void save_index(Index index) {
    std::sort(index.entries.begin(), index.entries.end(), [](const auto& a, const auto& b) {
        return std::tie(a.board, a.nvcc, a.digest) < std::tie(b.board, b.nvcc, b.digest);
    });
    std::ofstream out(golden_dir() / "facts_index.txt", std::ios::binary);
    for (const auto& l : index.kept) out << l << "\n";
    for (const auto& e : index.entries)
        out << e.board << " " << e.digest << " " << e.file << " " << e.nvcc << "\n";
}

long own_pid() {
#ifdef _WIN32
    return static_cast<long>(_getpid());
#else
    return static_cast<long>(getpid());
#endif
}

// Whether a process is certainly gone (POSIX; elsewhere a lock is never judged stale).
bool process_gone(long pid) {
#ifdef _WIN32
    (void)pid;
    return false;
#else
    return kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH;
#endif
}

// --update of several boards at once (one process each) rewrites one index: a directory holding
// its owner's pid is the lock. It is waited for 60 s; a lock whose owner no longer exists is
// removed (renamed away first, so one waiter wins); the destructor removes only a lock this
// process took. Without the lock the constructor throws.
struct IndexLock {
    std::filesystem::path dir   = golden_dir() / ".facts_index.lock";
    std::filesystem::path owner = dir / "owner";
    bool held                   = false;

    IndexLock() {
        const long me   = own_pid();
        long last_owner = 0;
        for (int i = 0; i < 600; ++i) {
            if (std::filesystem::create_directory(dir)) {
                std::ofstream(owner) << me << "\n";
                held = true;
                return;
            }
            long pid = 0;
            std::ifstream(owner) >> pid;
            last_owner = pid;
            if (pid > 0 && process_gone(pid)) {
                const auto grave = dir.string() + ".stale" + std::to_string(me);
                std::error_code ec;
                std::filesystem::rename(dir, grave, ec);
                if (!ec) {
                    long moved = 0;
                    std::ifstream(std::filesystem::path(grave) / "owner") >> moved;
                    if (moved == pid) {
                        std::filesystem::remove_all(grave, ec);
                    } else {
                        std::filesystem::rename(grave, dir, ec); // a live lock: give it back
                    }
                }
                continue;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        throw std::runtime_error("cannot take " + dir.string() + " within 60 s (held by pid " +
                                 std::to_string(last_owner) +
                                 "); remove it if that process is gone");
    }

    ~IndexLock() {
        if (!held) return;
        std::error_code ignored;
        std::filesystem::remove(owner, ignored);
        std::filesystem::remove(dir, ignored);
    }

    IndexLock(const IndexLock&)            = delete;
    IndexLock& operator=(const IndexLock&) = delete;
};

// A dump without its facts line: two toolkits whose kernels give the same choices share a file.
std::string without_facts(const std::string& dump) {
    std::istringstream in(dump);
    std::string out;
    for (std::string line; std::getline(in, line);) {
        if (line.rfind("# facts ", 0) != 0) out += line + "\n";
    }
    return out;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

std::filesystem::path golden_file(const GoldenBoard& b, const std::string& digest) {
    return golden_dir() / kBuildSet /
           (std::string(b.name) + (facts_keyed(b) ? "." + digest : std::string()) + ".txt");
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

// The board's dump with its facts line (a facts-priced board). The first dump of a board in a
// process reads the kernels' facts; a repeat is answered from the selector's per-device cache
// without reading, so the digest is kept per board.
std::string dump_board(const GoldenBoard& b, double sm_scale = 1.0, double memory_scale = 1.0) {
    const std::string body = dump_body(b, sm_scale, memory_scale);
    if (!facts_keyed(b)) return body;
    // The digest is the shared probe's (selector_facts.h), the one a release computes of its own
    // engine; it throws where the price reads no facts.
    static std::map<std::string, std::string> digests;
    if (digests.find(b.name) == digests.end()) digests[b.name] = generic::selector_facts_digest(b);
    const auto first = body.find('\n') + 1;
    return body.substr(0, first) + "# facts " + digests[b.name] + "\n" + body.substr(first);
}

// The digest line of a dump.
std::string digest_of(const std::string& dump) {
    constexpr std::string_view tag = "\n# facts ";
    const auto at                  = dump.find(tag);
    if (at == std::string::npos) return "";
    const auto from = at + tag.size();
    return dump.substr(from, dump.find('\n', from) - from);
}

// A board whose register-MMA price rests on the compiled kernels' registers and spill
// (rmma_terms_measured, generic_cost.cpp) is priced with the BUILD's kernels (the SASS of
// NINFERNO_BUILD_ARCH, which the host runs): another compute capability keeps other counts. Its
// golden is therefore reproducible only on a build of the board's own compute capability; for any
// other build it is skipped (neither written, compared nor run for clock invariance), and a set
// holds the file only for the capabilities its builds are made for. On its own capability the
// file is keyed by the digest of the kernels' facts (facts_digest: another nvcc compiles
// other registers, #556). The decisions are the
// selector's and route_query's own: nothing in the product changes for the test.
bool host_kernels_differ(const GoldenBoard& b) {
    if (!rmma_terms_measured(b.major, b.minor)) return false;
    return b.major != NINFERNO_BUILD_ARCH / 10 || b.minor != NINFERNO_BUILD_ARCH % 10;
}

int run_board(const GoldenBoard& b, bool update) {
    if (host_kernels_differ(b)) {
        std::cout << b.name << ": skipped, its register-MMA price needs a build for sm_" << b.major
                  << b.minor << " kernels\n";
        return 0;
    }
    const std::string now      = dump_board(b);
    const std::string d        = digest_of(now);
    std::filesystem::path path = golden_file(b, d);
    if (update) {
        // The directory may not exist yet in a fresh checkout; a write that fails is a failure.
        std::filesystem::create_directories(path.parent_path());
        const auto write = [&](const std::filesystem::path& to) {
            std::ofstream file(to, std::ios::binary);
            file << now;
            if (file.good()) return true;
            std::cerr << b.name << ": cannot write " << to.string() << '\n';
            return false;
        };
        if (!facts_keyed(b)) {
            if (!write(path)) return 1;
            std::cout << b.name << ": wrote " << path.string() << '\n';
            return 0;
        }
        const IndexLock lock;
        Index index = load_index();
        // The choices another toolkit's file already holds are not stored twice.
        std::string holder = d;
        for (const auto& entry : index.entries) {
            if (entry.board == board_key(b) && entry.digest != d &&
                without_facts(read_file(golden_file(b, entry.file))) == without_facts(now)) {
                holder = entry.file;
                break;
            }
        }
        if (holder == d && !write(path)) return 1;
        std::erase_if(index.entries, [&](const IndexEntry& e) {
            return e.board == board_key(b) && e.nvcc == NINFERNO_NVCC_VERSION;
        });
        index.entries.push_back({board_key(b), d, holder, NINFERNO_NVCC_VERSION});
        // The board's files no entry uses any more.
        const std::string prefix = std::string(b.name) + ".";
        for (const auto& item : std::filesystem::directory_iterator(path.parent_path())) {
            const std::string name = item.path().filename().string();
            if (name.rfind(prefix, 0) != 0 || item.path().extension() != ".txt") continue;
            const std::string used = name.substr(prefix.size(), name.size() - prefix.size() - 4);
            const bool kept =
                std::any_of(index.entries.begin(), index.entries.end(), [&](const IndexEntry& e) {
                    return e.board == board_key(b) && e.file == used;
                });
            if (!kept) std::filesystem::remove(item.path());
        }
        save_index(index);
        std::cout << b.name << ": facts " << d << " for nvcc " << NINFERNO_NVCC_VERSION << ", "
                  << (holder == d ? "wrote " : "same choices as ")
                  << golden_file(b, holder).string() << '\n';
        return 0;
    }
    if (facts_keyed(b)) {
        const Index index       = load_index();
        const IndexEntry* found = nullptr;
        for (const auto& entry : index.entries) {
            if (entry.board == board_key(b) && entry.digest == d &&
                (found == nullptr || entry.nvcc == NINFERNO_NVCC_VERSION)) {
                found = &entry;
            }
        }
        if (found == nullptr) {
            std::cerr << b.name << ": no golden for the compiled kernels' facts " << d << " (nvcc "
                      << NINFERNO_NVCC_VERSION
                      << "); run with --update, another toolchain keeps an entry of its own\n";
            return 1;
        }
        // The release's toolkit must give the facts the release entry names.
        if (NINFERNO_NVCC_VERSION == index.release && found->nvcc != index.release) {
            std::cerr << b.name << ": nvcc " << NINFERNO_NVCC_VERSION
                      << " is the release's, but its facts " << d
                      << " are not the release entry's; run with --update\n";
            return 1;
        }
        path = golden_file(b, found->file);
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << b.name << ": no golden file " << path.string() << " (run with --update)\n";
        return 1;
    }
    std::stringstream was_text;
    was_text << in.rdbuf();
    const std::string was_cmp = facts_keyed(b) ? without_facts(was_text.str()) : was_text.str();
    const std::string now_cmp = facts_keyed(b) ? without_facts(now) : now;
    std::stringstream was;
    was << was_cmp;
    if (was_cmp == now_cmp) {
        std::cout << b.name << ": choices as the golden file\n";
        return 0;
    }
    // The first lines that differ, from both sides.
    std::istringstream a(was.str()), c(now_cmp);
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
    if (b.pci_device_id == 0 || host_kernels_differ(b))
        return 0; // unknown ids are priced from the driver's values
    const auto body = [](const std::string& dump) {
        std::size_t at = 0; // past the header lines (the facts line is the first dump's only)
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
