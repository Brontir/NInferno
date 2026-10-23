#pragma once

// The generic projection path's choices on a substituted board (core/device_class.h), shared by the
// selector golden test and the selector property suite: the boards, the formats, shapes and widths
// asked about, one evaluation of every choice, and its text, which is exactly the golden file's.
//
// The route table and several heuristics cache what they read of the device on first use, so one
// process evaluates one board.
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/units.h"
#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_table.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::test::selector {

namespace generic = ninferno::ops::detail::generic;
using ops::LinearPolicy;

// Datasheet numbers as the CUDA device query reports them (boost SM clock, memory clock = data
// rate / 2, bus width).
struct TestBoard {
    const char* name;
    std::uint32_t pci_device_id; // 0: unknown to the id table, the chip follows from the SM count
    int major, minor, sms;
    std::size_t l2_kib, memory_gib;
    int sm_clock_mhz, memory_clock_mhz, bus_bits;
};

// The boards of the golden files (selector_golden/<set>/<board>.txt).
inline constexpr TestBoard kGoldenBoards[] = {
    {"rtx5090", 0x2B85, 12, 0, 170, 96 * 1024, 32, 2407, 14001, 512},
    {"rtx5080", 0x2C02, 12, 0, 84, 64 * 1024, 16, 2617, 15001, 256},
    {"rtx5070ti", 0x2C05, 12, 0, 70, 48 * 1024, 16, 2452, 14001, 256},
    {"rtx5060ti", 0x2D04, 12, 0, 36, 32 * 1024, 16, 2572, 14001, 128},
    {"rtx4090", 0x2684, 8, 9, 128, 72 * 1024, 24, 2520, 10501, 384},
    {"rtx4060ti", 0x2805, 8, 9, 34, 32 * 1024, 16, 2535, 9001, 128},
    {"l4", 0x27B8, 8, 9, 58, 48 * 1024, 24, 2040, 6251, 192},
    {"rtx3090", 0x2204, 8, 6, 82, 6 * 1024, 24, 1695, 9751, 384},
    {"a10", 0x2236, 8, 6, 72, 6 * 1024, 24, 1695, 6251, 384},
    {"a100", 0x20B0, 8, 0, 108, 40 * 1024, 40, 1410, 1215, 5120},
    {"rtx2080ti", 0x1E07, 7, 5, 68, 5 * 1024, 11, 1545, 7000, 352},
    {"v100", 0x1DB5, 7, 0, 80, 6 * 1024, 32, 1530, 877, 4096},
    // Not targets, but the rows where spellings of one generation fact used to disagree.
    {"h100-sxm", 0x2330, 9, 0, 132, 50 * 1024, 80, 1980, 2619, 5120},
    {"b300", 0, 10, 3, 148, 126 * 1024, 268, 2032, 3996, 8192},
    {"gb10-spark", 0, 12, 1, 48, 24 * 1024, 128, 2418, 4266, 256},
    {"unknown-sm120-48sm", 0, 12, 0, 48, 32 * 1024, 12, 2500, 14001, 192},
};

// Machines no golden file covers: every compute capability from 7.0 to 12.1 the feature table
// names, each at the smallest, a typical and the largest SM count of its generation's boards
// (one or two where the generation has few), unknown PCI ids (the chip follows from the SM count),
// the architecture's shared-memory limits (synthetic_device_class) and typical clocks and buses.
inline constexpr TestBoard kSyntheticBoards[] = {
    {"syn-sm70-80sm", 0, 7, 0, 80, 6144, 16, 1380, 877, 4096},
    {"syn-sm70-84sm", 0, 7, 0, 84, 6144, 32, 1597, 877, 4096},
    {"syn-sm72-8sm", 0, 7, 2, 8, 512, 16, 1377, 2133, 256},
    {"syn-sm75-14sm", 0, 7, 5, 14, 1024, 4, 1590, 4001, 128},
    {"syn-sm75-40sm", 0, 7, 5, 40, 4096, 8, 1770, 7001, 256},
    {"syn-sm75-72sm", 0, 7, 5, 72, 6144, 24, 1770, 7001, 384},
    {"syn-sm80-56sm", 0, 8, 0, 56, 24576, 24, 1440, 1215, 3072},
    {"syn-sm80-108sm", 0, 8, 0, 108, 40960, 80, 1410, 1593, 5120},
    {"syn-sm80-128sm", 0, 8, 0, 128, 49152, 80, 1410, 1593, 5120},
    {"syn-sm86-10sm", 0, 8, 6, 10, 2048, 16, 1770, 6251, 128},
    {"syn-sm86-46sm", 0, 8, 6, 46, 4096, 8, 1777, 7001, 256},
    {"syn-sm86-84sm", 0, 8, 6, 84, 6144, 24, 1860, 10501, 384},
    {"syn-sm87-8sm", 0, 8, 7, 8, 2048, 8, 1020, 3200, 128},
    {"syn-sm87-16sm", 0, 8, 7, 16, 4096, 64, 1300, 3200, 256},
    {"syn-sm89-24sm", 0, 8, 9, 24, 24576, 8, 2460, 8501, 128},
    {"syn-sm89-58sm", 0, 8, 9, 58, 49152, 12, 2610, 10501, 192},
    {"syn-sm89-142sm", 0, 8, 9, 142, 98304, 48, 2505, 9001, 384},
    {"syn-sm90-78sm", 0, 9, 0, 78, 61440, 96, 1755, 2619, 5120},
    {"syn-sm90-114sm", 0, 9, 0, 114, 51200, 80, 1755, 1593, 5120},
    {"syn-sm90-132sm", 0, 9, 0, 132, 51200, 94, 1980, 2619, 6144},
    {"syn-sm100-148sm", 0, 10, 0, 148, 129024, 180, 1965, 3996, 8192},
    {"syn-sm100-160sm", 0, 10, 0, 160, 129024, 192, 2100, 3996, 8192},
    {"syn-sm103-160sm", 0, 10, 3, 160, 129024, 288, 2100, 3996, 8192},
    {"syn-sm110-20sm", 0, 11, 0, 20, 32768, 128, 1575, 4266, 256},
    {"syn-sm120-20sm", 0, 12, 0, 20, 24576, 8, 2572, 10001, 128},
    {"syn-sm120-48sm", 0, 12, 0, 48, 32768, 16, 2617, 14001, 192},
    {"syn-sm120-188sm", 0, 12, 0, 188, 131072, 96, 2617, 14001, 512},
    {"syn-sm121-48sm", 0, 12, 1, 48, 24576, 128, 2418, 4266, 256},
};

inline constexpr QType kFormats[] = {
    QType::BF16,         QType::Q4_G64_FP16, QType::Q5_G64_FP16,
    QType::Q6_G64_FP16,  QType::Q8_G32_FP16, QType::FP8_E4M3FN_ROW_BF16,
    QType::NVFP4,        QType::MXFP4,       QType::U4_G128_ZP_FP16,
    QType::GGML_Q4_K,    QType::GGML_Q5_K,   QType::GGML_Q6_K,
    QType::GGML_Q3_K,    QType::GGML_Q2_K,   QType::GGML_IQ4_XS,
    QType::GGML_IQ3_XXS, QType::GGML_IQ3_S,  QType::GGML_IQ2_S,
    QType::GGML_Q4_0,    QType::GGML_IQ4_NL,
};

struct Shape {
    std::int32_t n, k;
};

inline constexpr Shape kShapes[]        = {{256, 2048},  {1024, 4096},  {2048, 1024},  {4096, 4096},
                                           {6144, 5120}, {17408, 5120}, {5120, 17408}, {28672, 8192}};
inline constexpr std::int32_t kTokens[] = {1,  2,  3,  4,  5,   8,   12,  16,  17,   24,
                                           32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048};

enum class Profile { Speed, Prefill, Invariant };

inline const char* profile_name(Profile p) {
    return p == Profile::Speed ? "speed" : p == Profile::Prefill ? "prefill" : "invariant";
}

// The route set a build compiles decides what a board can pick, so each set keeps its own files:
// sm_70 (FMA-emulated MMA, 96 KiB), sm_75 (64 KiB: no producer-warp W4A8 tiles), the full builds
// (120a / 121a / 120f, with NVFP4 and the cluster K-split schedules) and every other build (base;
// sm_80's deeper staged BF16 tiles change nothing here, the goldens pack no BF16 weights).
inline constexpr const char* kBuildSet =
#if NINFERNO_BUILD_ARCH == 70
    "sm70";
#elif NINFERNO_BUILD_ARCH == 75
    "sm75";
#elif defined(NINFERNO_WITHOUT_NVFP4)
    "base";
#else
    "full";
#endif

// The board's class as the device query would build it, with its clocks and memory bus.
inline DeviceClass device_class_of(const TestBoard& b) {
    DeviceClass d      = synthetic_device_class(b.pci_device_id, b.major, b.minor, b.sms,
                                                b.l2_kib * 1024, b.memory_gib * 1024 * kMiB);
    d.sm_clock_khz     = b.sm_clock_mhz * 1000;
    d.memory_clock_khz = b.memory_clock_mhz * 1000;
    d.memory_bus_bits  = b.bus_bits;
    d.facts            = machine_facts_of(d); // as the query derives them from these values
    return d;
}

// One choice: the route and its workspace, or the refusal select_route threw.
struct Outcome {
    bool threw = false;
    generic::Route route{};
    std::size_t workspace = 0;
    std::string error;

    [[nodiscard]] std::string text() const {
        return threw ? "threw: " + error
                     : generic::route_name(route) + " ws=" + std::to_string(workspace);
    }
};

// The choices of one (format, policy, epilogue, shape, profile) at every width asked about.
struct Row {
    QType qtype{};
    LinearPolicy policy{};
    generic::Epilogue epilogue{};
    std::int32_t rows = 0, k = 0;
    Profile profile{};
    std::vector<std::pair<std::int32_t, Outcome>> by_tokens;
};

// A format and shape the test packer cannot build is named, not silently dropped.
struct Item {
    std::string note; // "<format> <n>x<k> not packed: <why>" when the packer refused
    Row row;
};

struct Choices {
    std::string header; // "# <board> <class id> <arch>"
    std::vector<Item> items;
};

// Called for every choice that did not throw, with the weight it was made for; problem() once per
// problem before any of its choices (the property suite primes the dense tuner there).
struct ChoiceCheck {
    virtual ~ChoiceCheck() = default;

    virtual void problem(const generic::Problem& /*problem*/, const Weight& /*weight*/) {}

    virtual void check(const Row& row, std::int32_t tokens, const Weight& weight,
                       const Outcome& outcome) = 0;
};

// Every choice of the current device class (the caller substitutes the board), in golden order.
inline Choices evaluate(const char* board_name, ChoiceCheck* checker = nullptr) {
    Choices out;
    out.header =
        std::string("# ") + board_name + " " + device_class().id() + " " + device_class().arch();
    // Weights are never read here: an aligned address stands in for the payload.
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
            } catch (const std::invalid_argument& error) {
                Item item;
                item.note = std::string(generic::qtype_name(qtype)) + " " +
                            std::to_string(shape.n) + "x" + std::to_string(shape.k) +
                            " not packed: " + error.what();
                item.row.qtype = qtype;
                out.items.push_back(std::move(item));
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
                    if (checker != nullptr) checker->problem(problem, w);
                    for (const Profile profile :
                         {Profile::Speed, Profile::Prefill, Profile::Invariant}) {
                        std::optional<ops::ScopedInvariantRoutes> invariant;
                        if (profile == Profile::Invariant) invariant.emplace();
                        Item item;
                        item.row = {qtype, policy, epilogue, rows, shape.k, profile, {}};
                        for (const std::int32_t t : kTokens) {
                            const bool prefill = profile == Profile::Prefill;
                            if (prefill && t < 17) continue;
                            Outcome o;
                            try {
                                o.route     = generic::select_route(problem, w, t, prefill);
                                o.workspace = generic::route_workspace_bytes(o.route, shape.k, t);
                            } catch (const std::exception& error) {
                                o.threw = true;
                                o.error = error.what();
                            }
                            if (checker != nullptr && !o.threw) checker->check(item.row, t, w, o);
                            item.row.by_tokens.emplace_back(t, std::move(o));
                        }
                        out.items.push_back(std::move(item));
                    }
                }
            }
        }
    }
    return out;
}

// "<format> p<policy> <epilogue> <rows>x<k> <profile>": the row's key in the golden text.
inline std::string row_key(const Row& r) {
    return std::string(generic::qtype_name(r.qtype)) + " p" +
           std::to_string(static_cast<int>(r.policy)) + " " + generic::epilogue_name(r.epilogue) +
           " " + std::to_string(r.rows) + "x" + std::to_string(r.k) + " " + profile_name(r.profile);
}

// A row's lines: one per run of widths with the same route and workspace.
inline std::string render_row(const Row& r) {
    std::ostringstream out;
    const std::string key = row_key(r);
    std::string last;
    std::int32_t first_t = 0, last_t = 0;
    const auto flush = [&] {
        if (last.empty()) return;
        out << key << " T=" << first_t;
        if (last_t != first_t) out << ".." << last_t;
        out << " " << last << '\n';
    };
    for (const auto& [t, o] : r.by_tokens) {
        const std::string what = o.text();
        if (what != last) {
            flush();
            last    = what;
            first_t = t;
        }
        last_t = t;
    }
    flush();
    return out.str();
}

inline std::string render_item(const Item& item) {
    return item.note.empty() ? render_row(item.row) : item.note + '\n';
}

// The golden file's text: the header, then every item in order.
inline std::string render(const Choices& c) {
    std::string out = c.header + '\n';
    for (const Item& item : c.items) out += render_item(item);
    return out;
}

// 64-bit FNV-1a.
inline std::uint64_t fnv1a(std::string_view text, std::uint64_t h = 0xcbf29ce484222325ULL) {
    for (const unsigned char c : text) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

inline std::size_t count_lines(std::string_view text) {
    std::size_t n = 0;
    for (const char c : text) n += c == '\n';
    return n;
}

// The digest lines of a board: "<board> * <fnv64 of the golden text> <lines>", then one line per
// format with the digest of that format's part of the text, in format order.
inline std::vector<std::string> digest_lines(const char* board, const Choices& c) {
    char hex[17];
    const auto line = [&](const std::string& what, std::string_view text) {
        std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(fnv1a(text)));
        return std::string(board) + " " + what + " " + hex + " " +
               std::to_string(count_lines(text));
    };
    std::vector<std::string> lines;
    lines.push_back(line("*", render(c)));
    std::map<int, std::string> by_format;
    for (const Item& item : c.items)
        by_format[static_cast<int>(item.row.qtype)] += render_item(item);
    for (const QType q : kFormats) {
        const auto it = by_format.find(static_cast<int>(q));
        if (it != by_format.end()) lines.push_back(line(generic::qtype_name(q), it->second));
    }
    return lines;
}

} // namespace ninferno::test::selector
