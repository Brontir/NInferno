// The generic projection path's coverage as a lint (#619 step 2, K2): on every board this build
// serves, for every format a served model is stored in, under each activation policy it is served
// with and at every width band, at least one family takes the call on the device's own
// instructions. A cell whose only families are the scalar kernel or a tile whose MMA the device
// emulates is a hole. The known holes are listed below (kAllowed), each with its reason; a hole
// not listed fails, and so does a listed one that is now covered (the entry is stale). No GPU and
// no product change: the boards are substituted DeviceClasses and the weights are described by
// format and shape, their bytes never read.
//
// What is served, read from the tables that already say it:
// - boards: the golden boards (selector_facts.h kFactsBoards) of this build's compute capability,
//   the boards `ninferno-serve selector-facts` checks this build's kernels for
//   (selector_facts_main). A build of an architecture the release does not ship
//   (NINFERNO_CUDA_ARCHS, CMakeLists.txt) or that no golden board has skips (77).
// - formats: every row of the format table (core/format_facts.h) with a generic codec: an artifact
//   stores each one (the converter keeps the GGUF block formats as published and writes the
//   engine's own); FP32 and INT32 have none.
// - policies: A16Only for every format (--activations exact; an output head); AllowA8 where the
//   format's codes widen to int8 (FormatFacts::int8_input, what the converter's kept_block_policy
//   reads) and for NVFP4 (a site without its activation divisor runs AllowA8,
//   ops/weight_input.cpp); AllowA4 for NVFP4 (with the divisor, convert kept_policy).
// - widths: decode (1), the verify bands 2..4, 5..8 and 9..16, the prefill band 17..256 and past
//   it, each sampled at its ends and inside (17 tokens and more as the prefill profile runs them);
//   four model projections (selector_facts_probe's shapes), as a Store and as a SwiGLU.
//
// Which families take a cell: family_alternatives (generic_gemm.h: each family's route, or why it
// has none), each route resolved as the engine's call runs it (describe_route: resolve_route in
// the workspace the plan reserves, workspace_capacity_bytes, and Volta's CUTLASS copy or its
// scalar demotion), counted only where the build manifest (build_manifest) holds the route's
// kernel for the format. A route is emulated where the device has no tensor mma.sync
// (DeviceFeatures::mma_tensor false, Volta; or mma_slow, TU116 / TU117) and its family builds the
// MMA fragments from FP32 FMAs and dp4a over warp shuffles (ops/common/mma.cuh): the register MMA,
// except where Volta's own kernels take the call (the HMMA verify kernel, the decode GEMV, and a
// one-tile call of at most four tokens, which sums lane by lane: GemmArgs::lane_local), W4A8,
// W4A4, the staged MMA, I8 and the NVFP4 tiles. The tiled family (mma.m8n8k4, and CUTLASS over a
// decoded copy, on Volta), R8 and ggml's vector kernel (dp4a) and ggml's matrix kernel run the
// device's own instructions. Coverage is not speed: a covered cell may still pick a slow route.
//
// On failure the test prints the coverage table of each board with a new hole: a line per format
// and policy, a column per band ('.' covered, 'E' emulated only, 'S' scalar only; lower case where
// the hole is listed), and for each new hole the families of its first cell. --table prints every
// board's table, --cells every cell's families, --board <name> one golden board of another class
// under this build's kernels (a maintainer's look, not what the release runs there).
#include "core/device_class.h"
#include "core/format_facts.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "ops/generic/build_manifest.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;
using generic::Family;
using generic::Route;
using ops::LinearPolicy;

// ---------------------------------------------------------------------------------------------
// The known holes. Each entry: the boards' compute capability (major * 10 + minor), the policy,
// the widths, the formats, and why the hole is accepted. Checked only in a build that serves a
// board of that capability; an entry with a format no longer a hole there fails as stale.
struct Allowed {
    int capability;
    LinearPolicy policy;
    std::int32_t min_tokens, max_tokens;
    std::vector<QType> formats;
    const char* reason;
};

const std::vector<Allowed>& allowed_holes() {
    using Q                                    = QType;
    static const std::vector<Allowed> kAllowed = {
        // Volta (GV100, sm_70: no tensor mma.sync), the verify widths 5..16 with BF16 activations.
        // A format without a tiled tile of its own (tiled_supports; Q8 has one on Volta) reaches
        // the tensor cores through CUTLASS over a decoded copy, which starts at the prefill band
        // (kVoltaCutlassMinTokens = 17, so that a prefill token's bytes do not depend on its
        // call's width; below it a CUTLASS-only tiled call runs scalar, volta_tiled_runs_scalar,
        // and the selection keeps the register MMA); Volta's HMMA verify kernel reads Q4_K / Q5_K /
        // Q6_K / Q8 only (volta_hmma_codec). These calls run the register MMA's FMA-emulated
        // fragments: decode (1..4 tokens, lane by lane) and prefill are covered. Market weight:
        // one board class, and only an A16 call of a speculative verify: under the default policy
        // the ggml block formats take ggml's vector kernel through 16 tokens on Volta
        // (vec_max_tokens); A16 there is --activations exact or a model's output head.
        {70,
         LinearPolicy::A16Only,
         5,
         16,
         {Q::GGML_Q3_K, Q::GGML_Q2_K, Q::GGML_IQ4_XS, Q::GGML_IQ3_XXS, Q::GGML_IQ3_S, Q::GGML_Q4_0,
          Q::GGML_Q4_1, Q::GGML_Q5_0, Q::GGML_Q5_1, Q::GGML_IQ4_NL, Q::GGML_IQ2_XXS, Q::GGML_IQ2_XS,
          Q::GGML_IQ2_S, Q::GGML_IQ1_S, Q::GGML_IQ1_M},
         "Volta, A16 verify of a ggml block format: no own tile, no HMMA codec; the default "
         "AllowA8 takes ggml's vector kernel"},
        // The same widths for the formats whose every served policy is A16 there or whose int8
        // routes stop at decode: BF16 and the FP8 rows (no int8 family reads them), the ternary
        // and 1/2-bit block formats (TQ1_0, TQ2_0, Q1_0, Q2_0: kept A16Only, no int8 decoder),
        // and the FP4 formats (R8 takes them at decode only; W4A8 and W4A4 are emulated on
        // Volta). Market weight: one board class at verify widths; a BF16 projection there is an
        // unquantized checkpoint's or gpt-oss's attention, whose prefill CUTLASS takes.
        {70,
         LinearPolicy::A16Only,
         5,
         16,
         {Q::BF16, Q::FP8_E4M3FN_ROW_BF16, Q::FP8_E4M3FN_ROW_F32, Q::FP8_E4M3FN_B128_F32, Q::NVFP4,
          Q::MXFP4, Q::GGML_TQ1_0, Q::GGML_TQ2_0, Q::GGML_Q1_0, Q::GGML_Q2_0},
         "Volta, A16 verify of a format without own tile or HMMA codec (CUTLASS from 17 tokens)"},
        {70,
         LinearPolicy::AllowA8,
         5,
         16,
         {Q::NVFP4, Q::MXFP4},
         "Volta, int8 verify of an FP4 format: R8 at decode only, W4A8 emulated"},
        {70,
         LinearPolicy::AllowA4,
         5,
         16,
         {Q::NVFP4},
         "Volta, FP4 verify of NVFP4: W4A4 not built outside sm_120a, W4A8 emulated"},
    };
    return kAllowed;
}

// ---------------------------------------------------------------------------------------------
// The served cells.

struct Band {
    const char* name;
    std::int32_t lo, hi;
    std::vector<std::int32_t> tokens; // the widths sampled
};

const std::vector<Band>& bands() {
    static const std::vector<Band> kBands = {
        {"1", 1, 1, {1}},
        {"2-4", 2, 4, {2, 3, 4}},
        {"5-8", 5, 8, {5, 6, 8}},
        {"9-16", 9, 16, {9, 12, 16}},
        {"17-256", 17, 256, {17, 32, 64, 256}},
        {"257+", 257, 1 << 20, {512, 2048}},
    };
    return kBands;
}

struct Shape {
    std::int32_t n, k;
};

// selector_facts_probe's projections (selector_facts.cpp): a GQA k/v, a square, a gate/up and a
// down of the 27B..32B models; a SwiGLU call reads half of the rows as its gate.
constexpr Shape kShapes[] = {{1024, 4096}, {4096, 4096}, {17408, 5120}, {5120, 17408}};

std::vector<QType> served_formats() {
    std::vector<QType> out;
    for (const FormatFacts& f : kFormatFacts) {
        if (f.codec) out.push_back(f.q);
    }
    return out;
}

std::vector<LinearPolicy> served_policies(QType q) {
    std::vector<LinearPolicy> out{LinearPolicy::A16Only};
    if (format_facts(q).int8_input || q == QType::NVFP4) out.push_back(LinearPolicy::AllowA8);
    if (q == QType::NVFP4) out.push_back(LinearPolicy::AllowA4);
    return out;
}

const char* policy_name(LinearPolicy p) {
    return p == LinearPolicy::A16Only ? "a16" : p == LinearPolicy::AllowA8 ? "a8" : "a4";
}

// Whether the release ships this build's architecture (NINFERNO_CUDA_ARCHS).
bool release_arch() {
    const std::string_view list = NINFERNO_RELEASE_CUDA_ARCHS;
    std::size_t at              = 0;
    while (at <= list.size()) {
        const std::size_t end = std::min(list.find(',', at), list.size());
        if (list.substr(at, end - at) == NINFERNO_BUILD_ARCH_LABEL) return true;
        at = end + 1;
    }
    return false;
}

// selector_facts_main's boards of this build: its own compute capability.
bool serves(const generic::FactsBoard& b) { return b.major * 10 + b.minor == NINFERNO_BUILD_ARCH; }

// ---------------------------------------------------------------------------------------------
// One weight as the converter lays it out (route_query.cpp describe_weight): an aligned address
// stands in for its bytes. Its view and weight point into its parent.
struct DescribedWeight {
    WeightParent parent;
    WeightView view;
    Weight weight;

    DescribedWeight(QType q, generic::Epilogue epilogue, std::int32_t rows, std::int32_t k) {
        const std::uint64_t stored =
            static_cast<std::uint64_t>(epilogue == generic::Epilogue::SwiGlu ? 2 * rows : rows);
        const bool nvfp4             = q == QType::NVFP4;
        parent.data                  = reinterpret_cast<const std::byte*>(std::uintptr_t{1} << 32);
        parent.weight_scale_divisor  = nvfp4 ? 1.0F : 0.0F;
        const std::uint64_t shape[2] = {stored, static_cast<std::uint64_t>(k)};
        parent.geometry              = weight_geometry(q, default_weight_layout(q), shape);
        view.shape                   = {stored, static_cast<std::uint64_t>(k)};
        view.parts = {WeightRegion{&parent, 0, stored * static_cast<std::uint64_t>(k)}};
        weight     = native_weight(view, nvfp4 ? 1.0F : 0.0F);
    }

    DescribedWeight(const DescribedWeight&)            = delete;
    DescribedWeight& operator=(const DescribedWeight&) = delete;
};

enum class Kind : std::uint8_t { None, Scalar, Emulated, Native }; // ordered: the best wins

char kind_mark(Kind k, bool allowed) {
    const char c = k == Kind::Native     ? '.'
                   : k == Kind::Emulated ? 'E'
                   : k == Kind::Scalar   ? 'S'
                                         : '-';
    return allowed && c != '.' ? static_cast<char>(c - 'A' + 'a') : c;
}

const char* kind_name(Kind k) {
    return k == Kind::Native     ? "native"
           : k == Kind::Emulated ? "emulated"
           : k == Kind::Scalar   ? "scalar"
                                 : "none";
}

// Whether the build manifest holds the kernel of `route` for the format at this K.
bool manifest_holds(const generic::BuildManifest& m, Route route, QType q, std::int32_t k) {
    for (const generic::ManifestSchedule& s : m.schedules) {
        if (!(s.route == route)) continue;
        if (!s.compiled) return false;
        if (!s.only_formats.empty() &&
            std::find(s.only_formats.begin(), s.only_formats.end(), q) == s.only_formats.end()) {
            return false;
        }
        for (const generic::OmittedFormat& o : s.omitted_formats) {
            if (o.qtype == q && (o.k_multiple == 0 || k % o.k_multiple == 0)) return false;
        }
        return true;
    }
    return false;
}

// The instructions a route that runs (`executed`) multiplies on, on this device, for a call of
// `tokens` tokens of `w` (aligned and contiguous, as the engine passes it). `how` names Volta's
// own kernels where they take a register-MMA call.
Kind classify(Route executed, const Weight& w, std::int32_t tokens, std::string& how) {
    const DeviceFeatures& f = device_class().features;
    const bool emulating    = !f.mma_tensor || f.mma_slow;
    switch (executed.family) {
    case Family::Scalar:
        return Kind::Scalar;
    case Family::R8Mma:
    case Family::GgmlVec:
        return Kind::Native; // dp4a, no MMA
    case Family::TiledMma:
    case Family::GgmlMmq:
        // Volta's tiled kernel issues mma.m8n8k4 (and CUTLASS's); ggml's matrix kernel refuses
        // a device without tensor mma.sync (ggml_mmq::device_supported). TU116 / TU117 run
        // mma.sync on their FP16 units.
        return f.mma_slow ? Kind::Emulated : Kind::Native;
    case Family::RegisterMma: {
        if (!emulating) return Kind::Native;
        if (f.mma_slow) return Kind::Emulated;
        // run_route's order on Volta: the HMMA kernel, the GEMV, then the register MMA, lane by
        // lane for a one-tile call of at most four tokens (generic_gemm.cuh: kLocal).
        generic::GemmArgs args{};
        args.x                  = reinterpret_cast<const void*>(std::uintptr_t{1} << 32);
        args.ldx                = w.k;
        args.k                  = w.k;
        args.tokens             = tokens;
        args.lane_local         = 1;
        const WeightCodec codec = generic::codec_of(w);
        if (generic::volta_hmma_takes(codec, args)) {
            how = "volta_hmma";
            return Kind::Native;
        }
        if (generic::volta_gemv_takes(codec, args) && (args.ldx % 8) == 0) {
            how = "volta_gemv";
            return Kind::Native;
        }
        if (tokens <= 4 && generic::rmma_schedule_info(executed.schedule).n_tiles == 1) {
            how = "lane_local";
            return Kind::Native;
        }
        return Kind::Emulated;
    }
    case Family::W4a4Mma:
    case Family::W4a8Mma:
    case Family::StagedMma:
    case Family::Nvfp4TiledMma:
    case Family::I8Mma:
        return emulating ? Kind::Emulated : Kind::Native;
    }
    return Kind::None;
}

// What one call (format, policy, shape, epilogue, width) is covered by, and how each family fares.
struct CellAnswer {
    Kind kind = Kind::None;
    std::string families; // "<family>=<kind or refusal>" each
};

CellAnswer answer_cell(const DeviceClass& d, const generic::BuildManifest& manifest,
                       const Weight& w, const generic::Problem& problem, std::int32_t tokens,
                       int& manifest_failures, std::ostream& err) {
    CellAnswer out;
    const bool prefill          = tokens >= generic::kPrefillBandMinTokens;
    const std::size_t workspace = generic::workspace_capacity_bytes(problem, &w, tokens, tokens);
    std::ostringstream families;
    const auto note = [&](Family family, const std::string& what) {
        families << " " << generic::family_name(family) << "=" << what;
    };
    // Volta's HMMA and GEMV kernels are not the route's own schedule: the manifest lists neither.
    const auto take = [&](Route route, Kind kind, const std::string& how, Family family) {
        const bool own_kernel = how.rfind("volta_hmma", 0) == 0 || how.rfind("volta_gemv", 0) == 0;
        if (kind == Kind::Native && !own_kernel &&
            !manifest_holds(manifest, route, problem.qtype, problem.k)) {
            if (++manifest_failures <= 10) {
                err << "  " << generic::qtype_name(problem.qtype) << " T=" << tokens << ": "
                    << generic::route_name(route)
                    << " is admitted, but the build manifest does not hold its kernel\n";
            }
            kind = Kind::None;
        }
        note(family, std::string(kind_name(kind)) + (how.empty() ? "" : "(" + how + ")"));
        out.kind = std::max(out.kind, kind);
    };
    for (const generic::FamilyAlternative& a :
         generic::family_alternatives(problem, w, tokens, prefill)) {
        if (!a.route) {
            note(a.family, a.call != generic::R8CallRefusal::None
                               ? std::string(generic::r8_call_refusal_name(a.call))
                               : std::string(generic::schedule_refusal_name(a.refusal)));
            continue;
        }
        if (a.family == Family::RegisterMma) {
            // Every legal schedule: a one-tile one runs lane by lane on Volta where a wider pick
            // would emulate.
            const auto c = generic::rmma_coverage(problem, w, tokens, prefill);
            Kind best    = Kind::None;
            std::string best_how;
            Route best_route = *a.route;
            for (const generic::ScheduleCoverage& s : c.schedules) {
                if (s.refusal != generic::ScheduleRefusal::None) continue;
                const Route r{Family::RegisterMma, static_cast<std::uint8_t>(s.schedule)};
                std::string how;
                const Kind k = classify(r, w, tokens, how);
                if (k > best) {
                    best       = k;
                    best_how   = how;
                    best_route = r;
                }
            }
            take(best_route, best, best_how, a.family);
            continue;
        }
        const auto described =
            generic::describe_route(d, *a.route, problem.qtype, problem.policy, problem.epilogue,
                                    problem.n, problem.k, tokens, workspace);
        if (!described) {
            note(a.family, "undescribed");
            continue;
        }
        // A demoted route counts as what it runs (the manifest is asked of that route).
        const Route ran = described->resolved.executed;
        std::string how;
        const Kind k = classify(ran, w, tokens, how);
        if (described->resolved.reason != generic::Demotion::None &&
            described->resolved.reason != generic::Demotion::W4a8Admission) {
            how += (how.empty() ? "" : ", ") +
                   std::string(generic::demotion_name(described->resolved.reason)) + " to " +
                   generic::route_name(ran);
        }
        take(ran, k, how, a.family);
    }
    if (out.kind == Kind::None) out.kind = Kind::Scalar; // the scalar kernel takes what none does
    out.families = families.str();
    return out;
}

// One band of one (format, policy) on one board: the worst of its cells, and the first hole.
struct BandAnswer {
    Kind kind = Kind::Native;
    std::string first_hole; // where and what, for the report
};

// `table_always`: print every board's table, not only those with a new hole; `cells_always`: every
// cell's families. `only`: a golden board by name in place of this build's own (a maintainer's
// look at another class under this build's kernels, which is not what the release runs there).
int run(bool table_always, bool cells_always, std::string_view only) {
    if (only.empty() && !release_arch()) {
        std::cout << "coverage report: build sm_" << NINFERNO_BUILD_ARCH_LABEL
                  << " is not a release architecture (" << NINFERNO_RELEASE_CUDA_ARCHS
                  << "): skipped\n";
        return 77;
    }
    std::vector<const generic::FactsBoard*> boards;
    for (const generic::FactsBoard& b : generic::kFactsBoards) {
        if (only.empty() ? serves(b) : b.name == only) boards.push_back(&b);
    }
    if (boards.empty()) {
        std::cout << "coverage report: no golden board of sm_" << NINFERNO_BUILD_ARCH_LABEL
                  << ": skipped\n";
        return 77;
    }
    const auto& allowed = allowed_holes();
    // (entry, format) -> whether a served board of the entry's capability shows the hole.
    std::map<std::pair<std::size_t, QType>, bool> allowed_seen;
    std::map<int, bool> capability_served;
    for (const auto* b : boards) capability_served[b->major * 10 + b->minor] = true;
    const auto allowed_entry = [&](int capability, QType q, LinearPolicy p,
                                   std::int32_t t) -> std::optional<std::size_t> {
        for (std::size_t i = 0; i < allowed.size(); ++i) {
            const Allowed& a = allowed[i];
            if (a.capability == capability && a.policy == p && t >= a.min_tokens &&
                t <= a.max_tokens &&
                std::find(a.formats.begin(), a.formats.end(), q) != a.formats.end()) {
                return i;
            }
        }
        return std::nullopt;
    };

    int failures = 0, manifest_failures = 0;
    std::size_t cells = 0, holes = 0, allowed_holes_seen = 0;
    const auto started = std::chrono::steady_clock::now();
    std::ostringstream tables;
    std::ostringstream detail;
    for (const generic::FactsBoard* board : boards) {
        const DeviceClass d = generic::board_device_class(*board);
        const ScopedDeviceClass scoped(d);
        const generic::BuildManifest manifest = generic::build_manifest();
        const int capability                  = board->major * 10 + board->minor;
        const DeviceFeatures& f               = device_class().features;
        if (manifest.arch != NINFERNO_BUILD_ARCH) {
            std::cerr << "the build manifest names sm_" << manifest.arch << ", the build sm_"
                      << NINFERNO_BUILD_ARCH << '\n';
            ++failures;
        }
        std::ostringstream table;
        table << board->name << " (" << d.id() << ", sm_" << capability
              << (f.mma_tensor && !f.mma_slow ? "" : ", MMA emulated") << ")\n  format     pol ";
        for (const Band& band : bands()) table << " " << band.name;
        table << '\n';
        bool board_has_new = false;
        for (const QType q : served_formats()) {
            for (const LinearPolicy policy : served_policies(q)) {
                std::vector<BandAnswer> answers(bands().size());
                std::vector<bool> band_allowed(bands().size(), false);
                for (std::size_t bi = 0; bi < bands().size(); ++bi) {
                    const Band& band = bands()[bi];
                    for (const std::int32_t t : band.tokens) {
                        for (const Shape shape : kShapes) {
                            for (const generic::Epilogue epilogue :
                                 {generic::Epilogue::Store, generic::Epilogue::SwiGlu}) {
                                const std::int32_t rows =
                                    epilogue == generic::Epilogue::SwiGlu ? shape.n / 2 : shape.n;
                                std::optional<DescribedWeight> w;
                                try {
                                    w.emplace(q, epilogue, rows, shape.k);
                                } catch (const std::exception&) {
                                    continue; // a shape the format cannot hold
                                }
                                if (!generic::supports(w->weight)) continue;
                                const generic::Problem problem{q, policy, rows, shape.k, epilogue};
                                const CellAnswer a = answer_cell(d, manifest, w->weight, problem, t,
                                                                 manifest_failures, detail);
                                ++cells;
                                if (cells_always) {
                                    std::cout << board->name << " " << generic::qtype_name(q) << " "
                                              << policy_name(policy) << " T=" << t << " " << rows
                                              << "x" << shape.k << " "
                                              << generic::epilogue_name(epilogue) << " "
                                              << kind_name(a.kind) << ":" << a.families << '\n';
                                }
                                if (a.kind == Kind::Native) continue;
                                ++holes;
                                const auto entry = allowed_entry(capability, q, policy, t);
                                if (entry) {
                                    ++allowed_holes_seen;
                                    allowed_seen[{*entry, q}] = true;
                                    band_allowed[bi]          = true;
                                } else if (answers[bi].first_hole.empty()) {
                                    std::ostringstream where;
                                    where << board->name << " " << generic::qtype_name(q) << " "
                                          << policy_name(policy) << " T=" << t << " " << rows << "x"
                                          << shape.k << " " << generic::epilogue_name(epilogue)
                                          << ":" << a.families;
                                    answers[bi].first_hole = where.str();
                                }
                                answers[bi].kind = std::min(answers[bi].kind, a.kind);
                            }
                        }
                    }
                }
                std::string line = "  " + std::string(generic::qtype_name(q));
                line.resize(13, ' ');
                line += policy_name(policy);
                line.resize(17, ' ');
                for (std::size_t bi = 0; bi < bands().size(); ++bi) {
                    std::string cell(1,
                                     kind_mark(answers[bi].kind,
                                               band_allowed[bi] && answers[bi].first_hole.empty()));
                    cell.resize(std::string_view(bands()[bi].name).size(), ' ');
                    line += " " + cell;
                    if (!answers[bi].first_hole.empty()) {
                        ++failures;
                        board_has_new = true;
                        if (failures <= 40) {
                            detail << "new hole: " << answers[bi].first_hole << '\n';
                        }
                    }
                }
                table << line << '\n';
            }
        }
        if (board_has_new) tables << table.str();
        if (table_always && !board_has_new) std::cout << table.str();
        std::cout << board->name << ": checked\n";
    }
    failures += manifest_failures;
    for (std::size_t i = 0; i < allowed.size(); ++i) {
        const Allowed& a = allowed[i];
        if (!capability_served[a.capability]) continue;
        for (const QType q : a.formats) {
            if (allowed_seen[{i, q}]) continue;
            ++failures;
            detail << "stale allowlist entry: sm_" << a.capability << " " << generic::qtype_name(q)
                   << " " << policy_name(a.policy) << " T=" << a.min_tokens << ".." << a.max_tokens
                   << " is covered now (\"" << a.reason << "\"): remove it\n";
        }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (failures != 0) {
        std::cerr << tables.str() << detail.str();
        std::cerr << "coverage report: " << failures << " failures (" << cells << " cells, "
                  << holes << " holes, " << allowed_holes_seen << " of them listed)\n";
        return 1;
    }
    std::cout << "coverage report: PASS, " << boards.size() << " boards, " << cells << " cells, "
              << allowed_holes_seen << " listed holes, " << seconds << " s\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool table = false, cells = false;
        std::string_view board;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--table") {
                table = true;
            } else if (arg == "--cells") {
                cells = true;
            } else if (arg == "--board" && i + 1 < argc) {
                board = argv[++i];
            } else {
                std::cerr << "usage: " << argv[0]
                          << " [--table] [--cells] [--board <golden board>]\n";
                return 2;
            }
        }
        return run(table, cells, board);
    } catch (const std::exception& error) {
        std::cerr << "coverage report: " << error.what() << '\n';
        return 1;
    }
}
