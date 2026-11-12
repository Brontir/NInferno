// The generic projection path's coverage as a lint (#619 step 2, K2): on every board this build
// serves, for every format a served model is stored in, under each activation mode it is served
// with and at every width band, at least one family takes the call on the device's own
// instructions. A cell whose only families are the scalar kernel or a tile whose MMA the device
// emulates is a hole. The known holes are listed below (allowed_holes), each pinned to its bands
// and to what the cells fall to, with its reason. A hole not listed fails; so does a listed band
// where any sampled cell is covered or falls to another kind (the entry no longer describes it).
// The boards are substituted DeviceClasses and the weights are described by format and shape,
// their bytes never read; no GPU's timing is read, but one must be visible: ggml's matrix kernel
// reads the device's properties when it describes a route (#763).
//
// What is served, read from the rules that decide it:
// - boards: the golden boards (selector_facts.h kFactsBoards) of this build's compute capability,
//   the boards `ninferno-serve selector-facts` checks this build's kernels for
//   (selector_facts_main). A build of an architecture the release does not ship
//   (NINFERNO_CUDA_ARCHS, CMakeLists.txt) or that no golden board has skips (77).
// - formats: every row of the format table (core/format_facts.h) with a generic codec, an artifact
//   storing each (the converter keeps the GGUF block formats as published and writes the engine's
//   own); NVFP4 only where the planner offers it (product/plan/nvfp4_card.h). This follows the
//   planner, not the converter: its nvfp4 and round-plain modes can still write an NVFP4 artifact
//   for a card below sm_100, which this lint does not sample. FP32 and INT32 have no codec.
// - modes: the policy a site records (convert kept_policy, with and without an NVFP4 activation
//   divisor; head_policy for an output head; A16Only, the recipe default a format kept as
//   published keeps where neither rule reaches, such as a GGUF output head, convert.cpp) as each
//   --activations mode runs it (models::effective_policy): Fast keeps it; Int8 allows eight bits
//   on every linear but an A16Only head or embedding, which keeps BF16 (convert.cpp: the
//   engine's test of the name); Exact keeps BF16 activations under the invariant route profile
//   (ScopedInvariantRoutes, as the engine plans it). Under that profile one route answers a call
//   (the invariant-profile authority: the register MMA with the problem's K split, NVFP4's tiled
//   form of it, or scalar), and run_route clears GemmArgs::lane_local, so Volta's lane-local,
//   HMMA and GEMV forms do not run.
// - widths: decode (1), the verify bands 2..4, 5..8 and 9..16, the prefill band 17..256 and past
//   it, each sampled at its ends and inside. From 17 tokens a call is sampled under the prefill
//   profile and under the speed profile (a batched decode or verify that wide); a prompt shorter
//   than 17 tokens routes as the speed profile's calls of its width, which the bands below 17 are.
//   Four model projections (selector_facts_probe's shapes), each as a Store, an AddResidual and a
//   SwiGLU. A (board, format, mode, band) with no sampled cell fails.
//
// Which families take a cell outside the invariant profile: family_alternatives (generic_gemm.h:
// each family's route, or why it has none), each route resolved as the engine's call runs it
// (describe_route: resolve_route in the workspace the plan reserves, workspace_capacity_bytes, and
// Volta's CUTLASS copy or its scalar demotion), counted only where the build manifest
// (build_manifest) holds the route's kernel for the format. A route is emulated where the device
// has no tensor mma.sync (DeviceFeatures::mma_tensor false, Volta; or mma_slow, TU116 / TU117) and
// its family builds the MMA fragments from FP32 FMAs and dp4a over warp shuffles
// (ops/common/mma.cuh): the register MMA, except where Volta's own kernels take the call (the HMMA
// verify kernel, the decode GEMV, and a one-tile call of at most four tokens, which sums lane by
// lane), W4A8, W4A4, the staged MMA, I8 and the NVFP4 tiles. The tiled family (mma.m8n8k4, and
// CUTLASS over a decoded copy, on Volta), R8 and ggml's vector kernel (dp4a) and ggml's matrix
// kernel run the device's own instructions. Coverage is not speed: a covered cell may still pick
// a slow route.
//
// On failure the test prints the coverage table of each board with a new hole: a line per format
// and mode, a column per band ('.' covered, 'E' emulated only, 'S' scalar only; lower case where
// the band is listed), and for each new hole the families of its first cell. --table prints every
// board's table, --cells every cell's families, --board <name> one golden board of another class
// under this build's kernels (a maintainer's look, not what the release runs there).
#include "convert/formats.h"
#include "core/device_class.h"
#include "core/format_facts.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "models/common/weight_use.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/build_manifest.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"
#include "product/plan/nvfp4_card.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;
using generic::Family;
using generic::Route;
using ops::LinearPolicy;

enum class Kind : std::uint8_t { None, Scalar, Emulated, Native }; // ordered: the best wins

// A served activation mode of a linear: the policy its call runs under, and whether the invariant
// route profile is active (--activations exact).
struct Mode {
    LinearPolicy policy;
    bool invariant;

    bool operator==(const Mode&) const = default;
};

std::string mode_name(Mode m) {
    if (m.invariant) return "exact";
    return m.policy == LinearPolicy::A16Only   ? "a16"
           : m.policy == LinearPolicy::AllowA8 ? "a8"
                                               : "a4";
}

constexpr Mode kA16{LinearPolicy::A16Only, false};
constexpr Mode kA8{LinearPolicy::AllowA8, false};
constexpr Mode kExact{LinearPolicy::A16Only, true};

// ---------------------------------------------------------------------------------------------
// The known holes. Each entry: the boards' compute capability (major * 10 + minor), the mode, the
// formats (empty: every served format, for a hole the mode makes whatever the format), the bands
// (by name), what every sampled cell of those bands falls to, and why the hole is accepted. A
// cell is listed only where it falls to that kind; a listed band any of whose cells is covered or
// falls otherwise fails. Checked only in a build that serves a board of that capability.
struct Allowed {
    int capability;
    Mode mode;
    std::vector<QType> formats;
    std::vector<std::string_view> bands;
    Kind kind;
    const char* reason;
};

const std::vector<Allowed>& allowed_holes() {
    using Q = QType;
    // The verify widths 5..16 on Volta (GV100, sm_70: no tensor mma.sync). A format without a
    // tiled tile of its own (tiled_supports; Q8 has one on Volta) reaches the tensor cores through
    // CUTLASS over a decoded copy, which starts at the prefill band (kVoltaCutlassMinTokens = 17,
    // so that a prefill token's bytes do not depend on its call's width; below it a CUTLASS-only
    // tiled call runs scalar, volta_tiled_runs_scalar, and the selection keeps the register MMA);
    // Volta's HMMA verify kernel reads Q4_K / Q5_K / Q6_K / Q8 only (volta_hmma_codec). These calls
    // run the register MMA's FMA-emulated fragments; decode (1..4 tokens: lane by lane, or the
    // GEMV) and prefill are covered. Market weight: one board class, at verify widths only.
    static const std::vector<std::string_view> kVerify = {"5-8", "9-16"};
    static const std::vector<Allowed> kAllowed         = {
        // BF16 and the FP8 rows (an output head, a BF16 checkpoint's projections), and the
        // ternary and 1/2-bit block formats, which a site keeps at A16 (kept_block_policy). Not
        // MXFP4: no converter path records it A16Only (served_modes).
        {70,
                 kA16,
                 {Q::BF16, Q::FP8_E4M3FN_ROW_BF16, Q::FP8_E4M3FN_ROW_F32, Q::FP8_E4M3FN_B128_F32,
                  Q::GGML_TQ1_0, Q::GGML_TQ2_0, Q::GGML_Q1_0, Q::GGML_Q2_0},
                 kVerify,
                 Kind::Emulated,
                 "Volta A16 verify: no own tile, no HMMA codec, CUTLASS from 17 tokens"},
        // The ggml block formats with neither a tile of their own nor an HMMA codec, at A16: a
        // GGUF output head kept as published keeps the recipe's A16Only (convert.cpp), and keeps
        // it under --activations int8. Their sites elsewhere record AllowA8, which ggml's vector
        // kernel covers through 16 tokens on Volta (vec_max_tokens).
        {70,
                 kA16,
                 {Q::GGML_Q3_K, Q::GGML_Q2_K, Q::GGML_IQ4_XS, Q::GGML_IQ3_XXS, Q::GGML_IQ3_S, Q::GGML_Q4_0,
                  Q::GGML_Q4_1, Q::GGML_Q5_0, Q::GGML_Q5_1, Q::GGML_IQ4_NL, Q::GGML_IQ2_XXS, Q::GGML_IQ2_XS,
                  Q::GGML_IQ2_S, Q::GGML_IQ1_S, Q::GGML_IQ1_M},
                 kVerify,
                 Kind::Emulated,
                 "Volta A16 verify of a ggml block head: no own tile, no HMMA codec"},
        // The same under eight-bit activations (Fast for the FP8 rows and MXFP4, Int8 for every
        // format): no int8 family reads BF16, the FP8 rows or the ternary and 1/2-bit formats, and
        // MXFP4's int8 routes are R8 at decode only and W4A8, emulated on Volta. The other int8
        // formats are covered: ggml's vector kernel through 16 tokens (vec_max_tokens), the tiles.
        {70,
                 kA8,
                 {Q::BF16, Q::FP8_E4M3FN_ROW_BF16, Q::FP8_E4M3FN_ROW_F32, Q::FP8_E4M3FN_B128_F32, Q::MXFP4,
                  Q::GGML_TQ1_0, Q::GGML_TQ2_0, Q::GGML_Q1_0, Q::GGML_Q2_0},
                 kVerify,
                 Kind::Emulated,
                 "Volta A8 verify: no int8 route past decode, no own tile, CUTLASS from 17 tokens"},
        // --activations exact on Volta: the invariant profile answers before Volta's tiled and HMMA
        // authorities, with the register MMA at every width, and run_route clears lane_local
        // under it, so neither the lane-local sum, the GEMV nor the HMMA kernel takes the call:
        // every exact projection runs the FMA-emulated fragments, whatever its format. Market
        // weight: exact mode is opt-in (spec exact), on one board class.
        {70,
                 kExact,
                 {},
                 {"1", "2-4", "5-8", "9-16", "17-256", "257+"},
                 Kind::Emulated,
                 "Volta exact: the invariant profile's register MMA, lane_local cleared"},
    };
    return kAllowed;
}

// ---------------------------------------------------------------------------------------------
// The served cells.

struct Band {
    std::string_view name;
    std::vector<std::int32_t> tokens; // the widths sampled
};

const std::vector<Band>& bands() {
    static const std::vector<Band> kBands = {
        {"1", {1}},
        {"2-4", {2, 3, 4}},
        {"5-8", {5, 6, 8}},
        {"9-16", {9, 12, 16}},
        {"17-256", {17, 32, 64, 256}},
        {"257+", {512, 2048}},
    };
    return kBands;
}

struct Shape {
    std::int32_t n, k;
};

// selector_facts_probe's projections and epilogues (selector_facts.cpp): a GQA k/v, a square, a
// gate/up and a down of the 27B..32B models; a SwiGLU call reads half of the rows as its gate.
constexpr Shape kShapes[] = {{1024, 4096}, {4096, 4096}, {17408, 5120}, {5120, 17408}};
constexpr generic::Epilogue kEpilogues[] = {
    generic::Epilogue::Store, generic::Epilogue::AddResidual, generic::Epilogue::SwiGlu};

std::vector<QType> served_formats(int capability) {
    std::vector<QType> out;
    for (const FormatFacts& f : kFormatFacts) {
        if (!f.codec) continue;
        if (f.q == QType::NVFP4 && !product::nvfp4_planned_on(capability)) continue;
        out.push_back(f.q);
    }
    return out;
}

LinearPolicy linear_policy(artifact::ActivationPolicy p) {
    switch (p) {
    case artifact::ActivationPolicy::A16Only:
        return LinearPolicy::A16Only;
    case artifact::ActivationPolicy::AllowA8:
        return LinearPolicy::AllowA8;
    case artifact::ActivationPolicy::AllowA4:
        return LinearPolicy::AllowA4;
    }
    return LinearPolicy::A16Only;
}

// The modes a linear of the format is served under: each policy a site records (convert's
// kept_policy and head_policy, with and without an activation divisor; and the recipe's default
// A16Only where a format is kept as published, kept_where_decoded) under each --activations mode
// (models::effective_policy; Exact also under the invariant profile).
std::vector<Mode> served_modes(QType q) {
    std::vector<Mode> out;
    const auto add = [&](Mode m) {
        if (std::find(out.begin(), out.end(), m) == out.end()) out.push_back(m);
    };
    std::vector<artifact::ActivationPolicy> recorded;
    for (const bool divisor : {true, false}) {
        const convert::KeptEncoding site{q, divisor};
        recorded.push_back(convert::kept_policy(site));
        recorded.push_back(convert::head_policy(site));
    }
    // A format kept as published (GGUF Q8_0 and block formats, the u4 and 3-bit groupwise words)
    // keeps the recipe's default A16Only where convert.cpp sets no policy: a GGUF output head, a
    // Q8 grouped-absmax site (dense.cpp). MXFP4 is not such a format: it is written for expert
    // banks only, at kept_block_policy (AllowA8), and an MXFP4 head is kept as NVFP4.
    if (convert::kept_where_decoded(q)) recorded.push_back(artifact::ActivationPolicy::A16Only);
    for (const artifact::ActivationPolicy policy : recorded) {
        const LinearPolicy r = linear_policy(policy);
        add({models::effective_policy(r, ActivationMode::Fast), false});
        // Int8 runs every linear at AllowA8 but an A16Only head or embedding, which keeps BF16
        // (convert.cpp, the engine's test of the parameter's name): that one is Fast's row above.
        add({models::effective_policy(r, ActivationMode::Int8), false});
        add({models::effective_policy(r, ActivationMode::Exact), true});
    }
    std::sort(out.begin(), out.end(), [](Mode a, Mode b) {
        return std::tie(a.invariant, a.policy) < std::tie(b.invariant, b.policy);
    });
    return out;
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

char kind_mark(Kind k, bool listed) {
    const char c = k == Kind::Native ? '.' : k == Kind::Emulated ? 'E' : 'S';
    return listed && c != '.' ? static_cast<char>(c - 'A' + 'a') : c;
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
// `tokens` tokens of `w` (aligned and contiguous, as the engine passes it); `invariant`: under the
// invariant profile, where run_route clears lane_local. `how` names Volta's own kernels where
// they take a register-MMA call.
Kind classify(Route executed, const Weight& w, std::int32_t tokens, bool invariant,
              std::string& how) {
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
        // run_route's order without tensor mma.sync: the HMMA kernel, the GEMV, then the register
        // MMA, lane by lane for a one-tile call of at most four tokens (generic_gemm.cuh kLocal),
        // each only where lane_local is set (outside the invariant profile).
        generic::GemmArgs args{};
        args.x                  = reinterpret_cast<const void*>(std::uintptr_t{1} << 32);
        args.ldx                = w.k;
        args.k                  = w.k;
        args.tokens             = tokens;
        args.lane_local         = invariant ? 0 : 1;
        const WeightCodec codec = generic::codec_of(w);
        if (generic::volta_hmma_takes(codec, args)) {
            how = "volta_hmma";
            return Kind::Native;
        }
        if (generic::volta_gemv_takes(codec, args)) {
            how = "volta_gemv";
            return Kind::Native;
        }
        if (args.lane_local != 0 && tokens <= 4 &&
            generic::rmma_schedule_info(executed.schedule).n_tiles == 1) {
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

// What one call (format, mode, shape, epilogue, width, profile) is covered by, and how each family
// fares.
struct CellAnswer {
    Kind kind = Kind::None;
    std::string families; // "<family>=<kind or refusal>" each
};

struct CellContext {
    const DeviceClass& device;
    const generic::BuildManifest& manifest;
    int& manifest_failures;
    std::ostream& err;
};

CellAnswer answer_cell(const CellContext& c, const Weight& w, const generic::Problem& problem,
                       std::int32_t tokens, bool prefill, bool invariant) {
    CellAnswer out;
    const std::size_t workspace = generic::workspace_capacity_bytes(problem, &w, tokens, tokens);
    std::ostringstream families;
    const auto note = [&](Family family, const std::string& what) {
        families << " " << generic::family_name(family) << "=" << what;
    };
    // Volta's HMMA and GEMV kernels are not the route's own schedule: the manifest lists neither.
    const auto take = [&](Route route, Kind kind, const std::string& how) {
        const bool own_kernel = how.rfind("volta_hmma", 0) == 0 || how.rfind("volta_gemv", 0) == 0;
        // Native or emulated, a route the build does not hold runs nothing: an emulated one would
        // otherwise satisfy an entry that pins it.
        if ((kind == Kind::Native || kind == Kind::Emulated) && !own_kernel &&
            !manifest_holds(c.manifest, route, problem.qtype, problem.k)) {
            if (++c.manifest_failures <= 10) {
                c.err << "  " << generic::qtype_name(problem.qtype) << " T=" << tokens << ": "
                      << generic::route_name(route)
                      << " is admitted, but the build manifest does not hold its kernel\n";
            }
            kind = Kind::None;
        }
        note(route.family, std::string(kind_name(kind)) + (how.empty() ? "" : "(" + how + ")"));
        out.kind = std::max(out.kind, kind);
    };
    // The route as the engine's call runs it: a demoted route counts as what it runs.
    const auto take_resolved = [&](Route asked) {
        const auto described =
            generic::describe_route(c.device, asked, problem.qtype, problem.policy,
                                    problem.epilogue, problem.n, problem.k, tokens, workspace);
        if (!described) {
            note(asked.family, "undescribed");
            return;
        }
        const Route ran = described->resolved.executed;
        std::string how;
        const Kind k = classify(ran, w, tokens, invariant, how);
        if (described->resolved.reason != generic::Demotion::None &&
            described->resolved.reason != generic::Demotion::W4a8Admission) {
            how += (how.empty() ? "" : ", ") +
                   std::string(generic::demotion_name(described->resolved.reason)) + " from " +
                   generic::route_name(asked);
        }
        take(ran, k, how);
    };
    if (invariant) {
        // The invariant profile admits the one route its authority answers (invariant_profile).
        take_resolved(generic::select_route(problem, w, tokens, prefill));
    } else {
        for (const generic::FamilyAlternative& a :
             generic::family_alternatives(problem, w, tokens, prefill)) {
            if (!a.route) {
                note(a.family, a.call != generic::R8CallRefusal::None
                                   ? std::string(generic::r8_call_refusal_name(a.call))
                                   : std::string(generic::schedule_refusal_name(a.refusal)));
                continue;
            }
            if (a.family != Family::RegisterMma) {
                take_resolved(*a.route);
                continue;
            }
            // Every legal register-MMA schedule: a one-tile one runs lane by lane on Volta where
            // a wider pick would emulate.
            const auto coverage = generic::rmma_coverage(problem, w, tokens, prefill);
            Kind best           = Kind::None;
            std::string best_how;
            Route best_route = *a.route;
            for (const generic::ScheduleCoverage& s : coverage.schedules) {
                if (s.refusal != generic::ScheduleRefusal::None) continue;
                const Route r{Family::RegisterMma, static_cast<std::uint8_t>(s.schedule)};
                std::string how;
                const Kind k = classify(r, w, tokens, false, how);
                if (k > best) {
                    best       = k;
                    best_how   = how;
                    best_route = r;
                }
            }
            take(best_route, best, best_how);
        }
    }
    if (out.kind == Kind::None) out.kind = Kind::Scalar; // the scalar kernel takes what none does
    out.families = families.str();
    return out;
}

// One band of one (format, mode) on one board.
struct BandAnswer {
    Kind kind         = Kind::Native; // the worst of its cells
    std::size_t cells = 0, listed = 0;
    std::string first_hole; // the first hole not listed: where and what, for the report
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
    const auto listed   = [&](int capability, QType q, Mode m,
                            std::string_view band) -> const Allowed* {
        for (const Allowed& a : allowed) {
            if (a.capability == capability && a.mode == m &&
                std::find(a.bands.begin(), a.bands.end(), band) != a.bands.end() &&
                (a.formats.empty() ||
                 std::find(a.formats.begin(), a.formats.end(), q) != a.formats.end())) {
                return &a;
            }
        }
        return nullptr;
    };

    int failures = 0, manifest_failures = 0;
    std::size_t cells = 0, holes = 0, listed_holes = 0;
    const auto started = std::chrono::steady_clock::now();
    std::ostringstream tables, detail;
    // A band an entry names that is not a band of this test (a typo) lists nothing: stale.
    for (const Allowed& a : allowed) {
        for (const std::string_view name : a.bands) {
            if (std::none_of(bands().begin(), bands().end(),
                             [&](const Band& b) { return b.name == name; })) {
                ++failures;
                detail << "stale allowlist entry: band \"" << name << "\" is no band (\""
                       << a.reason << "\")\n";
            }
        }
    }
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
        const CellContext context{d, manifest, manifest_failures, detail};
        std::ostringstream table;
        table << board->name << " (" << d.id() << ", sm_" << capability
              << (f.mma_tensor && !f.mma_slow ? "" : ", MMA emulated") << ")\n  format     mode ";
        for (const Band& band : bands()) table << " " << band.name;
        table << '\n';
        bool board_has_new = false;
        const auto served  = served_formats(capability);
        for (const QType q : served) {
            for (const Mode mode : served_modes(q)) {
                std::optional<ops::ScopedInvariantRoutes> invariant;
                if (mode.invariant) invariant.emplace();
                std::string head = "  " + std::string(generic::qtype_name(q));
                head.resize(13, ' ');
                head += mode_name(mode);
                head.resize(18, ' ');
                table << head;
                std::vector<BandAnswer> answers(bands().size());
                for (std::size_t bi = 0; bi < bands().size(); ++bi) {
                    const Band& band     = bands()[bi];
                    const Allowed* entry = listed(capability, q, mode, band.name);
                    BandAnswer& answer   = answers[bi];
                    for (const std::int32_t t : band.tokens) {
                        for (const bool prefill : {false, true}) {
                            if (prefill && t < generic::kPrefillBandMinTokens) continue;
                            for (const Shape shape : kShapes) {
                                for (const generic::Epilogue epilogue : kEpilogues) {
                                    const std::int32_t rows = epilogue == generic::Epilogue::SwiGlu
                                                                  ? shape.n / 2
                                                                  : shape.n;
                                    std::optional<DescribedWeight> w;
                                    try {
                                        w.emplace(q, epilogue, rows, shape.k);
                                    } catch (const std::exception&) {
                                        continue; // a shape the format cannot hold
                                    }
                                    if (!generic::supports(w->weight)) continue;
                                    const generic::Problem problem{q, mode.policy, rows, shape.k,
                                                                   epilogue};
                                    const CellAnswer a = answer_cell(context, w->weight, problem, t,
                                                                     prefill, mode.invariant);
                                    ++cells;
                                    ++answer.cells;
                                    if (cells_always) {
                                        std::cout << board->name << " " << generic::qtype_name(q)
                                                  << " " << mode_name(mode) << " T=" << t
                                                  << (prefill ? " prefill " : " speed ") << rows
                                                  << "x" << shape.k << " "
                                                  << generic::epilogue_name(epilogue) << " "
                                                  << kind_name(a.kind) << ":" << a.families << '\n';
                                    }
                                    answer.kind = std::min(answer.kind, a.kind);
                                    if (a.kind == Kind::Native) continue;
                                    ++holes;
                                    if (entry != nullptr && a.kind == entry->kind) {
                                        ++listed_holes;
                                        ++answer.listed;
                                    } else if (answer.first_hole.empty()) {
                                        std::ostringstream where;
                                        where << board->name << " " << generic::qtype_name(q) << " "
                                              << mode_name(mode) << " T=" << t
                                              << (prefill ? " prefill " : " ") << rows << "x"
                                              << shape.k << " " << generic::epilogue_name(epilogue)
                                              << " " << kind_name(a.kind) << ":" << a.families;
                                        answer.first_hole = where.str();
                                    }
                                }
                            }
                        }
                    }
                    std::string mark(
                        1, kind_mark(answer.kind, entry != nullptr && answer.first_hole.empty()));
                    if (answer.cells == 0) {
                        ++failures;
                        board_has_new = true;
                        mark          = "0";
                        detail << "no sampled cell: " << board->name << " "
                               << generic::qtype_name(q) << " " << mode_name(mode) << " T "
                               << band.name << '\n';
                    } else if (!answer.first_hole.empty()) {
                        ++failures;
                        board_has_new = true;
                        if (failures <= 40) detail << "new hole: " << answer.first_hole << '\n';
                    } else if (entry != nullptr && answer.listed != answer.cells) {
                        // The entry pins every sampled cell of its band to its kind.
                        ++failures;
                        board_has_new = true;
                        detail << "listed band no longer a whole hole: " << board->name << " "
                               << generic::qtype_name(q) << " " << mode_name(mode) << " T "
                               << band.name << ": " << answer.listed << " of " << answer.cells
                               << " cells " << kind_name(entry->kind) << " (\"" << entry->reason
                               << "\"): narrow or remove the entry\n";
                    }
                    mark.resize(band.name.size(), ' ');
                    table << " " << mark;
                }
                table << '\n';
            }
        }
        // An entry naming a format this build does not serve on the board (NVFP4 below sm_100), or
        // a mode its sites are not served under (MXFP4 or Q4 G64 at A16), is stale too.
        for (const Allowed& a : allowed) {
            if (a.capability != capability) continue;
            for (const QType q : a.formats) {
                const auto modes = served_modes(q);
                if (std::find(served.begin(), served.end(), q) == served.end() ||
                    std::find(modes.begin(), modes.end(), a.mode) == modes.end()) {
                    ++failures;
                    detail << "stale allowlist entry: sm_" << a.capability << " "
                           << generic::qtype_name(q) << " " << mode_name(a.mode)
                           << " is not served (\"" << a.reason << "\"): remove it\n";
                }
            }
        }
        if (board_has_new) tables << table.str();
        if (table_always && !board_has_new) std::cout << table.str();
        std::cout << board->name << ": checked\n";
    }
    failures += manifest_failures;
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (failures != 0) {
        std::cerr << tables.str() << detail.str();
        std::cerr << "coverage report: " << failures << " failures (" << cells << " cells, "
                  << holes << " holes, " << listed_holes << " of them listed)\n";
        return 1;
    }
    std::cout << "coverage report: PASS, " << boards.size() << " boards, " << cells << " cells, "
              << listed_holes << " listed holes, " << seconds << " s\n";
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
