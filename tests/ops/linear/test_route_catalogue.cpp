// The route catalogue (#614, step A2): for every (family, schedule) the build compiles, on every
// facts board, the fields the selector reads about a route agree with one another:
//
//  - summation_order: the register order holds the register-MMA schedules that split no K and the
//    staged ones, the I8 order the I8 tiles, every other route is its own order; same_summation is
//    symmetric, and routes of one order agree on route_is_token_invariant;
//  - route_is_token_invariant: each family's answer is the one its kernel's contract gives, and a
//    route of the register or I8 order is invariant;
//  - the activation class of the family (A16, A8, A4): a route admitted under the A16-only policy
//    is an A16 route, and no route of the A4 class is admitted below AllowA4;
//  - the workspace need: A16 routes need none, the routes that quantize the activation do (R8 only
//    at its verify widths; ggml's matrix kernel has its own query, which reads the device, so it is
//    not asked here);
//  - the catalogue itself: the schedule counts per family are pinned in this file, and the names
//    of the routes are distinct and parse back.
//
// And profile_admits, the one profile filter (route_authorities.h), says what those fields say: in
// the prefill band it is route_is_token_invariant, under the other profiles and below the band it
// admits every route, profile_admits_as is same_summation under the prefill profile. Every
// routes/*.json on disk passes the filter (its prefill routes in the band, its prefill_bands as
// their prefill route), the check the loader makes at load. No GPU.
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_authorities.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <iterator>
#include <vector>

#ifndef NINFERNO_SOURCE_DIR
#    error "NINFERNO_SOURCE_DIR is defined by the test's CMake target"
#endif

namespace {

using namespace ninferno;
using namespace ninferno::test;
namespace generic = ninferno::ops::detail::generic;
using generic::Family;
using generic::Route;
using ops::LinearPolicy;

int g_failed = 0;

void fail(const std::string& what) {
    ++g_failed;
    if (g_failed <= 40) std::cerr << "FAIL: " << what << '\n';
}

enum class Activation { A16, A8, A4 };

const char* catalogue_family_name(Family f) {
    switch (f) {
    case Family::RegisterMma:
        return "rmma";
    case Family::Scalar:
        return "scalar";
    case Family::W4a4Mma:
        return "w4a4";
    case Family::TiledMma:
        return "tiled";
    case Family::W4a8Mma:
        return "w4a8";
    case Family::StagedMma:
        return "staged";
    case Family::Nvfp4TiledMma:
        return "nvfp4-tiled";
    case Family::R8Mma:
        return "r8";
    case Family::GgmlMmq:
        return "ggml-mmq";
    case Family::GgmlVec:
        return "ggml-vec";
    case Family::I8Mma:
        return "i8";
    }
    return "?";
}

// The activation class of a family, as its header comment states it.
Activation activation_of(Family f) {
    switch (f) {
    case Family::W4a4Mma:
        return Activation::A4;
    case Family::W4a8Mma:
    case Family::R8Mma:
    case Family::GgmlMmq:
    case Family::GgmlVec:
    case Family::I8Mma:
        return Activation::A8;
    case Family::RegisterMma:
    case Family::Scalar:
    case Family::TiledMma:
    case Family::StagedMma:
    case Family::Nvfp4TiledMma:
        return Activation::A16;
    }
    return Activation::A16;
}

// The schedules each family has, written down here and not read from the implementation: a
// schedule added, removed or moved to another family changes a count below, and the test then asks
// its author to say which checks the new route needs. W4A4 is compiled on the Blackwell build only
// (generic_gemm.cpp, w4a4_schedule_count).
#ifdef NINFERNO_WITHOUT_NVFP4
constexpr int kW4a4Schedules = 0;
#else
constexpr int kW4a4Schedules = 4;
#endif

struct FamilyCount {
    Family family;
    int count;
    int (*schedules)();
};

int one_schedule() { return 1; }

const FamilyCount kCatalogue[] = {
    {Family::RegisterMma, 20, generic::rmma_schedule_count},
    {Family::Scalar, 1, one_schedule},
    {Family::W4a4Mma, kW4a4Schedules, generic::w4a4_schedule_count},
    {Family::TiledMma, 3, generic::tiled_schedule_count},
    {Family::W4a8Mma, 6, generic::w4a8_schedule_count},
    {Family::StagedMma, 13, generic::staged_schedule_count},
    {Family::Nvfp4TiledMma, 3, generic::nvfp4_tiled_schedule_count},
    {Family::R8Mma, 4, generic::r8_schedule_count},
    {Family::GgmlMmq, 1, one_schedule},
    {Family::GgmlVec, 1, one_schedule},
    {Family::I8Mma, 3, generic::i8_schedule_count},
};

// The last family is the last of the enumeration: a family added makes this fail to compile.
static_assert(static_cast<std::size_t>(Family::I8Mma) + 1 == std::size(kCatalogue));

// Every (family, schedule) of this build, after the pinned counts have been compared with the
// implementation's.
std::vector<Route> catalogue() {
    std::vector<Route> routes;
    for (const FamilyCount& f : kCatalogue) {
        if (f.schedules() != f.count) {
            fail(std::string(catalogue_family_name(f.family)) + ": " +
                 std::to_string(f.schedules()) + " schedules, the catalogue pins " +
                 std::to_string(f.count));
        }
        for (int i = 0; i < f.count; ++i) {
            routes.push_back(Route{f.family, static_cast<std::uint8_t>(i)});
        }
    }
    return routes;
}

std::string label(Route r) {
    return std::string(catalogue_family_name(r.family)) + "#" + std::to_string(r.schedule);
}

// The names of the catalogue's routes are distinct and parse back to the route (a schedule moved
// to another family or renumbered keeps its name pointing at the same tile).
void check_names(const std::vector<Route>& routes, const std::string& board) {
    std::set<std::string> seen;
    for (const Route r : routes) {
        const std::string name = generic::route_name(r);
        if (!seen.insert(name).second)
            fail(board + " " + label(r) + ": the name '" + name + "' twice");
        const auto back = generic::parse_route(name);
        if (!back || !(*back == r))
            fail(board + " " + label(r) + ": '" + name + "' does not parse back");
    }
}

// ---- the fields of a route, on their own
// ---------------------------------------------------------

void check_summation(const std::vector<Route>& routes, const std::string& board) {
    using Kind = generic::SummationOrder::Kind;
    for (const Route r : routes) {
        const std::string where             = board + " " + label(r);
        const generic::SummationOrder order = generic::summation_order(r);
        const bool invariant                = generic::route_is_token_invariant(r);
        if (!generic::same_summation(r, r)) fail(where + ": not its own order");
        switch (order.kind) {
        case Kind::Register:
            if (r.family != Family::RegisterMma && r.family != Family::StagedMma) {
                fail(where + ": the register order held by another family");
            }
            if (!invariant) fail(where + ": the register order, not token-invariant");
            break;
        case Kind::I8:
            if (r.family != Family::I8Mma) fail(where + ": the I8 order held by another family");
            if (!invariant) fail(where + ": the I8 order, not token-invariant");
            break;
        case Kind::Own:
            if (!(order.own == r)) fail(where + ": an own order that names another route");
            if (r.family == Family::I8Mma || r.family == Family::StagedMma) {
                fail(where + ": a family of a shared order has an order of its own");
            }
            break;
        }
        if (r.family == Family::I8Mma && order.kind != Kind::I8) fail(where + ": not the I8 order");
        if (r.family == Family::StagedMma && order.kind != Kind::Register) {
            fail(where + ": staged does not sum in the register order");
        }
        if (r.family == Family::RegisterMma) {
            const bool one_k = generic::rmma_schedule_info(r.schedule).warps_k == 1;
            if (!one_k && (order.kind != Kind::Own || invariant)) {
                fail(where + ": a K-split tile in a shared order or token-invariant");
            }
            // WarpsK == 1 is invariant and in the register order, but for a lane-local tile
            // (Volta), which is neither.
            if (one_k && (invariant != (order.kind == Kind::Register))) {
                fail(where + ": token-invariance and the register order disagree");
            }
        }
    }
    // Routes of one order sum alike, so they agree on token invariance; the relation is symmetric.
    for (const Route a : routes) {
        for (const Route b : routes) {
            const bool same = generic::same_summation(a, b);
            if (same != generic::same_summation(b, a)) {
                fail(board + " " + label(a) + " / " + label(b) + ": same_summation not symmetric");
            }
            if (same &&
                generic::route_is_token_invariant(a) != generic::route_is_token_invariant(b)) {
                fail(board + " " + label(a) + " / " + label(b) +
                     ": one order, two answers on token invariance");
            }
        }
    }
}

void check_invariance(const std::vector<Route>& routes, const std::string& board) {
    for (const Route r : routes) {
        const std::string where = board + " " + label(r);
        const bool invariant    = generic::route_is_token_invariant(r);
        switch (r.family) {
        case Family::W4a4Mma:
        case Family::TiledMma:
        case Family::W4a8Mma:
        case Family::StagedMma:
        case Family::R8Mma:
        case Family::Scalar:
        case Family::GgmlVec:
        case Family::I8Mma:
            if (!invariant)
                fail(where + ": a family whose sums are a token's own is not invariant");
            break;
        case Family::GgmlMmq:
            if (invariant) fail(where + ": stream-k splits K by the call, yet token-invariant");
            break;
        case Family::Nvfp4TiledMma:
            if (invariant != (generic::nvfp4_tiled_schedule_info(r.schedule).classes == 1)) {
                fail(where + ": invariant exactly where the classes are 1");
            }
            break;
        case Family::RegisterMma:
            break; // checked against warps_k in check_summation
        }
    }
}

void check_workspace(const std::vector<Route>& routes, const std::string& board) {
    constexpr std::int32_t kK = 4096;
    for (const Route r : routes) {
        const std::string where = board + " " + label(r);
        for (const std::int32_t t : {1, 2, 8, 16, 17, 128}) {
            const std::size_t bytes = generic::route_workspace_bytes(r, kK, t);
            switch (r.family) {
            case Family::W4a4Mma:
            case Family::W4a8Mma:
            case Family::I8Mma:
            case Family::GgmlVec:
                if (bytes == 0 && t >= 2) {
                    fail(where + " T=" + std::to_string(t) + ": an A8/A4 route with no workspace");
                }
                break;
            case Family::R8Mma: {
                // R8 quantizes in its own CTA at decode; its verify widths read the W4A8 planes.
                const bool verify = t >= 2 && t <= generic::kR8VerifyTokens;
                if ((bytes != 0) != verify) {
                    fail(where + " T=" + std::to_string(t) + ": R8 workspace " +
                         std::to_string(bytes));
                }
                break;
            }
            case Family::GgmlMmq: // its own query (ggml_mmq_workspace_bytes), not asked here: the
                                  // query bounds its stream-k fixup by the device's SM count
                                  // (cudaGetDeviceProperties), which a host test has no board for
                if (bytes != 0) fail(where + ": ggml's matrix kernel in route_workspace_bytes");
                break;
            case Family::RegisterMma:
            case Family::Scalar:
            case Family::TiledMma:
            case Family::StagedMma:
            case Family::Nvfp4TiledMma:
                if (bytes != 0)
                    fail(where + " T=" + std::to_string(t) + ": an A16 route with a workspace");
                break;
            }
        }
    }
}

// A route admitted under a policy has the activation class the policy allows.
struct Format {
    QType qtype;
    std::int32_t n, k;
};

constexpr Format kFormats[] = {
    {QType::BF16, 4096, 4096},        {QType::Q4_G64_FP16, 4096, 4096},
    {QType::Q8_G32_FP16, 4096, 4096}, {QType::FP8_E4M3FN_ROW_BF16, 4096, 4096},
    {QType::NVFP4, 4096, 4096},       {QType::GGML_Q4_K, 4096, 4096},
    {QType::GGML_Q6_K, 4096, 4096},   {QType::GGML_IQ3_S, 4096, 4096},
    {QType::GGML_Q4_0, 4096, 4096},
};

// The host copy of a patterned weight of the format; the device view is made from it with a
// payload address nothing dereferences.
std::optional<quantized_weight::PackedWeight> host_weight(const Format& f) {
    quantized_weight::PatternedWeightOptions options;
    if (f.qtype == QType::NVFP4) options.weight_scale_divisor = options.input_scale_divisor = 1.0F;
    try {
        return quantized_weight::make_patterned_weight(f.qtype, f.n, f.k, 7U, options);
    } catch (const std::invalid_argument&) { return std::nullopt; }
}

void check_activation(const std::vector<Route>& routes, const std::string& board) {
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    for (const Format& f : kFormats) {
        const auto host = host_weight(f);
        if (!host) continue;
        const Weight w = host->device_weight(payload);
        if (!generic::supports(w)) continue;
        for (const Route r : routes) {
            const std::string where = board + " " + generic::qtype_name(f.qtype) + " " + label(r);
            try {
                if (generic::route_admits(r, w, LinearPolicy::A16Only) &&
                    activation_of(r.family) != Activation::A16) {
                    fail(where + ": admitted under A16Only, yet an int8/FP4 activation route");
                }
                if (generic::route_admits(r, w, LinearPolicy::AllowA8) &&
                    activation_of(r.family) == Activation::A4) {
                    fail(where + ": an A4 route admitted under AllowA8");
                }
            } catch (const std::exception& error) { fail(where + ": threw " + error.what()); }
        }
    }
}

// ---- the filter says what the fields say
// ---------------------------------------------------------

void check_filter(const std::vector<Route>& routes, const std::string& board) {
    const generic::Ask prefill_wide{generic::Profile::Prefill, 17, 1};
    const generic::Ask prefill_narrow{generic::Profile::Prefill, 16, 1};
    const generic::Ask speed{generic::Profile::Speed, 512, 1};
    const generic::Ask invariant{generic::Profile::Invariant, 512, 1};
    for (const Route a : routes) {
        const std::string where = board + " " + label(a);
        if (generic::profile_admits(a, prefill_wide) != generic::route_is_token_invariant(a)) {
            fail(where + ": the prefill band's filter is not token invariance");
        }
        for (const generic::Ask& open : {prefill_narrow, speed, invariant}) {
            if (!generic::profile_admits(a, open)) {
                fail(where + ": " + generic::profile_name(open.profile) +
                     " T=" + std::to_string(open.tokens) + " refuses a route");
            }
        }
        for (const Route b : routes) {
            const bool same = generic::same_summation(a, b);
            if (generic::profile_admits_as(a, prefill_wide, b) != same ||
                generic::profile_admits_as(a, prefill_narrow, b) != same) {
                fail(where + " as " + label(b) + ": the prefill filter is not same_summation");
            }
            if (!generic::profile_admits_as(a, speed, b)) {
                fail(where + " as " + label(b) + ": the speed profile refuses a stand-in");
            }
        }
    }
}

// With a weight, the prefill band also keeps a tiled-codec format on its tiled route: the answer is
// "token-invariant, and not a non-tiled route of a weight the tiled kernels read". Below the band
// and under the other profiles the weight changes nothing. The cases that matter must exist: a
// format with a tiled kernel refusing an invariant non-tiled route and admitting the tiled one,
// and a format without a tiled kernel admitting the invariant non-tiled route.
void check_filter_weighted(const std::vector<Route>& routes, const std::string& board) {
    const generic::Ask prefill_wide{generic::Profile::Prefill, 17, 1};
    const generic::Ask prefill_narrow{generic::Profile::Prefill, 16, 1};
    const generic::Ask speed{generic::Profile::Speed, 512, 1};
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    int tiled_refused = 0, tiled_kept = 0, plain_kept = 0;
    for (const Format& f : kFormats) {
        const auto host = host_weight(f);
        if (!host) continue;
        const Weight w = host->device_weight(payload);
        if (!generic::supports(w)) continue;
        const bool tiled = generic::tiled_supports(generic::codec_of(w));
        for (const Route r : routes) {
            const std::string where = board + " " + generic::qtype_name(f.qtype) + " " + label(r);
            const bool invariant    = generic::route_is_token_invariant(r);
            const bool want         = invariant && !(tiled && r.family != Family::TiledMma);
            if (generic::profile_admits(r, prefill_wide, &w) != want) {
                fail(where + ": the prefill band's filter with a weight is not (invariant, and "
                             "tiled where the codec has a tiled kernel)");
            }
            for (const generic::Ask& open : {prefill_narrow, speed}) {
                if (!generic::profile_admits(r, open, &w)) {
                    fail(where + ": " + generic::profile_name(open.profile) +
                         " refuses a route given a weight");
                }
            }
            if (invariant && tiled && r.family != Family::TiledMma) tiled_refused += want ? 0 : 1;
            if (invariant && tiled && r.family == Family::TiledMma) tiled_kept += want ? 1 : 0;
            if (invariant && !tiled && r.family != Family::TiledMma) plain_kept += want ? 1 : 0;
        }
    }
    if (tiled_refused == 0)
        fail(board + ": no tiled-codec format refused an invariant register route");
    if (tiled_kept == 0) fail(board + ": no tiled-codec format kept its tiled route");
    if (plain_kept == 0) fail(board + ": no format without a tiled kernel kept an invariant route");
}

void run_board(const generic::FactsBoard& b) {
    const ScopedDeviceClass scoped(generic::board_device_class(b));
    const std::vector<Route> routes = catalogue();
    if (routes.empty()) fail(std::string(b.name) + ": an empty catalogue");
    check_names(routes, b.name);
    check_summation(routes, b.name);
    check_invariance(routes, b.name);
    check_workspace(routes, b.name);
    check_activation(routes, b.name);
    check_filter(routes, b.name);
    check_filter_weighted(routes, b.name);
    std::cout << b.name << ": " << routes.size() << " routes\n";
}

// ---- the tables on disk
// --------------------------------------------------------------------------

const generic::FactsBoard& board_named(std::string_view name) {
    for (const generic::FactsBoard& b : generic::kFactsBoards) {
        if (b.name == name) return b;
    }
    throw std::runtime_error("no such facts board");
}

Route route_of(const std::string& name, const std::string& where) {
    const auto r = generic::parse_route(name);
    if (!r) {
        fail(where + ": unknown route '" + name + "'");
        return Route{};
    }
    return *r;
}

// A table is judged with the kernels of a device that runs them: a one-tile register MMA sums lane
// by lane without tensor mma.sync, so Volta's table on a Volta board and every other one on Ada.
void check_table(const std::filesystem::path& path) {
    std::ifstream in(path);
    const nlohmann::json root = nlohmann::json::parse(in);
    const std::string stem    = path.stem().string();
    const ScopedDeviceClass scoped(
        generic::board_device_class(board_named(stem.rfind("gv100", 0) == 0 ? "v100" : "rtx4090")));
    int prefill_routes = 0, prefill_bands = 0;
    for (const nlohmann::json& e : root.at("entries")) {
        if (!e.is_object() || e.contains("experts")) continue; // MoE layers name pairs of routes
        const std::string where = stem + " " + e.dump().substr(0, 80);
        if (!e.contains("prefill")) {
            if (e.contains("prefill_bands"))
                fail(where + ": prefill_bands without a prefill route");
            continue;
        }
        const Route prefill = route_of(e.at("prefill").get<std::string>(), where);
        ++prefill_routes;
        if (!generic::profile_admits(prefill, generic::Ask{generic::Profile::Prefill, 17, 1})) {
            fail(where + ": the prefill route does not pass the prefill band's filter");
        }
        for (const nlohmann::json& band : e.value("prefill_bands", nlohmann::json::array())) {
            const Route r = route_of(band.at(2).get<std::string>(), where);
            const generic::Ask ask{generic::Profile::Prefill, band.at(0).get<std::int32_t>(), 1};
            ++prefill_bands;
            if (ask.tokens < 17) fail(where + ": a prefill band below T = 17");
            if (!generic::profile_admits_as(r, ask, prefill)) {
                fail(where + ": a prefill band that does not sum as the prefill route");
            }
        }
    }
    std::cout << path.filename().string() << ": " << prefill_routes << " prefill routes, "
              << prefill_bands << " bands pass the filter\n";
}

void check_tables() {
    const std::filesystem::path dir = std::filesystem::path(NINFERNO_SOURCE_DIR) / "routes";
    int files                       = 0;
    for (const auto& f : std::filesystem::directory_iterator(dir)) {
        if (f.path().extension() != ".json") continue;
        ++files;
        try {
            check_table(f.path());
        } catch (const std::exception& error) {
            fail(f.path().filename().string() + ": " + error.what());
        }
    }
    if (files == 0) fail("no route table in " + dir.string());
}

} // namespace

int main() {
    try {
        for (const generic::FactsBoard& b : generic::kFactsBoards) run_board(b);
        check_tables();
    } catch (const std::exception& error) {
        std::cerr << "route catalogue: " << error.what() << '\n';
        return 1;
    }
    if (g_failed != 0) {
        std::cerr << "route catalogue: " << g_failed << " failures\n";
        return 1;
    }
    std::cout << "route catalogue: ok\n";
    return 0;
}
