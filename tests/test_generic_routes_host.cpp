// The host part of the generic selector (ops/generic/generic_routes.cpp, kernel_facts.cpp), linked
// without the CUDA runtime (#620 step 11): every route name parses back to its route and no two
// routes share a name, the register-MMA variant rules decide as their comments say, and every
// committed kernel-facts snapshot is readable. CPU, no toolkit.

#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_variant_rules.h"
#include "ops/generic/kernel_facts.h"

#include <iostream>
#include <set>
#include <string>

namespace gen = ninferno::ops::detail::generic;
namespace kf  = ninferno::ops::detail::generic::kernel_facts;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

struct FamilyCount {
    gen::Family family;
    int count;
};

} // namespace

int main() {
    // Route names: one name a route, parsed back to it.
    const FamilyCount families[] = {
        {gen::Family::RegisterMma, gen::rmma_schedule_count()},
        {gen::Family::W4a4Mma, gen::w4a4_schedule_names()},
        {gen::Family::TiledMma, gen::tiled_schedule_count()},
        {gen::Family::W4a8Mma, gen::w4a8_schedule_count()},
        {gen::Family::StagedMma, gen::staged_schedule_count()},
        {gen::Family::Nvfp4TiledMma, gen::nvfp4_tiled_schedule_count()},
        {gen::Family::R8Mma, gen::r8_schedule_count()},
        {gen::Family::I8Mma, gen::i8_schedule_count()},
    };
    std::set<std::string> names;
    int routes = 0;
    for (const FamilyCount& f : families) {
        for (int i = 0; i < f.count; ++i) {
            const gen::Route route{f.family, static_cast<std::uint8_t>(i)};
            const std::string name = gen::route_name(route);
            ++routes;
            check(names.insert(name).second, "route name unique: " + name);
            const auto parsed = gen::parse_route(name);
            check(parsed && *parsed == route, "route name parses back: " + name);
        }
    }
    for (const gen::Route fixed :
         {gen::Route{gen::Family::Scalar, 0}, gen::Route{gen::Family::GgmlMmq, 0},
          gen::Route{gen::Family::GgmlVec, 0}}) {
        const auto parsed = gen::parse_route(gen::route_name(fixed));
        check(parsed && *parsed == fixed, "fixed route parses back: " + gen::route_name(fixed));
    }
    check(!gen::parse_route("no_such_route").has_value(), "an unknown name parses to nothing");
    std::set<std::string> family_names;
    for (const FamilyCount& f : families) {
        family_names.insert(std::string(gen::family_name(f.family)));
    }
    check(family_names.size() == std::size(families), "each family has its own name");
    check(routes > 20, "the schedule tables are not empty");

    // The register-MMA variant rules (generic_routes.cpp).
    check(gen::rmma_variant_occupancy_rule(1000, 100, 2, 2), "equal residency: the variant stays");
    check(gen::rmma_variant_occupancy_rule(1000, 100, 3, 2), "more residency: the variant stays");
    check(!gen::rmma_variant_occupancy_rule(1000, 100, 1, 2),
          "fewer waves lost: the plain tile (10 waves against 5)");
    check(gen::rmma_variant_occupancy_rule(100, 100, 1, 2),
          "one wave either way: the variant stays");
    check(!gen::rmma_variant_occupancy_rule(1000, 0, 1, 1) &&
              !gen::rmma_variant_occupancy_rule(1000, 100, 0, 1),
          "no residency or no SMs: no variant");
    check(!gen::rmma_wide_rule(0.0, 89, true, 0, 0), "no DRAM rate: not wide");
    check(!gen::rmma_wide_rule(1e9, 89, true, 0, 0), "a rate above the wide threshold: not wide");

    // Every committed kernel-facts snapshot is readable.
    const auto archs = kf::snapshot_archs();
    check(!archs.empty(), "a kernel-facts snapshot is committed");
    for (const std::string_view arch : archs) {
        const auto header = kf::snapshot_header(arch);
        check(header && header->kernels > 0, "snapshot readable: " + std::string(arch));
    }

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "generic routes (host): " << routes << " routes, " << archs.size()
              << " snapshots: ok\n";
    return 0;
}
