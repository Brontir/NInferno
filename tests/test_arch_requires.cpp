// The arch requirements of the format predicates (core/arch_requires.h, ops/generic/
// format_requires.h; #615 F5), no GPU. For every board of the PCI-ID table
// (core/data/gpu_facts.json, the board registry's rows) and every build a tree can be (the
// architecture it names, the features its device code carries):
// - each declared requirement resolves from the registry row alone, and the row's facts equal the
//   facts of the device class built from the same board (synthetic_device_class, as the query
//   builds one);
// - each gives the answer of the code it replaced, kept below in `legacy` verbatim but for the
//   build architecture, which there was the macro NINFERNO_BUILD_ARCH and here is a parameter (0:
//   a tree that names none);
// - each answers yes on some board and build and no on another, so no requirement is vacuous.
// Then the selector's own predicates under each board as the substituted device, in this tree's
// build: w4a8_ws_extra for every named format, and current_arch_facts.
#include "core/arch_requires.h"
#include "core/board_registry.h"
#include "core/device_class.h"
#include "core/format_facts.h"
#include "core/generation.h"
#include "core/machine_facts.h"
#include "ops/generic/format_requires.h"
#include "ops/generic/format_traits.h"
#include "ops/generic/generic_select_detail.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;
namespace fr      = ninferno::ops::detail::generic::format_requires;

int failures    = 0;
int comparisons = 0;

void expect(bool ok, const std::string& what) {
    ++comparisons;
    if (!ok) {
        if (failures < 50) std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

namespace legacy {

// generic_select.cpp w4a8_ws_extra, its device part (inside its PerDevice cache).
bool w4a8_ws_tuned_arch(const DeviceClass& d) {
    return gb20x_cc(d.major, d.minor) || ada_cc(d.major, d.minor);
}

// generic_gemm.cpp qx_prefill_a8_pays.
bool qx_prefill_a8_pays(const DeviceClass& d) {
    const auto full_rate = fp32acc_full_rate_of(d.major, d.minor);
    return d.features.mma_tensor && !d.features.mma_slow && full_rate && !*full_rate;
}

// generic_select.cpp rules_r8: the branch of a card without native BF16 MMA.
bool r8_without_native_bf16(const DeviceClass& d) { return !d.features.mma_bf16_native; }

// generic_select.cpp rules_r8: r8_one_slot_build() && !device_class().features.mma_tensor, with
// r8_one_slot_build() `defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH < 75`.
bool r8_one_slot(const DeviceClass& d, int build_arch) {
    const bool one_slot_build = build_arch != 0 && build_arch < 75;
    return one_slot_build && !d.features.mma_tensor;
}

} // namespace legacy

bool legacy_answer(std::string_view name, const DeviceClass& d, int build_arch) {
    if (name == "w4a8_ws_tuned_arch") return legacy::w4a8_ws_tuned_arch(d);
    if (name == "qx_prefill_a8_pays") return legacy::qx_prefill_a8_pays(d);
    if (name == "r8_without_native_bf16") return legacy::r8_without_native_bf16(d);
    if (name == "r8_one_slot") return legacy::r8_one_slot(d, build_arch);
    expect(false, std::string(name) + ": a declared requirement with no legacy answer");
    return false;
}

// What a tree's device code can carry: build_features() of a tree built for each architecture
// (the block-scaled kinds only in an architecture-specific one, "120a"), and the set of a tree
// that names none, which limits nothing (the selector golden's substitutes carry it too).
struct Build {
    std::string name;
    DeviceFeatures features;
};

std::vector<Build> build_feature_sets() {
    std::vector<Build> out;
    out.push_back({"unlimited",
                   {.mma_bf16        = true,
                    .mma_bf16_native = true,
                    .mma_tensor      = true,
                    .mma_fp8         = true,
                    .mma_f8f6f4      = true,
                    .mma_mxf4nvf4    = true,
                    .tma             = true,
                    .pdl             = true,
                    .clusters        = true}});
    const std::pair<int, int> arches[] = {{7, 0}, {7, 5},  {8, 0},  {8, 6}, {8, 9},
                                          {9, 0}, {10, 0}, {12, 0}, {12, 1}};
    for (const auto& [major, minor] : arches) {
        DeviceFeatures specific = device_features_for(major, minor);
        DeviceFeatures plain    = specific;
        plain.mma_f8f6f4        = false;
        plain.mma_mxf4nvf4      = false;
        const std::string sm    = "sm_" + std::to_string(major * 10 + minor);
        out.push_back({sm, plain});
        out.push_back({sm + "a", specific});
    }
    return out;
}

constexpr int kBuildArches[] = {0, 70, 72, 75, 80, 86, 89, 90, 100, 120, 121};

bool same_features(const DeviceFeatures& a, const DeviceFeatures& b) {
    return has_features(a, b) && has_features(b, a) && a.mma_slow == b.mma_slow;
}

bool same_facts(const ArchFacts& a, const ArchFacts& b) {
    return a.major == b.major && a.minor == b.minor && same_features(a.features, b.features) &&
           a.fp32acc_full_rate == b.fp32acc_full_rate && a.build_arch == b.build_arch;
}

// The device class the query would build for the board (synthetic_device_class: the
// architecture's features, the board's mma_slow).
DeviceClass board_class(const board_registry::BoardRow& row) {
    return synthetic_device_class(row.pci_device_id, row.major, row.minor, row.sms,
                                  static_cast<std::size_t>(row.l2_mib * 1024 * 1024),
                                  static_cast<std::size_t>(row.vram_gib * 1024 * 1024 * 1024));
}

void requirements_equal_the_code_they_replaced() {
    const std::span<const GpuFactsRow> table = gpu_facts_table();
    expect(std::size(board_registry::kRows) == table.size(), "registry has one row per table row");
    const std::vector<Build> builds = build_feature_sets();
    std::map<std::string_view, std::pair<int, int>> answers; // name -> (yes, no)
    for (std::size_t i = 0; i < std::size(board_registry::kRows); ++i) {
        const board_registry::BoardRow& row = board_registry::kRows[i];
        const std::string board  = i < table.size() ? table[i].name : "row " + std::to_string(i);
        const DeviceClass probed = board_class(row);
        // The registry row's mma_slow is the one the query sets (board_mma_slow).
        expect(probed.features.mma_slow == row.mma_slow, board + ": mma_slow from the row");
        for (const Build& build : builds) {
            // DeviceClass::features as the query sets it: the device's row and the build's in
            // common, then the board's mma_slow.
            DeviceClass d       = probed;
            d.features          = common_features(probed.features, build.features);
            d.features.mma_slow = probed.features.mma_slow;
            for (const int build_arch : kBuildArches) {
                const std::string what = board + " / " + build.name + " features / build arch " +
                                         std::to_string(build_arch);
                const ArchFacts from_row   = arch_facts_of(row, build.features, build_arch);
                const ArchFacts from_class = arch_facts_of(d, build_arch);
                expect(same_facts(from_row, from_class), what + ": row facts == class facts");
                for (const fr::NamedRequires& r : fr::kDeclared) {
                    const bool old_answer = legacy_answer(r.name, d, build_arch);
                    const bool new_answer = satisfied(*r.arch, from_row);
                    expect(new_answer == old_answer,
                           what + ": " + std::string(r.name) + " " + (new_answer ? "yes" : "no") +
                               ", the replaced code " + (old_answer ? "yes" : "no"));
                    expect(satisfied(*r.arch, from_class) == new_answer,
                           what + ": " + std::string(r.name) + " from the class");
                    auto& [yes, no] = answers[r.name];
                    (new_answer ? yes : no) += 1;
                }
            }
        }
    }
    for (const fr::NamedRequires& r : fr::kDeclared) {
        const auto [yes, no] = answers[r.name];
        std::cout << r.name << ": yes " << yes << ", no " << no << '\n';
        expect(yes > 0 && no > 0, std::string(r.name) + ": answers both ways somewhere");
    }
}

// The selector's predicates under each board as this thread's device, in this tree's build.
void selector_reads_the_requirements() {
#if defined(NINFERNO_BUILD_ARCH)
    constexpr int build_arch = NINFERNO_BUILD_ARCH;
#else
    constexpr int build_arch = 0;
#endif
    const std::span<const GpuFactsRow> table = gpu_facts_table();
    for (std::size_t i = 0; i < std::size(board_registry::kRows); ++i) {
        const board_registry::BoardRow& row = board_registry::kRows[i];
        const std::string board = i < table.size() ? table[i].name : "row " + std::to_string(i);
        const DeviceClass d     = board_class(row);
        const ScopedDeviceClass scoped(d);
        expect(same_facts(generic::current_arch_facts(), arch_facts_of(d, build_arch)),
               board + ": current_arch_facts");
        for (const generic::FormatTraits& t : generic::kFormatTraits) {
            const bool old_answer = t.w4a8_ws_tuned && legacy::w4a8_ws_tuned_arch(d);
            expect(generic::w4a8_ws_extra(t.q) == old_answer,
                   board + ": w4a8_ws_extra(" + std::to_string(static_cast<int>(t.q)) + ")");
        }
    }
}

} // namespace

int main() {
    try {
        requirements_equal_the_code_they_replaced();
        selector_reads_the_requirements();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " of " << comparisons << " checks failed\n";
        return 1;
    }
    std::cout << "arch requirements: " << comparisons << " checks over "
              << std::size(board_registry::kRows) << " boards\n";
    return 0;
}
