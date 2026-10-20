#include "runtime/engine/warmup_widths.h"

#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/route_table.h"
#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"
#include "ops/generic/generic_gemm.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/linear_add/q8/q8_linear_add_kernels.h"
#include "ops/linear_add/q8/q8_linear_add_plan.h"
#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_pair/q8/q8_pair_plan.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ninferno::runtime::detail {

namespace op = ninferno::ops::detail;

static_assert(kWarmUpExactWidths >= static_cast<std::uint32_t>(std::max(
                                        {op::kQ8LinearAddLastExactCols,
                                         op::kQ8AttnInputLastTargetExactCols,
                                         op::kQ8Dflash2AttnInputLastSmallTokens,
                                         op::kQ8LinearSwiGluLastExactT})),
              "every per-exact-T launcher must lie inside the warm-up's exact band");

// Band edges above 48, each read out of the chooser its group names. An edge is a band's last
// width ("tokens <= N" gives N) or its first (">= N" gives N, "> N" gives N + 1); with one of the
// two for every band boundary, a gap between two consecutive edges lies inside one band of every
// chooser, which kernel_warm_up_widths relies on. Regenerate the literal groups with (from the
// repository root), then review the third by hand (a threshold on rows or on a model dimension is
// not a width edge):
//   grep -rhoE "tokens <= [0-9]+" src/ops/linear/*/shapes/ | grep -oE "[0-9]+" | sort -nu |
//     awk '$1>48'
//   grep -rhoE "\{\{?[0-9]+, *([0-9]+|kAnyCols)\}?," src/ops/{attn_input_proj,gdn_input_proj,
//     gdn_gating_proj,linear_add,linear_swiglu,linear_pair,candidate_selector,
//     dynamic_grouped_conv} | grep -oE "[0-9]+" | sort -nu | awk '$1>48'
//   grep -rnoE "\b(T|t|tokens|rows|cols|x\.ne\[1\]) *(<=|<|>=|>) *[0-9]+\b" src/ops
//     --include=*.cu --include=*.cpp | grep -v /shapes/
// The named constants below move with their sources. Values above kSparseMoePrefillSliceMax
// (4096) cannot be reached by a prefill call's slice (the MoE slices first), but they are kept: a
// future chunk above the cap keeps its coverage.
const std::vector<std::int32_t> kWarmUpCompiledBreakpoints = [] {
    std::vector<std::int32_t> edges{
        // shape-owned Linear ladders (bf16/q4/q5/q6/q8 shapes; the fp8/nvfp4 per-exact schedules
        // and the fp8 A8 threshold 25 sit inside 1..48)
        56, 64, 72, 76, 80, 96, 112, 120, 128, 160, 192, 224, 240, 256, 288, 320, 384, 448, 480,
        512, 576, 636, 640, 704, 768, 871, 896, 960, 1024, 1148, 1280, 1344, 1536, 1680, 2016,
        2048, 2112, 2560,
        // fused-op plan route bands, both table forms ({first, last, schedule} and
        // {{first, last}, schedule}), including the prefill-profile variants: attn_input_proj,
        // gdn_input_proj, gdn_gating_proj, linear_add, linear_swiglu, linear_pair,
        // candidate_selector, dynamic_grouped_conv
        49, 56, 63, 64, 65, 72, 80, 81, 85, 86, 88, 89, 96, 97, 104, 105, 112, 113, 127, 128, 129,
        160, 161, 168, 169, 191, 192, 193, 224, 225, 240, 241, 255, 256, 257, 264, 265, 288, 289,
        320, 321, 384, 385, 399, 400, 401, 447, 448, 449, 480, 481, 512, 513, 560, 561, 640, 641,
        642, 672, 673, 680, 681, 704, 705, 784, 785, 896, 897, 960, 961, 976, 977, 1023, 1024,
        1025, 1120, 1121, 1280, 1281, 1316, 1317, 1344, 1345, 1346, 1408, 1409, 1440, 1441, 1466,
        1467, 1680, 1681, 1708, 1709, 1791, 1792, 1793, 1919, 1920, 1921, 1922, 1923, 2016, 2017,
        2018, 2019, 2047, 2048, 2049, 2208, 2209, 2270, 2271, 4096, 4097,
        // launcher thresholds outside the tables: the fp8/nvfp4 attention and GDN input tiles, the
        // q4/q5 attention input, the q8 attention/GDN input split-K forms, the nvfp4/fp8/q4 linear
        // add tiles, the q8 linear_add capacity split-K (57), the q8 pair medium split-K tiles, the
        // generic MoE's wide plan (64), embed_gather (129; FP8 177), rmsnorm's gated wide rows
        // (171, rmsnorm.cu kRmsPrefetchBlocks), rope (401 for the DFlash split, 1021 a launch-shape
        // edge of the same kernel), and the fp8 A8 attention input's tail tile (145)
        57, 64, 80, 96, 97, 104, 128, 129, 144, 145, 160, 171, 176, 177, 192, 193, 224, 256, 384,
        400, 401, 512, 1020, 1021, 1024,
        // the GDN chunked path's first width, the paged KV append's page-tile form, the
        // sparse MoE wide plan, and the generic projection path's speed-profile heuristics
        op::gated_delta_net::chunked::kMinTokens, op::kKVCacheAppendPageTileMinTokens,
        op::kSparseMoePrefillWideMin, op::generic::kRmmaMidMinTokens,
        op::generic::kStreamMaxTokens + 1, op::generic::kTiledSpeedMinTokens,
        op::generic::kStagedSpeedMinTokens, op::generic::kW4a8WideMinTokens,
        // the q8 linear_add prefill small-T launcher's tile switch
        op::kQ8LinearAddPrefillSmallR16MaxCols, op::kQ8LinearAddPrefillSmallR16MaxCols + 1};
    // the q8 linear_add medium split-K launcher's bands (one tile per band)
    for (const std::int32_t last : op::kQ8LinearAddMediumBandLast) {
        edges.push_back(last);
        edges.push_back(last + 1);
    }
    return edges;
}();

const std::vector<std::int32_t> kWarmUpTokenTileCols{64, 80, 96, 112, 128, 256};

namespace {

std::int64_t class_value(int schedule, bool full, std::int32_t band) {
    return (static_cast<std::int64_t>(schedule) << 40) | (static_cast<std::int64_t>(full) << 32) |
           static_cast<std::uint32_t>(band);
}

struct ProbeShape {
    std::int32_t rows;
    std::int32_t k;
};

// The admitted problem shapes of each exported plan chooser.
constexpr ProbeShape kQ8LinearAddShapes[]{{2048, 4096}, {2048, 6144}, {5120, 6144}, {5120, 17408}};
constexpr ProbeShape kBf16GdnGatingShapes[]{{48, 5120}, {32, 2048}}; // heads x input rows
constexpr ProbeShape kQ8PairShapes[]{{1024, 5120}, {1024, 2048}};

std::string shape_name(const char* op, ProbeShape shape) {
    return std::string(op) + " " + std::to_string(shape.rows) + "x" + std::to_string(shape.k);
}

std::vector<WarmUpClassProbe> make_probes() {
    std::vector<WarmUpClassProbe> probes;
    for (const bool prefill : {false, true}) {
        const std::string profile = prefill ? " (prefill profile)" : " (speed profile)";
        for (const ProbeShape shape : kQ8LinearAddShapes) {
            const std::int32_t rows = shape.rows;
            const std::int32_t k    = shape.k;
            probes.push_back(
                {shape_name("q8 linear_add", shape) + profile,
                 [rows, k, prefill](std::int32_t cols) {
                     const ops::ScopedPrefillRoutes scope(prefill);
                     const op::Q8LinearAddProblem problem{rows, k, k, cols};
                     const op::Q8LinearAddKernelClass c = op::q8_linear_add_kernel_class(
                         op::q8_linear_add_resolve_plan(problem), problem);
                     return class_value(static_cast<int>(c.schedule), c.full, c.band);
                 }});
        }
        for (const ProbeShape shape : kBf16GdnGatingShapes) {
            const std::int32_t heads = shape.rows;
            const std::int32_t rows  = shape.k;
            probes.push_back(
                {shape_name("bf16 gdn_gating", shape) + profile,
                 [heads, rows, prefill](std::int32_t cols) {
                     const ops::ScopedPrefillRoutes scope(prefill);
                     const op::Bf16GdnGatingPlan plan =
                         op::bf16_gdn_gating_resolve_plan({heads, rows, cols});
                     return class_value(static_cast<int>(plan.schedule),
                                        plan.token_variant == op::Bf16GdnGatingTokenVariant::Full,
                                        0);
                 }});
        }
    }
    for (const ProbeShape shape : kQ8PairShapes) {
        const std::int32_t rows = shape.rows;
        const std::int32_t k    = shape.k;
        probes.push_back({shape_name("q8 linear_pair", shape), [rows, k](std::int32_t cols) {
                              const op::Q8PairProblem problem{rows, k, k, cols};
                              const op::Q8PairPlan plan = op::q8_pair_resolve_plan(problem);
                              return class_value(static_cast<int>(plan.schedule),
                                                 op::q8_pair_uses_full_tile(plan, problem), 0);
                          }});
    }
    return probes;
}

// One band of every chooser, lo..hi: its first odd width (no tile divides it: the partial-tile
// variants; a one-width gap takes that width) and a multiple of every tile width the gap holds
// (the full-tile variants), a width shared by several tiles where one exists.
void cover_gap(std::vector<char>& warm, std::int32_t lo, std::int32_t hi) {
    if (lo > hi) { return; }
    warm[static_cast<std::size_t>(lo % 2 != 0 || lo == hi ? lo : lo + 1)] = 1;
    std::vector<std::int32_t> need;
    for (const std::int32_t tile : kWarmUpTokenTileCols) {
        if (hi / tile * tile >= lo) { need.push_back(tile); }
    }
    while (!need.empty()) {
        std::int32_t best       = 0;
        std::size_t best_shared = 0;
        for (const std::int32_t tile : need) {
            for (std::int32_t width = (lo + tile - 1) / tile * tile; width <= hi; width += tile) {
                const auto shared = static_cast<std::size_t>(std::count_if(
                    need.begin(), need.end(), [&](std::int32_t t) { return width % t == 0; }));
                if (shared > best_shared || (shared == best_shared && width < best)) {
                    best        = width;
                    best_shared = shared;
                }
            }
        }
        warm[static_cast<std::size_t>(best)] = 1;
        std::erase_if(need, [&](std::int32_t t) { return best % t == 0; });
    }
}

} // namespace

const std::vector<WarmUpClassProbe>& warm_up_class_probes() {
    static const std::vector<WarmUpClassProbe> probes = make_probes();
    return probes;
}

std::vector<std::uint32_t> kernel_warm_up_widths(std::uint32_t chunk) {
    if (chunk <= kWarmUpExactWidths) {
        throw std::invalid_argument("the warm-up width set needs a chunk above " +
                                    std::to_string(kWarmUpExactWidths));
    }
    if (chunk > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("the warm-up width set needs a chunk that fits int32");
    }
    const auto last  = static_cast<std::int32_t>(chunk);
    const auto exact = static_cast<std::int32_t>(kWarmUpExactWidths);
    std::vector<char> warm(static_cast<std::size_t>(last) + 1, 0);
    std::fill_n(warm.begin() + 1, exact, char{1});

    // The band edges in (48, chunk]: the compiled choosers', the route table's, chunk itself.
    std::vector<std::int32_t> edges;
    for (const std::int32_t edge : kWarmUpCompiledBreakpoints) {
        if (edge > exact && edge <= last) { edges.push_back(edge); }
    }
    for (const std::int32_t edge : ops::route_table_breakpoints(exact + 1)) {
        if (edge <= last) { edges.push_back(edge); }
    }
    edges.push_back(last);
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    std::int32_t previous = exact;
    for (const std::int32_t edge : edges) {
        warm[static_cast<std::size_t>(edge)] = 1;
        cover_gap(warm, previous + 1, edge - 1);
        previous = edge;
    }

    // Every kernel class of the exported plan choosers: its first width unless one is warm.
    for (const WarmUpClassProbe& probe : warm_up_class_probes()) {
        std::unordered_set<std::int64_t> covered;
        std::unordered_map<std::int64_t, std::int32_t> first_width;
        for (std::int32_t width = exact + 1; width <= last; ++width) {
            const std::int64_t kernel_class = probe.kernel_class(width);
            if (warm[static_cast<std::size_t>(width)] != 0) {
                covered.insert(kernel_class);
            } else {
                first_width.emplace(kernel_class, width);
            }
        }
        for (const auto& [kernel_class, width] : first_width) {
            if (!covered.contains(kernel_class)) { warm[static_cast<std::size_t>(width)] = 1; }
        }
    }

    std::vector<std::uint32_t> widths;
    for (std::int32_t width = 1; width <= last; ++width) {
        if (warm[static_cast<std::size_t>(width)] != 0) {
            widths.push_back(static_cast<std::uint32_t>(width));
        }
    }
    return widths;
}

} // namespace ninferno::runtime::detail
