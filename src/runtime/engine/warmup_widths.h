#pragma once

// The kernel warm-up's prefill width set (construct_model): at least one width of every kernel
// class a served plan can launch at a width in 1..chunk, built from the route layer's own choice
// points, not from a guess. Widths 1..48 run exactly: every per-exact-T launcher in the tree lives
// there (kWarmUpExactWidths below), and an exact-T launcher is a distinct kernel instance for every
// T, so no class structure exists to collapse. Above 48 every chooser in the engine is a step
// function of T, and a tiled schedule launches its full-tile kernel variant only where T is a
// multiple of its tile's columns, its partial variant elsewhere. The set covers both:
//  - band edges: the C++ choosers' edges (kWarmUpCompiledBreakpoints, from the shape tables, the
//    fused-op plans and the launchers) and the loaded route table's (route_table_breakpoints);
//  - inside each gap between two consecutive edges (one band of every chooser): an odd width,
//    which no tile divides (the partial variant), and a multiple of every tile width of
//    kWarmUpTokenTileCols the gap holds (the full variant);
//  - the exported plan choosers (warm_up_class_probes): one width of every kernel class they
//    name, schedule and tile variant, at or below chunk.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ninferno::runtime::detail {

// The highest T any per-exact-T launcher in the tree admits (the q8 linear_add, q8 attention-input
// and q8 linear_swiglu split-K launchers and the q8 DFlash2 attention input's small-T form, each
// through 48; static_assert in warmup_widths.cpp). Widths above it are never served by a
// per-exact-T schedule.
inline constexpr std::uint32_t kWarmUpExactWidths = 48;

// The T values above kWarmUpExactWidths where some C++ chooser of the engine changes its kernel
// choice, each taken from the chooser it names (warmup_widths.cpp gives the sources and the
// regeneration recipe). Not sorted: kernel_warm_up_widths sorts. When a ladder changes, its edge
// moves with it in this list, or, for the named constants, by itself.
extern const std::vector<std::int32_t> kWarmUpCompiledBreakpoints;

// The token-tile widths (columns) of the tiled schedules above the exact band that choose a
// full-tile or a partial-tile kernel by T: the row-split MMA tiles of the shape tables (q4, q5,
// q6, q8, BF16, FP8) and of the fused ops (q8 linear_add, linear_pair, linear_swiglu, the GDN
// gating MMA). A narrower tile (16..56 or 72 columns) serves only bands above 48 that hold a band
// edge it divides (56, 64, 72 or 96), so the edges already cover its full variant.
extern const std::vector<std::int32_t> kWarmUpTokenTileCols;

// One exported plan chooser at one problem shape and route profile: the kernel class (schedule,
// tile variant, launcher band) a call of T columns runs, as an opaque value; equal values launch
// the same kernels. The warm-up runs at least one width of every class a probe reports at or
// below chunk.
struct WarmUpClassProbe {
    std::string name;
    std::function<std::int64_t(std::int32_t)> kernel_class;
};

[[nodiscard]] const std::vector<WarmUpClassProbe>& warm_up_class_probes();

// The widths the kernel warm-up runs for a plan whose prefill chunk is `chunk`: 1..48 (every
// per-exact-T schedule), then the band edges in (48, chunk], per gap the odd and the tile-multiple
// widths, one width of every probe class, and chunk; sorted ascending and deduplicated. The loaded
// route table's own breakpoints (route_table_breakpoints) join the compiled edges, so a tuned
// table's bands are covered exactly as written. `chunk` must be above kWarmUpExactWidths.
[[nodiscard]] std::vector<std::uint32_t> kernel_warm_up_widths(std::uint32_t chunk);

} // namespace ninferno::runtime::detail
