#pragma once

// The maintainer options of the family tools ninferno-dense and ninferno-gemma4, which run a
// model without an Engine: --route-table, --route-dir, --dense-tune, --tune-cache, --debug-poison
// and --debug-poison-skip, parsed as the Engine's products parse them (engine_flags.h) and applied
// to the process as an Engine applies them at construction (docs/maintainer/environment.md). The
// tool is the process's only user of these settings. --moe-tune is refused: neither tool times
// MoE schedules (only the Qwen 3.5 family's Engine load does, ops::tune_moe).

#include "core/debug_poison.h"
#include "ninferno/ops/route_table.h"
#include "ninferno/types.h"
#include "ops/generic/generic_gemm.h"
#include "product/engine_flags.h"
#include "product/tool_args.h"

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::product {

// The option keys (without "--") the tools accept for these settings.
inline constexpr std::array<std::string_view, 6> kToolMaintainerKeys{
    "route-table", "route-dir", "dense-tune", "tune-cache", "debug-poison", "debug-poison-skip"};

// The maintainer options among a tool's arguments; an invalid value, or --moe-tune, throws.
[[nodiscard]] inline MaintainerOptions tool_maintainer_options(const ToolArguments& args,
                                                               std::string_view program) {
    if (args.options.contains("moe-tune")) {
        throw std::invalid_argument(
            "--moe-tune does not apply: " + std::string(program) +
            " times no MoE schedules (the Qwen 3.5 family's Engine does, under ninferno and "
            "ninferno-serve)");
    }
    MaintainerOptions out;
    DecodePacingOptions unused;
    for (const std::string_view key : kToolMaintainerKeys) {
        const auto it = args.options.find(std::string(key));
        if (it == args.options.end()) { continue; }
        (void)parse_engine_flag("--" + std::string(key), [&] { return it->second; }, unused, out);
    }
    return out;
}

// Applies them to the process: the route-table lookup (and an explicit table), the dense tuner
// (its mode, and enabled as the Engine enables it for a loaded model, so that --dense-tune on
// times the first eager call of each problem) and the debug poison. Call after the device is
// current and before the model loads.
inline void apply_tool_maintainer_options(const MaintainerOptions& m) {
    ops::set_route_table_dir(m.route_dir);
    if (!m.route_table.empty()) {
        const ops::RouteTableStatus status = ops::load_route_table(m.route_table.string());
        if (!status.loaded) { throw std::invalid_argument("--route-table: " + status.message); }
    }
    ops::detail::generic::set_dense_tune(m.dense_tune, m.tune_cache);
    ops::detail::generic::enable_dense_tune(m.dense_tune != TuneMode::Off);
    set_debug_poison(m.debug_poison, m.debug_poison_skip);
}

} // namespace ninferno::product
