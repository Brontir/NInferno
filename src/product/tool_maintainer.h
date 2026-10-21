#pragma once

// The maintainer options of the family tools ninferno-dense and ninferno-gemma4, which run a
// model without an Engine: --route-table, --route-dir, --dense-tune, --moe-tune, --tune-cache,
// --debug-poison and --debug-poison-skip, parsed as the Engine's products parse them
// (engine_flags.h) and applied to the process as an Engine applies them at construction
// (docs/maintainer/environment.md). The tool is the process's only user of these settings.

#include "core/debug_poison.h"
#include "ninferno/ops/moe.h"
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
inline constexpr std::array<std::string_view, 7> kToolMaintainerKeys{
    "route-table", "route-dir", "dense-tune", "moe-tune", "tune-cache", "debug-poison",
    "debug-poison-skip"};

// The maintainer options among a tool's arguments; an invalid value throws.
[[nodiscard]] inline MaintainerOptions tool_maintainer_options(const ToolArguments& args) {
    MaintainerOptions out;
    DecodePacingOptions unused;
    for (const std::string_view key : kToolMaintainerKeys) {
        const auto it = args.options.find(std::string(key));
        if (it == args.options.end()) { continue; }
        (void)parse_engine_flag("--" + std::string(key), [&] { return it->second; }, unused, out);
    }
    return out;
}

// Applies them to the process: the route-table lookup (and an explicit table), the tuners and the
// debug poison. Call after the device is current.
inline void apply_tool_maintainer_options(const MaintainerOptions& m) {
    ops::set_route_table_dir(m.route_dir);
    if (!m.route_table.empty()) {
        const ops::RouteTableStatus status = ops::load_route_table(m.route_table.string());
        if (!status.loaded) { throw std::invalid_argument("--route-table: " + status.message); }
    }
    ops::detail::generic::set_dense_tune(m.dense_tune, m.tune_cache);
    ops::set_moe_tune(m.moe_tune, m.tune_cache);
    set_debug_poison(m.debug_poison, m.debug_poison_skip);
}

} // namespace ninferno::product
