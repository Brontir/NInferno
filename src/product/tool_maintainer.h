#pragma once

// The maintainer options of the family tools ninferno-dense and ninferno-gemma4, which run a
// model without an Engine: --route-table, --route-dir, --dense-tune, --tune-cache, --debug-poison
// and --debug-poison-skip, parsed as the Engine's products parse them (engine_flags.h) and applied
// to the process as an Engine applies them at construction (docs/maintainer/environment.md). The
// tool is the process's only user of these settings. --moe-tune is refused: the MoE schedule
// tuner runs in an Engine's model load (ModelInstance: tune_routes, ops::tune_moe, for the Qwen
// 3.5, Gemma 4, dense and Nemotron-H families alike), which neither tool goes through.

#include "core/debug_poison.h"
#include "ninferno/ops/route_table.h"
#include "ops/route_options.h"
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

// The maintainer options among a tool's arguments; an invalid value, --moe-tune, or a route
// directory or table that cannot be read (ops/route_options.h) throws, before anything is
// converted or loaded.
[[nodiscard]] inline MaintainerOptions tool_maintainer_options(const ToolArguments& args,
                                                               std::string_view program) {
    if (args.options.contains("moe-tune")) {
        throw std::invalid_argument(
            "--moe-tune does not apply: " + std::string(program) +
            " loads no model through an Engine, whose load runs the MoE schedule tuner (use "
            "ninferno or ninferno-serve)");
    }
    MaintainerOptions out;
    DecodePacingOptions unused;
    for (const std::string_view key : kToolMaintainerKeys) {
        const auto it = args.options.find(std::string(key));
        if (it == args.options.end()) { continue; }
        (void)parse_engine_flag("--" + std::string(key), [&] { return it->second; }, unused, out);
    }
    ops::check_route_options(out.route_table, out.route_dir);
    return out;
}

// Applies them to the process: the route-table lookup (and an explicit table), the dense tuner
// and the debug poison. Call after the device is current and before the model loads. The dense
// tuner is enabled whatever --dense-tune says, as an Engine enables it for every loaded model
// (model_instance.cpp): off, it times nothing and reads the release's measurements
// (routes/tuner/dense_tune.json in the route directory) over the route table, so a tool runs the
// routes ninferno and ninferno-serve run; on and retune also time the first eager call of each
// problem the measurements lack.
inline void apply_tool_maintainer_options(const MaintainerOptions& m) {
    if (!m.route_dir.empty()) { (void)ops::route_dir_files(m.route_dir); }
    ops::set_route_table_dir(m.route_dir);
    if (!m.route_table.empty()) { (void)ops::load_route_table_option(m.route_table); }
    ops::detail::generic::set_dense_tune(m.dense_tune, m.tune_cache);
    ops::detail::generic::enable_dense_tune(true);
    set_debug_poison(m.debug_poison, m.debug_poison_skip);
}

} // namespace ninferno::product
