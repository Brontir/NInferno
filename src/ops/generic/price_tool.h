#pragma once

// `ninferno-serve price`: the cost model's CPU replica (#613 PR-4). For a board of the facts
// table (core/data/gpu_facts.json), a format, a shape and a width it prints every named term of
// the kernel roofline (ops/generic/kernel_roofline.h) for each A16 tile this build prices, beside
// the selector's own price of the same call on that board (route_query.h priced_routes_query), so a
// model change is read term by term on any machine. The replica launches nothing and needs no
// GPU: the board is a substituted class built from its facts, and the compiled kernel facts
// (registers, spill) are taken as arguments until the per-arch snapshots carry them. The
// selector's column is the engine's own price: for a heavy decoder it reads the host build's
// compiled facts when a GPU is present (cudaFuncGetAttributes), so it can differ from machine to
// machine, and the tool marks such a row `!=`.

#include "core/device_class.h"
#include "core/weight.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::tools {

// Exit 0 printed; 2 on a usage error, an unknown board or format, or a --route the selector does
// not admit for the call on this build and board (route_query.h route_admitted_query, its own
// route_admits: no price is printed for a route it does not admit) or does not price
// (generic_gemm.h a16_route_priced: where the BF16 MMA is emulated, a pipelined tile or an 8-tile
// register-MMA schedule); 1 on any other error.
int price_main(int argc, char** argv);

// What `--format` takes: every name the usage text prints (the route table's qtype_name, case and
// underscores ignored by the parser) and the parser itself. One list serves both, and the test
// runs each printed name through the parser.
[[nodiscard]] std::vector<std::string> price_format_names();
[[nodiscard]] std::optional<QType> price_find_format(std::string_view name);
// The text printed on a usage error.
[[nodiscard]] std::string price_usage_text();

// The board the tool prices: core's board_class of the row, its driver view the row's own
// (registry_driver_view) and its memory the row's whole VRAM.
[[nodiscard]] DeviceClass price_board_class(const GpuFactsRow& row);

} // namespace ninferno::tools
