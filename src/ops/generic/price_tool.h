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

namespace ninferno::tools {

// Exit 0 printed; 2 on a usage error, an unknown board or format; 1 on any other error.
int price_main(int argc, char** argv);

} // namespace ninferno::tools
