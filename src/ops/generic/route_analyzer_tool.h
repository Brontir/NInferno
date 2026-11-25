#pragma once

// ninferno-serve route-analyzer: the route analyzer (route_analyzer.h) on a CPU, for boards of
// the facts table. Each line of the cells file (or stdin) is "<pci id> <format> <epilogue> <n> <k>
// <T>" (format as the route tables spell it, epilogue store / add / swiglu); for each, one line a
// priced candidate, cheapest first: "<pci id> <format> <epilogue> <n> <k> <T> <route> <ns>", and
// "... - -" where none is priced. --launch graph prices launches as graph nodes (default stream,
// as route_tune times them).

namespace ninferno::tools {

int route_analyzer_main(int argc, char** argv);

} // namespace ninferno::tools
