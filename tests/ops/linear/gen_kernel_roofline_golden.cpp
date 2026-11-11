// Generator of kernel_roofline_golden.inc (test_kernel_roofline.cpp). A build-only executable
// (ninferno_gen_kernel_roofline_golden, not in ctest, so it keeps compiling). The table is made
// once from the code BEFORE the roofline extraction (#613 PR-4), so it holds the old formula's
// outputs and nothing derived from the new code; running this program on the PR's code only
// shows that the new code still equals the table.
//
// To regenerate: check out the pre-refactor commit named in the .inc header, copy in the two files
// that expose the selector's price as a query (src/ops/generic/route_query.h and route_query.cpp
// from the PR branch: priced_routes_query, a pure addition that does not touch the formula) and
// this file with its tests.cmake registration, and build it TWICE, because which routes a build
// offers depends on the build set: in a full sm_120a build (NVFP4 on, so every register-MMA
// schedule including the cluster K splits launches; 99 KiB of shared memory per block) and in an
// sm_80 build (163 KiB, so the deep staged schedules 10-12 fit). Run each with the output path and
// the commit hash:
//   ninferno_gen_kernel_roofline_golden out.inc <commit>
// then merge the two outputs (the union of their routes; a route both price must agree):
//   merge_kernel_roofline_golden.py sm120a.inc sm80.inc kernel_roofline_golden.inc
// and run clang-format on the result (the pre-push hook does: the committed file is the formatted
// one). The grid below must match kGoldenFormats in test_kernel_roofline.cpp.
#include "core/device_class.h"
#include "ninferno/ops/linear.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace g = ninferno::ops::detail::generic;
using ninferno::QType;
using ninferno::ops::LinearPolicy;

namespace {

struct Shape {
    int n, k;
};

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s OUT.inc PRE_REFACTOR_COMMIT\n", argv[0]);
        return 2;
    }
    const QType formats[] = {QType::GGML_Q4_K, QType::BF16};
    const Shape shapes[]  = {{1024, 4096}, {4096, 4096}, {17408, 5120}, {5120, 17408}};
    const int tokens[]    = {17, 32, 64, 128, 256};
    std::string calls, routes;
    long nroutes = 0, ncalls = 0;
    int board_index = -1;
    for (const auto& board : g::kFactsBoards) {
        ++board_index;
        const auto d     = g::board_device_class(board);
        int format_index = -1;
        for (const QType q : formats) {
            ++format_index;
            for (const Shape s : shapes) {
                for (const g::Epilogue e : {g::Epilogue::Store, g::Epilogue::SwiGlu}) {
                    for (const int t : tokens) {
                        std::vector<g::PricedRouteQuery> priced;
                        try {
                            priced =
                                g::priced_routes_query(d, q, LinearPolicy::A16Only, e, s.n, s.k, t);
                        } catch (const std::exception&) { continue; }
                        // The register-MMA, tiled and staged routes only: the families the
                        // roofline prices (the A8 tiles are not moved).
                        const long first = nroutes;
                        char buf[160];
                        for (const auto& p : priced) {
                            if (p.route.family != g::Family::RegisterMma &&
                                p.route.family != g::Family::TiledMma &&
                                p.route.family != g::Family::StagedMma) {
                                continue;
                            }
                            std::snprintf(buf, sizeof buf, "{%d,%d,%.17g},",
                                          static_cast<int>(p.route.family),
                                          static_cast<int>(p.route.schedule), p.seconds);
                            routes += buf;
                            routes += '\n';
                            ++nroutes;
                        }
                        if (nroutes == first) continue;
                        std::snprintf(
                            buf, sizeof buf, "{%d,%d,%d,%d,%d,g::Epilogue::%s,%ld,%ld},\n",
                            board_index, format_index, s.n, s.k, t,
                            e == g::Epilogue::Store ? "Store" : "SwiGlu", first, nroutes - first);
                        calls += buf;
                        ++ncalls;
                    }
                }
            }
        }
    }
    std::FILE* f = std::fopen(argv[1], "w");
    if (!f) return 2;
    // The build set this output was made in (CMAKE_CUDA_ARCHITECTURES): which routes it holds
    // depends on it.
    const std::string arch(g::kernel_facts::build_arch());
    std::string boards;
    for (const auto& board : g::kFactsBoards) { boards += std::string("\"") + board.name + "\", "; }
    std::fprintf(
        f,
        "// Golden prices of the selector's register-MMA, tiled and staged routes from the\n"
        "// code before #613 PR-4 (kernel_roofline.h), commit %s, sm_%s build.\n"
        "constexpr const char* kGoldenBoards[] = {%s};\n"
        "// Regenerate with tests/ops/linear/gen_kernel_roofline_golden.cpp (see its header);\n"
        "// do not edit by hand.\n"
        "// kGoldenCalls: board (index into kFactsBoards), format (index into\n"
        "// kGoldenFormats), n, k, tokens, epilogue, first route, route count.\n"
        "// kGoldenRoutes: family, schedule, seconds (%%.17g).\n"
        "constexpr GoldenCall kGoldenCalls[] = {\n%s};\n"
        "constexpr GoldenRoute kGoldenRoutes[] = {\n%s};\n",
        argv[2], arch.c_str(), boards.c_str(), calls.c_str(), routes.c_str());
    std::fclose(f);
    std::printf("calls %ld routes %ld, sm_%s build\n", ncalls, nroutes, arch.c_str());
    return 0;
}
