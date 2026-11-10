// A route's description on Volta (#653, describe_route): a TiledMma call of a CUTLASS-only format
// runs CUTLASS over the decoded copy only when the build has it, the band takes the call and its
// workspace holds a chunk; otherwise run_route runs the scalar kernel and notes a demotion, and
// the description says so (executed Scalar, Demotion::VoltaCopy, the scalar route's workspace,
// tile and traffic). Host only, on a substituted GV100 class (one board per process: the
// selector caches what it reads of the device).
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/units.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/route_query.h"

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>

namespace {

using namespace ninferno;
using namespace ninferno::test;
namespace generic = ninferno::ops::detail::generic;
using ops::LinearPolicy;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    if (ninferno::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const ScopedDeviceClass scoped(
        synthetic_device_class(0x1DBA, 7, 0, 80, 6 * kMiB, std::size_t{32} * 1024 * kMiB));
    void* const payload      = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    constexpr std::int32_t n = 4096, k = 2048, tokens = 79;
    const generic::Route tiled{generic::Family::TiledMma, 0};
    const generic::Route scalar{generic::Family::Scalar, 0};
    int cutlass_only_formats = 0;
    for (const QType qtype : {QType::Q4_G64_FP16, QType::Q6_G64_FP16, QType::Q8_G32_FP16,
                              QType::FP8_E4M3FN_ROW_BF16, QType::GGML_Q4_K, QType::GGML_Q5_K}) {
        const auto host = quantized_weight::make_patterned_weight(qtype, n, k, 7U, {});
        const Weight w  = host.device_weight(payload);
        if (!generic::supports(w) || !generic::route_admits(tiled, w, LinearPolicy::A16Only))
            continue;
        const std::string name = std::to_string(static_cast<int>(qtype));
        const bool only        = generic::volta_cutlass_only(generic::codec_of(w));
        const auto ask         = [&](std::optional<std::size_t> free) {
            return generic::describe_route(device_class(), tiled, qtype, LinearPolicy::A16Only,
                                                   generic::Epilogue::Store, n, k, tokens, free);
        };
        // No workspace: the demotion is said for a CUTLASS-only format, and the tiled route stands
        // for one with a tile of its own.
        const auto none = ask(std::nullopt);
        check(none.has_value(), name + ": described without workspace");
        if (!none) continue;
        check(generic::volta_tiled_runs_scalar(generic::codec_of(w), n, k, tokens, false,
                                               std::nullopt) == only,
              name + ": the rule refuses a workspace-less call exactly for CUTLASS-only formats");
        if (only) {
            ++cutlass_only_formats;
            check(none->resolved.asked == tiled && none->resolved.executed == scalar &&
                      none->resolved.reason == generic::Demotion::VoltaCopy,
                  name + ": no workspace runs scalar and says why");
            check(
                none->workspace_bytes == 0 &&
                    none->tile.has_value() ==
                        generic::route_tile(scalar, generic::Epilogue::Store, tokens).has_value() &&
                    none->traffic.has_value(),
                name + ": the scalar route's workspace, tile and traffic");
            // With room for a chunk the copy runs where the build has it; else still scalar.
            const std::size_t room = std::size_t{1} << 30;
            const auto big         = ask(room);
            check(big.has_value(), name + ": described with workspace");
            if (big) {
                const bool runs = generic::volta_cutlass_call_runs(generic::codec_of(w), n, k,
                                                                   tokens, false, room);
                check((big->resolved.executed == tiled) == runs &&
                          (big->resolved.reason == generic::Demotion::None) == runs,
                      name + ": with room the description follows the shared rule");
                if (runs) {
                    check(big->workspace_bytes ==
                              generic::volta_cutlass_chunk_bytes(n, k, tokens, false, room),
                          name + ": the chunk that fits is the workspace it takes");
                }
            }
        } else {
            check(none->resolved.executed == tiled &&
                      none->resolved.reason == generic::Demotion::None,
                  name + ": a format with a tile keeps its tiled route");
        }
    }
    check(cutlass_only_formats > 0, "at least one CUTLASS-only format was exercised");
    check(std::string(generic::demotion_name(generic::Demotion::VoltaCopy)) == "volta_copy",
          "demotion name");
    if (failures == 0) std::cout << "route description (volta): ok\n";
    return failures == 0 ? 0 : 1;
}
