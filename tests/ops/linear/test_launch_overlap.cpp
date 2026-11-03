// A generic launch's overlap with its neighbours (#309, ops/generic/generic_gemm.h launch_overlap):
// join, release and prefetch each with the reason it is refused. Two checks: every refusal named
// where its rule says (the pre-quantized R8 form never releases or prefetches, release is not
// granted without PDL, a joined launch does not prefetch), and the flags it gives equal the rule
// run_route applied before (restated here) for every family, form, epilogue, width, ask and PDL.
// Host only: a launch is described, nothing runs.
#include "ops/generic/generic_gemm.h"

#include <iostream>
#include <string>

namespace g = ninferno::ops::detail::generic;
using R     = g::OverlapRefusal;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

g::ResolvedRoute route(g::Family family, bool prequantized) {
    g::ResolvedRoute r;
    r.asked = r.executed = g::Route{family, 0};
    r.r8_prequantized    = prequantized;
    return r;
}

} // namespace

int main() {
    using F = g::Family;
    // The refusals where their rules put them.
    {
        const auto pre = g::launch_overlap(route(F::R8Mma, true), g::Epilogue::Store, 4, false, true, true);
        check(pre.release == R::PrequantizedR8 && pre.prefetch == R::PrequantizedR8,
              "the pre-quantized R8 form neither releases nor prefetches");
        const auto r8 = g::launch_overlap(route(F::R8Mma, false), g::Epilogue::Store, 4, false, true, true);
        check(r8.releases() && r8.prefetches(), "R8 quantizing in its kernel releases and prefetches");
        const auto no_pdl = g::launch_overlap(route(F::RegisterMma, false), g::Epilogue::Store, 1, false, true, false);
        check(no_pdl.release == R::NoPdl && no_pdl.prefetch == R::NoPdl, "no PDL: no release, no prefetch");
        const auto joined = g::launch_overlap(route(F::RegisterMma, false), g::Epilogue::Store, 1, true, false, true);
        check(joined.joins() && joined.prefetch == R::Joined && joined.release == R::NotAsked,
              "a joined launch does not prefetch; an unasked release is said");
        const auto wide = g::launch_overlap(route(F::RegisterMma, false), g::Epilogue::Store,
                                            g::kPdlPrefetchMaxTokens + 1, false, false, true);
        check(wide.prefetch == R::Wide, "wider than the prefetch's tokens");
        const auto gated = g::launch_overlap(route(F::RegisterMma, false), g::Epilogue::SwiGlu, 1, true, true, true);
        check(gated.join == R::Epilogue && gated.release == R::Epilogue && gated.prefetches(),
              "a gated epilogue neither joins nor releases, yet prefetches");
        const auto tiled = g::launch_overlap(route(F::TiledMma, false), g::Epilogue::Store, 1, true, true, true);
        check(tiled.join == R::Family && tiled.release == R::Family && tiled.prefetch == R::Family,
              "another family runs in stream order");
    }
    // The flags equal run_route's rule before #309, for every case.
    int cases = 0;
    for (const F family : {F::RegisterMma, F::Scalar, F::W4a4Mma, F::TiledMma, F::W4a8Mma, F::StagedMma,
                           F::Nvfp4TiledMma, F::R8Mma, F::GgmlMmq, F::GgmlVec, F::I8Mma}) {
        for (const bool pre : {false, true}) {
            // resolve_route never marks another family pre-quantized.
            if (pre && family != F::R8Mma) continue;
            for (const auto epilogue : {g::Epilogue::Store, g::Epilogue::AddResidual, g::Epilogue::SwiGlu}) {
                for (const int tokens : {1, g::kPdlPrefetchMaxTokens, g::kPdlPrefetchMaxTokens + 1, 64}) {
                    for (const bool join : {false, true}) {
                        for (const bool release : {false, true}) {
                            for (const bool pdl : {false, true}) {
                                const auto o = g::launch_overlap(route(family, pre), epilogue, tokens, join,
                                                                 release, pdl);
                                const bool joinable = (family == F::RegisterMma || family == F::R8Mma) &&
                                                      epilogue == g::Epilogue::Store;
                                const bool old_join     = join && joinable;
                                const bool old_flag     = release && joinable;
                                const bool old_released = old_flag && !(family == F::R8Mma && pre) && pdl;
                                const bool old_prefetch =
                                    (family == F::RegisterMma || (family == F::R8Mma && !pre)) && !old_join &&
                                    tokens <= g::kPdlPrefetchMaxTokens && pdl;
                                const std::string at = "family " + std::to_string(static_cast<int>(family)) +
                                                       (pre ? " pre" : "") + " epilogue " +
                                                       std::to_string(static_cast<int>(epilogue)) + " T " +
                                                       std::to_string(tokens) + (join ? " join" : "") +
                                                       (release ? " release" : "") + (pdl ? " pdl" : "");
                                check(o.joins() == old_join, at + ": join as before");
                                check(o.releases() == old_released, at + ": released as before");
                                check(o.prefetches() == old_prefetch, at + ": prefetch as before");
                                // The kernel flag: as before, but cleared for the pre-quantized form,
                                // whose kernel never read it (generic_r8.cuh kPre).
                                const bool flag = o.releases() || o.release == R::NoPdl;
                                check(flag == (old_flag && !(family == F::R8Mma && pre)),
                                      at + ": the release flag as before");
                                ++cases;
                            }
                        }
                    }
                }
            }
        }
    }
    if (failures != 0) {
        std::cerr << "launch overlap: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "launch overlap: " << cases << " cases as run_route had them, each refusal named\n";
    return 0;
}
