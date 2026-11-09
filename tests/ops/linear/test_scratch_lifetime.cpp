// Scratch read lifetimes of generic launches (#309, ops/generic/scratch_lifetime.h): the hazard
// rule on hand-built chains, and the chains launch_overlap's refusals produce for the pre-quantized
// R8 form and its neighbours. Host only: launches are described, nothing runs, and CPU ordering
// tests do not prove CUDA ordering (the GPU race checks are separate).
#include "ops/generic/scratch_lifetime.h"

#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace g = ninferno::ops::detail::generic;
using H     = g::ScratchHazard;

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

// The prepared representation a family's launch reads, as prepared_rule says (nullopt: it reads x
// as BF16 or quantizes inside the kernel). The tests hand the rule in; the production caller passes
// prepared_rule(route, weight).
std::optional<g::PreparedRule> rule_of(const g::ResolvedRoute& r) {
    using P = g::PreparedRepresentation;
    switch (r.executed.family) {
    case g::Family::R8Mma:
        return r.r8_prequantized ? std::optional<g::PreparedRule>{{P::W4a8Blocks, 0, 0, false}}
                                 : std::nullopt;
    case g::Family::GgmlVec: return g::PreparedRule{P::GgmlQ8_1, 0, 0, false};
    case g::Family::GgmlMmq: return g::PreparedRule{P::GgmlQ8_1Mmq, 0, 0, false};
    case g::Family::W4a8Mma: return g::PreparedRule{P::W4a8Blocks, 0, 0, false};
    case g::Family::I8Mma: return g::PreparedRule{P::I8Blocks64, 0, 0, true};
    case g::Family::W4a4Mma: return g::PreparedRule{P::Nvfp4, 0, 0, true};
    default: return std::nullopt;
    }
}

constexpr g::ScratchRegion kA{1, 0, 4096};
constexpr g::ScratchRegion kB{2, 4096, 4096};     // disjoint from kA
constexpr g::ScratchRegion kHalfA{3, 2048, 4096}; // overlaps both

g::LaunchLifetime launch(std::string_view name, bool early, bool releases) {
    g::LaunchLifetime l;
    l.name         = name;
    l.starts_early = early;
    l.releases     = releases;
    return l;
}

// A quantizer that starts under its predecessor and writes `region` before its wait.
g::LaunchLifetime early_quantizer(const g::ScratchRegion& region) {
    auto q          = g::quantizer_lifetime("next_quantizer", region);
    q.starts_early  = true;
    q.uses[0].first = g::ScratchFirst::BeforeWait;
    return q;
}

} // namespace

int main() {
    // Region overlap is by bytes; an empty region overlaps nothing.
    check(g::regions_overlap(kA, kHalfA) && g::regions_overlap(kHalfA, kB), "overlapping ranges");
    check(!g::regions_overlap(kA, kB), "adjacent ranges do not overlap");
    check(!g::regions_overlap(kA, g::ScratchRegion{9, 100, 0}), "an empty range overlaps nothing");

    // The use rule.
    {
        using A = g::ScratchAccess;
        using F = g::ScratchFirst;
        using L = g::ScratchLast;
        const g::ScratchUse read_exit{kA, A::Read, F::AfterWait, L::AtExit};
        const g::ScratchUse read_trigger{kA, A::Read, F::AfterWait, L::AtTrigger};
        const g::ScratchUse write_early{kA, A::Write, F::BeforeWait, L::AtExit};
        const g::ScratchUse write_late{kA, A::Write, F::AfterWait, L::AtExit};
        const g::ScratchUse read_early{kA, A::Read, F::BeforeWait, L::AtExit};
        check(g::use_hazard(read_exit, write_early) == H::WriteAfterRead,
              "a write under a read that lasts to exit is a hazard");
        check(!g::use_hazard(read_trigger, write_early).has_value(),
              "a read finished at the trigger is over before the next launch begins");
        check(!g::use_hazard(read_exit, write_late).has_value(),
              "a write after the wait is ordered behind completion");
        check(!g::use_hazard(read_exit, read_early).has_value(), "two reads never conflict");
        check(g::use_hazard(write_late, read_early) == H::ReadAfterWrite,
              "release is not completion: the output is not there yet");
        check(g::use_hazard(write_late, write_early) == H::WriteAfterWrite, "two writers conflict");
        check(!g::use_hazard(read_exit, g::ScratchUse{kB, A::Write, F::BeforeWait, L::AtExit})
                   .has_value(),
              "disjoint regions never conflict");
    }

    // A pre-quantized R8 projection, then the next projection's quantizer into the same region.
    // launch_overlap gives the projection no release, so the quantizer cannot start early under
    // it; were the release granted, the hazard would show.
    {
        const auto pre        = route(g::Family::R8Mma, true);
        const auto overlap    = g::launch_overlap(pre, g::Epilogue::Store, 4, false, true, true);
        const auto projection = g::launch_lifetime("r8_pre", overlap, true, rule_of(pre), kA);
        check(projection.use_count == 1 && !projection.releases && !projection.starts_early,
              "pre-quantized R8: reads its prepared region to exit, no release, in stream order");

        // As launched today: stream order, the quantizer never starts early.
        check(!g::first_scratch_violation({projection, g::quantizer_lifetime("q", kA)}).has_value(),
              "serialized pre-quantized R8 then quantizer: no hazard");

        // A launch asked to start under a predecessor that does not release starts at its
        // completion (the trigger is implicit at exit): nothing overlaps, so nothing conflicts.
        auto v = g::first_scratch_violation({projection, early_quantizer(kA)});
        check(!v.has_value(), "an early start under a non-releasing launch is not a hazard");
        const auto mma     = route(g::Family::RegisterMma, false);
        const auto prefetch = g::launch_lifetime(
            "decode_prefetch", g::launch_overlap(mma, g::Epilogue::Store, 1, false, false, true),
            true, std::nullopt, std::nullopt);
        check(prefetch.starts_early &&
                  !g::first_scratch_violation({projection, prefetch}).has_value(),
              "pre-quantized R8 then a decode prefetch launch: clean");

        // Were the projection to release, a quantizer on the same region would clobber its read.
        auto releasing     = projection;
        releasing.releases = true;
        v                  = g::first_scratch_violation({releasing, early_quantizer(kA)});
        check(v && v->hazard == H::WriteAfterRead,
              "release is not completion: the same region would be clobbered");

        // The qualified alternative (#309): disjoint prepared storage until the consumer's last
        // read. The checker admits it; a partly overlapping region is still refused.
        check(!g::first_scratch_violation({releasing, early_quantizer(kB)}).has_value(),
              "disjoint prepared-activation storage admits the overlap");
        check(g::first_scratch_violation({releasing, early_quantizer(kHalfA)}).has_value(),
              "a partly overlapping region is refused");
    }

    // Two side-by-side register-MMA projections (no scratch): the chain launch_overlap grants is
    // clean.
    {
        const auto mma      = route(g::Family::RegisterMma, false);
        const auto first_o  = g::launch_overlap(mma, g::Epilogue::Store, 1, false, true, true);
        const auto second_o = g::launch_overlap(mma, g::Epilogue::Store, 1, true, false, true);
        const auto first    = g::launch_lifetime("proj0", first_o, true, std::nullopt, std::nullopt);
        const auto second   = g::launch_lifetime("proj1", second_o, true, std::nullopt, std::nullopt);
        check(first.releases && second.starts_early &&
                  !g::first_scratch_violation({first, second}).has_value(),
              "two register-MMA projections: join under a released predecessor, no scratch shared");
        // Without PDL a join falls back to stream order: it does not start early at all.
        const auto no_pdl = g::launch_overlap(mma, g::Epilogue::Store, 1, true, false, false);
        check(no_pdl.joins() &&
                  !g::launch_lifetime("proj1", no_pdl, false, std::nullopt, std::nullopt).starts_early,
              "join asked, no PDL: stream order");
        auto reader = g::launch_lifetime("reader", first_o, false, std::nullopt, std::nullopt);
        reader.add({kA, g::ScratchAccess::Read, g::ScratchFirst::AfterWait, g::ScratchLast::AtExit});
        reader.releases = true; // as if PDL were on
        auto joined     = g::launch_lifetime("joined", no_pdl, false, std::nullopt, std::nullopt);
        joined.add({kA, g::ScratchAccess::Write, g::ScratchFirst::BeforeWait, g::ScratchLast::AtExit});
        check(!g::first_scratch_violation({reader, joined}).has_value(),
              "join=true, pdl=false: clean");
    }

    // A chain of early starts is checked back past the middle launch: C would write what A reads
    // to exit while B (no scratch) lets C start before A completes.
    {
        auto a = launch("a", false, true);
        a.add({kA, g::ScratchAccess::Read, g::ScratchFirst::AfterWait, g::ScratchLast::AtExit});
        auto b = launch("b", true, true);
        auto c = launch("c", true, false);
        c.add({kA, g::ScratchAccess::Write, g::ScratchFirst::BeforeWait, g::ScratchLast::AtExit});
        const auto v = g::first_scratch_violation({a, b, c});
        check(v && v->launch == 2 && v->predecessor == 0 && v->hazard == H::WriteAfterRead,
              "early starts chain: the write is checked against the first running reader");
        b.starts_early = false;
        check(!g::first_scratch_violation({a, b, c}).has_value(),
              "a launch ordered after its predecessor's completion cuts the chain");
    }

    // Across every family, epilogue, width, ask and PDL setting: a launch that reads a prepared
    // scratch region never releases through launch_overlap (the restriction it keeps).
    int cases = 0;
    for (const auto family :
         {g::Family::RegisterMma, g::Family::R8Mma, g::Family::GgmlVec, g::Family::GgmlMmq,
          g::Family::TiledMma, g::Family::W4a8Mma, g::Family::I8Mma, g::Family::W4a4Mma,
          g::Family::StagedMma, g::Family::Nvfp4TiledMma, g::Family::Scalar}) {
        for (const bool pre : {false, true}) {
            if (pre && family != g::Family::R8Mma) continue;
            for (const auto epilogue : {g::Epilogue::Store, g::Epilogue::SwiGlu}) {
                for (const int tokens : {1, 4, 64}) {
                    for (const bool join : {false, true}) {
                        for (const bool release : {false, true}) {
                            for (const bool pdl : {false, true}) {
                                const auto r = route(family, pre);
                                const auto o =
                                    g::launch_overlap(r, epilogue, tokens, join, release, pdl);
                                const auto l = g::launch_lifetime("l", o, pdl, rule_of(r), kA);
                                if (l.use_count != 0) {
                                    check(!l.releases, "a prepared-region reader does not release");
                                }
                                ++cases;
                            }
                        }
                    }
                }
            }
        }
    }
    if (failures != 0) {
        std::cerr << "scratch lifetime: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "scratch lifetime: hazards named, " << cases << " launch_overlap cases checked\n";
    return 0;
}
