#include "ops/generic/route_authorities.h"

#include "core/device_class.h"
#include "core/measured_boards.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/common/route_probe.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/route_table.h"
#include "ops/route_pins.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ninferno::ops::detail::generic {

namespace {

RouteAnswer by_rule(Route route, std::string_view source) {
    return RouteAnswer{route, Evidence{Evidence::Kind::Ranker, Evidence::Ranker::Rule, source}};
}

RouteAnswer by_pin(Route route, std::string_view source) {
    return RouteAnswer{route, Evidence{Evidence::Kind::Pin, Evidence::Ranker::None, source}};
}

RouteAnswer measured_here(Route route, std::string_view source) {
    return RouteAnswer{route, Evidence{Evidence::Kind::Measured, Evidence::Ranker::None, source}};
}

// "8.9:128sm": a measured gate's donor, which the gate tables name by their board alone.
std::string board_label(const Board& b) {
    return std::to_string(b.major) + "." + std::to_string(b.minor) + ":" + std::to_string(b.sms) +
           "sm";
}

// ---- select: the profiles, the probe, Volta's fixed routes and the pins -----------------------

// The invariant profile: one summation order per problem at every width, BF16 activations
// whatever the policy allows. The scalar route (one lane order per element) takes what the
// register MMA cannot read.
std::optional<RouteAnswer> invariant_profile(const Ask& ask, const Problem& problem,
                                             const Weight& weight) {
    if (ask.profile != Profile::Invariant) return std::nullopt;
    const Route r{Family::RegisterMma,
                  static_cast<std::uint8_t>(invariant_rmma(problem.k, ask.tokens))};
    if (!route_admits(r, weight, problem.policy)) {
        return by_rule(Route{Family::Scalar, 0}, "invariant-profile");
    }
    // NVFP4 wide calls: the tiled kernel sums in the register MMA's order (the same bytes).
    if (const auto t = nvfp4_tiled_route(problem, weight, rmma_schedule_info(r.schedule).warps_k,
                                         ask.tokens)) {
        return by_rule(*t, "invariant-profile");
    }
    return by_rule(r, "invariant-profile");
}

// A route probe forcing one activation route for every width (route_probe.h).
std::optional<RouteAnswer> route_probe(const Ask& ask, const Problem& problem,
                                       const Weight& weight) {
    const std::int8_t forced = route_probe_state.quantized_act;
    if (forced < 0) return std::nullopt;
    const bool prefill_band = ask.prefill_band();
    if (forced == 1 && allows_w4a4(weight, problem.policy)) {
        const Route r{Family::W4a4Mma, static_cast<std::uint8_t>(
                                           heuristic_w4a4(problem.n, ask.tokens, prefill_band))};
        if (route_admits(r, weight, problem.policy)) return by_pin(r, "route-probe");
    }
    if (forced == 1 && r8_call(weight, problem.policy, ask.tokens)) {
        const Route r{Family::R8Mma, static_cast<std::uint8_t>(heuristic_r8(problem, ask.tokens))};
        if (route_admits(r, weight, problem.policy)) return by_pin(r, "route-probe");
    }
    if (forced == 1 && allows_w4a8(weight, problem.policy)) {
        if (const auto r = w4a8_route(problem, weight, ask.tokens))
            return by_pin(*r, "route-probe");
    }
    const Route r{Family::RegisterMma,
                  static_cast<std::uint8_t>(heuristic_rmma(problem, ask.tokens, prefill_band))};
    if (route_admits(r, weight, problem.policy)) return by_pin(r, "route-probe");
    return by_pin(Route{Family::Scalar, 0}, "route-probe");
}

// Volta's tiled route from 17 tokens (volta_tiled_route). Volta keeps it under a prefill pin (f16
// activations: the a16 family's there), and the ledger notes it, so a pin no prefill call can
// take is refused.
std::optional<RouteAnswer> volta_tiled(const Ask& ask, const Problem& problem,
                                       const Weight& weight) {
    const auto r = volta_tiled_route(weight, problem.policy, ask.tokens);
    if (!r) return std::nullopt;
    if (ask.tokens >= kPrefillPinMinTokens && route_pins().prefill != PrefillRoute::Auto) {
        note_prefill_pin(route_pins().prefill == PrefillRoute::A16, family_bit(PrefillRoute::A16));
    }
    return by_rule(*r, "volta-tiled");
}

// A maintainer's route pin (--prefill-route, --decode-route; ops/route_pins.h): the pinned family
// where it can compute the call, else the automatic route below. No pin: nothing here. It acts
// before Volta's HMMA verify route, the automatic choice of that band, so an A/B can pin vec or
// R8 against it.
std::optional<RouteAnswer> route_pin(const Ask& ask, const Problem& problem, const Weight& weight) {
    if (const auto r = pinned_route(problem, weight, ask.tokens, ask.prefill())) {
        return by_pin(*r, "route-pin");
    }
    return std::nullopt;
}

// Volta's HMMA route of the verify widths (volta_hmma_route).
std::optional<RouteAnswer> volta_hmma(const Ask& ask, const Problem& problem,
                                      const Weight& weight) {
    if (const auto r = volta_hmma_route(problem, weight, ask.tokens)) {
        return by_rule(*r, "volta-hmma");
    }
    return std::nullopt;
}

constexpr NamedAuthority kSelectAuthorities[] = {
    {"invariant-profile", invariant_profile},
    {"route-probe", route_probe},
    {"volta-tiled", volta_tiled},
    {"route-pin", route_pin},
    {"volta-hmma", volta_hmma},
};

// ---- auto: the measured answers ---------------------------------------------------------------

// An NVFP4 A16 call of 17 tokens or more: the scope of the half-L2 rule, and the calls the rest of
// the automatic selection treats as the prefill profile's (nvfp4_promoted).
bool nvfp4_a16_wide(const Ask& ask, const Problem& problem, const Weight& weight) {
    return problem.qtype == QType::NVFP4 && ask.tokens >= 17 &&
           !allows_w4a4(weight, problem.policy) && !allows_w4a8(weight, problem.policy);
}

// NVFP4 keeps the width invariance its own kernels give where a device has them (the shape
// tables of sm_100+; ops/linear/nvfp4): on sm_89/86 the same projections are the generic
// path's, whose speed bands split K by width (heuristic_rmma: 4, 8, 2 or 1 K-warps from 17
// tokens), so a prefill token's bytes followed its chunk's size (Gemma 4 --a16 on an RTX 4090,
// test_linear_gemma4_token_invariance). An A16 call of 17 tokens or more takes one summation
// order at every width; decode and verify (T <= 16) keep their routes. W4A4 and W4A8 sum each
// token over its own activation whatever the tile, so A4/A8 calls keep their speed tiles.
// A weight that fits half the L2 takes the invariant profile's register MMA (the decode split of
// K, 16 rows per CTA): its 32-token columns re-read the weight from L2, where the 128x128 prefill
// tile left a 1024-row projection eight CTAs (RTX 4090, Gemma 4 26B shapes 1024..8192 x
// 2112..8192 at T 17..128: 2-4x slower than the speed bands). A larger weight takes the prefill
// profile's route (31B 16384x5376, 43008x5376, 5376x21504: 0.3-0.8x the speed bands from 32
// tokens).
std::optional<RouteAnswer> nvfp4_half_l2(const Ask& ask, const Problem& problem,
                                         const Weight& weight) {
    if (!nvfp4_a16_wide(ask, problem, weight)) return std::nullopt;
    const auto weight_bytes =
        static_cast<std::size_t>(problem.n) * static_cast<std::size_t>(problem.k) * 9 / 16;
    if (2 * weight_bytes <= device_class().l2_bytes) {
        const Route r{Family::RegisterMma,
                      static_cast<std::uint8_t>(invariant_rmma(problem.k, ask.tokens))};
        if (route_admits(r, weight, problem.policy)) return by_rule(r, "nvfp4-half-l2");
    }
    return std::nullopt;
}

// The Ask the authorities after the half-L2 rule see: an NVFP4 A16 wide call as the prefill
// profile's.
Ask nvfp4_promoted(const Ask& ask, const Problem& problem, const Weight& weight) {
    if (!nvfp4_a16_wide(ask, problem, weight)) return ask;
    Ask out     = ask;
    out.profile = Profile::Prefill;
    return out;
}

// The int8 calls the route tables stand aside for: a W4A8 problem of 17 tokens or more takes the
// W4A8 heuristic (route tables hold A16 schedules only: compute_class keys these formats as a16
// whatever the policy), an R8 call likewise, and ggml's kernels stand in for the heuristic only.
bool a8_wide_call(const Ask& ask, const Problem& problem, const Weight& weight) {
    return ask.tokens >= 17 && allows_w4a8(weight, problem.policy);
}

bool a8_call(const Ask& ask, const Problem& problem, const Weight& weight) {
    return a8_wide_call(ask, problem, weight) || vec_rule(weight, problem.policy, ask.tokens) ||
           r8_call(weight, problem.policy, ask.tokens) ||
           mmq_verify_rule(weight, problem.policy, problem.n, ask.tokens, ask.lanes);
}

// The route table's entry for the exact problem (route_table.h): the nearest measured board's
// table, its gaps filled from the next nearest. Under the prefill profile a format with a tiled
// codec keeps its tiled route: the tiled and register-MMA families sum in different orders.
std::optional<RouteAnswer> route_table_entry(const Ask& ask, const Problem& problem,
                                             const Weight& weight) {
    if (a8_call(ask, problem, weight)) return std::nullopt;
    const auto hit = table_route_answer(problem, ask.tokens, ask.prefill());
    if (!hit) return std::nullopt;
    const bool prefill_band = ask.prefill_band();
    if (!route_admits(hit->route, weight, problem.policy) ||
        (prefill_band && !route_is_token_invariant(hit->route)) ||
        (prefill_band && supports(weight) && tiled_supports(codec_of(weight)) &&
         hit->route.family != Family::TiledMma)) {
        return std::nullopt;
    }
    Evidence e;
    e.ranker   = Evidence::Ranker::None;
    e.source   = "route-table";
    e.distance = hit->distance;
    if (!hit->lender.empty()) {
        e.kind  = Evidence::Kind::Borrowed;
        e.donor = std::string(hit->lender);
    } else if (hit->distance == 0) {
        e.kind = Evidence::Kind::Measured;
    } else {
        e.kind  = hit->distance <= measured::kCarryDistance ? Evidence::Kind::Carried
                                                            : Evidence::Kind::Transferred;
        e.donor = std::string(hit->gpu_class);
    }
    return RouteAnswer{hit->route, std::move(e)};
}

// The dense tuner's choice at decode widths (generic_tune.cpp): an A16 schedule for an A16 call,
// an R8 schedule for an R8 decode call; never a W4A8 call. Under the vector kernel's rule any
// timed route that beat it may stand in.
std::optional<RouteAnswer> dense_tune_entry(const Ask& ask, const Problem& problem,
                                            const Weight& weight) {
    if (a8_wide_call(ask, problem, weight)) return std::nullopt;
    const auto dense = tuned_dense_route(problem, ask.tokens);
    if (!dense || !route_admits(*dense, weight, problem.policy)) return std::nullopt;
    const bool r8 = r8_call(weight, problem.policy, ask.tokens);
    if (dense->family == Family::R8Mma) {
        if (!(r8 || r8_verify_tunable(weight, problem.policy, ask.tokens))) return std::nullopt;
    } else if (!(vec_rule(weight, problem.policy, ask.tokens) || !r8)) {
        return std::nullopt;
    }
    return measured_here(*dense, "dense-tune");
}

// The dense tuner's prefill bands, keyed by the rules' route: a W4A8 schedule for the W4A8
// heuristic (the same bytes at every tile), a route of the prefill route's summation order under
// the prefill profile.
std::optional<RouteAnswer> dense_tune_prefill_band(const Ask& ask, const Problem& problem,
                                                   const Weight& weight, Route heuristic) {
    const auto band = tuned_prefill_route(problem, ask.tokens, heuristic);
    if (!band || !route_admits(*band, weight, problem.policy)) return std::nullopt;
    if ((band->family == Family::W4a8Mma && heuristic.family == Family::W4a8Mma) ||
        (ask.prefill() && band->family != Family::W4a8Mma && same_summation(*band, heuristic))) {
        return measured_here(*band, "dense-tune-prefill-band");
    }
    return std::nullopt;
}

constexpr NamedAuthority kAutoAuthorities[] = {
    {"nvfp4-half-l2", nvfp4_half_l2},
    {"route-table", route_table_entry},
    {"dense-tune", dense_tune_entry},
};

// ---- rules: the shape-class heuristic -----------------------------------------------------------

// A4 only where the policy allows it, and never at T <= 16, which is where decode and speculative
// verify run.
std::optional<RouteAnswer> w4a4_rule(const Ask& ask, const Problem& problem, const Weight& weight) {
    if (ask.tokens < 17 || !allows_w4a4(weight, problem.policy)) return std::nullopt;
    const Route r{Family::W4a4Mma, static_cast<std::uint8_t>(
                                       heuristic_w4a4(problem.n, ask.tokens, ask.prefill_band()))};
    if (route_admits(r, weight, problem.policy)) return by_rule(r, "w4a4-rule");
    return std::nullopt;
}

// ggml's vector kernel for speculative verify (2..4 tokens) of the ggml block formats it decodes
// (vec_rule). It streams each weight row once for every token of the call and runs the epilogue
// in place: on the RTX 5090 ISTA IQ3_S MTP3 it beat the R8 and A16 schedules the dense tuner
// picked there (C1 166 -> 174 tok/s), whose L2-cold timings of one call at a time miss the stream
// of a verify step. The tuner still replaces it where a route beats it by its margin (Muse Glimmer
// 30B Q4_K_M: the rule alone lost C1 by 19 %).
std::optional<RouteAnswer> vec_kernel_rule(const Ask& ask, const Problem& problem,
                                           const Weight& weight) {
    if (vec_rule(weight, problem.policy, ask.tokens)) {
        return by_rule(Route{Family::GgmlVec, 0}, "vec-rule");
    }
    return std::nullopt;
}

// From 17 tokens outside the prefill profile the cost model's route of either activation class
// (cost_route, a8_candidates); the rules below where the device reports no clocks.
std::optional<RouteAnswer> cost_model(const Ask& ask, const Problem& problem,
                                      const Weight& weight) {
    if (ask.prefill_band() || ask.tokens < 17 || problem.qtype == QType::NVFP4) return std::nullopt;
    if (const auto r = cost_route(problem, weight, ask.tokens)) {
        return RouteAnswer{*r,
                           Evidence{Evidence::Kind::Ranker, Evidence::Ranker::Cost, "cost-model"}};
    }
    return std::nullopt;
}

// R8 for decode and for the verify of the formats with an R8 codec (r8_call): at one token the
// entry of the nearest measured board's decode table (tuned_r8, within kR8CarryDistance), else the
// R8 rules (heuristic_r8).
std::optional<RouteAnswer> r8_schedule(const Ask& ask, const Problem& problem,
                                       const Weight& weight) {
    if (!r8_call(weight, problem.policy, ask.tokens)) return std::nullopt;
    RouteAnswer answer = by_rule(r8_rules_route(problem, ask.tokens), "r8-rules");
    if (ask.tokens == 1) {
        if (const auto hit = tuned_r8(problem)) {
            answer.route           = Route{Family::R8Mma, static_cast<std::uint8_t>(hit->schedule)};
            answer.evidence.ranker = Evidence::Ranker::None;
            answer.evidence.source = "r8-table";
            answer.evidence.distance = hit->distance;
            if (hit->distance == 0) {
                answer.evidence.kind = Evidence::Kind::Measured;
            } else {
                answer.evidence.kind  = Evidence::Kind::Carried;
                answer.evidence.donor = std::string(hit->gpu);
            }
        }
    }
    if (route_admits(answer.route, weight, problem.policy)) return answer;
    return std::nullopt;
}

// ggml's matrix kernel for a prefill of the ggml block formats from 17 tokens, where the nearest
// measured board of the instruction set says it pays (mmq_prefill_wins; the gate carries at any
// distance).
std::optional<RouteAnswer> mmq_prefill_gate(const Ask& ask, const Problem& problem,
                                            const Weight& weight) {
    if (ask.tokens < 17 || !mmq_prefill_wins(problem, weight)) return std::nullopt;
    const measured::GateAnswer gate = mmq_prefill_device_gate();
    Evidence e;
    e.ranker   = Evidence::Ranker::None;
    e.source   = "mmq-prefill-gate";
    e.distance = gate.distance;
    using D    = measured::GateAnswer::Disposition;
    switch (gate.disposition) {
    case D::Measured:
        e.kind = Evidence::Kind::Measured;
        break;
    case D::Carried:
        e.kind = Evidence::Kind::Carried;
        break;
    case D::Transferred:
    case D::Incomplete:
    case D::None:
        e.kind = Evidence::Kind::Transferred;
        break;
    }
    if (gate.donor != nullptr && e.kind != Evidence::Kind::Measured) {
        e.donor = board_label(gate.donor->board);
    }
    return RouteAnswer{Route{Family::GgmlMmq, 0}, std::move(e)};
}

// The I8 tile of a call from 17 tokens (i8_route: the cost rule over its tiles).
std::optional<RouteAnswer> i8_rule(const Ask& ask, const Problem& problem, const Weight& weight) {
    if (ask.tokens < 17) return std::nullopt;
    if (const auto r = i8_route(problem, weight, ask.tokens)) return by_rule(*r, "i8-rule");
    return std::nullopt;
}

// The W4A8 tile from 17 tokens where the policy allows it (w4a8_route: the W4A8 rules' tile, else
// the first schedule that fits the build).
std::optional<RouteAnswer> w4a8_rule(const Ask& ask, const Problem& problem, const Weight& weight) {
    if (ask.tokens < 17 || !allows_w4a8(weight, problem.policy)) return std::nullopt;
    if (const auto r = w4a8_route(problem, weight, ask.tokens)) return by_rule(*r, "w4a8-rule");
    return std::nullopt;
}

// Where the BF16 MMA is emulated (Turing), the band between R8 / the vector kernel and W4A8 takes
// ggml's int8 matrix kernel for the formats it reads: the register MMA there runs on the
// emulation, about three times its native cost. With a native MMA the codebook formats' verify of
// several sequences (mmq_verify_rule).
std::optional<RouteAnswer> mmq_verify(const Ask& ask, const Problem& problem,
                                      const Weight& weight) {
    if (mmq_verify_rule(weight, problem.policy, problem.n, ask.tokens, ask.lanes)) {
        return by_rule(Route{Family::GgmlMmq, 0}, "mmq-verify-rule");
    }
    return std::nullopt;
}

// NVFP4 A16 outside the invariant profile stays on the register MMA: against its prefill
// schedule the tiled route loses on the Muse gate/up and synthetic shapes (+11..+20%)
// and wins on the down projection only from T = 2048. A route table may still name it.
// A format with a tiled codec takes it for every prefill-profile call of 17 tokens or more (one
// summation order for all of them) and, outside the profile, where its tiles fill the SMs
// (tiled_speed_pays). The first schedule whose tiles fit the build: the tuned 128x128 everywhere
// but Turing.
std::optional<RouteAnswer> tiled_rule(const Ask& ask, const Problem& problem,
                                      const Weight& weight) {
    if (!(ask.prefill_band() || tiled_speed_pays(problem.n, ask.tokens))) return std::nullopt;
    for (int i = 0; i < tiled_schedule_count(); ++i) {
        const Route r{Family::TiledMma, static_cast<std::uint8_t>(i)};
        if (route_admits(r, weight, problem.policy)) return by_rule(r, "tiled-rule");
    }
    return std::nullopt;
}

// BF16 weights take the staged register MMA (same bytes, operands through shared memory): under
// the prefill profile the first schedule that fits the build, and outside it once its 64x64 tiles
// cover half the SMs, with the largest tile whose grid still fills them (staged_speed_route). Only
// on the native BF16 MMA: where it is emulated its tiles lost to the register MMA, x2.5..4 on
// Turing (RTX 2080 Ti, BF16 N 1536 K 2048, T 17..1024: 142..1005 us against 35..402) and x6..8 on
// Volta's FP32 FMAs (V100, T 17..128: 942 us against 110).
std::optional<RouteAnswer> staged_rule(const Ask& ask, const Problem& problem,
                                       const Weight& weight) {
    if (!bf16_mma_native()) return std::nullopt;
    if (ask.prefill_band()) {
        for (int i = 0; i < staged_schedule_count(); ++i) {
            const Route r{Family::StagedMma, static_cast<std::uint8_t>(i)};
            if (route_admits(r, weight, problem.policy)) return by_rule(r, "staged-rule");
        }
        return std::nullopt;
    }
    if (const auto r = staged_speed_route(problem, weight, ask.tokens)) {
        return by_rule(*r, "staged-rule");
    }
    return std::nullopt;
}

// The register MMA's shape-class rules (heuristic_rmma).
std::optional<RouteAnswer> rmma_rule(const Ask& ask, const Problem& problem, const Weight& weight) {
    const Route r{Family::RegisterMma, static_cast<std::uint8_t>(heuristic_rmma(
                                           problem, ask.tokens, ask.prefill_band()))};
    if (route_admits(r, weight, problem.policy)) return by_rule(r, "rmma-rule");
    return std::nullopt;
}

// A register-MMA tile the build does not hold for the call: the narrower tiles the dense tuner
// picks where the BF16 MMA is emulated (m1.n2 and m1.n1 of four row warps: V100 and RTX 2080 Ti,
// BF16 N 1536 K 2048, T 17..2048), then the staged MMA, before the scalar route (V100, gpt-oss
// BF16 attention at T = 79: the scalar route took 7.3 ms a call).
std::optional<RouteAnswer> rmma_narrow_fallback(const Ask&, const Problem& problem,
                                                const Weight& weight) {
    for (const int n_tiles : {2, 1}) {
        const int index = rmma_index(4, 1, 1, n_tiles);
        const Route narrow{Family::RegisterMma, static_cast<std::uint8_t>(index)};
        if (route_admits(narrow, weight, problem.policy)) {
            return by_rule(narrow, "rmma-narrow-fallback");
        }
    }
    return std::nullopt;
}

std::optional<RouteAnswer> staged_fallback(const Ask& ask, const Problem& problem,
                                           const Weight& weight) {
    if (!(ask.prefill_band() || ask.tokens >= kStagedSpeedMinTokens)) return std::nullopt;
    for (int i = 0; i < staged_schedule_count(); ++i) {
        const Route staged{Family::StagedMma, static_cast<std::uint8_t>(i)};
        if (route_admits(staged, weight, problem.policy)) return by_rule(staged, "staged-fallback");
    }
    return std::nullopt;
}

std::optional<RouteAnswer> scalar_fallback(const Ask&, const Problem&, const Weight&) {
    return by_rule(Route{Family::Scalar, 0}, "scalar-fallback");
}

constexpr NamedAuthority kRuleAuthorities[] = {
    {"w4a4-rule", w4a4_rule},
    {"vec-rule", vec_kernel_rule},
    {"cost-model", cost_model},
    {"r8-rules", r8_schedule}, // "r8-table" where the decode table answers
    {"mmq-prefill-gate", mmq_prefill_gate},
    {"i8-rule", i8_rule},
    {"w4a8-rule", w4a8_rule},
    {"mmq-verify-rule", mmq_verify},
    {"tiled-rule", tiled_rule},
    {"staged-rule", staged_rule},
    {"rmma-rule", rmma_rule},
    {"rmma-narrow-fallback", rmma_narrow_fallback},
    {"staged-fallback", staged_fallback},
    {"scalar-fallback", scalar_fallback},
};

} // namespace

Ask resolve_ask(std::int32_t tokens, bool prefill) {
    return Ask{invariant_routes_active() ? Profile::Invariant
               : prefill                 ? Profile::Prefill
                                         : Profile::Speed,
               tokens, ops::round_lanes()};
}

Ask ask_below_profiles(std::int32_t tokens, bool prefill) {
    return Ask{prefill ? Profile::Prefill : Profile::Speed, tokens, ops::round_lanes()};
}

std::span<const NamedAuthority> select_authorities() noexcept { return kSelectAuthorities; }

std::span<const NamedAuthority> auto_authorities() noexcept { return kAutoAuthorities; }

std::span<const NamedAuthority> rule_authorities() noexcept { return kRuleAuthorities; }

RouteAnswer heuristic_answer(const Ask& ask, const Problem& problem, const Weight& weight) {
    for (const NamedAuthority& a : kRuleAuthorities) {
        if (auto answer = a.answer(ask, problem, weight)) return *answer;
    }
    return by_rule(Route{Family::Scalar, 0}, "scalar-fallback");
}

RouteAnswer auto_answer(const Ask& ask, const Problem& problem, const Weight& weight) {
    const Ask asked = nvfp4_promoted(ask, problem, weight);
    for (const NamedAuthority& a : kAutoAuthorities) {
        if (auto answer = a.answer(asked, problem, weight)) return *answer;
    }
    RouteAnswer answer = heuristic_answer(asked, problem, weight);
    if (auto band = dense_tune_prefill_band(asked, problem, weight, answer.route)) return *band;
    return answer;
}

RouteAnswer select_answer(const Problem& problem, const Weight& weight, std::int32_t tokens,
                          bool prefill) {
    const Ask ask = resolve_ask(tokens, prefill);
    for (const NamedAuthority& a : kSelectAuthorities) {
        if (auto answer = a.answer(ask, problem, weight)) return *answer;
    }
    return auto_answer(ask, problem, weight);
}

Route select_route(const Problem& problem, const Weight& weight, std::int32_t tokens,
                   bool prefill) {
    return select_answer(problem, weight, tokens, prefill).route;
}

Route auto_route(const Problem& problem, const Weight& weight, std::int32_t tokens, bool prefill) {
    return auto_answer(ask_below_profiles(tokens, prefill), problem, weight).route;
}

Route heuristic_route(const Problem& problem, const Weight& weight, std::int32_t tokens,
                      bool prefill) {
    return heuristic_answer(ask_below_profiles(tokens, prefill), problem, weight).route;
}

} // namespace ninferno::ops::detail::generic
