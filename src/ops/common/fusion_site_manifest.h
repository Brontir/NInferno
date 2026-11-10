#pragma once

// The manifest line of a fusion site (#617 step H): which branch the site takes for a board and a
// call width, and what the line evaluated. Host only, no device: a CPU reproduces it
// (docs/maintainer/deterministic-decisions.md). One line for every site of the registry
// (tools/audit/fusion_sites.json; fusion_site_table.h is its generated form), so a site cannot be
// added without a line; an emitter for the sites whose rule reads only the board and the call
// width says its branch, and the others say they are declared: their rule (the registry's
// `rule_today`) reads the weight, the model or the call's layout, which only the launch knows.
//
// Line grammar (one per site, in registry order):
//
//   site <id> <status> <branch> [<key>=<value> ...]
//
//   decided   the branch follows from the inputs listed on the line;
//   band      the call-width band alone is evaluated; the branch says what the band allows and the
//             rest of the rule (a weight's format, a tuner's timing) is the launch's;
//   declared  nothing is evaluated: the site has no host predicate yet, or the query lacks an input
//             its predicate reads (`needs=` lists them): "per_call", and the registry's rule
//             applies.
//
// Every evaluated site asks the function the launch asks, never a copy of its rule:
//   pdl_overlap      generic::launch_overlap (the launch's overlap decision)
//   r8_prequantized  generic::r8_prequantized_takes (resolve_route)
//   vec_gated_pair, mmq_gated_pair   generic::ggml_gated_pair_for (segmented_swiglu_pair)
//   qk_norm_rope     ops::rmsnorm_rope_text_takes_width (the Qwen 3.5 attention's dispatch)
//
// Not wired into the engine's startup manifest: that would add a section to every manifest. The
// engine adds `sites` when this lands in a manifest PR; until then the function is the contract and
// the test pins the lines. The regret format that scores each decided branch against its
// alternative is tools/bench/site_regret.py.

#include "core/device_class.h"
#include "ops/common/fusion_site_table.h"
#include "ninferno/ops/rmsnorm_rope.h"
#include "ops/generic/generic_gemm.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno::ops::detail::fusion {

// What a site's predicate may read. A field a predicate needs and the query leaves unset makes that
// site `declared` with the field named in `needs=`.
struct SiteQuery {
    const DeviceClass* device = nullptr;
    std::int32_t tokens = 1; // the call's width (a decode or verify width, or a prefill chunk)
    // The route the call runs (resolve_route's executed route), with the call's epilogue, whether
    // its launch asked to join the previous kernel or release the next, and for R8 the weight's
    // format, K and the workspace free to the call (nullopt: none).
    std::optional<generic::Route> route;
    generic::Epilogue epilogue = generic::Epilogue::Store;
    bool join_asked            = false;
    bool release_asked         = false;
    std::optional<QType> qtype;
    std::int32_t k = 0;
    std::optional<std::size_t> workspace_free;
    // The text attention geometry (query heads, key heads); 0: not given.
    std::int32_t query_heads = 0;
    std::int32_t key_heads   = 0;
};

struct SiteLine {
    std::string id;
    std::string status; // decided | band | declared
    std::string branch;
    std::vector<std::pair<std::string, std::string>> inputs;
};

[[nodiscard]] inline std::string text_of(const SiteLine& line) {
    std::string out = "site " + line.id + ' ' + line.status + ' ' + line.branch;
    for (const auto& [key, value] : line.inputs) { out += ' ' + key + '=' + value; }
    return out;
}

namespace detail_ {

inline SiteLine declared(std::string_view id, std::string needs = {}) {
    SiteLine line{std::string(id), "declared", "per_call", {}};
    if (!needs.empty()) { line.inputs.emplace_back("needs", std::move(needs)); }
    return line;
}

// The sites an emitter evaluates; every other row of the table is `declared`.
inline bool emit_evaluated(std::string_view id, const SiteQuery& q, SiteLine& out) {
    const std::string tokens = std::to_string(q.tokens);
    if (id == "pdl_overlap") {
        std::string needs;
        if (q.device == nullptr) { needs += "device,"; }
        if (!q.route) { needs += "route,"; }
        const bool r8 = q.route && q.route->family == generic::Family::R8Mma;
        if (r8 && (!q.qtype || q.k <= 0)) { needs += "qtype,k,"; }
        if (!needs.empty()) {
            needs.pop_back();
            out = declared(id, needs);
            return true;
        }
        generic::ResolvedRoute resolved{.asked = *q.route, .executed = *q.route};
        resolved.r8_prequantized = r8 && generic::r8_prequantized_takes(*q.route, *q.qtype, q.k,
                                                                        q.tokens, q.workspace_free);
        const generic::LaunchOverlap overlap = generic::launch_overlap(
            resolved, q.epilogue, q.tokens, q.join_asked, q.release_asked, q.device->features.pdl);
        out = {std::string(id),
               "decided",
               overlap.joins() || overlap.releases() || overlap.prefetches() ? "overlap"
                                                                             : "stream_order",
               {{"route", std::string(generic::route_name(*q.route))},
                {"join", std::string(generic::overlap_refusal_name(overlap.join))},
                {"release", std::string(generic::overlap_refusal_name(overlap.release))},
                {"prefetch", std::string(generic::overlap_refusal_name(overlap.prefetch))},
                {"pdl", q.device->features.pdl ? "1" : "0"},
                {"tokens", tokens}}};
        return true;
    }
    if (id == "vec_gated_pair" || id == "mmq_gated_pair") {
        const bool vec_band = generic::ggml_gated_pair_for(q.tokens) == generic::GgmlGatedPair::Vec;
        const bool mine     = (id == "vec_gated_pair") == vec_band;
        out                 = {
            std::string(id),
            "band",
            mine ? "tried" : "not_tried",
            {{"tokens", tokens}, {"vec_max_tokens", std::to_string(generic::kGgmlVecMaxTokens)}}};
        return true;
    }
    if (id == "r8_prequantized") {
        std::string needs;
        if (!q.route) { needs += "route,"; }
        if (!q.qtype || q.k <= 0) { needs += "qtype,k,"; }
        if (!needs.empty()) {
            needs.pop_back();
            out = declared(id, needs);
            return true;
        }
        const bool pre =
            generic::r8_prequantized_takes(*q.route, *q.qtype, q.k, q.tokens, q.workspace_free);
        out = {std::string(id),
               "decided",
               pre ? "prequantized" : "in_kernel",
               {{"route", std::string(generic::route_name(*q.route))},
                {"tokens", tokens},
                {"workspace_free", q.workspace_free ? std::to_string(*q.workspace_free) : "none"}}};
        return true;
    }
    if (id == "qk_norm_rope") {
        if (q.device == nullptr || q.query_heads <= 0 || q.key_heads <= 0) {
            out = declared(id, q.device == nullptr ? "device" : "query_heads,key_heads");
            return true;
        }
        const bool fused =
            rmsnorm_rope_text_takes_width(*q.device, q.query_heads, q.key_heads, q.tokens);
        out = {std::string(id),
               "band",
               fused ? "fused" : "decomposed",
               {{"geometry", std::to_string(q.query_heads) + '+' + std::to_string(q.key_heads)},
                {"tokens", tokens},
                {"max_tokens", std::to_string(rmsnorm_rope_text_fused_max_tokens(
                                   *q.device, q.query_heads, q.key_heads))}}};
        return true;
    }
    return false;
}

} // namespace detail_

// The line of one site; std::invalid_argument for an id the registry does not have.
[[nodiscard]] inline SiteLine site_line(std::string_view id, const SiteQuery& query) {
    bool known = false;
    for (const SiteRow& row : kSiteTable) { known = known || row.id == id; }
    if (!known) { throw std::invalid_argument("unknown fusion site: " + std::string(id)); }
    SiteLine line;
    return detail_::emit_evaluated(id, query, line) ? line : detail_::declared(id);
}

// Every site's line, in registry order.
[[nodiscard]] inline std::vector<SiteLine> all_site_lines(const SiteQuery& query) {
    std::vector<SiteLine> lines;
    lines.reserve(kSiteCount);
    for (const SiteRow& row : kSiteTable) { lines.push_back(site_line(row.id, query)); }
    return lines;
}

} // namespace ninferno::ops::detail::fusion
