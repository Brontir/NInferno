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
//   band      the call-width band alone is evaluated; the branch names what the band allows and the
//             rest of the rule (a weight's format, a workspace) is the launch's;
//   declared  nothing is evaluated at plan time ("per_call"): the registry's rule applies.
//
// Not wired into the engine's startup manifest: that would add a section to every manifest. The
// engine adds `sites` when this lands in a manifest PR; until then the function is the contract and
// the test pins the lines. The regret format that scores each decided branch against its
// alternative is tools/bench/site_regret.py.

#include "core/device_class.h"
#include "ops/common/fusion_site_table.h"
#include "ops/generic/generic_gemm.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno::ops::detail::fusion {

struct SiteQuery {
    const DeviceClass* device = nullptr;
    std::int32_t tokens = 1; // the call's width (a decode or verify width, or a prefill chunk)
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

inline SiteLine declared(std::string_view id) {
    return {std::string(id), "declared", "per_call", {}};
}

// The sites an emitter evaluates; every other row of the table is `declared`.
inline bool emit_evaluated(std::string_view id, const SiteQuery& q, SiteLine& out) {
    const std::string tokens = std::to_string(q.tokens);
    if (id == "pdl_overlap") {
        const bool pdl = q.device != nullptr && q.device->features.pdl;
        out            = {std::string(id),
                          "decided",
               pdl ? "overlap" : "stream_order",
                          {{"pdl", pdl ? "1" : "0"}}};
        return true;
    }
    if (id == "vec_gated_pair" || id == "mmq_gated_pair") {
        const bool vec    = q.tokens <= generic::kGgmlVecMaxTokens;
        const bool is_vec = id == "vec_gated_pair";
        out               = {
            std::string(id),
            "band",
            is_vec ? (vec ? "tried" : "not_tried") : (vec ? "after_vec" : "first"),
            {{"tokens", tokens}, {"vec_max_tokens", std::to_string(generic::kGgmlVecMaxTokens)}}};
        return true;
    }
    if (id == "r8_prequantized") {
        const bool in_band = q.tokens >= 2 && q.tokens <= generic::kR8VerifyTokens;
        out                = {
            std::string(id),
            "band",
            in_band ? "prequantized_if_fits" : "in_kernel",
            {{"tokens", tokens}, {"verify_max_tokens", std::to_string(generic::kR8VerifyTokens)}}};
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
