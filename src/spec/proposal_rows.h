#pragma once

// The rows of the target's output head a drafter's draft head reads, by speculation mode ("mtp",
// "dflash"): a converted Qwen target's proposal head (artifact component "text", proposal.rows;
// proposal.drafter_rows a drafter's own leading rows, MTP's since cache revision 8) serves its MTP
// and DFlash drafters with a restricted head (product model_draft_head). A mode not named reads the
// full head (DFlash2 keeps the whole domain; a target without a proposal head). Round cost prices a
// draft head at these rows (spec/round_cost.h).

#include <cstdint>
#include <map>
#include <string>
#include <string_view>

namespace ninferno::spec {

struct ProposalRows {
    std::map<std::string, std::uint64_t, std::less<>> rows;

    // The rows `mode` reads; 0: the full head.
    [[nodiscard]] std::uint64_t of(std::string_view mode) const {
        const auto it = rows.find(mode);
        return it == rows.end() ? 0 : it->second;
    }
    [[nodiscard]] bool empty() const noexcept { return rows.empty(); }

    friend bool operator==(const ProposalRows&, const ProposalRows&) = default;
};

} // namespace ninferno::spec
