#include "spec/draft_head.h"

#include <algorithm>
#include <functional>
#include <vector>

namespace ninferno::spec {

DraftHeadRows draft_head_rows(std::span<const std::int64_t> counts, double share,
                              std::int32_t depth) {
    DraftHeadRows best;
    const auto vocab = static_cast<std::uint64_t>(counts.size());
    if (vocab <= kDraftHeadRowStep || depth < 1 || !(share > 0.0 && share < 1.0)) { return best; }
    std::vector<std::int64_t> sorted(counts.begin(), counts.end());
    for (auto& c : sorted) { c = std::max<std::int64_t>(c, 0); }
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    double total = 0;
    for (const auto c : sorted) { total += static_cast<double>(c); }
    if (total <= 0) { return best; }
    double kept     = 0;
    std::uint64_t i = 0;
    for (std::uint64_t rows = kDraftHeadRowStep; rows < vocab; rows += kDraftHeadRowStep) {
        for (; i < rows; ++i) { kept += static_cast<double>(sorted[i]); }
        const double c = kept / total;
        double tokens  = 0; // sum_{j=0..k} c^j
        double power   = 1;
        for (std::int32_t j = 0; j <= depth; ++j) {
            tokens += power;
            power *= c;
        }
        tokens /= static_cast<double>(depth + 1);
        const double cost =
            1.0 - (1.0 - static_cast<double>(rows) / static_cast<double>(vocab)) * share;
        const double gain = tokens / cost - 1.0;
        if (gain > best.gain) { best = {rows, c, gain}; }
    }
    return best;
}

} // namespace ninferno::spec
