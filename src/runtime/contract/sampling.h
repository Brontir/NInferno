#pragma once

#include "ninferno/types.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::runtime {

// "<field> must be in [low,high]".
[[nodiscard]] inline std::string range_message(std::string_view field, SamplingRange range) {
    std::ostringstream text;
    text << field << " must be in [" << range.low << "," << range.high << "]";
    return text.str();
}

// Throws std::invalid_argument unless every field is finite and in its range, top_k in
// [1, kMaximumTopK]: what resolve_sampling returns, checked again where a Program plans a
// request (header-only, so the model layer checks without linking the Engine).
inline void check_resolved_sampling(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    const auto check = [](float value, SamplingRange range, std::string_view field) {
        if (!range.contains(value)) { throw std::invalid_argument(range_message(field, range)); }
    };
    check(sampling.temperature, kTemperatureRange, "temperature");
    check(sampling.top_p, kProbabilityRange, "top_p");
    check(sampling.min_p, kProbabilityRange, "min_p");
    check(sampling.presence_penalty, kPenaltyRange, "presence_penalty");
    check(sampling.frequency_penalty, kPenaltyRange, "frequency_penalty");
    if (sampling.top_k < 1 || sampling.top_k > kMaximumTopK) {
        throw std::invalid_argument("resolved top_k must be in [1," + std::to_string(kMaximumTopK) +
                                    "]");
    }
}

// Resolves one request at the Engine boundary. The registered preset supplies every omitted
// model-owned field; an omitted seed remains deterministic for direct Engine callers.
[[nodiscard]] ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                                          SamplingMode mode,
                                                          const SamplingOverrides& overrides);

} // namespace ninferno::runtime
