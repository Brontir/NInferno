#pragma once

// The sampling flags ninferno and ninferno-serve share, with one set of ranges.

#include "ninferno/types.h"
#include "product/numeric_options.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::product {

// When `arg` is a sampling flag, parses its value (value() reads it) into `fields` and returns
// true. `fields` is a SamplingConfig (ninferno) or ninferno-serve's optional overrides.
template <class Fields, class Value>
bool parse_sampling_flag(std::string_view arg, Value&& value, Fields& fields) {
    if (arg == "--temperature") {
        fields.temperature =
            parse_float_in(value(), "temperature", kTemperatureRange.low, kTemperatureRange.high);
    } else if (arg == "--top-p") {
        fields.top_p =
            parse_float_in(value(), "top-p", kProbabilityRange.low, kProbabilityRange.high);
    } else if (arg == "--top-k") {
        const int top_k = parse_nonnegative_int(value(), "top-k");
        if (top_k > kMaximumTopK) {
            const std::string limit = std::to_string(kMaximumTopK);
            throw std::invalid_argument("--top-k must be in [0," + limit +
                                        "]: the sampler keeps at most " + limit +
                                        " candidates (0 takes all of them)");
        }
        fields.top_k = static_cast<std::int32_t>(top_k);
    } else if (arg == "--min-p") {
        fields.min_p =
            parse_float_in(value(), "min-p", kProbabilityRange.low, kProbabilityRange.high);
    } else if (arg == "--presence-penalty") {
        fields.presence_penalty =
            parse_float_in(value(), "presence-penalty", kPenaltyRange.low, kPenaltyRange.high);
    } else if (arg == "--frequency-penalty") {
        fields.frequency_penalty =
            parse_float_in(value(), "frequency-penalty", kPenaltyRange.low, kPenaltyRange.high);
    } else if (arg == "--seed") {
        fields.seed = parse_u64(value(), "seed");
    } else {
        return false;
    }
    return true;
}

} // namespace ninferno::product
