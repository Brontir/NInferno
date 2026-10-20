#pragma once

// YaRN rotary frequencies (transformers' _compute_yarn_parameters, in binary32 as it computes
// them): each pair's frequency blends the original one (extrapolation, the fast pairs) and the
// one stretched by `factor` (interpolation, the slow pairs) along a ramp between the pairs that
// turn beta_fast and beta_slow times over the original context, and the attention logits gain
// mscale on cos and sin. Shared by the converter's dense rope groups and Qwen 3.5's
// --rope-yarn-factor.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <vector>

namespace ninferno {

struct YarnParameters {
    double factor   = 0; // the context stretch; > 0
    double original = 0; // original_max_position_embeddings; > 0
    std::optional<double> attention_factor;
    std::optional<double> mscale;         // with mscale_all_dim: DeepSeek's ratio form
    std::optional<double> mscale_all_dim;
    double beta_fast = 32;
    double beta_slow = 1;
    bool truncate    = true;
};

struct YarnRope {
    std::vector<float> inv_freq; // [dim / 2]
    double attention_scaling = 1.0;
};

inline YarnRope yarn_rope(double theta, std::uint64_t dim, const YarnParameters& p) {
    YarnRope out;
    const double factor   = p.factor;
    const double original = p.original;
    const auto mscale_of  = [](double scale, double m) {
        return scale <= 1 ? 1.0 : 0.1 * m * std::log(scale) + 1.0;
    };
    if (p.attention_factor) {
        out.attention_scaling = *p.attention_factor;
    } else {
        out.attention_scaling = p.mscale && p.mscale_all_dim
                                    ? mscale_of(factor, *p.mscale) /
                                          mscale_of(factor, *p.mscale_all_dim)
                                    : mscale_of(factor, 1.0);
    }
    const auto correction = [&](double rotations) {
        return (static_cast<double>(dim) *
                std::log(original / (rotations * 2 * std::numbers::pi))) /
               (2 * std::log(theta));
    };
    double low  = correction(p.beta_fast);
    double high = correction(p.beta_slow);
    if (p.truncate) {
        low  = std::floor(low);
        high = std::ceil(high);
    }
    low  = std::max(low, 0.0);
    high = std::min(high, static_cast<double>(dim - 1));
    if (low == high) { high += 0.001; }
    const auto low_f  = static_cast<float>(low);
    const auto high_f = static_cast<float>(high);
    const auto fac    = static_cast<float>(factor);
    for (std::uint64_t i = 0; i < dim; i += 2) {
        const float exponent      = static_cast<float>(i) / static_cast<float>(dim);
        const float pos_freq      = std::pow(static_cast<float>(theta), exponent);
        const float extrapolation = 1.0F / pos_freq;
        const float interpolation = 1.0F / (fac * pos_freq);
        const float ramp =
            std::clamp((static_cast<float>(i / 2) - low_f) / (high_f - low_f), 0.0F, 1.0F);
        const float extrapolation_factor = 1.0F - ramp;
        out.inv_freq.push_back(interpolation * (1.0F - extrapolation_factor) +
                               extrapolation * extrapolation_factor);
    }
    return out;
}

} // namespace ninferno
