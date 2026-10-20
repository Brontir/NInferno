// YaRN rotary frequencies (core/yarn.h): factor 1 is plain RoPE; a stretch keeps the fast pairs,
// divides the slow pairs by the factor and scales attention by 0.1 ln(factor) + 1. No GPU.

#include "core/yarn.h"

#include <cmath>
#include <cstdio>

namespace {
int check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); }
    return ok ? 0 : 1;
}

bool near(double a, double b) { return std::abs(a - b) <= 1e-6 * std::max(1.0, std::abs(b)); }
} // namespace

int main() {
    int failures = 0;
    constexpr double theta = 10000000.0;
    constexpr std::uint64_t dim = 64;

    const auto plain = ninferno::yarn_rope(theta, dim, {.factor = 1.0, .original = 262144.0});
    bool plain_ok    = plain.inv_freq.size() == dim / 2 && near(plain.attention_scaling, 1.0);
    for (std::uint64_t i = 0; i < dim; i += 2) {
        plain_ok = plain_ok && near(plain.inv_freq[i / 2],
                                    1.0 / std::pow(theta, static_cast<double>(i) / dim));
    }
    failures += check(plain_ok, "factor 1 must give the plain RoPE frequencies");

    const auto yarn = ninferno::yarn_rope(theta, dim, {.factor = 4.0, .original = 262144.0});
    failures += check(near(yarn.attention_scaling, 0.1 * std::log(4.0) + 1.0),
                      "the attention scaling must be 0.1 ln(factor) + 1");
    failures += check(near(yarn.inv_freq.front(), plain.inv_freq.front()),
                      "the fastest pair must keep its frequency");
    failures += check(near(yarn.inv_freq.back(), plain.inv_freq.back() / 4.0),
                      "the slowest pair must be stretched by the factor");
    bool between = true;
    for (std::size_t k = 0; k < yarn.inv_freq.size(); ++k) {
        between = between && yarn.inv_freq[k] <= plain.inv_freq[k] * (1 + 1e-6) &&
                  yarn.inv_freq[k] >= plain.inv_freq[k] / 4.0 * (1 - 1e-6);
    }
    failures += check(between, "every pair must lie between its original and stretched frequency");

    const auto given = ninferno::yarn_rope(theta, dim,
                                         {.factor = 4.0, .original = 262144.0, .attention_factor = 1.25});
    failures += check(near(given.attention_scaling, 1.25), "attention_factor must override mscale");

    if (failures == 0) { std::printf("yarn: all checks passed\n"); }
    return failures == 0 ? 0 : 1;
}
