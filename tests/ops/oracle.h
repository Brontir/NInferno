#pragma once

// FP64 reference pieces shared by Op tests. Each is the plain mathematical definition, evaluated in
// double from the represented inputs; tests compare Op outputs against them.

#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

namespace ninferno::test {

// gelu with the tanh approximation: 0.5 z (1 + tanh(sqrt(2/pi) (z + 0.044715 z^3))).
inline double gelu_tanh_oracle(double z) {
    const double inner = std::sqrt(2.0 / std::numbers::pi) * (z + 0.044715 * z * z * z);
    return 0.5 * z * (1.0 + std::tanh(inner));
}

// Uniform values in [low, high), each rounded to a representable BF16.
inline std::vector<float> random_bf16_values(std::size_t count, std::uint32_t seed, float low,
                                             float high) {
    std::vector<float> values(count);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& v : values) { v = bf16_to_f32(f32_to_bf16(distribution(generator))); }
    return values;
}

inline double max_abs_difference(const std::vector<double>& a, const std::vector<double>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) { worst = std::max(worst, std::abs(a[i] - b[i])); }
    return worst;
}

// x / sqrt(mean(x^2) + eps), times weight when one is given.
inline std::vector<double> rms_normalize(const float* x, int width, double eps,
                                         const float* weight = nullptr) {
    double squares = 0.0;
    for (int i = 0; i < width; ++i) { squares += double(x[i]) * x[i]; }
    const double inverse = 1.0 / std::sqrt(squares / width + eps);
    std::vector<double> out(width);
    for (int i = 0; i < width; ++i) {
        out[i] = x[i] * inverse * (weight != nullptr ? double(weight[i]) : 1.0);
    }
    return out;
}

} // namespace ninferno::test
