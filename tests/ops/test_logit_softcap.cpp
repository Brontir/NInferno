#include "core/device.h"
#include "ninferno/ops/logit_softcap.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

// The output lives in (-cap, cap), so an absolute floor scaled by the cap carries the BF16
// rounding of the result, and the relative term carries the interior where the transform is
// nearly the identity. BF16 has eight significand bits: 2^-8 = 3.9e-3.
constexpr double kRelative = 4.0e-3;

std::vector<float> make_input(std::size_t elements, std::uint32_t seed, float spread) {
    std::vector<float> input(elements);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-spread, spread);
    for (float& value : input) { value = bf16_to_f32(f32_to_bf16(distribution(generator))); }
    return input;
}

std::vector<std::uint16_t> to_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(),
                   [](float value) { return f32_to_bf16(value); });
    return bits;
}

// The sole oracle: the documented formula in FP64 from the represented input. It reproduces no
// reciprocal, no identity rewrite and no output rounding.
std::vector<double> oracle(const std::vector<float>& input, double cap, double scale = 1.0) {
    std::vector<double> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = cap * std::tanh(scale * static_cast<double>(input[index]) / cap);
    }
    return output;
}

int run_case(const std::string& label, std::size_t elements, float cap, float spread,
             std::uint32_t seed, float scale = 1.0F) {
    const auto input    = make_input(elements, seed, spread);
    const auto bits     = to_bits(input);
    const auto expected = oracle(input, static_cast<double>(cap), static_cast<double>(scale));

    GuardedDeviceBuffer device(bits.size() * sizeof(std::uint16_t));
    device.copy_from_host(bits.data(), device.bytes());
    Tensor tensor(device.data(), DType::BF16, {static_cast<std::int32_t>(elements)});
    ops::logit_softcap(tensor, cap, nullptr, scale);
    cuda_synchronize();

    const auto got = from_device<std::uint16_t>(device.data(), bits.size());
    std::vector<double> actual(got.size());
    std::transform(got.begin(), got.end(), actual.begin(),
                   [](std::uint16_t word) { return static_cast<double>(bf16_to_f32(word)); });

    const PointwiseCriterion criterion{static_cast<double>(cap) * kRelative, kRelative};
    const auto stats = compute_pointwise_stats(actual.data(), expected.data(),
                                               static_cast<std::int64_t>(actual.size()), criterion);
    report_pointwise_stats(label, static_cast<std::int64_t>(actual.size()), stats, criterion);
    int failures = 0;
    if (!pointwise_passes(stats, static_cast<std::int64_t>(actual.size()))) {
        std::cerr << label << ": pointwise criterion violated\n";
        ++failures;
    }
    // The transform is a bound, not just an approximation: nothing may leave the interval, and
    // saturation must land on the cap rather than overflow.
    for (double value : actual) {
        if (!std::isfinite(value) || std::abs(value) > static_cast<double>(cap)) {
            std::cerr << label << ": output " << value << " escaped the cap " << cap << '\n';
            ++failures;
            break;
        }
    }
    failures += device.verify_guards(label + " guards");
    return failures;
}

// Two arms that must fail, because "output is inside the cap" is satisfied by a kernel that
// writes zeros, and "output equals input" is satisfied by a kernel that does nothing.
int run_controls() {
    constexpr std::size_t kElements = 4096;
    constexpr float kCap            = 30.0F;
    const auto input                = make_input(kElements, 0x77U, 120.0F);
    const auto expected             = oracle(input, kCap);

    double worst_identity = 0.0;
    double worst_zero     = 0.0;
    for (std::size_t index = 0; index < input.size(); ++index) {
        worst_identity = std::max(worst_identity,
                                  std::abs(expected[index] - static_cast<double>(input[index])));
        worst_zero     = std::max(worst_zero, std::abs(expected[index]));
    }
    std::cout << "control: softcap vs identity max_abs=" << worst_identity
              << ", vs zeros max_abs=" << worst_zero << '\n';
    int failures = 0;
    if (worst_identity < 1.0) {
        std::cerr << "control: this input never leaves the near-linear region, so the suite "
                     "cannot tell the transform from the identity\n";
        ++failures;
    }
    if (worst_zero < 1.0) {
        std::cerr << "control: this input is indistinguishable from zeros\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    // Gemma 4 caps at 30 over a 262144-row vocabulary; one token's logits is the shipped shape.
    failures += run_case("gemma4 vocab, interior", 262'144, 30.0F, 12.0F, 0x1234U);
    // Spread well past the cap, so most of the tensor is in saturation rather than in the
    // near-linear region where any monotone function would look right.
    failures += run_case("gemma4 vocab, saturating", 262'144, 30.0F, 300.0F, 0x2345U);
    // An odd count exercises the scalar tail of the paired kernel.
    failures += run_case("odd tail", 262'145, 30.0F, 90.0F, 0x3456U);
    // A single element takes the scalar path outright.
    failures += run_case("single element", 1, 30.0F, 90.0F, 0x4567U);
    // A different cap, so nothing can be tuned to 30.
    failures += run_case("cap 7.5", 8192, 7.5F, 60.0F, 0x5678U);
    // A logit multiplier before the cap (Muse Glimmer: 1/sqrt(6656/256) under a cap of 20), odd
    // count for the paired kernel's tail, spread into saturation after the multiplier.
    failures += run_case("scale 0.196, cap 20", 202'049, 20.0F, 300.0F, 0x6789U,
                         0.19611613513818404F);
    failures += run_case("scale 3, cap 30", 4096, 30.0F, 40.0F, 0x789AU, 3.0F);
    failures += run_controls();

    if (failures != 0) {
        std::cerr << "logit_softcap: " << failures << " checks failed\n";
        return 1;
    }
    std::cout << "logit_softcap: all checks passed\n";
    return 0;
}
