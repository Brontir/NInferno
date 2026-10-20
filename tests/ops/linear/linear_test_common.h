#pragma once

#include "ninferno/ops/linear.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninferno::test::linear {

enum class ActivationCompute : std::uint8_t {
    A16,
    A8,
    A4,
};

enum class CallForm : std::uint8_t {
    Policy,
    A16Convenience,
};

enum class Comparison : std::uint8_t {
    Full,
    Sampled,
};

struct Invocation {
    std::int32_t t;
    CallForm call_form       = CallForm::Policy;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    bool graph_replay        = false;
};

struct ShapeCase {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    Comparison comparison;
    bool verify_input_preservation;
    std::span<const Invocation> invocations;
};

using WeightGenerator = quantized_weight::PackedWeight (*)(std::int32_t, std::int32_t,
                                                           std::uint32_t);

quantized_weight::PackedWeight make_q4_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed);
quantized_weight::PackedWeight make_q5_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed);
quantized_weight::PackedWeight make_q6_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed);
quantized_weight::PackedWeight make_q8_g32_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed);
quantized_weight::PackedWeight make_nvfp4_weight(std::int32_t n, std::int32_t k,
                                                 std::uint32_t seed);
quantized_weight::PackedWeight make_fp8_weight(std::int32_t n, std::int32_t k, std::uint32_t seed);

// Whether this build runs the weights `generator` makes: every format does, through the generic
// path where no native kernel is built.
inline bool generator_runs(WeightGenerator, std::string_view) { return true; }

// As above, under `policy`: AllowA8 over FP8 weights is the E4M3 mma.sync route (sm_89 or later).
// Without it the call takes the A16 route, which a check of the A8 route does not describe.
inline bool generator_runs(WeightGenerator generator, ops::LinearPolicy policy,
                           std::string_view label) {
    if (!generator_runs(generator, label)) { return false; }
    if (generator != make_fp8_weight || policy != ops::LinearPolicy::AllowA8 ||
        fp8_mma_available()) {
        return true;
    }
    std::cout << "SKIP " << label << ": FP8 A8 needs the E4M3 mma.sync of sm_89 or later\n";
    return false;
}

void cpu_linear_gemm_fp64(const float* weight, const float* activation, double* output,
                          std::int32_t n, std::int32_t k, std::int32_t t);

bool cuda_available();

int run_shape(std::string_view label, ActivationCompute activation_compute,
              WeightGenerator generator, const ShapeCase& shape);

// A declared token interval must reserve enough space for every public point it contains.
int verify_workspace_envelopes(QType qtype, std::int32_t n, std::int32_t k);

// The named criterion run_shape applies to an activation-compute path, for tests that build their
// own cases (control arms, isolated K segments) and must judge them by the same rule.
ReductionCriterion linear_criterion(ActivationCompute activation_compute);

// The activation run_shape feeds a path: token-major BF16 bits, T rows of K values.
std::vector<std::uint16_t> make_linear_activation(std::int32_t k, std::int32_t t,
                                                  std::uint32_t seed,
                                                  ActivationCompute activation_compute);

} // namespace ninferno::test::linear
