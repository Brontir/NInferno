#include "core/weight.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "ninferno/ops/linear_swiglu.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/workspace_query.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninferno::test::linear_swiglu {
namespace {

// The criterion belongs to the activation-compute profile, not the weight storage format or a
// private materialized/fused implementation.
constexpr ReductionCriterion tolerance_for(ActivationCompute activation_compute) {
    switch (activation_compute) {
    case ActivationCompute::A16:
        return {3.3e-3, 5.0e-3, 6.3e-3};
    case ActivationCompute::A8:
        // Both independently A8-quantized projections feed the nonlinear product, so this profile
        // allows twice Linear's relative-L2 quantization allowance plus a bounded gross tail.
        return {8.0e-2, 1.0e-2, 1.2e-1};
    case ActivationCompute::A4:
        return {1.6e-1, 1.0e-2, 1.6e-1};
    }
    throw std::invalid_argument("linear_swiglu test: unknown activation compute profile");
}

bool cuda_available() { return !test::cuda_unavailable(); }

std::size_t checked_elements(std::int32_t first, std::int32_t second, const char* label) {
    if (first <= 0 || second <= 0) {
        throw std::invalid_argument(std::string("linear_swiglu test: invalid ") + label);
    }
    const std::size_t a = static_cast<std::size_t>(first);
    const std::size_t b = static_cast<std::size_t>(second);
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("linear_swiglu test: ") + label + " overflows");
    }
    return a * b;
}

std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::vector<std::uint16_t> make_activation(const Profile& profile, std::int32_t tokens) {
    std::vector<std::uint16_t> activation(
        checked_elements(profile.input_rows, tokens, "activation size"), 0);

    // Token zero is dense, so every logical K contributes to every production implementation.
    // Later tokens use exact zeros plus a rotating set of nonzeros. This keeps a full-output,
    // full-formula oracle practical at large registered T boundaries without adopting any
    // production staging or reduction behavior.
    const bool native_float_weight =
        profile.qtype == QType::NVFP4 || profile.qtype == QType::FP8_E4M3FN_ROW_BF16;
    const float dense_scale =
        profile.qtype == QType::Q4_G64_FP16 ? 1.25e-4F : (native_float_weight ? 1.0e-3F : 1.0e-5F);
    for (std::int32_t column = 0; column < profile.input_rows; ++column) {
        const std::uint64_t mixed = mix64((static_cast<std::uint64_t>(profile.seed) << 32) |
                                          static_cast<std::uint32_t>(column));
        int numerator             = static_cast<int>((mixed >> 48) % 63U) - 31;
        if (numerator == 0) { numerator = (column & 1) == 0 ? 1 : -1; }
        const float value                            = static_cast<float>(numerator) * dense_scale;
        activation[static_cast<std::size_t>(column)] = test::f32_to_bf16(value);
    }

    constexpr std::int32_t kNonzerosPerSparseToken = 4;
    const float sparse_scale =
        profile.qtype == QType::Q4_G64_FP16 ? 1.5e-2F : (native_float_weight ? 2.0e-2F : 1.5e-3F);
    for (std::int32_t token = 1; token < tokens; ++token) {
        for (std::int32_t lane = 0; lane < kNonzerosPerSparseToken; ++lane) {
            const std::uint64_t mixed =
                mix64((static_cast<std::uint64_t>(profile.seed + 1U) << 32) |
                      static_cast<std::uint32_t>(token * kNonzerosPerSparseToken + lane));
            const std::int32_t column =
                static_cast<std::int32_t>(mixed % static_cast<std::uint64_t>(profile.input_rows));
            int numerator = static_cast<int>((mixed >> 48) % 31U) - 15;
            if (numerator == 0) { numerator = (lane & 1) == 0 ? 1 : -1; }
            const float value = static_cast<float>(numerator) * sparse_scale;
            activation[static_cast<std::size_t>(token) * profile.input_rows + column] =
                test::f32_to_bf16(value);
        }
    }
    return activation;
}

struct ActiveValue {
    std::int32_t token;
    double value;
};

std::vector<std::vector<ActiveValue>>
index_nonzero_activations(const std::vector<std::uint16_t>& activation, std::int32_t input_rows,
                          std::int32_t tokens) {
    std::vector<std::vector<ActiveValue>> by_column(static_cast<std::size_t>(input_rows));
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < input_rows; ++column) {
            const std::uint16_t bits =
                activation[static_cast<std::size_t>(token) * input_rows + column];
            if ((bits & 0x7fffU) == 0U) { continue; }
            by_column[static_cast<std::size_t>(column)].push_back(
                {token, static_cast<double>(test::bf16_to_f32(bits))});
        }
    }
    return by_column;
}

// Round-to-nearest-even onto a sorted non-negative grid, saturating at its top: the host image of
// cvt.rn.satfinite to e2m1 and to e4m3.
float round_to_grid(float magnitude, const std::vector<float>& grid) {
    if (magnitude >= grid.back()) { return grid.back(); }
    const auto upper = std::lower_bound(grid.begin(), grid.end(), magnitude);
    if (*upper == magnitude || upper == grid.begin()) { return *upper; }
    const auto lower  = upper - 1;
    const float below = magnitude - *lower;
    const float above = *upper - magnitude;
    if (below != above) { return below < above ? *lower : *upper; }
    return ((lower - grid.begin()) % 2) == 0 ? *lower : *upper;
}

const std::vector<float>& e2m1_grid() {
    static const std::vector<float> grid{0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    return grid;
}

const std::vector<float>& e4m3_grid() {
    static const std::vector<float> grid = [] {
        std::vector<float> values;
        for (int exponent = 0; exponent < 16; ++exponent) {
            for (int mantissa = 0; mantissa < 8; ++mantissa) {
                if (exponent == 15 && mantissa == 7) { continue; } // NaN
                values.push_back(
                    exponent == 0
                        ? std::ldexp(static_cast<float>(mantissa) / 8.0F, -6)
                        : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7));
            }
        }
        return values;
    }();
    return grid;
}

// The activation as the NVFP4 W4A4 routes see it: every group of 16 quantized to e2m1 codes under
// an e4m3 block scale, as quantize_nvfp4_k16_bits does it, and decoded back. An oracle over these
// values and the logical weights differs from the kernel only by accumulation order and the
// epilogue's roundings, so it can hold an A4 route to an A16 bound.
std::vector<double> nvfp4_a4_activation(const std::vector<std::uint16_t>& activation,
                                        float input_scale_divisor) {
    std::vector<double> decoded(activation.size(), 0.0);
    for (std::size_t group = 0; group < activation.size(); group += 16) {
        float max_abs = 0.0F;
        for (std::size_t i = 0; i < 16; ++i) {
            max_abs = std::max(max_abs, std::fabs(test::bf16_to_f32(activation[group + i])));
        }
        const float scale = round_to_grid((input_scale_divisor * max_abs) / 6.0F, e4m3_grid());
        if (scale == 0.0F) { continue; }
        for (std::size_t i = 0; i < 16; ++i) {
            const float value = test::bf16_to_f32(activation[group + i]);
            const float code =
                round_to_grid(std::fabs((value * input_scale_divisor) / scale), e2m1_grid());
            decoded[group + i] =
                std::copysign(static_cast<double>(code), static_cast<double>(value)) *
                static_cast<double>(scale) / static_cast<double>(input_scale_divisor);
        }
    }
    return decoded;
}

// One token's silu(gate) * up and silu(up) * gate over an activation given as doubles.
void order_oracle_fp64(const Profile& profile, const quantized_weight::PackedWeight& weight,
                       const std::vector<double>& x, std::vector<double>& direct,
                       std::vector<double>& swapped);

double silu_fp64(double value) {
    if (value >= 0.0) { return value / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return value * exponential / (1.0 + exponential);
}

// `swapped`, when given, receives silu(up) * gate: the control for the [gate; up] row order the
// converter writes and the kernel reads.
std::vector<double> linear_swiglu_oracle_fp64(const Profile& profile,
                                              const quantized_weight::PackedWeight& weight,
                                              const std::vector<std::uint16_t>& activation,
                                              std::int32_t tokens,
                                              std::vector<double>* swapped = nullptr) {
    const auto active_by_column = index_nonzero_activations(activation, profile.input_rows, tokens);
    std::vector<double> output(checked_elements(profile.output_rows, tokens, "oracle output size"));
    if (swapped) { swapped->assign(output.size(), 0.0); }

    const unsigned hardware_threads = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t thread_count =
        std::min(profile.output_rows, static_cast<std::int32_t>(hardware_threads));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(thread_count));
    for (std::int32_t thread = 0; thread < thread_count; ++thread) {
        const std::int32_t row_begin = static_cast<std::int32_t>(
            static_cast<std::int64_t>(profile.output_rows) * thread / thread_count);
        const std::int32_t row_end = static_cast<std::int32_t>(
            static_cast<std::int64_t>(profile.output_rows) * (thread + 1) / thread_count);
        workers.emplace_back([&, row_begin, row_end] {
            std::vector<double> gate(static_cast<std::size_t>(tokens));
            std::vector<double> up(static_cast<std::size_t>(tokens));
            for (std::int32_t row = row_begin; row < row_end; ++row) {
                std::fill(gate.begin(), gate.end(), 0.0);
                std::fill(up.begin(), up.end(), 0.0);
                const std::int32_t up_row = profile.output_rows + row;
                for (std::int32_t column = 0; column < profile.input_rows; ++column) {
                    const double gate_weight =
                        quantized_weight::logical_weight_fp64(weight, row, column);
                    const double up_weight =
                        quantized_weight::logical_weight_fp64(weight, up_row, column);
                    for (const ActiveValue active :
                         active_by_column[static_cast<std::size_t>(column)]) {
                        gate[static_cast<std::size_t>(active.token)] += gate_weight * active.value;
                        up[static_cast<std::size_t>(active.token)] += up_weight * active.value;
                    }
                }
                for (std::int32_t token = 0; token < tokens; ++token) {
                    const double fused = silu_fp64(gate[static_cast<std::size_t>(token)]) *
                                         up[static_cast<std::size_t>(token)];
                    output[static_cast<std::size_t>(token) * profile.output_rows + row] = fused;
                    if (swapped) {
                        (*swapped)[static_cast<std::size_t>(token) * profile.output_rows + row] =
                            silu_fp64(up[static_cast<std::size_t>(token)]) *
                            gate[static_cast<std::size_t>(token)];
                    }
                }
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
    return output;
}

void order_oracle_fp64(const Profile& profile, const quantized_weight::PackedWeight& weight,
                       const std::vector<double>& x, std::vector<double>& direct,
                       std::vector<double>& swapped) {
    direct.assign(static_cast<std::size_t>(profile.output_rows), 0.0);
    swapped.assign(direct.size(), 0.0);
    const unsigned hardware_threads = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t thread_count =
        std::min(profile.output_rows, static_cast<std::int32_t>(hardware_threads));
    std::vector<std::thread> workers;
    for (std::int32_t thread = 0; thread < thread_count; ++thread) {
        const std::int32_t row_begin = static_cast<std::int32_t>(
            static_cast<std::int64_t>(profile.output_rows) * thread / thread_count);
        const std::int32_t row_end = static_cast<std::int32_t>(
            static_cast<std::int64_t>(profile.output_rows) * (thread + 1) / thread_count);
        workers.emplace_back([&, row_begin, row_end] {
            for (std::int32_t row = row_begin; row < row_end; ++row) {
                double gate = 0.0;
                double up   = 0.0;
                for (std::int32_t column = 0; column < profile.input_rows; ++column) {
                    const double value = x[static_cast<std::size_t>(column)];
                    gate += quantized_weight::logical_weight_fp64(weight, row, column) * value;
                    up += quantized_weight::logical_weight_fp64(weight, profile.output_rows + row,
                                                                column) *
                          value;
                }
                direct[static_cast<std::size_t>(row)]  = silu_fp64(gate) * up;
                swapped[static_cast<std::size_t>(row)] = silu_fp64(up) * gate;
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
}

std::vector<double> read_bf16_output(const test::GuardedDeviceBuffer& output,
                                     std::size_t elements) {
    std::vector<std::uint16_t> bits(elements);
    cuda_check(cudaMemcpy(bits.data(), output.data(), elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy LinearSwiGLU output");
    std::vector<double> values(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        values[index] = static_cast<double>(test::bf16_to_f32(bits[index]));
    }
    return values;
}

int compare_output(std::string_view label, const std::vector<double>& actual,
                   const double* reference, ActivationCompute activation_compute) {
    if (actual.empty() || reference == nullptr) {
        std::cerr << label << ": invalid numerical comparison\n";
        return 1;
    }

    return verify_reduction(label, actual, std::span<const double>(reference, actual.size()),
                            tolerance_for(activation_compute));
}

int verify_unchanged(std::string_view label, const test::GuardedDeviceBuffer& device,
                     const void* expected, std::size_t bytes) {
    if (device.bytes() != bytes) {
        std::cerr << label << ": read-only input size mismatch\n";
        return 1;
    }
    std::vector<std::uint8_t> actual(bytes);
    device.copy_to_host(actual.data(), actual.size());
    if (std::memcmp(actual.data(), expected, bytes) == 0) { return 0; }
    std::cerr << label << ": read-only input was modified\n";
    return 1;
}

void validate_profile(const Profile& profile) {
    const bool q4 = profile.qtype == QType::Q4_G64_FP16 && profile.gate_up_rows == 34816 &&
                    profile.input_rows == 5120 && profile.output_rows == 17408;
    const bool q8_companion = profile.qtype == QType::Q8_G32_FP16 &&
                              profile.gate_up_rows == 12288 && profile.input_rows == 2048 &&
                              profile.output_rows == 6144;
    const bool q8_dflash2 = profile.qtype == QType::Q8_G32_FP16 && profile.gate_up_rows == 34816 &&
                            profile.input_rows == 5120 && profile.output_rows == 17408;
    const bool nvfp4 = profile.qtype == QType::NVFP4 && profile.gate_up_rows == 34816 &&
                       profile.input_rows == 5120 && profile.output_rows == 17408;
    const bool fp8 = profile.qtype == QType::FP8_E4M3FN_ROW_BF16 && profile.gate_up_rows == 34816 &&
                     profile.input_rows == 5120 && profile.output_rows == 17408;
    if ((!q4 && !q8_companion && !q8_dflash2 && !nvfp4 && !fp8) ||
        profile.gate_up_rows != 2 * profile.output_rows) {
        throw std::invalid_argument("linear_swiglu test: profile is not registered");
    }
    if ((nvfp4 && profile.activation_compute != ActivationCompute::A16 &&
         profile.activation_compute != ActivationCompute::A4) ||
        (fp8 && profile.activation_compute != ActivationCompute::A16 &&
         profile.activation_compute != ActivationCompute::A8) ||
        (!nvfp4 && !fp8 && profile.activation_compute != ActivationCompute::A16)) {
        throw std::invalid_argument("linear_swiglu test: invalid activation-compute profile");
    }
}

} // namespace

int run_profile(std::string_view label, const Profile& profile,
                std::span<const std::int32_t> token_cases,
                std::span<const std::int32_t> graph_cases) {
    validate_profile(profile);
    if (token_cases.empty()) { throw std::invalid_argument("linear_swiglu test: no token cases"); }
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    for (std::size_t index = 0; index < token_cases.size(); ++index) {
        if (token_cases[index] <= 0 ||
            (index != 0 && token_cases[index] <= token_cases[index - 1])) {
            throw std::invalid_argument(
                "linear_swiglu test: token cases must be positive and increasing");
        }
    }
    const std::int32_t maximum_tokens = token_cases.back();
    for (int tokens : graph_cases)
        if (tokens <= 0 || tokens > maximum_tokens)
            throw std::invalid_argument("graph case outside oracle extent");

    quantized_weight::PatternedWeightOptions weight_options;
    if (profile.qtype == QType::NVFP4) {
        weight_options.weight_scale_divisor = 0.125F;
        weight_options.input_scale_divisor  = 3.5F;
    }
    quantized_weight::PackedWeight host_weight = quantized_weight::make_patterned_weight(
        profile.qtype, profile.gate_up_rows, profile.input_rows, profile.seed, weight_options);
    const std::vector<std::uint16_t> host_activation = make_activation(profile, maximum_tokens);
    std::vector<double> swapped_reference;
    const std::vector<double> reference = linear_swiglu_oracle_fp64(
        profile, host_weight, host_activation, maximum_tokens, &swapped_reference);
    std::vector<std::uint16_t> negative_activation;
    std::vector<double> negative_reference;
    std::optional<DeviceContext> graph_context;
    if (!graph_cases.empty()) {
        negative_activation = host_activation;
        for (auto& bits : negative_activation) bits ^= 0x8000;
        negative_reference =
            linear_swiglu_oracle_fp64(profile, host_weight, negative_activation, maximum_tokens);
        graph_context.emplace();
    }

    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    test::GuardedDeviceBuffer device_activation(host_activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(host_activation.data(),
                                     host_activation.size() * sizeof(std::uint16_t));

    const ops::LinearPolicy policy =
        profile.activation_compute == ActivationCompute::A4
            ? ops::LinearPolicy::AllowA4
            : (profile.activation_compute == ActivationCompute::A8 ? ops::LinearPolicy::AllowA8
                                                                   : ops::LinearPolicy::A16Only);
    const std::size_t workspace_bytes = ops::linear_swiglu_workspace_capacity_bytes(
        profile.qtype, profile.gate_up_rows, profile.input_rows, policy, 1, maximum_tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    int failures        = 0;
    const auto run_case = [&](int tokens, bool replay) {
        const auto elements = checked_elements(profile.output_rows, tokens, "output size");
        test::GuardedDeviceBuffer output(elements * sizeof(std::uint16_t));
        Tensor x(device_activation.data(), DType::BF16, {profile.input_rows, tokens});
        Tensor destination(output.data(), DType::BF16, {profile.output_rows, tokens});
        workspace.reset();
        workspace.reset_peak();
        const cudaStream_t stream = replay ? graph_context->stream : nullptr;
        const auto launch         = [&] {
            ops::linear_swiglu(x, weight, destination, policy, workspace, stream);
        };
        const std::string label_case =
            std::string(label) + " T=" + std::to_string(tokens) + (replay ? " graph" : " eager");
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        try {
            test::cuda_check(cudaDeviceSynchronize(), "finish fixture initialization");
            if (replay) {
                definition.capture(stream, launch);
                graph.instantiate(definition);
            }
            for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
                const auto& input_bits = phase ? negative_activation : host_activation;
                const auto& expected   = phase ? negative_reference : reference;
                if (replay)
                    device_activation.copy_from_host(input_bits.data(),
                                                     input_bits.size() * sizeof(std::uint16_t));
                output.fill(0xff);
                test::cuda_check(cudaDeviceSynchronize(), "finish replay input update");
                if (replay)
                    graph.launch(stream);
                else
                    launch();
                test::cuda_check(cudaStreamSynchronize(stream), "synchronize LinearSwiGLU");
                const auto exact = ops::linear_swiglu_workspace_capacity_bytes(
                    profile.qtype, profile.gate_up_rows, profile.input_rows, policy, tokens,
                    tokens);
                if (workspace.used() != 0 ||
                    !test::workspace_query_holds(
                        workspace.peak_used(), exact,
                        test::qx_query_covers_prefill_a8(profile.qtype, tokens))) {
                    std::cerr << label_case << ": exact workspace query/execution mismatch (used " << workspace.used()
                              << ", peak " << workspace.peak_used() << ", query " << exact << ")\n";
                    ++failures;
                }
                failures += output.verify_guards(label_case);
                const auto actual = read_bf16_output(output, elements);
                failures +=
                    compare_output(label_case, actual, expected.data(), profile.activation_compute);
                if (!replay) {
                    // How far silu(up) * gate is from this case's output. Reported only: with
                    // the fixture's small projections the two orders can agree to within a
                    // bound (the A4 bound is 0.16); the order case below must fail.
                    const ReductionStats control =
                        compute_reduction_stats(actual.data(), swapped_reference.data(),
                                                static_cast<std::int64_t>(actual.size()));
                    const bool passes =
                        reduction_passes(control, static_cast<std::int64_t>(actual.size()),
                                         tolerance_for(profile.activation_compute));
                    std::printf("  SWAP %s silu(up)*gate: rel_l2 vs output=%.4e %s\n",
                                label_case.c_str(), control.relative_l2,
                                passes ? "(within the bound: this case cannot see the order)"
                                       : "(outside the bound)");
                }
                if (replay)
                    failures += verify_unchanged(label_case + " input", device_activation,
                                                 input_bits.data(),
                                                 input_bits.size() * sizeof(std::uint16_t));
            }
        } catch (const std::exception& error) {
            std::cerr << label_case << ": unexpected exception: " << error.what() << '\n';
            ++failures;
        }
    };
    for (int tokens : token_cases) run_case(tokens, false);
    if (!graph_cases.empty()) {
        // Check eager inputs before graph fixtures overwrite them with the next represented input.
        failures += verify_unchanged(std::string(label) + " eager input", device_activation,
                                     host_activation.data(),
                                     host_activation.size() * sizeof(std::uint16_t));
    }
    for (int tokens : graph_cases) run_case(tokens, true);

    // The [gate; up] order case. The fixture's activations keep the projections small,
    // where silu(g)*u and silu(u)*g agree to the first order (silu(z) = z/2 + z^2/4 + ...), so a
    // wide bound (A4) cannot see a swapped plane. Here the dense token is rescaled so the gate
    // projections have an RMS of 2, where the two orders differ by far more than any profile's
    // bound; every token of the case is that token. The output must meet the bound against
    // silu(gate)*up and fail it against silu(up)*gate, at T=1, T=8 and, on the A4 profile, on
    // the W4A4 routes (T=8 fused, T=129 Linear-then-SiLU, T=256 TMA). A W4A4 route is held to
    // the A16 bound against an oracle over the activation it quantizes (nvfp4_a4_activation).
    {
        quantized_weight::PatternedWeightOptions order_options = weight_options;
        // The NVFP4 pattern repeats every 16 rows and each half has 17408 rows, a multiple of 16:
        // with one seed the gate and up halves are the same plane and no input can tell the two
        // orders apart. The main cases above keep that fixture; this one seeds the up half apart.
        if (profile.qtype == QType::NVFP4) { order_options.upper_half_seed = 5U; }
        const quantized_weight::PackedWeight order_host = quantized_weight::make_patterned_weight(
            profile.qtype, profile.gate_up_rows, profile.input_rows, profile.seed, order_options);
        test::GuardedDeviceBuffer order_weight_buffer(order_host.payload.size());
        order_weight_buffer.copy_from_host(order_host.payload.data(), order_host.payload.size());
        const Weight order_weight = order_host.device_weight(order_weight_buffer.data());

        std::vector<std::uint16_t> order_activation(host_activation.begin(),
                                                    host_activation.begin() + profile.input_rows);
        double square = 0.0;
        for (std::int32_t row = 0; row < profile.output_rows; row += 17) {
            double gate = 0.0;
            for (std::int32_t column = 0; column < profile.input_rows; ++column) {
                gate += quantized_weight::logical_weight_fp64(order_host, row, column) *
                        static_cast<double>(test::bf16_to_f32(order_activation[column]));
            }
            square += gate * gate;
        }
        const double rms = std::sqrt(square / static_cast<double>((profile.output_rows + 16) / 17));
        const float factor = static_cast<float>(2.0 / std::max(rms, 1e-30));
        for (auto& bits : order_activation) {
            bits = test::f32_to_bf16(test::bf16_to_f32(bits) * factor);
        }
        std::vector<double> plain(order_activation.size());
        for (std::size_t i = 0; i < plain.size(); ++i) {
            plain[i] = static_cast<double>(test::bf16_to_f32(order_activation[i]));
        }
        std::vector<double> plain_direct, plain_swapped, a4_direct, a4_swapped;
        order_oracle_fp64(profile, order_host, plain, plain_direct, plain_swapped);
        const bool a4 = profile.activation_compute == ActivationCompute::A4;
        if (a4) {
            order_oracle_fp64(
                profile, order_host,
                nvfp4_a4_activation(order_activation, weight_options.input_scale_divisor),
                a4_direct, a4_swapped);
        }

        std::vector<std::int32_t> order_tokens{1, 8};
        if (a4) { order_tokens.insert(order_tokens.end(), {129, 256}); }
        for (const std::int32_t tokens : order_tokens) {
            if (tokens > maximum_tokens) { continue; }
            const std::size_t elements = checked_elements(profile.output_rows, tokens, "order");
            std::vector<std::uint16_t> input;
            input.reserve(static_cast<std::size_t>(tokens) * order_activation.size());
            for (std::int32_t token = 0; token < tokens; ++token) {
                input.insert(input.end(), order_activation.begin(), order_activation.end());
            }
            test::GuardedDeviceBuffer order_input(input.size() * sizeof(std::uint16_t));
            order_input.copy_from_host(input.data(), input.size() * sizeof(std::uint16_t));
            test::GuardedDeviceBuffer order_output(elements * sizeof(std::uint16_t));
            Tensor x(order_input.data(), DType::BF16, {profile.input_rows, tokens});
            Tensor destination(order_output.data(), DType::BF16, {profile.output_rows, tokens});
            workspace.reset();
            ops::linear_swiglu(x, order_weight, destination, policy, workspace, nullptr);
            test::cuda_check(cudaDeviceSynchronize(), "synchronize LinearSwiGLU order case");
            const auto actual = read_bf16_output(order_output, elements);

            // The NVFP4 A4 profile quantizes the activation from T=9 on: T = 1 runs the fused
            // decode route and speculative verify (2..8 tokens) the exact one, both A16.
            const bool quantized    = a4 && tokens >= 9;
            const auto& one_direct  = quantized ? a4_direct : plain_direct;
            const auto& one_swapped = quantized ? a4_swapped : plain_swapped;
            std::vector<double> direct_reference, swapped_reference_t;
            for (std::int32_t token = 0; token < tokens; ++token) {
                direct_reference.insert(direct_reference.end(), one_direct.begin(),
                                        one_direct.end());
                swapped_reference_t.insert(swapped_reference_t.end(), one_swapped.begin(),
                                           one_swapped.end());
            }
            const ActivationCompute bound =
                quantized ? ActivationCompute::A16 : profile.activation_compute;
            const std::string tag = std::string(label) + " order case T=" + std::to_string(tokens) +
                                    " (gate RMS " + std::to_string(rms * factor) +
                                    (quantized ? ", oracle over the A4-quantized input" : "") + ")";
            failures += compare_output(tag, actual, direct_reference.data(), bound);
            const ReductionStats control = compute_reduction_stats(
                actual.data(), swapped_reference_t.data(), static_cast<std::int64_t>(elements));
            const ReductionStats direct = compute_reduction_stats(
                actual.data(), direct_reference.data(), static_cast<std::int64_t>(elements));
            const bool passes = reduction_passes(control, static_cast<std::int64_t>(elements),
                                                 tolerance_for(bound));
            std::printf("  ORDER %s: rel_l2 vs silu(gate)*up=%.4e, vs silu(up)*gate=%.4e %s\n",
                        tag.c_str(), direct.relative_l2, control.relative_l2,
                        passes ? "CONTROL PASSED - the bound cannot see a swapped plane"
                               : "(control fails as it must)");
            if (quantized) {
                // How far the oracle over the unquantized input is: this input's A4 error.
                const ReductionStats unquantized =
                    compute_reduction_stats(actual.data(), plain_direct.data(),
                                            static_cast<std::int64_t>(profile.output_rows));
                std::printf("        token 0 vs the oracle over the unquantized input: rel_l2 "
                            "%.4e, max abs %.4e\n",
                            unquantized.relative_l2, unquantized.maximum_absolute_error);
            }
            failures += passes ? 1 : 0;
            failures += order_output.verify_guards(tag);
        }
    }
    const auto& final_input = graph_cases.empty() ? host_activation : negative_activation;
    failures += device_activation.verify_guards(std::string(label) + " activation");
    failures += device_weight.verify_guards(std::string(label) + " weight");
    failures += verify_unchanged(std::string(label) + " activation", device_activation,
                                 final_input.data(), final_input.size() * sizeof(std::uint16_t));
    failures += verify_unchanged(std::string(label) + " weight", device_weight,
                                 host_weight.payload.data(), host_weight.payload.size());
    return failures;
}

} // namespace ninferno::test::linear_swiglu
