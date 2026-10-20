#include "ninferno/ops/gated_rmsnorm.h"
#include "ninferno/ops/scatter.h"
#include "core/device.h"
#include "ops/norm_test_common.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;
using namespace ninferno::test::norm;

namespace {

constexpr ReductionCriterion gated_rmsnorm_bf16_criterion() {
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 4.5e-5,
            /*gross_relative_to_max_reference*/ 2.8e-3};
}

std::vector<double> gated_rmsnorm_oracle(const std::vector<float>& input,
                                         const std::vector<float>& weight,
                                         const std::vector<float>& gate, const Shape& shape) {
    std::vector<double> output(input.size());
    const auto row_count = static_cast<std::int64_t>(shape.rows) * shape.tokens;
    for (std::int64_t row = 0; row < row_count; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * shape.d;
        double sum_squares     = 0.0;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double value = input[base + column];
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares / static_cast<double>(shape.d) + kEps);
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double gate_value = gate[base + column];
            const double silu       = gate_value / (1.0 + std::exp(-gate_value));
            output[base + column]   = static_cast<double>(input[base + column]) * inverse *
                                    static_cast<double>(weight[column]) * silu;
        }
    }
    return output;
}

int run_case(DeviceExecutionView execution, const char* label, const Shape& shape,
             std::uint32_t seed, float input_scale = 4.0F, bool bf16x2_unaligned = false,
             bool graph = false) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d), gate(count);
    fill_uniform(input, seed, -input_scale, input_scale);
    fill_uniform(weight, seed + 1U, 0.25F, 1.75F);
    fill_uniform(gate, seed + 2U, -5.0F, 5.0F);
    gate.front() = 0.0F;
    gate.back()  = -0.0F;
    round_to_bf16(input);
    round_to_bf16(weight);
    round_to_bf16(gate);
    std::vector<double> reference = gated_rmsnorm_oracle(input, weight, gate, shape);

    DeviceInput device_input  = make_input(input, bf16x2_unaligned);
    DeviceInput device_weight = make_input(weight, bf16x2_unaligned);
    DeviceInput device_gate   = make_input(gate, bf16x2_unaligned);
    const std::size_t leading = bf16x2_unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + count * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor input_tensor = tensor_for(device_input.data, shape);
    Tensor weight_tensor(device_weight.data, DType::BF16, {shape.d});
    Tensor gate_tensor   = tensor_for(device_gate.data, shape);
    Tensor output_tensor = tensor_for(output_data, shape);
    ops::gated_rmsnorm(input_tensor, weight_tensor, gate_tensor, kEps, output_tensor, execution);
    cuda_synchronize();

    if (graph) {
        cudaStream_t stream;
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::gated_rmsnorm(input_tensor, weight_tensor, gate_tensor, kEps, output_tensor,
                           execution.on_stream(stream));
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (std::size_t i = 0; i < gate.size(); ++i) {
            gate[i]                                                   = -gate[i];
            device_gate.expected[i + leading / sizeof(std::uint16_t)] = f32_to_bf16(gate[i]);
        }
        CUDA_CHECK(cudaMemcpyAsync(device_gate.storage.p, device_gate.expected.data(),
                                   device_gate.storage.bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
        reference = gated_rmsnorm_oracle(input, weight, gate, shape);
    }

    int failures = verify_reduction(label, from_device_bf16(output_data, count), reference,
                                    gated_rmsnorm_bf16_criterion());
    failures += verify_output_storage(std::string(label) + " output", output, bf16x2_unaligned);
    failures += verify_preserved(std::string(label) + " preserves input", device_input);
    failures += verify_preserved(std::string(label) + " preserves weight", device_weight);
    failures += verify_preserved(std::string(label) + " preserves gate", device_gate);
    return failures;
}

// gated_rmsnorm_regrouped against gated_rmsnorm followed by regroup_bf16_heads: the same bytes.
int run_regrouped(DeviceExecutionView execution, const Shape& shape, std::int32_t groups,
                  std::uint32_t seed) {
    const std::string label = "gated_rmsnorm_regrouped [" + std::to_string(shape.d) + "," +
                              std::to_string(shape.rows) + "," + std::to_string(shape.tokens) +
                              "] groups " + std::to_string(groups);
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d), gate(count);
    fill_uniform(input, seed, -4.0F, 4.0F);
    fill_uniform(weight, seed + 1U, 0.25F, 1.75F);
    fill_uniform(gate, seed + 2U, -5.0F, 5.0F);
    round_to_bf16(input);
    round_to_bf16(weight);
    round_to_bf16(gate);
    DeviceInput device_input  = make_input(input, false);
    DeviceInput device_weight = make_input(weight, false);
    DeviceInput device_gate   = make_input(gate, false);
    GuardedDeviceBuffer normed(count * sizeof(std::uint16_t)), tiled(count * sizeof(std::uint16_t)),
        fused(count * sizeof(std::uint16_t));
    normed.fill(0xff);
    tiled.fill(0xff);
    fused.fill(0xff);
    // Three dimensions also for one token: the regroup reads the heads from ne[1].
    const auto as_3d = [&](void* data) {
        return Tensor(data, DType::BF16, {shape.d, shape.rows, shape.tokens});
    };
    const Tensor x = as_3d(device_input.data), z = as_3d(device_gate.data);
    const Tensor w(device_weight.data, DType::BF16, {shape.d});
    Tensor normed_tensor = as_3d(normed.data()), tiled_tensor = as_3d(tiled.data()),
           fused_tensor = as_3d(fused.data());
    ops::gated_rmsnorm(x, w, z, kEps, normed_tensor, execution);
    ops::regroup_bf16_heads(normed_tensor, groups, tiled_tensor, execution.stream);
    ops::gated_rmsnorm_regrouped(x, w, z, kEps, groups, fused_tensor, execution);
    cuda_synchronize();
    const auto want = from_device_bf16(tiled.data(), count);
    const auto got  = from_device_bf16(fused.data(), count);
    int failures    = 0;
    if (std::memcmp(want.data(), got.data(), count * sizeof(want[0])) != 0) {
        std::cerr << "FAIL " << label << ": differs from gated_rmsnorm + regroup_bf16_heads\n";
        ++failures;
    }
    failures += verify_output_storage(label + " output", fused, false);
    failures += verify_preserved(label + " preserves input", device_input);
    failures += verify_preserved(label + " preserves gate", device_gate);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    DeviceContext device;
    // Fixture initialization uses the default stream; keep the eager call ordered with it.
    const auto execution = device.execution_view().on_stream(nullptr);
    int failures         = 0;
    failures += run_case(execution, "gated_rmsnorm [128,48,1]", {128, 48}, 1401U);
    for (int columns : {2, 7, 16, 48, 56, 57, 64, 96, 128}) {
        const std::string label = "gated_rmsnorm target [128,48," + std::to_string(columns) + "]";
        failures += run_case(execution, label.c_str(), {128, 48, columns}, 1420U + columns, 4.0F,
                             false, columns == 56 || columns == 57 || columns == 128);
    }
    failures += run_case(execution, "gated_rmsnorm [128,32,7]", {128, 32, 7}, 1402U);
    failures += run_case(execution, "gated_rmsnorm [128,32,128]", {128, 32, 128}, 1403U);
    failures += run_case(execution, "gated_rmsnorm near-zero [128,32]", {128, 32}, 1404U, 1.0e-5F);
    failures +=
        run_case(execution, "gated_rmsnorm unaligned [128,48]", {128, 48}, 1405U, 4.0F, true);
    // Qwen3.5's linear-attention heads (value heads over key-head groups), every warp-route D,
    // decode, verify and prefill widths.
    for (const auto& [d, heads, groups] : std::vector<std::array<std::int32_t, 3>>{{128, 48, 16},
                                                                                   {128, 32, 16},
                                                                                   {128, 64, 16},
                                                                                   {64, 32, 8},
                                                                                   {192, 12, 4},
                                                                                   {256, 16, 8}}) {
        for (const std::int32_t tokens : {1, 2, 4, 7, 64, 513}) {
            failures += run_regrouped(execution, {d, heads, tokens}, groups,
                                      1500U + static_cast<std::uint32_t>(d + heads + tokens));
        }
    }
    try {
        failures += run_regrouped(execution, {512, 16, 2}, 8, 1600U);
        std::cerr << "FAIL gated_rmsnorm_regrouped D=512 must be refused\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    std::cout << (failures ? "FAIL" : "OK") << " gated_rmsnorm\n";
    return failures ? 1 : 0;
}
