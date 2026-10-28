#include "core/weight.h"
#include "ninferno/ops/gdn_gating_proj.h"
#include "ninferno/ops/segmented_projection.h"
#include "ninferno/ops/weight_input.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "core/decode_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <variant>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

struct Geometry {
    const char* label;
    std::int32_t hidden;
    std::int32_t heads;
    bool parent_weight;
};

constexpr Geometry kQwen27{"qwen3_6_27b", 5120, 48, false};
constexpr Geometry kQwen38Parent{"qwen3_8_27b_parent", 5120, 48, true};
constexpr Geometry kQwen35{"qwen3_6_35b_a3b", 2048, 32, true};
// Published quantized a/b (qwen3_5_published): separate parents of their own. The 9B's geometry
// has no fused registration; the 27B's and 35B's do, but only for BF16.
constexpr Geometry kQwen27Published{"qwen3_6_27b", 5120, 48, false};
constexpr Geometry kQwen35Published{"qwen3_6_35b_a3b", 2048, 32, false};
constexpr Geometry kQwen9Published{"qwen3_5_9b", 4096, 32, false};

constexpr ReductionCriterion kGdnProjectionFp32{/*relative_l2=*/1.4e-6,
                                                /*gross_absolute=*/5.0e-7,
                                                /*gross_relative_to_max_reference=*/2.5e-6};
constexpr ReductionCriterion kGdnNormOutputBf16{/*relative_l2=*/1.75e-3,
                                                /*gross_absolute=*/1.0e-4,
                                                /*gross_relative_to_max_reference=*/4.0e-3};
// FP32 public controls permit 16-bit private operands/materialization. The gross cap includes
// propagated BF16 staging error; relative L2 remains the accuracy gate for the full formula.
constexpr ReductionCriterion kGdnNormControlFp32{/*relative_l2=*/8.0e-4,
                                                 /*gross_absolute=*/1.5e-4,
                                                 /*gross_relative_to_max_reference=*/2.0e-3};

// The generic path quantizes h for the kept matrices' activation widths: the linear tests'
// allowances for A4 (NVFP4 W4A4) and A8 (FP8 rows), against an FP64 oracle over the decoded words.
constexpr ReductionCriterion kPublishedA4Control{/*relative_l2=*/0.16,
                                                 /*gross_absolute=*/1.0 / 256.0,
                                                 /*gross_relative_to_max_reference=*/0.16};
constexpr ReductionCriterion kPublishedA8Control{/*relative_l2=*/0.04,
                                                 /*gross_absolute=*/1.0 / 256.0,
                                                 /*gross_relative_to_max_reference=*/0.06};

double softplus(double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double e = std::exp(value);
    return e / (1.0 + e);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> read_fp32(const void* device, std::size_t elements) {
    const std::vector<float> values = from_device<float>(device, elements);
    return {values.begin(), values.end()};
}

int verify_normwise(const std::string& label, const std::vector<double>& actual,
                    const std::vector<double>& expected, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), actual, expected, criterion);
}

Weight bf16_weight(void* data, std::int32_t rows, std::int32_t hidden) {
    Weight weight{};
    weight.qtype           = QType::BF16;
    weight.layout          = QuantLayout::Contiguous;
    weight.payload         = data;
    weight.payload_bytes   = static_cast<std::uint64_t>(rows) * hidden * sizeof(std::uint16_t);
    weight.qdata           = data;
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = hidden;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = hidden;
    weight.n               = rows;
    weight.k               = hidden;
    return weight;
}

std::vector<std::int32_t> oracle_tokens(std::int32_t tokens) {
    std::vector<std::int32_t> selected;
    if (tokens <= 128) {
        selected.reserve(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) { selected.push_back(token); }
    } else {
        selected = {0, tokens / 2, tokens - 1};
    }
    return selected;
}

void projection_oracle(const Geometry& geometry, const std::vector<float>& x,
                       const std::vector<float>& a_weight, const std::vector<float>& b_weight,
                       const std::vector<float>& a_log, const std::vector<float>& dt_bias,
                       const std::vector<std::int32_t>& selected_tokens, std::vector<double>& g,
                       std::vector<double>& beta) {
    const std::size_t output_elements =
        static_cast<std::size_t>(geometry.heads) * selected_tokens.size();
    g.resize(output_elements);
    beta.resize(output_elements);

    for (std::size_t sample = 0; sample < selected_tokens.size(); ++sample) {
        const std::size_t x_base =
            static_cast<std::size_t>(selected_tokens[sample]) * geometry.hidden;
        for (std::int32_t head = 0; head < geometry.heads; ++head) {
            const std::size_t weight_base = static_cast<std::size_t>(head) * geometry.hidden;
            double projected_a            = 0.0;
            double projected_b            = 0.0;
            for (std::int32_t k = 0; k < geometry.hidden; ++k) {
                const double value = static_cast<double>(x[x_base + static_cast<std::size_t>(k)]);
                projected_a +=
                    static_cast<double>(a_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
                projected_b +=
                    static_cast<double>(b_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
            }
            const std::size_t output = sample * geometry.heads + head;
            g[output]                = -std::exp(static_cast<double>(a_log[head])) *
                        softplus(projected_a + static_cast<double>(dt_bias[head]));
            beta[output] = sigmoid(projected_b);
        }
    }
}

void norm_projection_oracle(const Geometry& geometry, const std::vector<float>& x,
                            const std::vector<float>& norm_weight,
                            const std::vector<float>& a_weight, const std::vector<float>& b_weight,
                            const std::vector<float>& a_log, const std::vector<float>& dt_bias,
                            std::int32_t tokens, double eps, std::vector<double>& h,
                            std::vector<double>& g, std::vector<double>& beta) {
    h.resize(static_cast<std::size_t>(geometry.hidden) * tokens);
    g.resize(static_cast<std::size_t>(geometry.heads) * tokens);
    beta.resize(static_cast<std::size_t>(geometry.heads) * tokens);
    std::vector<double> normalized(static_cast<std::size_t>(geometry.hidden));

    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t x_base = static_cast<std::size_t>(token) * geometry.hidden;
        double sum_squares       = 0.0;
        for (std::int32_t k = 0; k < geometry.hidden; ++k) {
            const double value = static_cast<double>(x[x_base + static_cast<std::size_t>(k)]);
            sum_squares += value * value;
        }
        const double inverse_rms =
            1.0 / std::sqrt(sum_squares / static_cast<double>(geometry.hidden) + eps);
        for (std::int32_t k = 0; k < geometry.hidden; ++k) {
            const double value = static_cast<double>(x[x_base + static_cast<std::size_t>(k)]) *
                                 inverse_rms * (1.0 + static_cast<double>(norm_weight[k]));
            normalized[static_cast<std::size_t>(k)] = value;
            h[x_base + static_cast<std::size_t>(k)] = value;
        }

        for (std::int32_t head = 0; head < geometry.heads; ++head) {
            const std::size_t weight_base = static_cast<std::size_t>(head) * geometry.hidden;
            double projected_a            = 0.0;
            double projected_b            = 0.0;
            for (std::int32_t k = 0; k < geometry.hidden; ++k) {
                const double value = normalized[static_cast<std::size_t>(k)];
                projected_a +=
                    static_cast<double>(a_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
                projected_b +=
                    static_cast<double>(b_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
            }
            const std::size_t output = static_cast<std::size_t>(token) * geometry.heads + head;
            g[output]                = -std::exp(static_cast<double>(a_log[head])) *
                        softplus(projected_a + static_cast<double>(dt_bias[head]));
            beta[output] = sigmoid(projected_b);
        }
    }
}

std::vector<double> select_tokens(const std::vector<double>& full,
                                  const std::vector<std::int32_t>& selected_tokens,
                                  std::int32_t rows) {
    std::vector<double> selected(static_cast<std::size_t>(rows) * selected_tokens.size());
    for (std::size_t sample = 0; sample < selected_tokens.size(); ++sample) {
        const std::size_t source = static_cast<std::size_t>(selected_tokens[sample]) * rows;
        std::copy_n(full.begin() + static_cast<std::ptrdiff_t>(source), rows,
                    selected.begin() + static_cast<std::ptrdiff_t>(sample * rows));
    }
    return selected;
}

int require_all_finite(const std::string& label, const std::vector<double>& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            std::cerr << label << ": output element " << i
                      << " was not written to a finite value\n";
            return 1;
        }
    }
    return 0;
}

int verify_inputs_unchanged(const std::string& label, const DeviceBuffer& device_x,
                            const std::vector<std::uint16_t>& x_bits,
                            const DeviceBuffer& device_weight,
                            const std::vector<std::uint16_t>& weight_bits,
                            const DeviceBuffer& device_a_log, const std::vector<float>& a_log,
                            const DeviceBuffer& device_dt_bias, const std::vector<float>& dt_bias) {
    int failures = 0;
    failures += verify_exact((label + " x immutable").c_str(),
                             from_device<std::uint16_t>(device_x, x_bits.size()), x_bits);
    failures +=
        verify_exact((label + " weight immutable").c_str(),
                     from_device<std::uint16_t>(device_weight, weight_bits.size()), weight_bits);
    failures += verify_exact((label + " A_log immutable").c_str(),
                             from_device<float>(device_a_log, a_log.size()), a_log);
    failures += verify_exact((label + " dt_bias immutable").c_str(),
                             from_device<float>(device_dt_bias, dt_bias.size()), dt_bias);
    return failures;
}

int run_projection_case(const Geometry& geometry, std::int32_t tokens, std::uint32_t seed,
                        DeviceExecutionView execution) {
    std::vector<float> x(static_cast<std::size_t>(geometry.hidden) * tokens);
    std::vector<float> a_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> b_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0F, 1.0F);
    fill_uniform(a_weight, seed + 1u, -0.015F, 0.015F);
    fill_uniform(b_weight, seed + 2u, -0.015F, 0.015F);
    fill_uniform(a_log, seed + 3u, -2.0F, 1.0F);
    fill_uniform(dt_bias, seed + 4u, -1.0F, 1.0F);
    round_to_bf16(x);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);

    const std::vector<std::int32_t> selected = oracle_tokens(tokens);
    std::vector<double> reference_g, reference_beta;
    projection_oracle(geometry, x, a_weight, b_weight, a_log, dt_bias, selected, reference_g,
                      reference_beta);

    const std::vector<std::uint16_t> x_bits        = bf16_bits(x);
    std::vector<std::uint16_t> weight_bits         = bf16_bits(a_weight);
    const std::vector<std::uint16_t> b_weight_bits = bf16_bits(b_weight);
    if (geometry.parent_weight) {
        weight_bits.insert(weight_bits.end(), b_weight_bits.begin(), b_weight_bits.end());
    }
    DeviceBuffer device_x      = to_device(x_bits);
    DeviceBuffer device_weight = to_device(weight_bits);
    DeviceBuffer device_b_weight;
    if (!geometry.parent_weight) { device_b_weight = to_device(b_weight_bits); }
    DeviceBuffer device_a_log         = to_device(a_log);
    DeviceBuffer device_dt_bias       = to_device(dt_bias);
    const std::size_t output_elements = static_cast<std::size_t>(geometry.heads) * tokens;
    GuardedDeviceBuffer device_g(output_elements * sizeof(float));
    GuardedDeviceBuffer device_beta(output_elements * sizeof(float));
    device_g.fill(0xff);
    device_beta.fill(0xff);

    Tensor tensor_x(device_x.p, DType::BF16, {geometry.hidden, tokens});
    Tensor tensor_a_log(device_a_log.p, DType::FP32, {geometry.heads});
    Tensor tensor_dt_bias(device_dt_bias.p, DType::FP32, {geometry.heads});
    Tensor tensor_g(device_g.data(), DType::FP32, {geometry.heads, tokens});
    Tensor tensor_beta(device_beta.data(), DType::FP32, {geometry.heads, tokens});
    const std::size_t workspace_bytes = ops::gdn_gating_proj_workspace_capacity_bytes(
        geometry.heads, geometry.hidden, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    if (geometry.parent_weight) {
        // A/B use a contiguous reshape of the same stored words, with a different parent K.
        const std::array shape{static_cast<std::uint64_t>(4 * geometry.heads),
                               static_cast<std::uint64_t>(geometry.hidden / 2)};
        const WeightParent parent{weight_geometry(QType::BF16, QuantLayout::Contiguous, shape),
                                  static_cast<const std::byte*>(device_weight.p)};
        const auto count = static_cast<std::uint64_t>(geometry.heads) * geometry.hidden;
        const std::vector<std::uint64_t> logical_shape{static_cast<std::uint64_t>(geometry.heads),
                                                       static_cast<std::uint64_t>(geometry.hidden)};
        const WeightView a{logical_shape, {{&parent, 0, count}}};
        const WeightView b{logical_shape, {{&parent, count, 2 * count}}};
        const auto prepared =
            std::get<ops::SingleProjectionWeight>(ops::prepare_gdn_gating_proj_weights({a}, {b}));
        ops::gdn_gating_proj(tensor_x, prepared.weight, tensor_a_log, tensor_dt_bias, workspace,
                             tensor_g, tensor_beta, execution);
    } else {
        Weight weight_a = bf16_weight(device_weight.p, geometry.heads, geometry.hidden);
        Weight weight_b = bf16_weight(device_b_weight.p, geometry.heads, geometry.hidden);
        ops::gdn_gating_proj(tensor_x, weight_a, weight_b, tensor_a_log, tensor_dt_bias, workspace,
                             tensor_g, tensor_beta, execution);
    }
    cuda_synchronize();

    const std::vector<double> full_g    = read_fp32(device_g.data(), output_elements);
    const std::vector<double> full_beta = read_fp32(device_beta.data(), output_elements);
    const std::string label =
        std::string("gdn_gating_proj ") + geometry.label + " T=" + std::to_string(tokens);
    int failures = 0;
    failures += require_all_finite(label + " g", full_g);
    failures += require_all_finite(label + " beta", full_beta);
    failures += verify_normwise(label + " g", select_tokens(full_g, selected, geometry.heads),
                                reference_g, kGdnProjectionFp32);
    failures += verify_normwise(label + " beta", select_tokens(full_beta, selected, geometry.heads),
                                reference_beta, kGdnProjectionFp32);
    failures += device_g.verify_guards((label + " g").c_str());
    failures += device_beta.verify_guards((label + " beta").c_str());
    failures += verify_inputs_unchanged(label, device_x, x_bits, device_weight, weight_bits,
                                        device_a_log, a_log, device_dt_bias, dt_bias);
    if (!geometry.parent_weight) {
        failures += verify_exact((label + " b_weight immutable").c_str(),
                                 from_device<std::uint16_t>(device_b_weight, b_weight_bits.size()),
                                 b_weight_bits);
    }
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_norm_projection_case(const Geometry& geometry, std::int32_t tokens, std::uint32_t seed,
                             DeviceExecutionView execution, bool replay = false, int mode = 0) {
    constexpr float kEps               = 1.0e-6f;
    const std::size_t h_elements       = std::size_t(geometry.hidden) * tokens;
    const std::size_t control_elements = std::size_t(geometry.heads) * tokens;
    std::vector<float> x(h_elements), norm_weight(geometry.hidden);
    std::vector<float> a_weight(std::size_t(geometry.heads) * geometry.hidden),
        b_weight(a_weight.size());
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0f, 1.0f);
    fill_uniform(norm_weight, seed + 1, -0.2f, 0.2f);
    fill_uniform(a_weight, seed + 2, -0.015f, 0.015f);
    fill_uniform(b_weight, seed + 3, -0.015f, 0.015f);
    fill_uniform(a_log, seed + 4, -2.0f, 1.0f);
    fill_uniform(dt_bias, seed + 5, -1.0f, 1.0f);
    norm_weight[0] = -1.0f;
    norm_weight[1] = -1.5f;
    if (mode == 1) std::fill(x.begin(), x.end(), 0.0f);
    if (mode == 2)
        for (auto& v : x) v *= 1.0e-5f;
    if (mode == 3)
        for (auto& v : x) v *= 512.0f;
    if (mode == 4) std::fill(norm_weight.begin(), norm_weight.end(), -1.0f);
    round_to_bf16(x);
    round_to_bf16(norm_weight);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);
    auto x_bits = bf16_bits(x), norm_weight_bits = bf16_bits(norm_weight),
         weight_bits         = bf16_bits(a_weight);
    const auto b_weight_bits = bf16_bits(b_weight);
    if (geometry.parent_weight)
        weight_bits.insert(weight_bits.end(), b_weight_bits.begin(), b_weight_bits.end());
    DeviceBuffer device_x = to_device(x_bits), device_norm_weight = to_device(norm_weight_bits),
                 device_weight = to_device(weight_bits);
    DeviceBuffer device_b_weight;
    if (!geometry.parent_weight) device_b_weight = to_device(b_weight_bits);
    DeviceBuffer device_a_log = to_device(a_log), device_dt_bias = to_device(dt_bias);
    GuardedDeviceBuffer device_h(h_elements * 2), device_g(control_elements * 4),
        device_beta(control_elements * 4);
    Tensor tx(device_x.p, DType::BF16, {geometry.hidden, tokens}),
        tn(device_norm_weight.p, DType::BF16, {geometry.hidden});
    Tensor th(device_h.data(), DType::BF16, {geometry.hidden, tokens}),
        tg(device_g.data(), DType::FP32, {geometry.heads, tokens}),
        tb(device_beta.data(), DType::FP32, {geometry.heads, tokens});
    Tensor ta(device_a_log.p, DType::FP32, {geometry.heads}),
        td(device_dt_bias.p, DType::FP32, {geometry.heads});
    const auto capacity = ops::gdn_norm_gating_proj_workspace_capacity_bytes(
        geometry.heads, geometry.hidden, tokens, tokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 256));
    WorkspaceArena workspace(DeviceSpan{scratch.data(), std::max<std::size_t>(capacity, 256)});
    const auto wa = bf16_weight(device_weight.p, (geometry.parent_weight ? 2 : 1) * geometry.heads,
                                geometry.hidden);
    const auto wb = geometry.parent_weight
                        ? Weight{}
                        : bf16_weight(device_b_weight.p, geometry.heads, geometry.hidden);
    const auto launch = [&] {
        if (geometry.parent_weight)
            ops::gdn_norm_gating_proj(tx, tn, kEps, wa, ta, td, workspace, th, tg, tb, execution);
        else
            ops::gdn_norm_gating_proj(tx, tn, kEps, wa, wb, ta, td, workspace, th, tg, tb,
                                      execution);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (replay) {
        launch();
        cuda_synchronize(execution.stream);
    }
    int failures = 0;
    for (int phase = 0; phase < (replay ? 3 : 1); ++phase) {
        if (phase) {
            for (auto& v : x) v *= -0.5f;
            for (auto& v : norm_weight) v = -v;
            for (auto& v : a_log) v += 0.125f;
            for (auto& v : dt_bias) v = -v;
            round_to_bf16(x);
            round_to_bf16(norm_weight);
            x_bits           = bf16_bits(x);
            norm_weight_bits = bf16_bits(norm_weight);
            device_x.copy_from_host(x_bits.data(), x_bits.size() * 2);
            device_norm_weight.copy_from_host(norm_weight_bits.data(), norm_weight_bits.size() * 2);
            device_a_log.copy_from_host(a_log.data(), a_log.size() * 4);
            device_dt_bias.copy_from_host(dt_bias.data(), dt_bias.size() * 4);
        }
        device_h.fill(0xff);
        device_g.fill(0xff);
        device_beta.fill(0xff);
        scratch.fill(0xff);
        cuda_synchronize();
        if (replay && phase == 0) {
            definition.capture(execution.stream, launch);
            graph.instantiate(definition);
        }
        if (replay)
            graph.launch(execution.stream);
        else
            launch();
        cuda_synchronize(execution.stream);
        std::vector<double> rh, rg, rb;
        norm_projection_oracle(geometry, x, norm_weight, a_weight, b_weight, a_log, dt_bias, tokens,
                               kEps, rh, rg, rb);
        const std::string label = std::string("gdn_norm_gating_proj ") + geometry.label +
                                  " T=" + std::to_string(tokens) + " mode=" + std::to_string(mode) +
                                  " phase=" + std::to_string(phase);
        failures += verify_normwise(label + " h", from_device_bf16(device_h.data(), h_elements), rh,
                                    kGdnNormOutputBf16);
        failures += verify_normwise(label + " g", read_fp32(device_g.data(), control_elements), rg,
                                    kGdnNormControlFp32);
        failures +=
            verify_normwise(label + " beta", read_fp32(device_beta.data(), control_elements), rb,
                            kGdnNormControlFp32);
        failures += device_h.verify_guards((label + " h").c_str());
        failures += device_g.verify_guards((label + " g").c_str());
        failures += device_beta.verify_guards((label + " beta").c_str());
        failures += scratch.verify_guards((label + " scratch").c_str());
        failures += verify_inputs_unchanged(label, device_x, x_bits, device_weight, weight_bits,
                                            device_a_log, a_log, device_dt_bias, dt_bias);
        failures +=
            verify_exact((label + " norm weight immutable").c_str(),
                         from_device<std::uint16_t>(device_norm_weight, norm_weight_bits.size()),
                         norm_weight_bits);
        if (!geometry.parent_weight)
            failures += verify_exact(
                (label + " b weight immutable").c_str(),
                from_device<std::uint16_t>(device_b_weight, b_weight_bits.size()), b_weight_bits);
        if (workspace.used() != 0 || workspace.peak_used() != capacity) {
            std::cerr << label << ": workspace query/peak mismatch\n";
            ++failures;
        }
    }
    return failures;
}

enum class PublishedFormat { Nvfp4A4, Fp8Row };

// One published a or b matrix: the fixture's packed words on the device, the parent the artifact
// reader would bind and its decoded values. Not movable: the view points at the parent.
struct PublishedMatrix {
    PublishedMatrix(PublishedFormat format, std::int32_t heads, std::int32_t hidden,
                    std::uint32_t seed) {
        if (format == PublishedFormat::Nvfp4A4) {
            quantized_weight::PatternedWeightOptions options;
            options.weight_scale_divisor = 16.0F; // |w| <= 6 * 0.1 / 16
            options.input_scale_divisor  = 1.0F;  // unused: the use carries the divisor
            host =
                quantized_weight::make_patterned_weight(QType::NVFP4, heads, hidden, seed, options);
        } else {
            host = quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, heads,
                                                           hidden, seed);
            // Row scales near 2^-15 (|w| <= 448 * 2^-15 * 1.75), the BF16 test's magnitude.
            for (std::int32_t row = 0; row < heads; ++row) {
                const float scale = std::ldexp(1.0F + 0.25F * static_cast<float>(row % 4), -15);
                quantized_weight::detail::store_u16_le(
                    host.payload, host.scale_plane_offset + static_cast<std::size_t>(row) * 2,
                    f32_to_bf16(scale));
            }
        }
        values.resize(static_cast<std::size_t>(heads) * hidden);
        for (std::int32_t row = 0; row < heads; ++row) {
            for (std::int32_t column = 0; column < hidden; ++column) {
                values[static_cast<std::size_t>(row) * hidden + column] =
                    static_cast<float>(quantized_weight::logical_weight_fp64(host, row, column));
            }
        }
        device = to_device(host.payload);
        const std::array shape{static_cast<std::uint64_t>(heads),
                               static_cast<std::uint64_t>(hidden)};
        parent = WeightParent{weight_geometry(host.weight.qtype, host.weight.layout, shape),
                              static_cast<const std::byte*>(device.p),
                              format == PublishedFormat::Nvfp4A4 ? 16.0F : 0.0F};
        if (parent.geometry.bytes != host.payload.size()) {
            throw std::logic_error("published a/b fixture: payload differs from its geometry");
        }
        view = WeightView{{shape[0], shape[1]}, {{&parent, 0, shape[0] * shape[1]}}};
    }

    PublishedMatrix(const PublishedMatrix&)            = delete;
    PublishedMatrix& operator=(const PublishedMatrix&) = delete;

    quantized_weight::PackedWeight host;
    DeviceBuffer device;
    WeightParent parent;
    WeightView view;
    std::vector<float> values;
};

// a/b kept as published (separate NVFP4 parents with calibrated activation divisors, or FP8
// rows): prepare_gdn_gating_proj_weights must leave the fused BF16 kernel for the segmented
// generic path at every geometry, and g/beta must match the decoded weights' oracle.
int run_published_control_case(const Geometry& geometry, PublishedFormat format,
                               std::int32_t tokens, std::uint32_t seed, cudaStream_t stream) {
    constexpr float kEps    = 1.0e-6F;
    const bool nvfp4        = format == PublishedFormat::Nvfp4A4;
    const std::string label = std::string("published ") + (nvfp4 ? "nvfp4 a4 " : "fp8 row ") +
                              geometry.label + " [" + std::to_string(geometry.heads) + ", " +
                              std::to_string(geometry.hidden) + "] T=" + std::to_string(tokens);
    const std::size_t h_elements       = std::size_t(geometry.hidden) * tokens;
    const std::size_t control_elements = std::size_t(geometry.heads) * tokens;
    std::vector<float> x(h_elements), norm_weight(geometry.hidden);
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0F, 1.0F);
    fill_uniform(norm_weight, seed + 1u, -0.2F, 0.2F);
    fill_uniform(a_log, seed + 2u, -2.0F, 1.0F);
    fill_uniform(dt_bias, seed + 3u, -1.0F, 1.0F);
    round_to_bf16(x);
    round_to_bf16(norm_weight);
    const auto a =
        std::make_unique<PublishedMatrix>(format, geometry.heads, geometry.hidden, seed + 4u);
    const auto b =
        std::make_unique<PublishedMatrix>(format, geometry.heads, geometry.hidden, seed + 5u);

    std::vector<double> rh, rg, rb;
    norm_projection_oracle(geometry, x, norm_weight, a->values, b->values, a_log, dt_bias, tokens,
                           kEps, rh, rg, rb);
    // ModelOpt's calibration: input_scale = amax / (6 * 448), the divisor its reciprocal; b's is
    // another, so the two slices cannot share one activation quantization.
    double amax = 0.0;
    for (const double value : rh) amax = std::max(amax, std::abs(value));
    const float a_divisor = static_cast<float>(6.0 * 448.0 / amax);
    const float b_divisor = 0.75F * a_divisor;
    const ops::WeightInput a_input =
        nvfp4 ? ops::WeightInput{a->view, ops::LinearPolicy::AllowA4, a_divisor}
              : ops::WeightInput{a->view, ops::LinearPolicy::AllowA8, std::nullopt};
    const ops::WeightInput b_input =
        nvfp4 ? ops::WeightInput{b->view, ops::LinearPolicy::AllowA4, b_divisor}
              : ops::WeightInput{b->view, ops::LinearPolicy::AllowA8, std::nullopt};

    int failures                          = 0;
    const ops::ProjectionWeights prepared = ops::prepare_gdn_gating_proj_weights(a_input, b_input);
    const auto* segmented                 = std::get_if<ops::SegmentedProjectionWeights>(&prepared);
    if (segmented == nullptr) {
        std::cerr << label << ": prepare_gdn_gating_proj_weights did not take the segmented path\n";
        return 1;
    }
    const auto want_policy = nvfp4 ? ops::LinearPolicy::AllowA4 : ops::LinearPolicy::AllowA8;
    if (segmented->slices.size() != 2 || segmented->rows() != 2 * geometry.heads ||
        segmented->columns() != geometry.hidden) {
        std::cerr << label << ": segmented a/b bank has " << segmented->slices.size()
                  << " slices of " << segmented->rows() << " rows\n";
        return 1;
    }
    for (const auto& slice : segmented->slices) {
        if (slice.parent.qtype == QType::BF16 || slice.policy != want_policy ||
            slice.rows != geometry.heads || slice.row_begin != 0) {
            std::cerr << label << ": segmented slice is not the published parent at its policy\n";
            ++failures;
        }
    }
    if (nvfp4 && (segmented->slices[0].parent.input_scale_divisor != a_divisor ||
                  segmented->slices[1].parent.input_scale_divisor != b_divisor)) {
        std::cerr << label << ": a/b slices lost their own activation divisors\n";
        ++failures;
    }

    const auto x_bits = bf16_bits(x), norm_bits = bf16_bits(norm_weight);
    DeviceBuffer device_x = to_device(x_bits), device_norm = to_device(norm_bits);
    DeviceBuffer device_a_log = to_device(a_log), device_dt_bias = to_device(dt_bias);
    GuardedDeviceBuffer device_h(h_elements * 2), device_g(control_elements * 4),
        device_beta(control_elements * 4);
    device_h.fill(0xff);
    device_g.fill(0xff);
    device_beta.fill(0xff);
    Tensor tx(device_x.p, DType::BF16, {geometry.hidden, tokens}),
        tn(device_norm.p, DType::BF16, {geometry.hidden});
    Tensor th(device_h.data(), DType::BF16, {geometry.hidden, tokens}),
        tg(device_g.data(), DType::FP32, {geometry.heads, tokens}),
        tb(device_beta.data(), DType::FP32, {geometry.heads, tokens});
    Tensor ta(device_a_log.p, DType::FP32, {geometry.heads}),
        td(device_dt_bias.p, DType::FP32, {geometry.heads});
    const std::size_t capacity = std::max<std::size_t>(
        256, ops::segmented_gdn_norm_gating_workspace_capacity_bytes(*segmented, tokens, tokens));
    GuardedDeviceBuffer scratch(capacity);
    WorkspaceArena workspace(DeviceSpan{scratch.data(), capacity});
    ops::segmented_gdn_norm_gating_proj(tx, tn, kEps, *segmented, ta, td, workspace, th, tg, tb,
                                        stream);
    cuda_synchronize(stream);

    const ReductionCriterion& control = nvfp4 ? kPublishedA4Control : kPublishedA8Control;
    const std::vector<double> g       = read_fp32(device_g.data(), control_elements);
    const std::vector<double> beta    = read_fp32(device_beta.data(), control_elements);
    failures += require_all_finite(label + " g", g);
    failures += require_all_finite(label + " beta", beta);
    failures += verify_normwise(label + " h", from_device_bf16(device_h.data(), h_elements), rh,
                                kGdnNormOutputBf16);
    failures += verify_normwise(label + " g", g, rg, control);
    failures += verify_normwise(label + " beta", beta, rb, control);
    failures += device_h.verify_guards((label + " h").c_str());
    failures += device_g.verify_guards((label + " g").c_str());
    failures += device_beta.verify_guards((label + " beta").c_str());
    failures += scratch.verify_guards((label + " scratch").c_str());
    for (const auto* matrix : {a.get(), b.get()}) {
        if (from_device<std::uint8_t>(matrix->device, matrix->host.payload.size()) !=
            matrix->host.payload) {
            std::cerr << label << ": a published weight was modified\n";
            ++failures;
        }
    }
    // The error the A4/A8 activations leave against the decoded weights, for the record.
    const auto relative_l2 = [](const std::vector<double>& got, const std::vector<double>& ref) {
        double error = 0.0, norm = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            error += (got[i] - ref[i]) * (got[i] - ref[i]);
            norm += ref[i] * ref[i];
        }
        return std::sqrt(error / std::max(norm, 1e-300));
    };
    std::cout << label << ": segmented, g rel_l2 " << relative_l2(g, rg) << ", beta rel_l2 "
              << relative_l2(beta, rb) << (failures == 0 ? " ok" : " FAILED") << '\n';
    return failures;
}

// BF16 a/b at an unregistered geometry (the Qwen3.5 0.8B..9B sizes): the segmented generic path.
// At decode and verify widths (T <= kSegmentedFusedControlMaxTokens) it runs the fused norm and
// control kernel, wider calls the composed form; both against the FP64 oracle. In the fused
// widths a token's h, g and beta must not depend on the other columns of its call (a verify pass
// and the decode step of the same token agree). `one_parent` binds a and b as consecutive rows of
// one [2H, D] parent, otherwise as two parents.
int run_segmented_bf16_control_case(const Geometry& geometry, bool one_parent, std::uint32_t seed,
                                    cudaStream_t stream) {
    constexpr float kEps         = 1.0e-6F;
    constexpr std::int32_t kMaxT = 64;
    const std::string base_label = std::string("segmented bf16 ") + geometry.label + " [" +
                                   std::to_string(geometry.heads) + ", " +
                                   std::to_string(geometry.hidden) + "]" +
                                   (one_parent ? " one parent" : " two parents");
    const std::size_t rows_elements = std::size_t(geometry.heads) * geometry.hidden;
    std::vector<float> x(std::size_t(geometry.hidden) * kMaxT), norm_weight(geometry.hidden);
    std::vector<float> a_weight(rows_elements), b_weight(rows_elements);
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0F, 1.0F);
    fill_uniform(norm_weight, seed + 1u, -0.2F, 0.2F);
    fill_uniform(a_weight, seed + 2u, -0.015F, 0.015F);
    fill_uniform(b_weight, seed + 3u, -0.015F, 0.015F);
    fill_uniform(a_log, seed + 4u, -2.0F, 1.0F);
    fill_uniform(dt_bias, seed + 5u, -1.0F, 1.0F);
    round_to_bf16(x);
    round_to_bf16(norm_weight);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);
    auto ab_bits             = bf16_bits(a_weight);
    const auto b_weight_bits = bf16_bits(b_weight);
    ab_bits.insert(ab_bits.end(), b_weight_bits.begin(), b_weight_bits.end());
    DeviceBuffer device_ab = to_device(ab_bits);
    const std::vector<std::uint64_t> a_shape{static_cast<std::uint64_t>(geometry.heads),
                                             static_cast<std::uint64_t>(geometry.hidden)};
    const std::vector<std::uint64_t> ab_shape{static_cast<std::uint64_t>(2 * geometry.heads),
                                              static_cast<std::uint64_t>(geometry.hidden)};
    const auto* ab_data = static_cast<const std::byte*>(device_ab.p);
    const WeightParent ab_parent{weight_geometry(QType::BF16, QuantLayout::Contiguous, ab_shape),
                                 ab_data};
    const WeightParent a_parent{weight_geometry(QType::BF16, QuantLayout::Contiguous, a_shape),
                                ab_data};
    const WeightParent b_parent{weight_geometry(QType::BF16, QuantLayout::Contiguous, a_shape),
                                ab_data + rows_elements * 2};
    const WeightView a_view{a_shape, {{one_parent ? &ab_parent : &a_parent, 0, rows_elements}}};
    const WeightView b_view{a_shape,
                            {one_parent ? WeightRegion{&ab_parent, rows_elements, 2 * rows_elements}
                                        : WeightRegion{&b_parent, 0, rows_elements}}};
    const ops::ProjectionWeights prepared = ops::prepare_gdn_gating_proj_weights(
        ops::WeightInput{a_view, ops::LinearPolicy::A16Only, std::nullopt},
        ops::WeightInput{b_view, ops::LinearPolicy::A16Only, std::nullopt});
    const auto* segmented = std::get_if<ops::SegmentedProjectionWeights>(&prepared);
    if (segmented == nullptr || segmented->rows() != 2 * geometry.heads) {
        std::cerr << base_label
                  << ": prepare_gdn_gating_proj_weights did not take the segmented path\n";
        return 1;
    }
    const auto x_bits = bf16_bits(x), norm_bits = bf16_bits(norm_weight);
    DeviceBuffer device_x = to_device(x_bits), device_norm = to_device(norm_bits);
    DeviceBuffer device_a_log = to_device(a_log), device_dt_bias = to_device(dt_bias);
    Tensor tn(device_norm.p, DType::BF16, {geometry.hidden});
    Tensor ta(device_a_log.p, DType::FP32, {geometry.heads}),
        td(device_dt_bias.p, DType::FP32, {geometry.heads});

    struct Outputs {
        std::vector<std::uint16_t> h;
        std::vector<float> g, beta;
    };

    int failures   = 0;
    const auto run = [&](std::int32_t tokens) {
        const std::string label        = base_label + " T=" + std::to_string(tokens);
        const std::size_t h_elements   = std::size_t(geometry.hidden) * tokens;
        const std::size_t ctl_elements = std::size_t(geometry.heads) * tokens;
        GuardedDeviceBuffer device_h(h_elements * 2), device_g(ctl_elements * 4),
            device_beta(ctl_elements * 4);
        device_h.fill(0xff);
        device_g.fill(0xff);
        device_beta.fill(0xff);
        Tensor tx(device_x.p, DType::BF16, {geometry.hidden, tokens});
        Tensor th(device_h.data(), DType::BF16, {geometry.hidden, tokens}),
            tg(device_g.data(), DType::FP32, {geometry.heads, tokens}),
            tb(device_beta.data(), DType::FP32, {geometry.heads, tokens});
        const std::size_t capacity = std::max<std::size_t>(
            256,
            ops::segmented_gdn_norm_gating_workspace_capacity_bytes(*segmented, tokens, tokens));
        GuardedDeviceBuffer scratch(capacity);
        WorkspaceArena workspace(DeviceSpan{scratch.data(), capacity});
        ops::segmented_gdn_norm_gating_proj(tx, tn, kEps, *segmented, ta, td, workspace, th, tg, tb,
                                            stream);
        cuda_synchronize(stream);
        std::vector<double> rh, rg, rb;
        const std::vector<float> xt(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(h_elements));
        norm_projection_oracle(geometry, xt, norm_weight, a_weight, b_weight, a_log, dt_bias,
                               tokens, kEps, rh, rg, rb);
        Outputs out{from_device<std::uint16_t>(device_h.data(), h_elements),
                    from_device<float>(device_g.data(), ctl_elements),
                    from_device<float>(device_beta.data(), ctl_elements)};
        failures += verify_normwise(label + " h", from_device_bf16(device_h.data(), h_elements), rh,
                                    kGdnNormOutputBf16);
        failures +=
            verify_normwise(label + " g", {out.g.begin(), out.g.end()}, rg, kGdnNormControlFp32);
        failures += verify_normwise(label + " beta", {out.beta.begin(), out.beta.end()}, rb,
                                    kGdnNormControlFp32);
        failures += device_h.verify_guards((label + " h").c_str());
        failures += device_g.verify_guards((label + " g").c_str());
        failures += device_beta.verify_guards((label + " beta").c_str());
        failures += scratch.verify_guards((label + " scratch").c_str());
        return out;
    };
    const Outputs widest = run(ops::kSegmentedFusedControlMaxTokens);
    for (const std::int32_t tokens : {1, 2, 3, 5, 8, 13}) {
        const Outputs narrow = run(tokens);
        const auto prefix    = [&](const auto& v, std::int32_t rows) {
            return std::vector(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(rows) * tokens);
        };
        if (narrow.h != prefix(widest.h, geometry.hidden) ||
            narrow.g != prefix(widest.g, geometry.heads) ||
            narrow.beta != prefix(widest.beta, geometry.heads)) {
            std::cerr << base_label << " T=" << tokens
                      << ": a token's bytes follow the width of its call\n";
            ++failures;
        }
    }
    for (const std::int32_t tokens : {ops::kSegmentedFusedControlMaxTokens + 1, kMaxT})
        (void)run(tokens);
    if (from_device<std::uint16_t>(device_ab.p, ab_bits.size()) != ab_bits) {
        std::cerr << base_label << ": a/b weights were modified\n";
        ++failures;
    }
    std::cout << base_label << (failures == 0 ? " ok" : " FAILED") << '\n';
    return failures;
}

int verify_workspace_capacity_contract(const Geometry& geometry,
                                       std::initializer_list<std::int32_t> route_endpoints) {
    const std::int32_t last = *std::max_element(route_endpoints.begin(), route_endpoints.end());
    const std::size_t interval =
        ops::gdn_gating_proj_workspace_capacity_bytes(geometry.heads, geometry.hidden, 1, last);
    std::size_t witness = 0;
    for (const std::int32_t tokens : route_endpoints) {
        witness = std::max(witness, ops::gdn_gating_proj_workspace_capacity_bytes(
                                        geometry.heads, geometry.hidden, tokens, tokens));
    }
    int failures = 0;
    if (interval != witness) {
        std::cerr << geometry.label << ": GDN control interval missed a route endpoint\n";
        ++failures;
    }
    for (const auto bounds :
         std::vector<std::pair<int, int>>{{1, 42}, {40, 43}, {1, 128}, {120, 129}, {1, 256}}) {
        std::size_t witness = 0;
        for (int t = bounds.first; t <= bounds.second; ++t)
            witness = std::max(witness, ops::gdn_norm_gating_proj_workspace_capacity_bytes(
                                            geometry.heads, geometry.hidden, t, t));
        const auto capacity = ops::gdn_norm_gating_proj_workspace_capacity_bytes(
            geometry.heads, geometry.hidden, bounds.first, bounds.second);
        if (capacity != witness) {
            std::cerr << geometry.label << ": norm/control interval missed an extent\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    DeviceContext device;
    const DeviceExecutionView execution{nullptr, device.multiprocessor_count()};
    const DeviceExecutionView norm_execution{device.stream, device.multiprocessor_count()};
    int failures = 0;
    failures += verify_workspace_capacity_contract(kQwen27, {1, 8, 1024, 2048, 4096, 4097});
    failures += verify_workspace_capacity_contract(kQwen35, {1, 127, 1024, 2048, 4096, 4097});

    // Every registered 27B projection route, including predicated and full token tiles.
    for (const std::int32_t tokens : {1, 8, 9, 1024, 1025, 2049, 4097}) {
        failures += run_projection_case(kQwen27, tokens,
                                        0x1000u + static_cast<std::uint32_t>(tokens), execution);
    }
    // The Qwen3.8 parent changes only the public storage boundary. One direct oracle case proves
    // its [A,B] row partition; the split 27B cases above cover every unchanged execution route.
    failures += run_projection_case(kQwen38Parent, 1, 0x1801u, execution);
    // Every registered 35B projection route and its contiguous-parent storage contract.
    for (const std::int32_t tokens : {1, 127, 128, 1024, 1025, 2049, 4097}) {
        failures += run_projection_case(kQwen35, tokens,
                                        0x2000u + static_cast<std::uint32_t>(tokens), execution);
    }

    // Direct complete norm/control oracles across variable widths and the existing 35B profile.
    failures += run_norm_projection_case(kQwen27, 1, 0x3001u, norm_execution);
    failures += run_norm_projection_case(kQwen27, 9, 0x3009u, norm_execution);
    failures += run_norm_projection_case(kQwen27, 64, 0x3040u, norm_execution);
    for (int tokens = 1; tokens <= 128; ++tokens)
        failures +=
            run_norm_projection_case(kQwen38Parent, tokens, 0x3800u + tokens, norm_execution);
    for (int tokens : {1, 2, 3, 8, 9, 14, 15, 16, 28, 29, 32, 42, 43, 64, 96, 128, 129, 256})
        failures +=
            run_norm_projection_case(kQwen38Parent, tokens, 0x4800u + tokens, norm_execution, true);
    for (int mode : {1, 2, 3, 4})
        failures +=
            run_norm_projection_case(kQwen38Parent, 16, 0x5800u + mode, norm_execution, true, mode);
    failures += run_norm_projection_case(kQwen35, 1, 0x4001u, norm_execution);
    failures += run_norm_projection_case(kQwen35, 16, 0x4010u, norm_execution);
    failures += run_norm_projection_case(kQwen35, 17, 0x4011u, norm_execution);
    failures += run_norm_projection_case(kQwen35, 64, 0x4040u, norm_execution);

    // Requalify the retained BF16-staging profile at every prefill reduction boundary.
    for (int tokens : {1024, 1025, 2048, 2049, 4097})
        failures +=
            run_norm_projection_case(kQwen38Parent, tokens, 0x6800u + tokens, norm_execution);
    for (int tokens : {2, 8, 15, 127, 128, 1024, 1025, 2048, 2049, 4097})
        failures += run_norm_projection_case(kQwen35, tokens, 0x7800u + tokens, norm_execution,
                                             tokens == 15);

    // Published quantized a/b (qwen3_5_published): NVFP4 W4A4 and FP8 rows, at the registered
    // fused geometries (which run only BF16) and the 9B's unregistered one, decode and prefill.
    std::uint32_t published_seed = 0x9000u;
    for (const Geometry* geometry : {&kQwen27Published, &kQwen35Published, &kQwen9Published})
        for (const PublishedFormat format : {PublishedFormat::Nvfp4A4, PublishedFormat::Fp8Row})
            for (const std::int32_t tokens : {1, 64})
                failures += run_published_control_case(
                    *geometry, format, tokens, published_seed += 0x10u, norm_execution.stream);

    // BF16 a/b at the Qwen3.5 sizes no fused registration owns (9B, 4B, 0.8B).
    std::uint32_t segmented_seed = 0xa000u;
    for (const Geometry& geometry :
         {Geometry{"qwen3_5_9b", 4096, 32, false}, Geometry{"qwen3_5_4b", 2560, 32, false},
          Geometry{"qwen3_5_0_8b", 1024, 16, false}})
        for (const bool one_parent : {false, true})
            failures += run_segmented_bf16_control_case(
                geometry, one_parent, segmented_seed += 0x10u, norm_execution.stream);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_gating_proj correctness\n";
    return failures == 0 ? 0 : 1;
}
