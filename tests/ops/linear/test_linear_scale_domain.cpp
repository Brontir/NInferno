// The Linear operator oracle over the scale domain. The other Linear tests run
// fixed scales (NVFP4 divisor 0.125, block scale words 0x04..0x1d, row-split scales within 4x),
// where a kernel that applies a scale twice, in the wrong direction or not at all per block can
// still pass. Here, for every weight format and activation path:
//
//   - the tensor scale takes 3e-4, 0.37, 1 and 2.5 (the NVFP4 weight divisor; a factor on every
//     FP8 row scale and every Q4..Q8 group scale);
//   - block scales vary from block to block and span more than 2^16 within one tensor: NVFP4 uses
//     every E4M3 exponent including the subnormal words 2^-9..7*2^-9 up to 448; FP8 row scales
//     and Q group scales step through 2^-9..2^8 times the tensor scale (FP16 subnormals included);
//   - N = 640 and K = 832 are not multiples of 512, and M takes 1, 3, 127 and 129.
//
// The reference is the FP64 product of the activation and the weight decoded from the stored words
// (scales from the payload, the divisor as stored); the bound is the activation path's Linear
// criterion. Every output row must be written (no row of zeros where the reference has none).
// Control, every case: the same output against the reference with the tensor scale applied twice
// (NVFP4: the divisor doubled) must fail the bound.

#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::test;
using namespace ninferno::test::linear;
namespace qw = ninferno::test::quantized_weight;

constexpr std::int32_t kN        = 640; // 5 x 128, not a multiple of 512
constexpr std::int32_t kK        = 832; // 13 x 64, not a multiple of 512 (nor of 128)
constexpr std::int32_t kTokens[] = {1, 3, 127, 129};
constexpr float kTensorScales[]  = {3.0e-4F, 0.37F, 1.0F, 2.5F};

std::uint64_t mix(std::uint64_t v) {
    v += 0x9e3779b97f4a7c15ULL;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    return v ^ (v >> 31);
}

// Every positive finite E4M3 word from 0x01 (2^-9, subnormal) to 0x7e (448), chosen per block.
std::uint8_t e4m3_word(std::int32_t row, std::int32_t group) {
    return static_cast<std::uint8_t>(
        1 + mix((static_cast<std::uint64_t>(row) << 20) ^ static_cast<std::uint64_t>(group)) % 126);
}

struct Stats {
    double min_scale = INFINITY, max_scale = 0.0;
};

// Rewrites the block/group/row scales of a patterned weight for tensor scale `s`.
qw::PackedWeight scaled_weight(QType qtype, float s, Stats& stats) {
    if (qtype == QType::NVFP4) {
        qw::PatternedWeightOptions options;
        options.weight_scale_divisor = s;
        options.input_scale_divisor  = 3.5F;
        auto packed                = qw::make_patterned_weight(QType::NVFP4, kN, kK, 901U, options);
        const std::int32_t k_tiles = kK / 64;
        for (std::int32_t row = 0; row < kN; ++row) {
            for (std::int32_t group = 0; group < kK / 16; ++group) {
                const std::size_t at =
                    packed.scale_plane_offset +
                    static_cast<std::size_t>((row / 128) * k_tiles + group / 4) * 512U +
                    static_cast<std::size_t>((row % 128) % 32) * 16U +
                    static_cast<std::size_t>((row % 128) / 32) * 4U +
                    static_cast<std::size_t>(group % 4);
                const std::uint8_t word = row == 0 && group == 0   ? 0x01
                                          : row == 0 && group == 1 ? 0x7e
                                                                   : e4m3_word(row, group);
                packed.payload[at]      = word;
                const double value      = qw::detail::decode_e4m3fn(word);
                stats.min_scale         = std::min(stats.min_scale, value);
                stats.max_scale         = std::max(stats.max_scale, value);
            }
        }
        return packed;
    }
    auto packed = qw::make_patterned_weight(
        qtype, kN, kK, 903U,
        qtype == QType::FP8_E4M3FN_ROW_BF16
            ? qw::PatternedWeightOptions{}
            : qw::PatternedWeightOptions{qw::RowSplitScalePattern::Unit,
                                         qw::RowSplitCodePattern::Hashed});
    if (qtype == QType::FP8_E4M3FN_ROW_BF16) {
        for (std::int32_t row = 0; row < kN; ++row) {
            const int exponent = -9 + static_cast<int>(mix(static_cast<std::uint64_t>(row)) % 18);
            const float value  = std::ldexp(s, exponent);
            const std::uint16_t bits = f32_to_bf16(value);
            qw::detail::store_u16_le(packed.payload,
                                     packed.scale_plane_offset + static_cast<std::size_t>(row) * 2,
                                     bits);
            stats.min_scale = std::min(stats.min_scale, static_cast<double>(bf16_to_f32(bits)));
            stats.max_scale = std::max(stats.max_scale, static_cast<double>(bf16_to_f32(bits)));
        }
        return packed;
    }
    const auto spec            = qw::detail::quant_spec(qtype);
    const std::int32_t groups  = packed.weight.padded_shape[1] / spec.group_size;
    const std::int32_t logical = (kK + spec.group_size - 1) / spec.group_size;
    for (std::int32_t row = 0; row < kN; ++row) {
        for (std::int32_t group = 0; group < logical; ++group) {
            const int exponent =
                -9 + static_cast<int>(mix((static_cast<std::uint64_t>(row) << 16) ^ group) % 18);
            const std::uint16_t bits = qw::detail::f32_to_f16(std::ldexp(s, exponent));
            qw::detail::store_u16_le(packed.payload,
                                     packed.scale_plane_offset +
                                         (static_cast<std::size_t>(row) * groups + group) *
                                             sizeof(std::uint16_t),
                                     bits);
            const double value = qw::detail::f16_to_f32(bits);
            if (value > 0) { stats.min_scale = std::min(stats.min_scale, value); }
            stats.max_scale = std::max(stats.max_scale, value);
        }
    }
    return packed;
}

struct Path {
    const char* name;
    QType qtype;
    ops::LinearPolicy policy;
    ActivationCompute compute;
};

int run_path(const Path& path) {
    int failures = 0;
    for (const float s : kTensorScales) {
        Stats stats;
        const qw::PackedWeight host = scaled_weight(path.qtype, s, stats);
        std::vector<std::int32_t> rows(kN);
        std::iota(rows.begin(), rows.end(), 0);
        const std::vector<float> dequant = qw::materialize_rows_fp32(host, rows);
        DeviceBuffer device_weight(host.payload.size());
        device_weight.copy_from_host(host.payload.data(), host.payload.size());
        const Weight weight = host.device_weight(device_weight.p);
        for (const std::int32_t t : kTokens) {
            const auto bits =
                make_linear_activation(kK, t, 907U + static_cast<std::uint32_t>(t), path.compute);
            std::vector<float> activation(bits.size());
            for (std::size_t i = 0; i < bits.size(); ++i) { activation[i] = bf16_to_f32(bits[i]); }
            std::vector<double> reference(static_cast<std::size_t>(kN) * t);
            cpu_linear_gemm_fp64(dequant.data(), activation.data(), reference.data(), kN, kK, t);

            DeviceBuffer device_activation = to_device(bits);
            GuardedDeviceBuffer output(static_cast<std::size_t>(kN) * t * sizeof(std::uint16_t));
            output.fill(0xff);
            Tensor x(device_activation.p, DType::BF16, {kK, t});
            Tensor out(output.data(), DType::BF16, {kN, t});
            const std::size_t capacity =
                ops::linear_workspace_capacity_bytes(path.qtype, kN, kK, path.policy, t, t);
            DeviceArena workspace(std::max<std::size_t>(capacity, 256));
            const std::string label =
                std::string(path.name) + " s=" + std::to_string(s) + " M=" + std::to_string(t);
            try {
                ops::linear(x, weight, out, path.policy, workspace, nullptr);
                cuda_check(cudaDeviceSynchronize(), "linear");
            } catch (const std::exception& error) {
                std::cerr << label << ": " << error.what() << '\n';
                ++failures;
                continue;
            }
            const std::vector<double> got      = from_device_bf16(output.data(), reference.size());
            const ReductionCriterion criterion = linear_criterion(path.compute);
            failures += verify_reduction(label, got, reference, criterion);
            failures += output.verify_guards(label);
            // Every row written: no all-zero (or NaN) row where the reference is nonzero.
            int bad_rows = 0;
            for (std::int32_t row = 0; row < kN; ++row) {
                bool zero = true, reference_zero = true;
                for (std::int32_t token = 0; token < t; ++token) {
                    const std::size_t at = static_cast<std::size_t>(token) * kN + row;
                    zero                 = zero && got[at] == 0.0;
                    reference_zero       = reference_zero && reference[at] == 0.0;
                    if (!std::isfinite(got[at])) { zero = true; }
                }
                bad_rows += zero && !reference_zero ? 1 : 0;
            }
            failures += bad_rows != 0 ? 1 : 0;
            // Control: the tensor scale applied twice. For NVFP4 the stored divisor divides, so
            // doubling it halves the reference; elsewhere the scale multiplies.
            std::vector<double> doubled(reference);
            const double factor = path.qtype == QType::NVFP4 ? 0.5 : 2.0;
            for (double& value : doubled) { value *= factor; }
            const ReductionStats control = compute_reduction_stats(
                got.data(), doubled.data(), static_cast<std::int64_t>(got.size()));
            const bool control_passes =
                reduction_passes(control, static_cast<std::int64_t>(got.size()), criterion);
            const ReductionStats direct = compute_reduction_stats(
                got.data(), reference.data(), static_cast<std::int64_t>(got.size()));
            std::printf(
                "CASE %-26s rel_l2=%.3e (limit %.1e) zero_rows=%d scales %.3g..%.3g (x%.3g) "
                "| CONTROL doubled scale rel_l2=%.3e %s\n",
                label.c_str(), direct.relative_l2, criterion.relative_l2, bad_rows, stats.min_scale,
                stats.max_scale, stats.max_scale / stats.min_scale, control.relative_l2,
                control_passes ? "PASSED - blind" : "fails as it must");
            failures += control_passes ? 1 : 0;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const Path paths[] = {
        {"NVFP4 W4A16", QType::NVFP4, ops::LinearPolicy::A16Only, ActivationCompute::A16},
        {"NVFP4 W4A4", QType::NVFP4, ops::LinearPolicy::AllowA4, ActivationCompute::A4},
        {"FP8 row A16", QType::FP8_E4M3FN_ROW_BF16, ops::LinearPolicy::A16Only,
         ActivationCompute::A16},
        {"FP8 row A8", QType::FP8_E4M3FN_ROW_BF16, ops::LinearPolicy::AllowA8,
         ActivationCompute::A8},
        {"Q4 g64", QType::Q4_G64_FP16, ops::LinearPolicy::A16Only, ActivationCompute::A16},
        {"Q5 g64", QType::Q5_G64_FP16, ops::LinearPolicy::A16Only, ActivationCompute::A16},
        {"Q6 g64", QType::Q6_G64_FP16, ops::LinearPolicy::A16Only, ActivationCompute::A16},
        {"Q8 g32", QType::Q8_G32_FP16, ops::LinearPolicy::A16Only, ActivationCompute::A16},
    };
    int failures = 0;
    try {
        for (const Path& path : paths) { failures += run_path(path); }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " Linear scale domain\n";
    return failures == 0 ? 0 : 1;
}
