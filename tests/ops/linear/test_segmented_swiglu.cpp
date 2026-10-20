// segmented_swiglu (ninferno/ops/segmented_projection.h): the SwiGLU / GeGLU of a gate and an up
// weight stored apart (a GGUF layer's ffn_gate and ffn_up, often of two formats). At decode and
// verify widths where both halves take ggml's vector kernel, x is quantized once and the
// activation is the up product's epilogue (generic::ggml_vec_gated_pair: one gated launch for a
// pair of one format, the gate's FP32 plane and a GateProduct epilogue for two); elsewhere the
// projection and a separate activation kernel. Every pair below, of one format and of two, against
// the FP64 oracle of the unquantized x under the A8 allowance; and, wherever the pair path runs at
// two widths, each token's bytes equal at both (the vector kernel sums a token alone).
#include "ninferno/ops/linear.h"
#include "ninferno/ops/segmented_projection.h"

#include "core/arena.h"
#include "ops/generic/generic_gemm.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;
using namespace ninferno::test::linear;

namespace {

namespace generic = ops::detail::generic;

struct Pair {
    QType gate;
    QType up;
};

const char* name(QType q) {
    switch (q) {
    case QType::GGML_Q2_K:
        return "q2_k";
    case QType::GGML_Q4_K:
        return "q4_k";
    case QType::GGML_Q5_K:
        return "q5_k";
    case QType::GGML_Q6_K:
        return "q6_k";
    case QType::GGML_IQ4_XS:
        return "iq4_xs";
    case QType::GGML_IQ3_XXS:
        return "iq3_xxs";
    case QType::GGML_IQ3_S:
        return "iq3_s";
    case QType::GGML_Q3_K:
        return "q3_k";
    case QType::GGML_IQ2_XXS:
        return "iq2_xxs";
    case QType::GGML_IQ2_XS:
        return "iq2_xs";
    case QType::GGML_IQ2_S:
        return "iq2_s";
    case QType::GGML_IQ1_S:
        return "iq1_s";
    case QType::GGML_IQ1_M:
        return "iq1_m";
    case QType::GGML_Q4_0:
        return "q4_0";
    case QType::GGML_Q4_1:
        return "q4_1";
    case QType::GGML_Q5_0:
        return "q5_0";
    case QType::GGML_Q5_1:
        return "q5_1";
    case QType::GGML_IQ4_NL:
        return "iq4_nl";
    default:
        return "?";
    }
}

struct Result {
    int failures = 0;
    bool paired  = false;
    std::vector<std::uint16_t> bits; // [rows, T]
};

Result check(const Pair& p, std::int32_t rows, std::int32_t k, std::int32_t t, bool gelu,
             std::uint32_t seed) {
    const auto policy       = ops::LinearPolicy::AllowA8;
    const std::string label = std::string(name(p.gate)) + "/" + name(p.up) +
                              (gelu ? " geglu" : " swiglu") + " [" + std::to_string(rows) + "," +
                              std::to_string(k) + "] T=" + std::to_string(t);
    Result r;
    const auto gate_host = quantized_weight::make_patterned_weight(p.gate, rows, k, seed);
    const auto up_host   = quantized_weight::make_patterned_weight(p.up, rows, k, seed + 1U);
    DeviceBuffer gate_payload(gate_host.payload.size()), up_payload(up_host.payload.size());
    gate_payload.copy_from_host(gate_host.payload.data(), gate_payload.bytes);
    up_payload.copy_from_host(up_host.payload.data(), up_payload.bytes);
    const Weight gate = gate_host.device_weight(gate_payload.p);
    const Weight up   = up_host.device_weight(up_payload.p);

    const auto activation = make_linear_activation(k, t, seed + 7U, ActivationCompute::A8);
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    const Tensor tx(x.p, DType::BF16, {k, t});
    DeviceBuffer out(static_cast<std::size_t>(rows) * t * 2);
    cuda_check(cudaMemset(out.p, 0xff, out.bytes), "poison");
    Tensor tout(out.p, DType::BF16, {rows, t});

    ops::SegmentedProjectionWeights weights;
    weights.slices.push_back({gate, 0, rows, policy});
    weights.slices.push_back({up, 0, rows, policy});
    DeviceArena workspace(ops::segmented_swiglu_workspace_capacity_bytes(weights, t, t));
    const auto takes = [&](const Weight& w) {
        return generic::ggml_vec_takes({w.qtype, policy, rows, k, generic::Epilogue::Store}, w, t);
    };
    // Past the vector kernel's widths the pair is ggml's matrix kernel's, where both halves take
    // it.
    const auto takes_mmq = [&](const Weight& w) {
        return generic::ggml_mmq_takes_tuned(tx, w, policy, 0, rows, nullptr);
    };
    // A prefill width takes the pair in slices where the kernel's prefill route holds.
    const auto takes_prefill = [&](const Weight& w) {
        return generic::ggml_mmq_slice_tokens(w.qtype) > generic::kGgmlMmqMaxTokens;
    };
    r.paired = t <= generic::kGgmlVecMaxTokens   ? takes(gate) && takes(up)
               : t <= generic::kGgmlMmqMaxTokens ? takes_mmq(gate) && takes_mmq(up)
                                                 : takes_prefill(gate) && takes_prefill(up);
    ops::segmented_swiglu(tx, weights, gelu, tout, workspace, nullptr);
    cuda_check(cudaDeviceSynchronize(), "run");
    r.bits = from_device<std::uint16_t>(out, static_cast<std::size_t>(rows) * t);

    std::vector<std::int32_t> all(static_cast<std::size_t>(rows));
    std::iota(all.begin(), all.end(), 0);
    const auto wg = quantized_weight::materialize_rows_fp32(gate_host, all);
    const auto wu = quantized_weight::materialize_rows_fp32(up_host, all);
    std::vector<float> xa(activation.size());
    for (std::size_t i = 0; i < activation.size(); ++i) xa[i] = bf16_to_f32(activation[i]);
    std::vector<double> g(static_cast<std::size_t>(rows) * t), u(g.size());
    cpu_linear_gemm_fp64(wg.data(), xa.data(), g.data(), rows, k, t);
    cpu_linear_gemm_fp64(wu.data(), xa.data(), u.data(), rows, k, t);
    std::vector<double> got(g.size()), ref(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
        const double v = g[i];
        const double a =
            gelu ? 0.5 * v * (1.0 + std::tanh(0.79788456080286535588 * (v + 0.044715 * v * v * v)))
                 : v / (1.0 + std::exp(-v));
        ref[i] = a * u[i];
        if (r.bits[i] == 0xffffU) {
            std::cerr << label << ": output retains poison\n";
            ++r.failures;
            return r;
        }
        got[i] = bf16_to_f32(r.bits[i]);
    }
    r.failures += verify_reduction(label + (r.paired ? " pair" : " separate"), got, ref,
                                   linear_criterion(ActivationCompute::A8));
    return r;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "SKIP segmented swiglu: no CUDA device\n";
        return 77;
    }
    const std::vector<Pair> pairs = {
        {QType::GGML_Q2_K, QType::GGML_Q2_K},     {QType::GGML_Q4_K, QType::GGML_Q4_K},
        {QType::GGML_Q5_K, QType::GGML_Q5_K},     {QType::GGML_Q6_K, QType::GGML_Q6_K},
        {QType::GGML_IQ4_XS, QType::GGML_IQ4_XS}, {QType::GGML_IQ3_XXS, QType::GGML_IQ3_XXS},
        {QType::GGML_IQ3_S, QType::GGML_IQ3_S},   {QType::GGML_IQ3_XXS, QType::GGML_IQ3_S},
        {QType::GGML_IQ3_S, QType::GGML_IQ4_XS},  {QType::GGML_IQ4_XS, QType::GGML_IQ3_XXS},
        {QType::GGML_Q4_K, QType::GGML_Q6_K},     {QType::GGML_Q6_K, QType::GGML_Q4_K},
        {QType::GGML_Q2_K, QType::GGML_IQ3_XXS},  {QType::GGML_IQ4_XS, QType::GGML_Q5_K},
        {QType::GGML_Q3_K, QType::GGML_Q3_K},     {QType::GGML_IQ2_XXS, QType::GGML_IQ2_XXS},
        {QType::GGML_IQ2_XS, QType::GGML_IQ2_XS}, {QType::GGML_IQ2_S, QType::GGML_IQ2_S},
        {QType::GGML_IQ1_S, QType::GGML_IQ1_S},   {QType::GGML_IQ2_XXS, QType::GGML_IQ2_S},
        {QType::GGML_IQ1_S, QType::GGML_Q3_K},    {QType::GGML_Q3_K, QType::GGML_IQ4_XS},
        {QType::GGML_IQ1_M, QType::GGML_IQ1_M},   {QType::GGML_IQ1_M, QType::GGML_IQ2_XXS},
        {QType::GGML_Q4_0, QType::GGML_Q4_0},     {QType::GGML_Q4_1, QType::GGML_Q4_1},
        {QType::GGML_Q5_0, QType::GGML_Q5_0},     {QType::GGML_Q5_1, QType::GGML_Q5_1},
        {QType::GGML_IQ4_NL, QType::GGML_IQ4_NL}, {QType::GGML_Q4_0, QType::GGML_Q5_1},
        {QType::GGML_IQ4_NL, QType::GGML_Q4_K},   {QType::GGML_Q5_0, QType::GGML_IQ4_XS},
    };
    int failures = 0, paired = 0, invariant = 0, matrix_paired = 0;
    for (const auto& p : pairs) {
        int pair_paired = 0;
        for (const auto& [rows, k] : std::vector<std::pair<std::int32_t, std::int32_t>>{
                 {640, 5120}, {336, 2560}, {200, 1280}, {128, 2880}}) {
            // K 2880 (gpt-oss): whole blocks of the 32-value formats only.
            const auto small = [](QType q) {
                return q == QType::GGML_Q4_0 || q == QType::GGML_Q4_1 || q == QType::GGML_Q5_0 ||
                       q == QType::GGML_Q5_1 || q == QType::GGML_IQ4_NL;
            };
            if (k % 256 != 0 && !(small(p.gate) && small(p.up))) continue;
            for (const bool gelu : {false, true}) {
                if (gelu && p.gate != p.up && p.gate != QType::GGML_IQ3_XXS) continue;
                const std::uint32_t seed = 101U + static_cast<std::uint32_t>(k);
                const Result wide        = check(p, rows, k, 8, gelu, seed);
                failures += wide.failures;
                paired += wide.paired ? 1 : 0;
                pair_paired += wide.paired ? 1 : 0;
                // The matrix kernel's band (whole 128-row tiles): against the FP64 reference only,
                // the tuner picks its route per width.
                for (const std::int32_t t : {9, 12, 16}) {
                    if (rows % 128 != 0) continue;
                    const Result matrix = check(p, rows, k, t, gelu, seed);
                    failures += matrix.failures;
                    matrix_paired += matrix.paired ? 1 : 0;
                }
                // A prefill of two of the matrix kernel's slices (one shape: the FP64
                // reference grows with T).
                if (rows == 640 && k == 5120) {
                    const Result prefill = check(p, 256, 1024, 520, gelu, seed);
                    failures += prefill.failures;
                }
                for (const std::int32_t t : {1, 2, 3, 4, 5, 6, 7}) {
                    const Result narrow = check(p, rows, k, t, gelu, seed);
                    failures += narrow.failures;
                    paired += narrow.paired ? 1 : 0;
                    pair_paired += narrow.paired ? 1 : 0;
                    if (!(narrow.paired && wide.paired)) continue;
                    // The activation of token j is the same at both widths (seeded by column).
                    const auto a8 = make_linear_activation(k, 8, seed + 7U, ActivationCompute::A8);
                    const auto at = make_linear_activation(k, t, seed + 7U, ActivationCompute::A8);
                    if (!std::equal(at.begin(), at.end(), a8.begin())) continue;
                    ++invariant;
                    if (std::memcmp(narrow.bits.data(), wide.bits.data(), narrow.bits.size() * 2) !=
                        0) {
                        std::cerr << "FAIL " << name(p.gate) << "/" << name(p.up) << " [" << rows
                                  << "," << k << "]: T=" << t
                                  << " tokens differ from the same tokens at T=8\n";
                        ++failures;
                    }
                }
            }
        }
        // The tuner may route a width elsewhere, never every width of every shape.
        if (pair_paired == 0) {
            std::cerr << "FAIL " << name(p.gate) << "/" << name(p.up)
                      << ": never took the vector pair\n";
            ++failures;
        }
    }
    std::cout << "segmented swiglu: " << paired << " paired calls, " << matrix_paired
              << " matrix-kernel pairs, " << invariant << " width comparisons\n";
    if (failures != 0) {
        std::cerr << "segmented swiglu: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "segmented swiglu: all cases passed\n";
    return 0;
}
