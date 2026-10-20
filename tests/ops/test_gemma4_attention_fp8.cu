// gemma4_attention and gemma4_kv_append over FP8 planes (Gemma4KvLayer's FP8 storage: E4M3 codes
// with an FP32 scale per row).
//
// Every call appends its history and its own rows through gemma4_kv_append, attends, and reads
// the planes back. Two checks:
//   - the codec: every stored row against the BF16 row it came from (scale = absmax / 448 and
//     each value within half an E4M3 step of the code times the scale);
//   - the attention: against the FP64 formula of include/ninferno/ops/gemma4_attention.h over the
//     values the kernel stages, BF16(FP32(code) * scale) from the codes and scales read back,
//     under the BF16 op criterion (a BF16 cache is checked over its BF16 rows the same way).
// Geometries cover the four head widths, full and sliding layers, power-of-two and runtime GQA
// groups; T covers the decode route (T <= 16) and the prefill route, calls on and off the
// position grid.

#include "core/device.h"
#include "ninferno/ops/gemma4_attention.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

constexpr double kUnit     = 1.0 / 256.0;
constexpr double kSlack    = 17.0 / 16.0;
constexpr double kRelL2Max = kUnit;

struct Geometry {
    const char* label;
    int head_dim;
    int q_heads;
    int kv_heads;
    int window;
};

constexpr Geometry kGeometries[] = {
    {"D512 16q/2kv full", 512, 16, 2, 0},
    {"D512 12q/2kv full (group 6)", 512, 12, 2, 0},
    {"D256 16q/8kv W1024", 256, 16, 8, 1024},
    {"D256 12q/4kv W1024 (group 3)", 256, 12, 4, 1024},
    {"D128 32q/8kv full", 128, 32, 8, 0},
    {"D128 24q/8kv W512 (group 3)", 128, 24, 8, 512},
    {"D64 64q/8kv W128", 64, 64, 8, 128},
    {"D64 32q/8kv full", 64, 32, 8, 0},
};

struct Call {
    int tokens;
    int base;
};

constexpr Call kCalls[] = {{1, 0}, {1, 777}, {5, 1234}, {16, 2000}, {37, 0}, {300, 1100}, {129, 3001}};

float bf16_round(float value) { return bf16_to_f32(f32_to_bf16(value)); }

double e4m3_value(std::uint8_t code) {
    __nv_fp8_e4m3 v;
    v.__x = code;
    return static_cast<double>(static_cast<float>(v));
}

int run(const Geometry& g, const Call& call) {
    const int D = g.head_dim, Hq = g.q_heads, Hkv = g.kv_heads, W = g.window, T = call.tokens;
    const int p0 = call.base, top = p0 + T;
    const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(top), T};
    const int capacity = W > 0 ? W + T - 1 : top;
    const int pos_lo   = W > 0 ? std::max(0, p0 - capacity) : 0;
    const int G        = Hq / Hkv;
    std::mt19937 rng(static_cast<std::uint32_t>(D * 131 + Hq * 17 + T * 7 + p0));
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);

    // Rows of positions [pos_lo, top): K near unit scale with an occasional outlier feature, V
    // with a per-(head, feature) bias.
    const int rows = top - pos_lo;
    std::vector<float> k(static_cast<std::size_t>(rows) * Hkv * D), v(k.size());
    std::vector<float> bias(static_cast<std::size_t>(Hkv) * D);
    for (float& b : bias) { b = 0.5F * unit(rng); }
    for (std::size_t i = 0; i < k.size(); ++i) {
        const float outlier = (i % 97 == 5) ? 12.0F : 1.0F;
        k[i] = bf16_round(outlier * unit(rng));
        v[i] = bf16_round(bias[i % bias.size()] + unit(rng));
    }
    std::vector<float> q(static_cast<std::size_t>(T) * Hq * D);
    const float q_amp = 6.0F / std::sqrt(static_cast<float>(D));
    for (float& x : q) { x = bf16_round(q_amp * unit(rng)); }

    DeviceBuffer ck(static_cast<std::size_t>(D) * capacity * Hkv);
    DeviceBuffer cv(ck.bytes);
    DeviceBuffer ks(sizeof(float) * capacity * Hkv), vs(ks.bytes);
    CUDA_CHECK(cudaMemset(ck.p, 0, ck.bytes));
    CUDA_CHECK(cudaMemset(cv.p, 0, cv.bytes));
    CUDA_CHECK(cudaMemset(ks.p, 0, ks.bytes));
    CUDA_CHECK(cudaMemset(vs.p, 0, vs.bytes));
    const ops::Gemma4KvLayer layer{
        .k       = Tensor(ck.p, DType::FP8_E4M3FN, {D, capacity, Hkv}),
        .v       = Tensor(cv.p, DType::FP8_E4M3FN, {D, capacity, Hkv}),
        .window  = W,
        .k_scale = Tensor(ks.p, DType::FP32, {capacity, Hkv}),
        .v_scale = Tensor(vs.p, DType::FP32, {capacity, Hkv}),
    };
    // Appends positions [from, to) in chunks the ring holds.
    const auto append = [&](int from, int to) {
        const int chunk = std::min(capacity, 256);
        for (int a = from; a < to; a += chunk) {
            const int n = std::min(chunk, to - a);
            std::vector<std::uint16_t> kb(static_cast<std::size_t>(n) * Hkv * D), vb(kb.size());
            std::vector<int> pos(static_cast<std::size_t>(n));
            for (int t = 0; t < n; ++t) {
                pos[static_cast<std::size_t>(t)] = a + t;
                for (std::size_t i = 0; i < static_cast<std::size_t>(Hkv) * D; ++i) {
                    const std::size_t src = static_cast<std::size_t>(a + t - pos_lo) * Hkv * D + i;
                    kb[static_cast<std::size_t>(t) * Hkv * D + i] = f32_to_bf16(k[src]);
                    vb[static_cast<std::size_t>(t) * Hkv * D + i] = f32_to_bf16(v[src]);
                }
            }
            DeviceBuffer dk = to_device(kb), dv = to_device(vb), dp = to_device(pos);
            ops::gemma4_kv_append(Tensor(dk.p, DType::BF16, {D, Hkv, n}),
                                  Tensor(dv.p, DType::BF16, {D, Hkv, n}),
                                  Tensor(dp.p, DType::I32, {n}), layer, nullptr);
            cuda_synchronize();
        }
    };
    append(pos_lo, p0);
    append(p0, top); // the call's own rows (one append of T <= capacity rows)

    std::vector<std::uint16_t> qb(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) { qb[i] = f32_to_bf16(q[i]); }
    std::vector<int> positions(static_cast<std::size_t>(T));
    for (int t = 0; t < T; ++t) { positions[static_cast<std::size_t>(t)] = p0 + t; }
    DeviceBuffer dq = to_device(qb), dpos = to_device(positions);
    DeviceBuffer dout(static_cast<std::size_t>(T) * Hq * D * 2);
    WorkspaceArena workspace(
        std::max<std::size_t>(1, ops::gemma4_attention_workspace_bytes(D, Hq, Hkv, envelope)));
    Tensor out(dout.p, DType::BF16, {D, Hq, T});
    ops::gemma4_attention(Tensor(dq.p, DType::BF16, {D, Hq, T}), Tensor(dpos.p, DType::I32, {T}),
                          layer, 1.0F, envelope, workspace, out, nullptr);
    cuda_synchronize();

    // Represented rows of the positions the ring holds, and the codec check against the source.
    const auto codes_k  = from_device<std::uint8_t>(ck, ck.bytes);
    const auto codes_v  = from_device<std::uint8_t>(cv, cv.bytes);
    const auto scales_k = from_device<float>(ks, static_cast<std::size_t>(capacity) * Hkv);
    const auto scales_v = from_device<float>(vs, static_cast<std::size_t>(capacity) * Hkv);
    const int held_lo   = std::max(pos_lo, top - capacity);
    std::vector<double> rk(static_cast<std::size_t>(rows) * Hkv * D), rv(rk.size());
    int codec_bad = 0;
    for (int p = held_lo; p < top; ++p) {
        for (int h = 0; h < Hkv; ++h) {
            const std::size_t row  = static_cast<std::size_t>(h) * capacity + p % capacity;
            const std::size_t src  = (static_cast<std::size_t>(p - pos_lo) * Hkv + h) * D;
            for (int plane = 0; plane < 2; ++plane) {
                const auto& codes  = plane == 0 ? codes_k : codes_v;
                const float scale  = (plane == 0 ? scales_k : scales_v)[row];
                const auto& orig   = plane == 0 ? k : v;
                auto& represented  = plane == 0 ? rk : rv;
                double absmax      = 0.0;
                for (int d = 0; d < D; ++d) { absmax = std::max(absmax, std::abs(double(orig[src + d]))); }
                if (std::abs(double(scale) - absmax / 448.0) > 1e-6 * absmax / 448.0 + 1e-30) {
                    if (codec_bad++ < 4) {
                        std::printf("    scale of p=%d h=%d plane %d: %.9g, expected %.9g\n", p, h,
                                    plane, double(scale), absmax / 448.0);
                    }
                }
                for (int d = 0; d < D; ++d) {
                    const double value = double(scale) * e4m3_value(codes[row * D + d]);
                    // The kernel stages BF16(FP32(code) * scale) into its BF16 tile, as the BF16
                    // cache holds BF16 rows: the oracle attends over exactly those values, and
                    // the BF16 profile's criterion applies unchanged.
                    represented[src + d] =
                        bf16_round(static_cast<float>(e4m3_value(codes[row * D + d])) * scale);
                    const double x       = orig[src + d];
                    const double limit   = std::max(std::abs(x) * 0x1p-4, double(scale) * 0x1p-9) * 1.0001;
                    if (!(std::abs(value - x) <= limit)) {
                        if (codec_bad++ < 4) {
                            std::printf("    code of p=%d h=%d d=%d plane %d: %.6g for %.6g\n", p,
                                        h, d, plane, value, x);
                        }
                    }
                }
            }
        }
    }

    // FP64 oracle over the staged values.
    const auto got   = from_device_bf16(dout, static_cast<std::size_t>(T) * Hq * D);
    double v_max     = 0.0;
    for (const double x : rv) { v_max = std::max(v_max, std::abs(x)); }
    long double err_sq = 0.0L, ideal_sq = 0.0L;
    double worst       = 0.0;
    std::int64_t bad = 0, non_finite = 0;
    std::vector<double> weights, sum(static_cast<std::size_t>(D));
    for (int t = 0; t < T; ++t) {
        const int p    = p0 + t;
        const int j_lo = W > 0 ? std::max(0, p - W + 1) : 0;
        for (int head = 0; head < Hq; ++head) {
            const int kvh = head / G;
            const float* qr = q.data() + (static_cast<std::size_t>(t) * Hq + head) * D;
            weights.assign(static_cast<std::size_t>(p - j_lo + 1), 0.0);
            double maximum = -1e300;
            for (int j = j_lo; j <= p; ++j) {
                const double* kr = rk.data() + (static_cast<std::size_t>(j - pos_lo) * Hkv + kvh) * D;
                double dot = 0.0;
                for (int d = 0; d < D; ++d) { dot += double(qr[d]) * kr[d]; }
                weights[static_cast<std::size_t>(j - j_lo)] = dot;
                maximum = std::max(maximum, dot);
            }
            double denominator = 0.0;
            for (double& w : weights) {
                w = std::exp(w - maximum);
                denominator += w;
            }
            std::fill(sum.begin(), sum.end(), 0.0);
            for (int j = j_lo; j <= p; ++j) {
                const double* vr = rv.data() + (static_cast<std::size_t>(j - pos_lo) * Hkv + kvh) * D;
                const double w   = weights[static_cast<std::size_t>(j - j_lo)];
                for (int d = 0; d < D; ++d) { sum[static_cast<std::size_t>(d)] += w * vr[d]; }
            }
            for (int d = 0; d < D; ++d) {
                const double ideal = sum[static_cast<std::size_t>(d)] / denominator;
                const double value = got[(static_cast<std::size_t>(t) * Hq + head) * D + d];
                if (!std::isfinite(value)) {
                    ++non_finite;
                    continue;
                }
                const double error = std::abs(value - ideal);
                const double limit = kSlack * kUnit * (v_max + std::abs(ideal));
                worst              = std::max(worst, error / limit);
                bad += error > limit ? 1 : 0;
                err_sq += static_cast<long double>(error) * error;
                ideal_sq += static_cast<long double>(ideal) * ideal;
            }
        }
    }
    const double rel_l2 = std::sqrt(double(err_sq)) / std::max(std::sqrt(double(ideal_sq)), 1e-300);
    const bool pass     = bad == 0 && non_finite == 0 && rel_l2 <= kRelL2Max && codec_bad == 0;
    std::printf("  %-32s T=%-4d base=%-5d C=%-5d worst=%.3f rel_l2=%.2e codec_bad=%d%s\n", g.label,
                T, p0, capacity, worst, rel_l2, codec_bad, pass ? "" : "  FAIL");
    return pass ? 0 : 1;
}

} // namespace

int main() {
    if (ninferno::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        for (const Geometry& g : kGeometries) {
            for (const Call& c : kCalls) { failures += run(g, c); }
        }
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << '\n';
        return 1;
    }
    std::printf("gemma4_attention fp8: %s (%d failures)\n", failures == 0 ? "passed" : "FAILED",
                failures);
    return failures == 0 ? 0 : 1;
}
