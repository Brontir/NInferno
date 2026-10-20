#include "core/device.h"
#include "ninferno/ops/gemma4_vision.h"
#include "ops/op_tester.h"
#include "ops/oracle.h"

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

// One BF16 output rounding from an FP32 interior (8-bit significand: 3.9e-3 relative); the
// absolute floor covers rotated values that cancel toward zero.
const PointwiseCriterion kCriterion{1.0e-2, 8.0e-3};

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    std::transform(values.begin(), values.end(), out.begin(), f32_to_bf16);
    return out;
}

template <class T>
GuardedDeviceBuffer upload(const std::vector<T>& host) {
    GuardedDeviceBuffer device(host.size() * sizeof(T));
    device.copy_from_host(host.data(), device.bytes());
    return device;
}

std::vector<double> download_bf16(const GuardedDeviceBuffer& device, std::size_t count) {
    const auto raw = from_device<std::uint16_t>(device.data(), count);
    std::vector<double> out(count);
    std::transform(raw.begin(), raw.end(), out.begin(),
                   [](std::uint16_t w) { return static_cast<double>(bf16_to_f32(w)); });
    return out;
}

double round_bf16(double value) { return bf16_to_f32(f32_to_bf16(static_cast<float>(value))); }

// The tower's 31B geometry (D72, 16 heads) and a small one, on one packed [q; k; v] projection
// output whose token stride is the three planes; positions on a patch grid.
int run_qkv(const std::string& label, int d, int heads, int tokens, int grid_width,
            float q_scale) {
    const int rows    = d * heads;
    const auto packed = random_bf16_values(std::size_t(3) * rows * tokens, 11 + d, -3.0f, 3.0f);
    const auto qw     = random_bf16_values(d, 13 + d, 0.5f, 1.5f);
    const auto kw     = random_bf16_values(d, 17 + d, 0.5f, 1.5f);
    std::vector<std::int32_t> pos(std::size_t(2) * tokens);
    for (int t = 0; t < tokens; ++t) {
        pos[2 * t]     = t % grid_width;
        pos[2 * t + 1] = t / grid_width;
    }
    auto d_packed = upload(bits(packed));
    auto d_qw = upload(bits(qw)), d_kw = upload(bits(kw));
    auto d_pos = upload(pos);
    const auto plane = std::size_t(rows) * tokens;
    GuardedDeviceBuffer d_q(plane * 2), d_k(plane * 2), d_v(plane * 2);
    Tensor all(d_packed.data(), DType::BF16, {3 * rows, tokens});
    const Tensor q_src = all.slice(0, 0, rows), k_src = all.slice(0, rows, rows),
                 v_src = all.slice(0, 2 * rows, rows);
    Tensor tpos(d_pos.data(), DType::I32, {2, tokens});
    Tensor tqw(d_qw.data(), DType::BF16, {d}), tkw(d_kw.data(), DType::BF16, {d});
    Tensor q(d_q.data(), DType::BF16, {d, heads, tokens}), k(d_k.data(), DType::BF16, {d, heads, tokens}),
        v(d_v.data(), DType::BF16, {d, heads, tokens});
    const double eps = 1.0e-6, theta = 100.0;
    ops::gemma4_vision_qkv_prepare(q_src, k_src, v_src, d, tpos, tqw, tkw, float(eps),
                                   float(theta), q_scale, q, k, v, nullptr);
    cuda_synchronize();
    std::vector<double> ref_q(plane), ref_k(plane), ref_v(plane), swapped_k(plane);
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            for (int which = 0; which < 3; ++which) {
                const float* x = packed.data() + std::size_t(t) * 3 * rows +
                                 std::size_t(which) * rows + std::size_t(h) * d;
                double ss = 0;
                for (int i = 0; i < d; ++i) { ss += double(x[i]) * x[i]; }
                const double inv = 1.0 / std::sqrt(ss / d + eps);
                const auto at    = std::size_t(t) * rows + std::size_t(h) * d;
                if (which == 2) {
                    for (int i = 0; i < d; ++i) { ref_v[at + i] = x[i] * inv; }
                    continue;
                }
                const auto& w = which == 0 ? qw : kw;
                for (int axes = 0; axes < 2; ++axes) {
                    auto& dst = which == 0 ? ref_q : (axes == 0 ? ref_k : swapped_k);
                    if (which == 0 && axes == 1) { continue; }
                    for (int i = 0; i < d / 2; ++i) {
                        const double a = x[i] * inv * w[i];
                        const double b = x[i + d / 2] * inv * w[i + d / 2];
                        int axis       = i < d / 4 ? 0 : 1;
                        if (axes == 1) { axis = 1 - axis; }
                        const int j      = i < d / 4 ? i : i - d / 4;
                        const double phi = pos[2 * t + axis] * std::pow(theta, -4.0 * j / d);
                        const double s = which == 0 ? q_scale : 1.0;
                        dst[at + i]         = (a * std::cos(phi) - b * std::sin(phi)) * s;
                        dst[at + i + d / 2] = (b * std::cos(phi) + a * std::sin(phi)) * s;
                    }
                }
            }
        }
    }
    int failures = verify_pointwise(label + " q", download_bf16(d_q, plane), ref_q, kCriterion);
    const auto got_k = download_bf16(d_k, plane);
    failures += verify_pointwise(label + " k", got_k, ref_k, kCriterion);
    failures += verify_pointwise(label + " v", download_bf16(d_v, plane), ref_v, kCriterion);
    failures += d_q.verify_guards(label + " q guards") + d_k.verify_guards(label + " k guards") +
                d_v.verify_guards(label + " v guards");
    const double distance = max_abs_difference(got_k, swapped_k);
    std::cout << label << " control: k vs the axes swapped max_abs=" << distance << '\n';
    // One token sits at (0, 0): both axes rotate by zero, nothing to tell apart.
    if (tokens > 1 && distance < 0.25) {
        std::cerr << label << " control: the x and y axes are indistinguishable\n";
        ++failures;
    }
    return failures;
}

int run_pool(const std::string& label, int hidden, int width, int rows, int kernel) {
    const int patches = width * rows;
    const int out_w = width / kernel, tokens = out_w * (rows / kernel);
    const auto x     = random_bf16_values(std::size_t(hidden) * patches, 31 + width, -2.0f, 2.0f);
    const auto bias  = random_bf16_values(hidden, 37, -5.0f, 5.0f);
    const auto scale = random_bf16_values(hidden, 41, 0.05f, 0.5f);
    auto d_x = upload(bits(x)), d_b = upload(bits(bias)), d_s = upload(bits(scale));
    GuardedDeviceBuffer d_out(std::size_t(hidden) * tokens * 2);
    Tensor tx(d_x.data(), DType::BF16, {hidden, patches});
    Tensor tb(d_b.data(), DType::BF16, {hidden}), ts(d_s.data(), DType::BF16, {hidden});
    Tensor out(d_out.data(), DType::BF16, {hidden, tokens});
    const double root = std::sqrt(double(hidden)), eps = 1.0e-6;
    ops::gemma4_vision_pool(tx, width, kernel, tb, ts, float(root), float(eps), out, nullptr);
    cuda_synchronize();
    std::vector<double> ref(std::size_t(hidden) * tokens), transposed(ref.size());
    for (int pass = 0; pass < 2; ++pass) {
        auto& dst = pass == 0 ? ref : transposed;
        for (int n = 0; n < tokens; ++n) {
            // pass 1: the grid read column-major (a wrong raster order)
            const int ny = pass == 0 ? n / out_w : n % (rows / kernel);
            const int nx = pass == 0 ? n % out_w : n / (rows / kernel);
            std::vector<double> s(hidden);
            double ss = 0;
            for (int dd = 0; dd < hidden; ++dd) {
                double acc = 0;
                for (int dy = 0; dy < kernel; ++dy) {
                    for (int dx = 0; dx < kernel; ++dx) {
                        const auto p = std::size_t(kernel * ny + dy) * width + kernel * nx + dx;
                        acc += x[p * hidden + dd];
                    }
                }
                const double m = round_bf16(acc / (kernel * kernel));
                s[dd]          = round_bf16((m * float(root) - bias[dd]) * scale[dd]);
                ss += s[dd] * s[dd];
            }
            const double inv = 1.0 / std::sqrt(ss / hidden + eps);
            for (int dd = 0; dd < hidden; ++dd) {
                dst[std::size_t(n) * hidden + dd] = s[dd] * inv;
            }
        }
    }
    const auto got = download_bf16(d_out, ref.size());
    int failures   = verify_pointwise(label, got, ref, kCriterion);
    failures += d_out.verify_guards(label + " guards");
    if (width != rows) {
        const double distance = max_abs_difference(got, transposed);
        std::cout << label << " control: output vs a column-major grid max_abs=" << distance
                  << '\n';
        if (distance < 0.25) {
            std::cerr << label << " control: the raster order is not what was read\n";
            ++failures;
        }
    }
    return failures;
}

template <class F>
int expect_invalid(const std::string& label, F f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": no invalid_argument\n";
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_qkv("qkv 31B tower D72 16 heads", 72, 16, 37, 9, std::sqrt(72.0f));
    failures += run_qkv("qkv D72 one token", 72, 16, 1, 1, 1.0f);
    failures += run_qkv("qkv D64 4 heads", 64, 4, 130, 13, 1.0f);
    failures += run_pool("pool 31B 1152 12x9", 1152, 12, 9, 3);
    failures += run_pool("pool 1152 48x48", 1152, 48, 48, 3);
    failures += run_pool("pool odd width 33x6 k3", 72, 33, 6, 3);
    failures += run_pool("pool k1", 64, 4, 2, 1);
    {
        GuardedDeviceBuffer buf(1 << 16);
        Tensor x(buf.data(), DType::BF16, {72, 9});
        Tensor o(buf.data(), DType::BF16, {72, 1});
        failures += expect_invalid("pool width not a multiple of k", [&] {
            ops::gemma4_vision_pool(x, 2, 3, o, o, 1.0f, 1e-6f, o, nullptr);
        });
        Tensor src(buf.data(), DType::BF16, {70, 2});
        Tensor out(buf.data(), DType::BF16, {70, 1, 2});
        Tensor pos(buf.data(), DType::I32, {2, 2});
        failures += expect_invalid("qkv D70", [&] {
            ops::gemma4_vision_qkv_prepare(src, src, src, 70, pos, out, out, 1e-6f, 100.0f, 1.0f,
                                           out, out, out, nullptr);
        });
    }
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "gemma4_vision: ok\n";
    return 0;
}
