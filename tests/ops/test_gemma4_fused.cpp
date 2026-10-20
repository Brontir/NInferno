#include "core/device.h"
#include "ninferno/ops/gemma4_fused.h"
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

// BF16 carries eight significand bits (2^-8 = 3.9e-3). Every output here is rounded once from
// an FP32 interior; the relative term carries that rounding and the FP32 reduction, the absolute
// floor carries values that cancel toward zero inside a rotation or a sum of two branches.
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

// The cache-writing form of a prepare against its dense form run on the same positions: the
// queries must be bit-identical, every cache slot a token addresses must hold that token's dense
// K/V row bit for bit, and every other slot must keep its prior bytes. Positions start two slots
// before the ring's end, so the rows wrap.
template <class Dense, class Cached>
int verify_cached_prepare(const std::string& label, int d, int q_heads, int kv_heads, int t_count,
                          Dense dense, Cached cached) {
    const int capacity = t_count + 3;
    std::vector<std::int32_t> ring(t_count);
    for (int t = 0; t < t_count; ++t) { ring[t] = capacity - 2 + t; }
    auto d_ring       = upload(ring);
    const auto plane  = std::size_t(d) * capacity * kv_heads;
    const auto fill_k = bits(random_bf16_values(plane, 71, -1.0f, 1.0f));
    const auto fill_v = bits(random_bf16_values(plane, 73, -1.0f, 1.0f));
    auto d_ck         = upload(fill_k);
    auto d_cv         = upload(fill_v);
    // A context-only form has no queries; its q tensors are placeholders the Op never sees.
    const int q_shape = std::max(q_heads, 1);
    GuardedDeviceBuffer d_q1(std::size_t(q_shape) * d * t_count * 2);
    GuardedDeviceBuffer d_q2(std::size_t(q_shape) * d * t_count * 2);
    GuardedDeviceBuffer d_k(std::size_t(kv_heads) * d * t_count * 2);
    GuardedDeviceBuffer d_v(std::size_t(kv_heads) * d * t_count * 2);
    Tensor pos(d_ring.data(), DType::I32, {t_count});
    Tensor q1(d_q1.data(), DType::BF16, {d, q_shape, t_count});
    Tensor q2(d_q2.data(), DType::BF16, {d, q_shape, t_count});
    Tensor k(d_k.data(), DType::BF16, {d, kv_heads, t_count});
    Tensor v(d_v.data(), DType::BF16, {d, kv_heads, t_count});
    ops::Gemma4KvLayer cache;
    cache.k = Tensor(d_ck.data(), DType::BF16, {d, capacity, kv_heads});
    cache.v = Tensor(d_cv.data(), DType::BF16, {d, capacity, kv_heads});
    dense(pos, q1, k, v);
    cached(pos, q2, cache);
    cuda_synchronize();
    int failures       = 0;
    const auto q_count = std::size_t(q_heads) * d * t_count;
    if (q_heads > 0 && from_device<std::uint16_t>(d_q1.data(), q_count) !=
                           from_device<std::uint16_t>(d_q2.data(), q_count)) {
        std::cerr << label << " cached: queries differ from the dense form\n";
        ++failures;
    }
    const auto dk = from_device<std::uint16_t>(d_k.data(), std::size_t(kv_heads) * d * t_count);
    const auto dv = from_device<std::uint16_t>(d_v.data(), std::size_t(kv_heads) * d * t_count);
    auto want_k   = fill_k;
    auto want_v   = fill_v;
    for (int t = 0; t < t_count; ++t) {
        const int slot = ring[t] % capacity;
        for (int g = 0; g < kv_heads; ++g) {
            const auto src = (std::size_t(t) * kv_heads + g) * d;
            const auto dst = (std::size_t(g) * capacity + slot) * d;
            std::copy_n(dk.begin() + src, d, want_k.begin() + dst);
            std::copy_n(dv.begin() + src, d, want_v.begin() + dst);
        }
    }
    if (from_device<std::uint16_t>(d_ck.data(), plane) != want_k ||
        from_device<std::uint16_t>(d_cv.data(), plane) != want_v) {
        std::cerr << label << " cached: cache planes differ from dense rows + untouched slots\n";
        ++failures;
    }
    for (auto* buffer : {&d_ck, &d_cv, &d_q2}) {
        failures += buffer->verify_guards(label + " cached guards");
    }
    return failures;
}

// ---- Q/K/V preparation ------------------------------------------------------------------------

struct QkvCase {
    std::string label;
    int head_dim;
    int q_heads;
    int kv_heads;
    int tokens;
    int base_position;
    double theta;
    int rotary_pairs;
    bool value_from_key; // Gemma's attention_k_eq_v
};

enum class Ladder { Gemma, OverRotated };

// Normalizes, scales and rotates one head row in FP64. `ladder` selects the frequency base: Gemma
// takes it over D; the control takes it over the rotated width, the convention the engine's
// existing `rope` uses and the one a naive port would carry over.
std::vector<double> reference_row(const float* x, const float* weight, int d, double eps,
                                  bool rotate, double position, double theta, int pairs,
                                  Ladder ladder) {
    std::vector<double> n = rms_normalize(x, d, eps, weight);
    if (!rotate) { return n; }
    std::vector<double> out = n;
    const double base       = ladder == Ladder::Gemma ? d : 2.0 * pairs;
    for (int i = 0; i < pairs; ++i) {
        const double phi = position * std::pow(theta, -2.0 * i / base);
        out[i]           = n[i] * std::cos(phi) - n[i + d / 2] * std::sin(phi);
        out[i + d / 2]   = n[i + d / 2] * std::cos(phi) + n[i] * std::sin(phi);
    }
    return out;
}

int run_qkv(const QkvCase& c, bool report_controls) {
    const int d = c.head_dim, t_count = c.tokens;
    const int q_width = c.q_heads * d, kv_width = c.kv_heads * d;
    const int packed = q_width + kv_width + (c.value_from_key ? 0 : kv_width);
    // A padded token stride, as a row slice of a wider projection output has.
    const int stride = packed + 64;
    const auto source =
        random_bf16_values(static_cast<std::size_t>(stride) * t_count, 11 + d, -3.0f, 3.0f);
    const auto q_weight = random_bf16_values(d, 21 + d, 0.2f, 1.8f);
    const auto k_weight = random_bf16_values(d, 31 + d, 0.2f, 1.8f);
    const double eps    = 1e-6;
    std::vector<std::int32_t> positions(t_count);
    for (int t = 0; t < t_count; ++t) { positions[t] = c.base_position + t; }

    auto d_source    = upload(bits(source));
    auto d_q_weight  = upload(bits(q_weight));
    auto d_k_weight  = upload(bits(k_weight));
    auto d_positions = upload(positions);
    GuardedDeviceBuffer d_q(std::size_t(q_width) * t_count * 2);
    GuardedDeviceBuffer d_k(std::size_t(kv_width) * t_count * 2);
    GuardedDeviceBuffer d_v(std::size_t(kv_width) * t_count * 2);

    auto* base       = static_cast<std::uint16_t*>(d_source.data());
    const auto slice = [&](int offset, int width) {
        Tensor t(base + offset, DType::BF16, {width, t_count});
        t.nb[1] = static_cast<std::int64_t>(stride) * 2;
        return t;
    };
    const Tensor q_src = slice(0, q_width);
    const Tensor k_src = slice(q_width, kv_width);
    const Tensor v_src = c.value_from_key ? k_src : slice(q_width + kv_width, kv_width);
    Tensor q_w(d_q_weight.data(), DType::BF16, {d});
    Tensor k_w(d_k_weight.data(), DType::BF16, {d});
    Tensor pos(d_positions.data(), DType::I32, {t_count});
    Tensor q_out(d_q.data(), DType::BF16, {d, c.q_heads, t_count});
    Tensor k_out(d_k.data(), DType::BF16, {d, c.kv_heads, t_count});
    Tensor v_out(d_v.data(), DType::BF16, {d, c.kv_heads, t_count});
    ops::gemma4_qkv_prepare(q_src, k_src, v_src, d, q_w, k_w, float(eps), pos, float(c.theta),
                            c.rotary_pairs, q_out, k_out, v_out, nullptr);
    cuda_synchronize();

    std::vector<double> ref_q, ref_k, ref_v, wrong_q;
    for (int t = 0; t < t_count; ++t) {
        const float* row = source.data() + std::size_t(t) * stride;
        for (int h = 0; h < c.q_heads; ++h) {
            auto r = reference_row(row + h * d, q_weight.data(), d, eps, true, positions[t],
                                   c.theta, c.rotary_pairs, Ladder::Gemma);
            ref_q.insert(ref_q.end(), r.begin(), r.end());
            auto w = reference_row(row + h * d, q_weight.data(), d, eps, true, positions[t],
                                   c.theta, c.rotary_pairs, Ladder::OverRotated);
            wrong_q.insert(wrong_q.end(), w.begin(), w.end());
        }
        for (int g = 0; g < c.kv_heads; ++g) {
            auto r = reference_row(row + q_width + g * d, k_weight.data(), d, eps, true,
                                   positions[t], c.theta, c.rotary_pairs, Ladder::Gemma);
            ref_k.insert(ref_k.end(), r.begin(), r.end());
            const float* v_row = row + (c.value_from_key ? q_width : q_width + kv_width) + g * d;
            auto v = reference_row(v_row, nullptr, d, eps, false, 0, 1, 1, Ladder::Gemma);
            ref_v.insert(ref_v.end(), v.begin(), v.end());
        }
    }
    const auto got_q = download_bf16(d_q, ref_q.size());
    const auto got_k = download_bf16(d_k, ref_k.size());
    const auto got_v = download_bf16(d_v, ref_v.size());
    int failures     = 0;
    {
        // The query-only form runs the same rows through the same kernel, so it must reproduce
        // the query half bit for bit.
        GuardedDeviceBuffer d_q_only(std::size_t(q_width) * t_count * 2);
        Tensor q_only(d_q_only.data(), DType::BF16, {d, c.q_heads, t_count});
        ops::gemma4_q_prepare(q_src, d, q_w, float(eps), pos, float(c.theta), c.rotary_pairs,
                              q_only, nullptr);
        cuda_synchronize();
        const auto a = from_device<std::uint16_t>(d_q.data(), ref_q.size());
        const auto b = from_device<std::uint16_t>(d_q_only.data(), ref_q.size());
        if (a != b) {
            std::cerr << c.label << ": query-only preparation differs from the query half\n";
            ++failures;
        }
        failures += d_q_only.verify_guards(c.label + " q-only guards");
    }
    failures += verify_pointwise(c.label + " q", got_q, ref_q, kCriterion);
    failures += verify_pointwise(c.label + " k", got_k, ref_k, kCriterion);
    failures += verify_pointwise(c.label + " v", got_v, ref_v, kCriterion);
    failures += verify_cached_prepare(
        c.label, d, c.q_heads, c.kv_heads, t_count,
        [&](const Tensor& p, Tensor& q, Tensor& k, Tensor& v) {
            ops::gemma4_qkv_prepare(q_src, k_src, v_src, d, q_w, k_w, float(eps), p, float(c.theta),
                                    c.rotary_pairs, q, k, v, nullptr);
        },
        [&](const Tensor& p, Tensor& q, const ops::Gemma4KvLayer& cache) {
            // Joined (a programmatic dependent of the kernel before): the bytes of the plain form.
            ops::gemma4_qkv_prepare(q_src, k_src, v_src, d, q_w, k_w, float(eps), p, float(c.theta),
                                    c.rotary_pairs, q, cache, nullptr, true);
        });
    failures += d_q.verify_guards(c.label + " q guards");
    failures += d_k.verify_guards(c.label + " k guards");
    failures += d_v.verify_guards(c.label + " v guards");

    if (report_controls) {
        // The control arm: the kernel's output against the other frequency ladder must be far
        // outside the criterion, or this suite could not tell the two conventions apart. With
        // every pair rotated the two ladders are the same formula, so the arm exists only for
        // the partial rotation of the full layers.
        if (c.rotary_pairs < d / 2) {
            const double distance = max_abs_difference(got_q, wrong_q);
            std::cout << c.label << " control: output vs over-rotated ladder max_abs=" << distance
                      << '\n';
            if (distance < 0.25) {
                std::cerr << c.label << " control: ladders indistinguishable on this input\n";
                ++failures;
            }
        }
        // And the value stream must not have been normalized with the key weight.
        std::vector<double> v_with_weight;
        for (int t = 0; t < t_count; ++t) {
            const float* row   = source.data() + std::size_t(t) * stride;
            const float* v_row = row + (c.value_from_key ? q_width : q_width + kv_width);
            for (int g = 0; g < c.kv_heads; ++g) {
                auto v = reference_row(v_row + g * d, k_weight.data(), d, eps, false, 0, 1, 1,
                                       Ladder::Gemma);
                v_with_weight.insert(v_with_weight.end(), v.begin(), v.end());
            }
        }
        const double v_distance = max_abs_difference(got_v, v_with_weight);
        std::cout << c.label << " control: v vs weighted v max_abs=" << v_distance << '\n';
        if (v_distance < 0.25) {
            std::cerr << c.label << " control: weighted and unweighted v indistinguishable\n";
            ++failures;
        }
    }
    return failures;
}

// DFlash drafter layers (Qwen3 style, D = 128): weighted q/k norm, whole-head RoPE, raw V. The
// context-only form (no queries) must reproduce the K/V halves bit for bit.
int run_dflash_qkv(const std::string& label, int q_heads, int kv_heads, int t_count,
                   int base_position) {
    constexpr int d    = 128;
    const double theta = 1.0e6;
    const int q_width = q_heads * d, kv_width = kv_heads * d;
    const int stride    = q_width + 2 * kv_width + 64;
    const auto source   = random_bf16_values(std::size_t(stride) * t_count, 41, -3.0f, 3.0f);
    const auto q_weight = random_bf16_values(d, 43, 0.2f, 1.8f);
    const auto k_weight = random_bf16_values(d, 47, 0.2f, 1.8f);
    const double eps    = 1e-6;
    std::vector<std::int32_t> positions(t_count);
    for (int t = 0; t < t_count; ++t) { positions[t] = base_position + t; }

    auto d_source    = upload(bits(source));
    auto d_q_weight  = upload(bits(q_weight));
    auto d_k_weight  = upload(bits(k_weight));
    auto d_positions = upload(positions);
    GuardedDeviceBuffer d_q(std::size_t(q_width) * t_count * 2);
    GuardedDeviceBuffer d_k(std::size_t(kv_width) * t_count * 2);
    GuardedDeviceBuffer d_v(std::size_t(kv_width) * t_count * 2);
    GuardedDeviceBuffer d_k2(std::size_t(kv_width) * t_count * 2);
    GuardedDeviceBuffer d_v2(std::size_t(kv_width) * t_count * 2);
    auto* base       = static_cast<std::uint16_t*>(d_source.data());
    const auto slice = [&](int offset, int width) {
        Tensor t(base + offset, DType::BF16, {width, t_count});
        t.nb[1] = static_cast<std::int64_t>(stride) * 2;
        return t;
    };
    const Tensor q_src = slice(0, q_width);
    const Tensor k_src = slice(q_width, kv_width);
    const Tensor v_src = slice(q_width + kv_width, kv_width);
    Tensor q_w(d_q_weight.data(), DType::BF16, {d});
    Tensor k_w(d_k_weight.data(), DType::BF16, {d});
    Tensor pos(d_positions.data(), DType::I32, {t_count});
    Tensor q_out(d_q.data(), DType::BF16, {d, q_heads, t_count});
    Tensor k_out(d_k.data(), DType::BF16, {d, kv_heads, t_count});
    Tensor v_out(d_v.data(), DType::BF16, {d, kv_heads, t_count});
    Tensor k_ctx(d_k2.data(), DType::BF16, {d, kv_heads, t_count});
    Tensor v_ctx(d_v2.data(), DType::BF16, {d, kv_heads, t_count});
    ops::gemma4_dflash_qkv_prepare(&q_src, k_src, v_src, &q_w, k_w, float(eps), pos, float(theta),
                                   &q_out, k_out, v_out, nullptr);
    ops::gemma4_dflash_qkv_prepare(nullptr, k_src, v_src, nullptr, k_w, float(eps), pos,
                                   float(theta), nullptr, k_ctx, v_ctx, nullptr);
    cuda_synchronize();

    std::vector<double> ref_q, ref_k, ref_v, normed_v;
    std::vector<std::uint16_t> raw_v;
    for (int t = 0; t < t_count; ++t) {
        const float* row = source.data() + std::size_t(t) * stride;
        for (int h = 0; h < q_heads; ++h) {
            auto r = reference_row(row + h * d, q_weight.data(), d, eps, true, positions[t], theta,
                                   d / 2, Ladder::Gemma);
            ref_q.insert(ref_q.end(), r.begin(), r.end());
        }
        for (int g = 0; g < kv_heads; ++g) {
            auto r = reference_row(row + q_width + g * d, k_weight.data(), d, eps, true,
                                   positions[t], theta, d / 2, Ladder::Gemma);
            ref_k.insert(ref_k.end(), r.begin(), r.end());
            const float* v_row = row + q_width + kv_width + g * d;
            for (int i = 0; i < d; ++i) { ref_v.push_back(v_row[i]); }
            const auto v_bits = bits(std::vector<float>(v_row, v_row + d));
            raw_v.insert(raw_v.end(), v_bits.begin(), v_bits.end());
            auto n = reference_row(v_row, nullptr, d, eps, false, 0, 1, 1, Ladder::Gemma);
            normed_v.insert(normed_v.end(), n.begin(), n.end());
        }
    }
    int failures     = 0;
    const auto got_q = download_bf16(d_q, ref_q.size());
    const auto got_k = download_bf16(d_k, ref_k.size());
    const auto got_v = download_bf16(d_v, ref_v.size());
    failures += verify_pointwise(label + " q", got_q, ref_q, kCriterion);
    failures += verify_pointwise(label + " k", got_k, ref_k, kCriterion);
    if (from_device<std::uint16_t>(d_v.data(), raw_v.size()) != raw_v) {
        std::cerr << label << ": v is not a bit-exact copy of v_src\n";
        ++failures;
    }
    if (from_device<std::uint16_t>(d_k2.data(), ref_k.size()) !=
            from_device<std::uint16_t>(d_k.data(), ref_k.size()) ||
        from_device<std::uint16_t>(d_v2.data(), raw_v.size()) != raw_v) {
        std::cerr << label << ": the context-only form differs from the full form's K/V\n";
        ++failures;
    }
    failures += verify_cached_prepare(
        label + " block", d, q_heads, kv_heads, t_count,
        [&](const Tensor& p, Tensor& q, Tensor& k, Tensor& v) {
            ops::gemma4_dflash_qkv_prepare(&q_src, k_src, v_src, &q_w, k_w, float(eps), p,
                                           float(theta), &q, k, v, nullptr);
        },
        [&](const Tensor& p, Tensor& q, const ops::Gemma4KvLayer& cache) {
            ops::gemma4_dflash_qkv_prepare(&q_src, k_src, v_src, &q_w, k_w, float(eps), p,
                                           float(theta), &q, cache, nullptr);
        });
    failures += verify_cached_prepare(
        label + " context", d, 0, kv_heads, t_count,
        [&](const Tensor& p, Tensor&, Tensor& k, Tensor& v) {
            ops::gemma4_dflash_qkv_prepare(nullptr, k_src, v_src, nullptr, k_w, float(eps), p,
                                           float(theta), nullptr, k, v, nullptr);
        },
        [&](const Tensor& p, Tensor&, const ops::Gemma4KvLayer& cache) {
            ops::gemma4_dflash_qkv_prepare(nullptr, k_src, v_src, nullptr, k_w, float(eps), p,
                                           float(theta), nullptr, cache, nullptr);
        });
    // Control: a Gemma-style normalized V must be far from what this op writes.
    const double v_distance = max_abs_difference(got_v, normed_v);
    std::cout << label << " control: raw v vs normalized v max_abs=" << v_distance << '\n';
    if (v_distance < 0.25) {
        std::cerr << label << " control: raw and normalized v indistinguishable\n";
        ++failures;
    }
    for (auto* buffer : {&d_q, &d_k, &d_v, &d_k2, &d_v2}) {
        failures += buffer->verify_guards(label + " guards");
    }
    return failures;
}

// ---- GeGLU ------------------------------------------------------------------------------------

int run_geglu(const std::string& label, int width, int tokens) {
    const auto gate_up =
        random_bf16_values(std::size_t(2) * width * tokens, 41 + width, -4.0f, 4.0f);
    auto d_in = upload(bits(gate_up));
    GuardedDeviceBuffer d_out(std::size_t(width) * tokens * 2);
    Tensor in(d_in.data(), DType::BF16, {2 * width, tokens});
    Tensor out(d_out.data(), DType::BF16, {width, tokens});
    ops::gelu_tanh_mul(in, out, nullptr);
    cuda_synchronize();
    std::vector<double> ref(std::size_t(width) * tokens), swapped(ref.size());
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < width; ++i) {
            const double g                      = gate_up[std::size_t(t) * 2 * width + i];
            const double u                      = gate_up[std::size_t(t) * 2 * width + width + i];
            ref[std::size_t(t) * width + i]     = gelu_tanh_oracle(g) * u;
            swapped[std::size_t(t) * width + i] = gelu_tanh_oracle(u) * g;
        }
    }
    const auto got = download_bf16(d_out, ref.size());
    int failures   = verify_pointwise(label, got, ref, kCriterion);
    failures += d_out.verify_guards(label + " guards");
    const double distance = max_abs_difference(got, swapped);
    std::cout << label << " control: output vs gelu-on-up max_abs=" << distance << '\n';
    if (distance < 0.25) {
        std::cerr << label << " control: gate/up roles indistinguishable\n";
        ++failures;
    }
    return failures;
}

// GeGLU over separate planes, the multiplier one layer's rows of a wider per-layer plane (a
// padded token stride), as the per-layer embedding block reads it.
int run_geglu_planes(const std::string& label, int width, int slices, int slice, int tokens) {
    const auto gate = random_bf16_values(std::size_t(width) * tokens, 43 + width, -4.0f, 4.0f);
    const auto wide =
        random_bf16_values(std::size_t(width) * slices * tokens, 47 + width, -4.0f, 4.0f);
    auto d_gate = upload(bits(gate)), d_wide = upload(bits(wide));
    GuardedDeviceBuffer d_out(std::size_t(width) * tokens * 2);
    Tensor tgate(d_gate.data(), DType::BF16, {width, tokens});
    Tensor twide(d_wide.data(), DType::BF16, {width * slices, tokens});
    Tensor out(d_out.data(), DType::BF16, {width, tokens});
    ops::gelu_tanh_mul(tgate, twide.slice(0, slice * width, width), out, nullptr);
    cuda_synchronize();
    std::vector<double> ref(std::size_t(width) * tokens), other(ref.size());
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < width; ++i) {
            const double g  = gate[std::size_t(t) * width + i];
            const auto row  = std::size_t(t) * width * slices + i;
            const double u  = wide[row + std::size_t(slice) * width];
            const double u0 = wide[row + std::size_t((slice + 1) % slices) * width];
            ref[std::size_t(t) * width + i]   = gelu_tanh_oracle(g) * u;
            other[std::size_t(t) * width + i] = gelu_tanh_oracle(g) * u0;
        }
    }
    const auto got = download_bf16(d_out, ref.size());
    int failures   = verify_pointwise(label, got, ref, kCriterion);
    failures += d_out.verify_guards(label + " guards");
    const double distance = max_abs_difference(got, other);
    std::cout << label << " control: output vs another layer's rows max_abs=" << distance << '\n';
    if (distance < 0.25) {
        std::cerr << label << " control: the layer slice is not what was read\n";
        ++failures;
    }
    return failures;
}

// ---- per-layer embedding inputs ---------------------------------------------------------------

double round_bf16(double value) { return bf16_to_f32(f32_to_bf16(static_cast<float>(value))); }

// The embedding normalizer on gathered rows: bf16(x * s) for a BF16 s is one correctly rounded
// product (both factors have 8-bit significands, so the FP32 product is exact), bit for bit.
int run_scale(const std::string& label, int width, int tokens) {
    const auto x     = random_bf16_values(std::size_t(width) * tokens, 239 + width, -3.0f, 3.0f);
    const float root = bf16_to_f32(f32_to_bf16(std::sqrt(float(width))));
    GuardedDeviceBuffer d_x(x.size() * 2);
    CUDA_CHECK(cudaMemcpy(d_x.data(), bits(x).data(), x.size() * 2, cudaMemcpyHostToDevice));
    Tensor tx(d_x.data(), DType::BF16, {width, tokens});
    ops::scale_bf16(tx, root, nullptr);
    cuda_synchronize();
    const auto got = download_bf16(d_x, x.size());
    int failures   = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double want = round_bf16(double(x[i]) * double(root));
        if (got[i] != want) {
            if (failures < 4) {
                std::cerr << label << ": [" << i << "] " << got[i] << " != " << want << '\n';
            }
            ++failures;
        }
    }
    failures += d_x.verify_guards(label + " guards");
    std::cout << label << ": scale " << root << ", " << x.size() << " values, " << failures
              << " mismatches\n";
    return failures ? 1 : 0;
}

int run_ple_inputs(const std::string& label, int width, int slices, int tokens, int hidden) {
    const auto rows       = std::size_t(width) * slices;
    const auto projection = random_bf16_values(rows * tokens, 211 + width, -40.0f, 40.0f);
    const auto table      = random_bf16_values(rows * tokens, 223 + width, -0.5f, 0.5f);
    const auto weight     = random_bf16_values(width, 227 + width, 0.1f, 2.0f);
    auto d_proj = upload(bits(projection)), d_table = upload(bits(table));
    auto d_weight = upload(bits(weight));
    Tensor tproj(d_proj.data(), DType::BF16, {int(rows), tokens});
    Tensor ttable(d_table.data(), DType::BF16, {int(rows), tokens});
    Tensor tweight(d_weight.data(), DType::BF16, {width});
    const double eps = 1e-6;
    const float projection_scale = static_cast<float>(1.0 / std::sqrt(double(hidden)));
    const float embedding_scale  = bf16_to_f32(f32_to_bf16(std::sqrt(float(width))));
    const float input_scale      = static_cast<float>(1.0 / std::sqrt(2.0));
    // In place, as the forward pass calls it.
    ops::gemma4_ple_inputs(tproj, ttable, tweight, float(eps), projection_scale, embedding_scale,
                           input_scale, tproj, nullptr);
    cuda_synchronize();
    std::vector<double> ref(rows * tokens), unscaled(ref.size());
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < slices; ++s) {
            const std::size_t base = std::size_t(t) * rows + std::size_t(s) * width;
            std::vector<double> a(width);
            double sum = 0;
            for (int i = 0; i < width; ++i) {
                a[i] = round_bf16(double(projection[base + i]) * projection_scale);
                sum += a[i] * a[i];
            }
            const double inverse = 1.0 / std::sqrt(sum / width + eps);
            for (int i = 0; i < width; ++i) {
                const double n = round_bf16(a[i] * inverse * weight[i]);
                const double e = round_bf16(double(table[base + i]) * embedding_scale);
                ref[base + i]      = round_bf16(n + e) * input_scale;
                unscaled[base + i] = round_bf16(n + double(table[base + i])) * input_scale;
            }
        }
    }
    const auto got = download_bf16(d_proj, ref.size());
    int failures   = verify_pointwise(label, got, ref, kCriterion);
    failures += d_proj.verify_guards(label + " guards");
    // Control: the table rows without their sqrt(P) scale must be far outside the criterion.
    const double distance = max_abs_difference(got, unscaled);
    std::cout << label << " control: output vs unscaled table rows max_abs=" << distance << '\n';
    if (distance < 0.5) {
        std::cerr << label << " control: the table scale is not observable\n";
        ++failures;
    }
    return failures;
}

// ---- sandwich norms ---------------------------------------------------------------------------

std::vector<double> normed(const float* x, const float* w, int width, double eps) {
    return rms_normalize(x, width, eps, w);
}

int run_residual_norm(const std::string& label, int width, int tokens, bool two_branches,
                      float scale, bool two_outputs, bool update_only = false,
                      bool three_outputs = false) {
    const auto x  = random_bf16_values(std::size_t(width) * tokens, 121 + width, -6.0f, 6.0f);
    const auto a  = random_bf16_values(std::size_t(width) * tokens, 131 + width, -30.0f, 30.0f);
    const auto b  = random_bf16_values(std::size_t(width) * tokens, 141 + width, -0.5f, 0.5f);
    const auto wa = random_bf16_values(width, 151 + width, 0.1f, 2.0f);
    const auto wb = random_bf16_values(width, 161 + width, 0.1f, 2.0f);
    const auto wp = random_bf16_values(width, 171 + width, 0.1f, 2.0f);
    const auto w1 = random_bf16_values(width, 181 + width, 0.1f, 20.0f);
    const auto w2 = random_bf16_values(width, 191 + width, 0.1f, 2.0f);
    const auto w3 = random_bf16_values(width, 197 + width, 0.01f, 0.2f);
    auto d_x = upload(bits(x)), d_a = upload(bits(a)), d_b = upload(bits(b));
    auto d_wa = upload(bits(wa)), d_wb = upload(bits(wb)), d_wp = upload(bits(wp));
    auto d_w1 = upload(bits(w1)), d_w2 = upload(bits(w2)), d_w3 = upload(bits(w3));
    GuardedDeviceBuffer d_o1(std::size_t(width) * tokens * 2),
        d_o2(std::size_t(width) * tokens * 2), d_o3(std::size_t(width) * tokens * 2);
    Tensor tx(d_x.data(), DType::BF16, {width, tokens});
    Tensor ta(d_a.data(), DType::BF16, {width, tokens});
    Tensor tb(d_b.data(), DType::BF16, {width, tokens});
    Tensor twa(d_wa.data(), DType::BF16, {width}), twb(d_wb.data(), DType::BF16, {width});
    Tensor twp(d_wp.data(), DType::BF16, {width});
    Tensor tw1(d_w1.data(), DType::BF16, {width}), tw2(d_w2.data(), DType::BF16, {width});
    Tensor tw3(d_w3.data(), DType::BF16, {width});
    Tensor to1(d_o1.data(), DType::BF16, {width, tokens}),
        to2(d_o2.data(), DType::BF16, {width, tokens}),
        to3(d_o3.data(), DType::BF16, {width, tokens});
    const double eps = 1e-6;
    const ops::ResidualBranch branch =
        two_branches ? ops::ResidualBranch{ta, &twa, &tb, &twb} : ops::ResidualBranch{ta};
    const ops::ResidualNormOutput outputs[] = {{tw1, to1}, {tw2, to2}, {tw3, to3}};
    const std::size_t used = update_only ? 0 : three_outputs ? 3 : two_outputs ? 2 : 1;
    // The same update launched joined (join_previous: a programmatic dependent of the kernel
    // before it, here the plain call) on a copy of x, into outputs of its own.
    auto d_jx = upload(bits(x));
    GuardedDeviceBuffer d_j1(std::size_t(width) * tokens * 2),
        d_j2(std::size_t(width) * tokens * 2), d_j3(std::size_t(width) * tokens * 2);
    Tensor tjx(d_jx.data(), DType::BF16, {width, tokens});
    Tensor tj1(d_j1.data(), DType::BF16, {width, tokens}),
        tj2(d_j2.data(), DType::BF16, {width, tokens}),
        tj3(d_j3.data(), DType::BF16, {width, tokens});
    const ops::ResidualNormOutput joined_outputs[] = {{tw1, tj1}, {tw2, tj2}, {tw3, tj3}};
    ops::gemma4_residual_norm(tx, branch, twp, float(eps), scale, std::span(outputs, used),
                              nullptr);
    ops::gemma4_residual_norm(tjx, branch, twp, float(eps), scale, std::span(joined_outputs, used),
                              nullptr, true);
    cuda_synchronize();

    std::vector<double> ref_x, ref_o1, ref_o2, ref_o3, stale_o1;
    for (int t = 0; t < tokens; ++t) {
        const std::size_t off = std::size_t(t) * width;
        std::vector<double> branch(width);
        if (two_branches) {
            const auto na = normed(a.data() + off, wa.data(), width, eps);
            const auto nb = normed(b.data() + off, wb.data(), width, eps);
            for (int i = 0; i < width; ++i) { branch[i] = na[i] + nb[i]; }
        } else {
            for (int i = 0; i < width; ++i) { branch[i] = a[off + i]; }
        }
        double sb = 0;
        for (const double v : branch) { sb += v * v; }
        const double ib = 1.0 / std::sqrt(sb / width + eps);
        std::vector<float> updated(width);
        for (int i = 0; i < width; ++i) {
            const double value = (x[off + i] + branch[i] * ib * wp[i]) * scale;
            // x' crosses its BF16 storage boundary before anything reads it again.
            updated[i] = bf16_to_f32(f32_to_bf16(static_cast<float>(value)));
            ref_x.push_back(value);
        }
        const auto o1 = normed(updated.data(), w1.data(), width, eps);
        const auto o2 = normed(updated.data(), w2.data(), width, eps);
        const auto o3 = normed(updated.data(), w3.data(), width, eps);
        const auto s1 = normed(x.data() + off, w1.data(), width, eps);
        ref_o1.insert(ref_o1.end(), o1.begin(), o1.end());
        ref_o2.insert(ref_o2.end(), o2.begin(), o2.end());
        ref_o3.insert(ref_o3.end(), o3.begin(), o3.end());
        stale_o1.insert(stale_o1.end(), s1.begin(), s1.end());
    }
    const auto got_x  = download_bf16(d_x, ref_x.size());
    const auto got_o1 = download_bf16(d_o1, ref_o1.size());
    int failures      = verify_pointwise(label + " x", got_x, ref_x, kCriterion);
    if (update_only) {
        // No normalized output was asked for: out1 keeps the allocation's 0xa5 fill.
        const auto untouched = from_device<std::uint16_t>(d_o1.data(), ref_o1.size());
        if (std::any_of(untouched.begin(), untouched.end(),
                        [](std::uint16_t w) { return w != 0xa5a5; })) {
            std::cerr << label << ": the update-only form wrote a normalized output\n";
            ++failures;
        }
        return failures + d_o1.verify_guards(label + " out1 guards");
    }
    // Output 1 carries a weight up to 20, so its absolute floor scales with it.
    failures += verify_pointwise(label + " out1", got_o1, ref_o1, PointwiseCriterion{0.2, 8.0e-3});
    if (two_outputs) {
        failures += verify_pointwise(label + " out2", download_bf16(d_o2, ref_o2.size()), ref_o2,
                                     kCriterion);
    }
    if (three_outputs) {
        // The router's weight (router_scale * H^-0.5) is small: its values sit far below 1.
        failures += verify_pointwise(label + " out3", download_bf16(d_o3, ref_o3.size()), ref_o3,
                                     kCriterion);
    }
    failures += d_o1.verify_guards(label + " out1 guards");
    failures += d_o2.verify_guards(label + " out2 guards");
    failures += d_o3.verify_guards(label + " out3 guards");
    {
        const std::size_t count = std::size_t(width) * tokens;
        bool same               = from_device<std::uint16_t>(d_x.data(), count) ==
                    from_device<std::uint16_t>(d_jx.data(), count);
        GuardedDeviceBuffer* plain[]  = {&d_o1, &d_o2, &d_o3};
        GuardedDeviceBuffer* joined[] = {&d_j1, &d_j2, &d_j3};
        for (std::size_t i = 0; i < used; ++i) {
            same = same && from_device<std::uint16_t>(plain[i]->data(), count) ==
                               from_device<std::uint16_t>(joined[i]->data(), count);
        }
        if (!same) {
            std::cerr << label << ": the joined launch differs from the plain one\n";
            ++failures;
        }
        failures += d_j1.verify_guards(label + " joined out1 guards");
    }
    // Control: a norm of the stream *before* this update — the order a fusion is most likely to
    // get wrong — must be far outside the criterion.
    const double distance = max_abs_difference(got_o1, stale_o1);
    std::cout << label << " control: out1 vs norm of the pre-update stream max_abs=" << distance
              << '\n';
    if (distance < 1.0) {
        std::cerr << label << " control: pre- and post-update norms indistinguishable\n";
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
    // The four shipped geometries: 26B sliding/full, 31B sliding/full.
    failures += run_qkv({"26B sliding D256 16/8", 256, 16, 8, 5, 0, 1.0e4, 128, false}, true);
    failures += run_qkv({"26B full D512 16/2 k=v", 512, 16, 2, 5, 3000, 1.0e6, 64, true}, true);
    failures +=
        run_qkv({"31B sliding D256 32/16", 256, 32, 16, 3, 262000, 1.0e4, 128, false}, false);
    failures += run_qkv({"31B full D512 32/4 k=v", 512, 32, 4, 17, 70000, 1.0e6, 64, true}, true);
    // DFlash drafters: the 16-token block of 26B (32/8) and 31B (64/8), and a context append.
    failures += run_dflash_qkv("dflash 26B block D128 32/8", 32, 8, 16, 1000);
    failures += run_dflash_qkv("dflash 31B block D128 64/8", 64, 8, 16, 250000);
    failures += run_dflash_qkv("dflash context D128 32/8", 32, 8, 37, 0);
    failures += run_geglu("geglu 26B dense", 2112, 7);
    failures += run_geglu("geglu 31B dense", 21504, 3);
    failures += run_geglu("geglu single", 704, 1);
    failures += run_residual_norm("residual_norm 26B after attention", 2816, 5, false, 1.0f, true);
    failures += run_residual_norm("residual_norm 26B GGUF router norm", 2816, 5, false, 1.0f, true,
                                  false, true);
    failures += run_residual_norm("residual_norm decode router norm", 2816, 1, false, 1.0f, true,
                                  false, true);
    failures += run_residual_norm("residual_norm scalar width router norm", 2820, 3, false, 1.0f,
                                  true, false, true);
    failures += run_residual_norm("residual_norm 26B MoE join", 2816, 3, true, 0.0703125f, false);
    failures += run_residual_norm("residual_norm 31B dense join", 5376, 2, false, 0.8125f, false);
    failures += run_residual_norm("residual_norm decode column", 2816, 1, true, 0.5f, true);
    // A width that is not whole 16-byte vectors takes the scalar schedule.
    failures += run_residual_norm("residual_norm scalar width", 2820, 3, true, 0.5f, true);
    // Gemma 4 E2B/E4B: the update alone before the per-layer embedding block (both schedules),
    // the block's GeGLU over one layer's rows of the per-layer inputs, and those inputs.
    failures += run_residual_norm("residual_norm E2B update only", 1536, 4, false, 1.0f, false,
                                  true);
    failures += run_residual_norm("residual_norm update only scalar width", 1540, 2, false, 1.0f,
                                  false, true);
    failures += run_geglu_planes("geglu E2B per-layer slice", 256, 35, 17, 5);
    failures += run_geglu_planes("geglu E4B per-layer slice decode", 256, 42, 41, 1);
    failures += run_ple_inputs("ple_inputs E2B", 256, 35, 3, 1536);
    failures += run_ple_inputs("ple_inputs E4B decode", 256, 42, 1, 2560);
    failures += run_ple_inputs("ple_inputs narrow", 18, 4, 7, 288);
    failures += run_scale("scale E2B embedding rows", 1536, 5);
    failures += run_scale("scale E4B embedding decode", 2560, 1);

    if (failures != 0) {
        std::cerr << "gemma4_fused: " << failures << " checks failed\n";
        return 1;
    }
    std::cout << "gemma4_fused: all checks passed\n";
    return 0;
}
