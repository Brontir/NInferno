#include "core/device.h"
#include "ninferno/ops/dense_fused.h"
#include "ops/op_tester.h"
#include "ops/oracle.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

// Every output is rounded once from an FP32 interior; the relative term carries that rounding and
// the FP32 reductions, the absolute floor values that cancel toward zero inside a rotation.
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

std::vector<double> download_bf16(const void* device, std::size_t count) {
    const auto raw = from_device<std::uint16_t>(device, count);
    std::vector<double> out(count);
    std::transform(raw.begin(), raw.end(), out.begin(),
                   [](std::uint16_t w) { return static_cast<double>(bf16_to_f32(w)); });
    return out;
}

// ---- Q/K/V preparation ------------------------------------------------------------------------

struct QkvCase {
    std::string label;
    int d, q_heads, kv_heads, tokens, base_position;
    ops::QkNorm norm;
    int rotary_dim; // 0: NoPE
    double theta;
    float rotary_scale;
    bool bias;
    float query_beta;
    int query_period;
    bool unit_offset = false; // Gemma's RMSNorm gain 1 + w
    // Keys and values into k_out / v_out columns [D, Hkv, T] instead of a cache (a DFlash
    // drafter's block), and a call without queries (its committed context).
    bool columns    = false;
    bool no_queries = false;
};

// The rotation convention a naive port of Gemma's rope would carry over: pair i with i + D/2
// whatever the rotated width. Only a partial rotary width tells the two apart.
enum class Partner { RotaryHalf, HeadHalf };

// Reference outputs of one token, in FP64, for the given partner convention.
struct TokenRef {
    std::vector<double> q, k, v;
};

TokenRef reference_token(const QkvCase& c, const float* row, const std::vector<float>& bq,
                         const std::vector<float>& bk, const std::vector<float>& bv,
                         const std::vector<float>& wq, const std::vector<float>& wk,
                         const std::vector<float>& inv_freq, int position, Partner partner) {
    const int d = c.d, qw = c.q_heads * d, kw = c.kv_heads * d;
    const double eps = 1e-5;
    const auto biased = [&](const float* src, const std::vector<float>& b, int width) {
        std::vector<double> out(width);
        for (int i = 0; i < width; ++i) { out[i] = double(src[i]) + (c.bias ? b[i] : 0.0); }
        return out;
    };
    auto q = biased(row, bq, qw);
    auto k = biased(row + qw, bk, kw);
    auto v = biased(row + qw + kw, bv, kw);
    const auto norm = [&](std::vector<double>& x, int heads, const std::vector<float>& weights) {
        std::vector<double> w(weights.begin(), weights.end());
        if (c.unit_offset) {
            for (double& g : w) { g += 1.0; }
        }
        if (c.norm == ops::QkNorm::Head) {
            for (int h = 0; h < heads; ++h) {
                double s = 0;
                for (int i = 0; i < d; ++i) { s += x[h * d + i] * x[h * d + i]; }
                const double inv = 1.0 / std::sqrt(s / d + eps);
                for (int i = 0; i < d; ++i) { x[h * d + i] *= inv * w[i]; }
            }
        } else if (c.norm == ops::QkNorm::Projection) {
            double s = 0;
            for (const double value : x) { s += value * value; }
            const double inv = 1.0 / std::sqrt(s / x.size() + eps);
            for (std::size_t i = 0; i < x.size(); ++i) { x[i] *= inv * w[i]; }
        }
    };
    norm(q, c.q_heads, wq);
    norm(k, c.kv_heads, wk);
    const auto rotate = [&](std::vector<double>& x, int heads) {
        if (c.rotary_dim == 0) { return; }
        const int half   = c.rotary_dim / 2;
        const int offset = partner == Partner::RotaryHalf ? half : d / 2;
        for (int h = 0; h < heads; ++h) {
            std::vector<double> in(x.begin() + h * d, x.begin() + (h + 1) * d);
            for (int i = 0; i < half; ++i) {
                const double phi = double(position) * double(inv_freq[i]);
                const double cs  = c.rotary_scale * std::cos(phi);
                const double sn  = c.rotary_scale * std::sin(phi);
                x[h * d + i]          = in[i] * cs - in[i + offset] * sn;
                x[h * d + i + offset] = in[i + offset] * cs + in[i] * sn;
            }
            if (partner == Partner::RotaryHalf) {
                for (int i = c.rotary_dim; i < d; ++i) { x[h * d + i] = in[i]; }
            }
        }
    };
    rotate(q, c.q_heads);
    rotate(k, c.kv_heads);
    if (c.query_period > 0) {
        const double factor =
            1.0 + c.query_beta * std::log(1.0 + std::floor(double(position) / c.query_period));
        for (double& value : q) { value *= factor; }
    }
    return {q, k, v};
}

int run_qkv(const QkvCase& c) {
    const int d = c.d, t_count = c.tokens;
    const int qw = c.q_heads * d, kw = c.kv_heads * d;
    const int stride = qw + 2 * kw + 64; // a padded token stride, as a row slice has
    const auto source = random_bf16_values(std::size_t(stride) * t_count, 11 + d, -3.0f, 3.0f);
    const auto bq     = random_bf16_values(qw, 13, -1.0f, 1.0f);
    const auto bk     = random_bf16_values(kw, 17, -1.0f, 1.0f);
    const auto bv     = random_bf16_values(kw, 19, -1.0f, 1.0f);
    const bool head   = c.norm == ops::QkNorm::Head;
    const auto wq     = random_bf16_values(head ? d : qw, 23, 0.2f, 1.8f);
    const auto wk     = random_bf16_values(head ? d : kw, 29, 0.2f, 1.8f);
    std::vector<float> inv_freq(std::max(c.rotary_dim / 2, 1));
    for (int i = 0; i < c.rotary_dim / 2; ++i) {
        // A YaRN-like ladder: every other frequency divided by a factor, so no closed form holds.
        inv_freq[i] = float(std::pow(c.theta, -2.0 * i / c.rotary_dim) / (i % 2 ? 7.0 : 1.0));
    }
    const int capacity = t_count + 3;
    std::vector<std::int32_t> positions(t_count);
    for (int t = 0; t < t_count; ++t) { positions[t] = c.base_position + t; }

    auto d_source    = upload(bits(source));
    auto d_bq        = upload(bits(bq));
    auto d_bk        = upload(bits(bk));
    auto d_bv        = upload(bits(bv));
    auto d_wq        = upload(bits(wq));
    auto d_wk        = upload(bits(wk));
    auto d_freq      = upload(inv_freq);
    auto d_positions = upload(positions);
    const auto plane = std::size_t(d) * capacity * c.kv_heads;
    const auto fill_k = bits(random_bf16_values(plane, 71, -1.0f, 1.0f));
    const auto fill_v = bits(random_bf16_values(plane, 73, -1.0f, 1.0f));
    auto d_ck        = upload(fill_k);
    auto d_cv        = upload(fill_v);
    GuardedDeviceBuffer d_q(std::size_t(qw) * t_count * 2);
    const auto column_plane = std::size_t(kw) * t_count;
    GuardedDeviceBuffer d_k_out(column_plane * 2);
    GuardedDeviceBuffer d_v_out(column_plane * 2);

    auto* base       = static_cast<std::uint16_t*>(d_source.data());
    const auto slice = [&](int offset, int width) {
        Tensor t(base + offset, DType::BF16, {width, t_count});
        t.nb[1] = static_cast<std::int64_t>(stride) * 2;
        return t;
    };
    ops::DenseQkvPrepare a;
    if (!c.no_queries) { a.q_src = slice(0, qw); }
    a.k_src = slice(qw, kw);
    a.v_src = slice(qw + kw, kw);
    if (c.bias) {
        if (!c.no_queries) { a.q_bias = Tensor(d_bq.data(), DType::BF16, {qw}); }
        a.k_bias = Tensor(d_bk.data(), DType::BF16, {kw});
        a.v_bias = Tensor(d_bv.data(), DType::BF16, {kw});
    }
    a.qk_norm          = c.norm;
    a.norm_unit_offset = c.unit_offset;
    if (c.norm != ops::QkNorm::None) {
        if (!c.no_queries) { a.q_norm = Tensor(d_wq.data(), DType::BF16, {head ? d : qw}); }
        a.k_norm = Tensor(d_wk.data(), DType::BF16, {head ? d : kw});
    }
    a.eps = 1e-5f;
    if (c.rotary_dim > 0) {
        a.inv_freq   = Tensor(d_freq.data(), DType::FP32, {c.rotary_dim / 2});
        a.rotary_dim = c.rotary_dim;
    }
    a.rotary_scale = c.rotary_scale;
    a.query_beta   = c.query_beta;
    a.query_period = c.query_period;
    a.positions    = Tensor(d_positions.data(), DType::I32, {t_count});
    if (!c.no_queries) { a.q_out = Tensor(d_q.data(), DType::BF16, {d, c.q_heads, t_count}); }
    if (c.columns) {
        a.k_out = Tensor(d_k_out.data(), DType::BF16, {d, c.kv_heads, t_count});
        a.v_out = Tensor(d_v_out.data(), DType::BF16, {d, c.kv_heads, t_count});
    } else {
        a.cache.k = Tensor(d_ck.data(), DType::BF16, {d, capacity, c.kv_heads});
        a.cache.v = Tensor(d_cv.data(), DType::BF16, {d, capacity, c.kv_heads});
    }
    ops::dense_qkv_prepare(a, nullptr);
    cuda_synchronize();

    std::vector<double> ref_q, wrong_q;
    std::vector<double> want_k(plane), want_v(plane);
    std::vector<double> want_k_columns(column_plane), want_v_columns(column_plane);
    for (std::size_t i = 0; i < plane; ++i) {
        want_k[i] = bf16_to_f32(fill_k[i]);
        want_v[i] = bf16_to_f32(fill_v[i]);
    }
    for (int t = 0; t < t_count; ++t) {
        const float* row = source.data() + std::size_t(t) * stride;
        const auto r = reference_token(c, row, bq, bk, bv, wq, wk, inv_freq, positions[t],
                                       Partner::RotaryHalf);
        const auto w = reference_token(c, row, bq, bk, bv, wq, wk, inv_freq, positions[t],
                                       Partner::HeadHalf);
        ref_q.insert(ref_q.end(), r.q.begin(), r.q.end());
        wrong_q.insert(wrong_q.end(), w.q.begin(), w.q.end());
        const int slot = positions[t] % capacity;
        for (int g = 0; g < c.kv_heads; ++g) {
            for (int i = 0; i < d; ++i) {
                want_k[(std::size_t(g) * capacity + slot) * d + i] = r.k[g * d + i];
                want_v[(std::size_t(g) * capacity + slot) * d + i] = r.v[g * d + i];
                want_k_columns[(std::size_t(t) * c.kv_heads + g) * d + i] = r.k[g * d + i];
                want_v_columns[(std::size_t(t) * c.kv_heads + g) * d + i] = r.v[g * d + i];
            }
        }
    }
    int failures     = 0;
    const auto got_q = download_bf16(d_q.data(), ref_q.size());
    if (!c.no_queries) { failures += verify_pointwise(c.label + " q", got_q, ref_q, kCriterion); }
    if (c.columns) {
        failures += verify_pointwise(c.label + " k columns",
                                     download_bf16(d_k_out.data(), column_plane), want_k_columns,
                                     kCriterion);
        failures += verify_pointwise(c.label + " v columns",
                                     download_bf16(d_v_out.data(), column_plane), want_v_columns,
                                     kCriterion);
        // The cache planes were not handed to the call and stay as they were.
        if (from_device<std::uint16_t>(d_ck.data(), plane) != fill_k ||
            from_device<std::uint16_t>(d_cv.data(), plane) != fill_v) {
            std::cerr << c.label << ": a columns call wrote cache planes\n";
            ++failures;
        }
        failures += d_k_out.verify_guards(c.label + " k columns guards");
        failures += d_v_out.verify_guards(c.label + " v columns guards");
        failures += d_q.verify_guards(c.label + " q guards");
        return failures;
    }
    failures += verify_pointwise(c.label + " cache k", download_bf16(d_ck.data(), plane), want_k,
                                 kCriterion);
    failures += verify_pointwise(c.label + " cache v", download_bf16(d_cv.data(), plane), want_v,
                                 kCriterion);
    if (!c.bias) {
        // Without a bias the values are copied, not recomputed: bit for bit.
        std::vector<std::uint16_t> want_bits = fill_v;
        for (int t = 0; t < t_count; ++t) {
            const int slot = positions[t] % capacity;
            for (int g = 0; g < c.kv_heads; ++g) {
                std::copy_n(bits(std::vector<float>(
                                     source.begin() + std::size_t(t) * stride + qw + kw + g * d,
                                     source.begin() + std::size_t(t) * stride + qw + kw +
                                         (g + 1) * d))
                                .begin(),
                            d, want_bits.begin() + (std::size_t(g) * capacity + slot) * d);
            }
        }
        if (from_device<std::uint16_t>(d_cv.data(), plane) != want_bits) {
            std::cerr << c.label << ": values are not a bit-exact copy\n";
            ++failures;
        }
    }
    failures += d_q.verify_guards(c.label + " q guards");
    failures += d_ck.verify_guards(c.label + " k guards");
    failures += d_cv.verify_guards(c.label + " v guards");
    // Control: under a partial rotary width the other partner convention is far away.
    if (c.rotary_dim > 0 && c.rotary_dim < d) {
        const double distance = max_abs_difference(got_q, wrong_q);
        std::cout << c.label << " control: q vs head-half partners max_abs=" << distance << '\n';
        if (distance < 0.25) {
            std::cerr << c.label << " control: partner conventions indistinguishable\n";
            ++failures;
        }
    }
    return failures;
}

// Control: a projection-wide norm and a per-head norm of the same rows must differ, or the scope
// switch is not observable.
int run_norm_scope_control() {
    const QkvCase projection{"control projection", 128, 4, 2, 3, 5, ops::QkNorm::Projection, 128,
                             1e4, 1.0f, false, 0.0f, 0};
    const int d = 128, qw = 4 * d, kw = 2 * d, stride = qw + 2 * kw;
    const auto source = random_bf16_values(std::size_t(stride) * 3, 5, -3.0f, 3.0f);
    // Make the heads' scales differ, so the two statistics are far apart.
    std::vector<float> scaled = source;
    for (int t = 0; t < 3; ++t) {
        for (int i = 0; i < d; ++i) { scaled[std::size_t(t) * stride + i] *= 8.0f; }
    }
    std::vector<float> w(qw, 1.0f), wk(kw, 1.0f), none;
    std::vector<float> freq(64);
    for (int i = 0; i < 64; ++i) { freq[i] = float(std::pow(1e4, -2.0 * i / 128)); }
    QkvCase head = projection;
    head.norm    = ops::QkNorm::Head;
    std::vector<float> w_head(d, 1.0f);
    const auto a = reference_token(projection, scaled.data(), none, none, none, w, wk, freq, 5,
                                   Partner::RotaryHalf);
    const auto b = reference_token(head, scaled.data(), none, none, none, w_head, w_head, freq, 5,
                                   Partner::RotaryHalf);
    const double distance = max_abs_difference(a.q, b.q);
    std::cout << "control: projection vs head norm max_abs=" << distance << '\n';
    return distance < 0.25 ? 1 : 0;
}

// A batched call over bound caches writes, for each batch row, the bytes the single-sequence call
// writes into slot storage, only elsewhere: its ring image, or its block-table row of a page pool.
// Two rows at different positions, bound in reverse order; a projection-wide norm and a per-head
// norm with biases.
int run_qkv_bound(const std::string& label, ops::QkNorm norm, bool bias, bool paged) {
    constexpr int d = 128, q_heads = 8, kv_heads = 2, row_tokens = 5, rows = 2;
    const int qw = q_heads * d, kw = kv_heads * d, stride = qw + 2 * kw;
    const int columns = row_tokens * rows;
    const std::int32_t row_base[rows] = {61, 190}; // row 0 crosses a page boundary at 64
    const auto source = random_bf16_values(std::size_t(stride) * columns, 211, -3.0f, 3.0f);
    const bool head   = norm == ops::QkNorm::Head;
    const auto bq     = random_bf16_values(qw, 213, -1.0f, 1.0f);
    const auto bk     = random_bf16_values(kw, 217, -1.0f, 1.0f);
    const auto bv     = random_bf16_values(kw, 219, -1.0f, 1.0f);
    const auto wq     = random_bf16_values(head ? d : qw, 223, 0.2f, 1.8f);
    const auto wk     = random_bf16_values(head ? d : kw, 229, 0.2f, 1.8f);
    std::vector<float> inv_freq(d / 2);
    for (int i = 0; i < d / 2; ++i) { inv_freq[i] = float(std::pow(5e5, -2.0 * i / d)); }
    std::vector<std::int32_t> positions(columns);
    for (int b = 0; b < rows; ++b) {
        for (int t = 0; t < row_tokens; ++t) { positions[b * row_tokens + t] = row_base[b] + t; }
    }
    auto d_source    = upload(bits(source));
    auto d_bq        = upload(bits(bq));
    auto d_bk        = upload(bits(bk));
    auto d_bv        = upload(bits(bv));
    auto d_wq        = upload(bits(wq));
    auto d_wk        = upload(bits(wk));
    auto d_freq      = upload(inv_freq);
    auto d_positions = upload(positions);
    auto* base       = static_cast<std::uint16_t*>(d_source.data());
    const auto prepare = [&](int first, int count, const Tensor& pos, void* q_out,
                             const ops::Gemma4KvLayer& cache) {
        const auto slice = [&](int offset, int width) {
            Tensor t(base + std::size_t(first) * stride + offset, DType::BF16, {width, count});
            t.nb[1] = static_cast<std::int64_t>(stride) * 2;
            return t;
        };
        ops::DenseQkvPrepare a;
        a.q_src = slice(0, qw);
        a.k_src = slice(qw, kw);
        a.v_src = slice(qw + kw, kw);
        if (bias) {
            a.q_bias = Tensor(d_bq.data(), DType::BF16, {qw});
            a.k_bias = Tensor(d_bk.data(), DType::BF16, {kw});
            a.v_bias = Tensor(d_bv.data(), DType::BF16, {kw});
        }
        a.qk_norm    = norm;
        a.q_norm     = Tensor(d_wq.data(), DType::BF16, {head ? d : qw});
        a.k_norm     = Tensor(d_wk.data(), DType::BF16, {head ? d : kw});
        a.eps        = 1e-6f;
        a.inv_freq   = Tensor(d_freq.data(), DType::FP32, {d / 2});
        a.rotary_dim = d;
        a.positions  = pos;
        a.q_out      = Tensor(q_out, DType::BF16, {d, q_heads, count});
        a.cache      = cache;
        ops::dense_qkv_prepare(a, nullptr);
    };

    // The single-sequence calls: each row alone into its own slot storage.
    constexpr int reference_capacity = 256;
    const auto reference_plane       = std::size_t(d) * reference_capacity * kv_heads;
    std::vector<GuardedDeviceBuffer> ref_k, ref_v;
    GuardedDeviceBuffer ref_q(std::size_t(qw) * columns * 2);
    for (int b = 0; b < rows; ++b) {
        ref_k.push_back(upload(std::vector<std::uint16_t>(reference_plane, 0)));
        ref_v.push_back(upload(std::vector<std::uint16_t>(reference_plane, 0)));
        ops::Gemma4KvLayer cache;
        cache.k = Tensor(ref_k.back().data(), DType::BF16, {d, reference_capacity, kv_heads});
        cache.v = Tensor(ref_v.back().data(), DType::BF16, {d, reference_capacity, kv_heads});
        const Tensor pos(static_cast<std::int32_t*>(d_positions.data()) + b * row_tokens,
                         DType::I32, {row_tokens});
        prepare(b * row_tokens, row_tokens,
                pos, static_cast<std::uint16_t*>(ref_q.data()) + std::size_t(qw) * b * row_tokens,
                cache);
    }

    // The batched call: row b bound to binding 1 - b.
    const std::vector<std::int32_t> bound_rows{1, 0};
    auto d_rows = upload(bound_rows);
    constexpr int ring = 16, pages = 8, pages_per_row = 4;
    // Row 0 (binding 1, positions 61..65) takes pages 5 and 2; row 1 (binding 0, 190..194) pages 3
    // and 1.
    const std::vector<std::int32_t> tables{0, 7, 3, 1, /* table row 1: */ 5, 2, 6, 4};
    auto d_tables = upload(tables);
    const auto bound_elements = paged ? std::size_t(d) * 64 * pages * kv_heads
                                      : std::size_t(d) * ring * kv_heads * rows;
    auto bound_k = upload(std::vector<std::uint16_t>(bound_elements, 0));
    auto bound_v = upload(std::vector<std::uint16_t>(bound_elements, 0));
    GuardedDeviceBuffer bound_q(std::size_t(qw) * columns * 2);
    ops::Gemma4KvLayer cache;
    if (paged) {
        cache.k            = Tensor(bound_k.data(), DType::BF16, {d, 64, pages, kv_heads});
        cache.v            = Tensor(bound_v.data(), DType::BF16, {d, 64, pages, kv_heads});
        cache.block_tables = Tensor(d_tables.data(), DType::I32, {pages_per_row, 2});
    } else {
        cache.k      = Tensor(bound_k.data(), DType::BF16, {d, ring, kv_heads, rows});
        cache.v      = Tensor(bound_v.data(), DType::BF16, {d, ring, kv_heads, rows});
        cache.window = ring - row_tokens;
    }
    cache.rows = Tensor(d_rows.data(), DType::I32, {rows});
    prepare(0, columns, Tensor(d_positions.data(), DType::I32, {columns}), bound_q.data(), cache);
    cuda_synchronize();

    int failures = 0;
    if (from_device<std::uint16_t>(bound_q.data(), std::size_t(qw) * columns) !=
        from_device<std::uint16_t>(ref_q.data(), std::size_t(qw) * columns)) {
        std::cerr << label << ": batched queries differ from the single-sequence calls\n";
        ++failures;
    }
    const auto got_k = from_device<std::uint16_t>(bound_k.data(), bound_elements);
    const auto got_v = from_device<std::uint16_t>(bound_v.data(), bound_elements);
    std::size_t written = 0;
    for (int b = 0; b < rows; ++b) {
        const auto want_k = from_device<std::uint16_t>(ref_k[b].data(), reference_plane);
        const auto want_v = from_device<std::uint16_t>(ref_v[b].data(), reference_plane);
        const int binding = bound_rows[b];
        for (int t = 0; t < row_tokens; ++t) {
            const int p = row_base[b] + t;
            for (int g = 0; g < kv_heads; ++g) {
                std::size_t at;
                if (paged) {
                    const int page = tables[std::size_t(binding) * pages_per_row + p / 64];
                    at = (std::size_t(g) * 64 * pages + std::size_t(page) * 64 + p % 64) * d;
                } else {
                    at = ((std::size_t(binding) * kv_heads + g) * ring + p % ring) * d;
                }
                const std::size_t from = (std::size_t(g) * reference_capacity + p) * d;
                if (!std::equal(want_k.begin() + from, want_k.begin() + from + d,
                                got_k.begin() + at) ||
                    !std::equal(want_v.begin() + from, want_v.begin() + from + d,
                                got_v.begin() + at)) {
                    std::cerr << label << ": row " << b << " position " << p << " head " << g
                              << " differs from the single-sequence call\n";
                    ++failures;
                }
                written += 2 * std::size_t(d);
            }
        }
    }
    // Exactly the addressed rows: every other element keeps its zero fill.
    const auto nonzero = [](const std::vector<std::uint16_t>& v) {
        return static_cast<std::size_t>(
            std::count_if(v.begin(), v.end(), [](std::uint16_t w) { return w != 0; }));
    };
    if (nonzero(got_k) + nonzero(got_v) > written) {
        std::cerr << label << ": writes outside the addressed rows\n";
        ++failures;
    }
    failures += bound_q.verify_guards(label + " q guards");
    failures += bound_k.verify_guards(label + " k guards");
    failures += bound_v.verify_guards(label + " v guards");
    std::cout << label << (failures == 0 ? ": ok\n" : ": FAILED\n");
    return failures;
}

// ---- residual_norm ----------------------------------------------------------------------------

// post_eps > 0: the post-norm's own epsilon; out_offset 0/1: the output norm's own gain
// convention (-1: the post-norm's).
int run_residual(const std::string& label, int width, int tokens, bool bias, bool post,
                 float scale, bool emit, bool unit_offset = false, double post_eps = 0,
                 int out_offset = -1) {
    const bool out_unit = out_offset < 0 ? unit_offset : out_offset == 1;
    const auto x  = random_bf16_values(std::size_t(width) * tokens, 121 + width, -6.0f, 6.0f);
    const auto b  = random_bf16_values(std::size_t(width) * tokens, 131 + width, -30.0f, 30.0f);
    const auto bb = random_bf16_values(width, 141 + width, -2.0f, 2.0f);
    // Gemma's norm weights sit around zero: the gain 1 + w is what keeps them positive.
    const float lo = unit_offset ? -0.9f : 0.1f, hi = unit_offset ? 1.0f : 2.0f;
    const auto wp = random_bf16_values(width, 151 + width, lo, hi);
    const auto wo = random_bf16_values(width, 161 + width, out_unit ? -0.9f : 0.1f,
                                       out_unit ? 1.0f : 2.0f);
    const double offset     = unit_offset ? 1.0 : 0.0;
    const double out_gain_0 = out_unit ? 1.0 : 0.0;
    auto d_x = upload(bits(x)), d_b = upload(bits(b)), d_bb = upload(bits(bb));
    auto d_wp = upload(bits(wp)), d_wo = upload(bits(wo));
    GuardedDeviceBuffer d_o(std::size_t(width) * tokens * 2);
    Tensor tx(d_x.data(), DType::BF16, {width, tokens});
    Tensor tb(d_b.data(), DType::BF16, {width, tokens});
    Tensor tbb(d_bb.data(), DType::BF16, {width});
    Tensor twp(d_wp.data(), DType::BF16, {width});
    Tensor two(d_wo.data(), DType::BF16, {width});
    Tensor to(d_o.data(), DType::BF16, {width, tokens});
    const double eps = 1e-6;
    const double branch_eps = post_eps > 0 ? post_eps : eps;
    ops::residual_norm(
        {.x               = tx,
         .branch          = tb,
         .bias            = bias ? tbb : Tensor{},
         .post_weight     = post ? twp : Tensor{},
         .scale           = scale,
         .eps             = float(eps),
         .post_eps        = post_eps > 0 ? std::optional<float>(float(post_eps)) : std::nullopt,
         .unit_offset     = unit_offset,
         .out_weight      = emit ? two : Tensor{},
         .out             = emit ? to : Tensor{},
         .out_unit_offset = out_offset < 0 ? std::nullopt : std::optional<bool>(out_unit)},
        nullptr);
    cuda_synchronize();
    std::vector<double> ref_x, ref_o, unscaled_x;
    for (int t = 0; t < tokens; ++t) {
        const std::size_t off = std::size_t(t) * width;
        std::vector<double> branch(width);
        for (int i = 0; i < width; ++i) { branch[i] = double(b[off + i]) + (bias ? bb[i] : 0.0); }
        double factor = scale;
        if (post) {
            double s = 0;
            for (const double v : branch) { s += v * v; }
            factor /= std::sqrt(s / width + branch_eps);
        }
        std::vector<float> updated(width);
        for (int i = 0; i < width; ++i) {
            const double value = x[off + i] + branch[i] * factor * (post ? wp[i] + offset : 1.0);
            updated[i]         = bf16_to_f32(f32_to_bf16(static_cast<float>(value)));
            ref_x.push_back(value);
            unscaled_x.push_back(x[off + i] + branch[i] * (factor / scale) *
                                                   (post ? wp[i] + offset : 1.0));
        }
        std::vector<float> gain(wo);
        for (float& g : gain) { g = static_cast<float>(g + out_gain_0); }
        const auto o = rms_normalize(updated.data(), width, eps, gain.data());
        ref_o.insert(ref_o.end(), o.begin(), o.end());
    }
    const auto got_x = download_bf16(d_x.data(), ref_x.size());
    int failures     = verify_pointwise(label + " x", got_x, ref_x, kCriterion);
    if (emit) {
        failures += verify_pointwise(label + " out", download_bf16(d_o.data(), ref_o.size()), ref_o,
                                     kCriterion);
    } else if (from_device<std::uint8_t>(d_o.data(), d_o.bytes()) !=
               std::vector<std::uint8_t>(d_o.bytes(), 0xa5)) {
        std::cerr << label << ": out written although not requested\n";
        ++failures;
    }
    failures += d_o.verify_guards(label + " guards");
    if (scale != 1.0f) {
        // Control: the residual multiplier must be observable.
        const double distance = max_abs_difference(got_x, unscaled_x);
        std::cout << label << " control: x vs unscaled update max_abs=" << distance << '\n';
        if (distance < 0.25) {
            std::cerr << label << " control: scale not observable\n";
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
    int failures = 0;
    // Llama / Mistral / Granite: no q/k norm, full rotary, GQA 4.
    failures += run_qkv({"llama D128 32/8", 128, 32, 8, 5, 0, ops::QkNorm::None, 128, 5e5, 1.0f,
                         false, 0.0f, 0});
    // A prefill-width call (one block per token) far into the context.
    failures += run_qkv({"llama D128 32/8 prefill", 128, 32, 8, 40, 200000, ops::QkNorm::None,
                         128, 1e7, 1.0f, false, 0.0f, 0});
    // OLMo 3: projection-wide q/k norm, MHA, YaRN attention factor.
    failures += run_qkv({"olmo D128 8/8 projection norm", 128, 8, 8, 3, 9000,
                         ops::QkNorm::Projection, 128, 5e5, 1.2079f, false, 0.0f, 0});
    failures += run_qkv({"olmo prefill projection norm", 128, 4, 4, 37, 100,
                         ops::QkNorm::Projection, 128, 5e5, 1.0f, false, 0.0f, 0});
    // Qwen3-style per-head norm with biases (Qwen2/Seed-OSS carry q/k/v biases).
    failures += run_qkv({"head norm + bias D128 16/2", 128, 16, 2, 7, 33, ops::QkNorm::Head, 128,
                         1e6, 1.0f, true, 0.0f, 0});
    // Partial rotary (a quarter of the head), biases, no norm.
    failures += run_qkv({"partial rotary D128 8/2", 128, 8, 2, 3, 700, ops::QkNorm::None, 32, 1e4,
                         1.0f, true, 0.0f, 0});
    // D256, NoPE layer, and the Mistral 3 query temperature across several periods.
    failures += run_qkv({"nope D256 8/4 temperature", 256, 8, 4, 9, 30, ops::QkNorm::None, 0, 1e4,
                         1.0f, false, 0.1f, 4});
    failures += run_qkv({"mistral3 D128 32/8 temperature", 128, 32, 8, 4, 16383, ops::QkNorm::None,
                         128, 1e6, 1.0f, false, 0.1f, 16384});
    // gpt-oss: a 64-wide head, q/k/v biases, YaRN's attention factor, decode and prefill widths.
    failures += run_qkv({"gpt-oss D64 64/8 yarn + bias", 64, 64, 8, 1, 131000, ops::QkNorm::None,
                         64, 1.5e5, 1.3466f, true, 0.0f, 0});
    failures += run_qkv({"gpt-oss D64 64/8 prefill", 64, 64, 8, 33, 70, ops::QkNorm::None, 64,
                         1.5e5, 1.3466f, true, 0.0f, 0});
    // gpt-oss's DFlash drafter: per-head q/k norm after the biases, YaRN, keys and values as
    // columns: its block (with queries, two rows of eight) and its committed context (none).
    {
        QkvCase block{"dflash D64 64/8 block columns", 64, 64, 8, 16, 4090, ops::QkNorm::Head, 64,
                      1.5e5, 1.3466f, true, 0.0f, 0};
        block.columns = true;
        failures += run_qkv(block);
        QkvCase context{"dflash D64 64/8 context columns", 64, 64, 8, 37, 12, ops::QkNorm::Head, 64,
                        1.5e5, 1.3466f, true, 0.0f, 0};
        context.columns    = true;
        context.no_queries = true;
        failures += run_qkv(context);
        QkvCase muse{"dflash D128 32/8 block columns", 128, 32, 8, 8, 100000, ops::QkNorm::Head,
                     128, 1e6, 1.0f, false, 0.0f, 0};
        muse.columns = true;
        failures += run_qkv(muse);
    }
    // Gemma 3: per-head q/k norm with the unit-offset gain, D256, one KV head (1B) and two (4B).
    {
        QkvCase gemma{"gemma3 D256 4/1 head norm 1+w", 256, 4, 1, 6, 510, ops::QkNorm::Head, 256,
                      1e4, 1.0f, false, 0.0f, 0};
        gemma.unit_offset = true;
        failures += run_qkv(gemma);
        gemma = {"gemma3 D256 8/4 prefill 1+w", 256, 8, 4, 40, 3000, ops::QkNorm::Head, 256, 1e6,
                 1.0f, false, 0.0f, 0};
        gemma.unit_offset = true;
        failures += run_qkv(gemma);
    }
    failures += run_norm_scope_control();
    failures += run_qkv_bound("bound images, head norm + bias", ops::QkNorm::Head, true, false);
    failures += run_qkv_bound("bound pages, head norm", ops::QkNorm::Head, false, true);
    failures += run_qkv_bound("bound pages, projection norm", ops::QkNorm::Projection, false,
                              true);
    failures += run_residual("residual pre-norm 4096", 4096, 3, false, false, 1.0f, true);
    failures += run_residual("residual post-norm 3072", 3072, 17, false, true, 1.0f, false);
    failures += run_residual("residual granite scale 2048", 2048, 2, true, true, 0.22f, true);
    failures += run_residual("residual decode 8192", 8192, 1, false, false, 0.5f, true);
    // Gemma 3's sandwich: post-norm of the branch and the next pre-norm, both gains 1 + w.
    failures += run_residual("residual sandwich 1+w 1152", 1152, 5, false, true, 1.0f, true, true);
    failures += run_residual("residual sandwich 1+w decode 2560", 2560, 1, false, true, 1.0f, true,
                             true);
    // Muse Glimmer: 1+w post-norm with its own eps 1e-8, and the last layer's update feeding a
    // final norm with the plain gain w.
    failures += run_residual("residual muse post eps, plain final 6656", 6656, 3, false, true, 1.0f,
                             true, true, 1e-8, 0);
    failures += run_residual("residual muse post eps, 1+w next 6656", 6656, 1, false, true, 1.0f,
                             true, true, 1e-8, 1);
    if (failures != 0) {
        std::cerr << "dense_fused: " << failures << " checks failed\n";
        return 1;
    }
    std::cout << "dense_fused: all checks passed\n";
    return 0;
}
