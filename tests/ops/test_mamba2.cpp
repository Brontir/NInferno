// The Mamba2 selective state scan (ops::mamba2_scan) against an FP64 reference, and the gated
// group norm (ops::mamba2_gated_norm, check_gated_norm).
//
// Geometries: NemotronH's mixer (H 64, P 64, G 8, N 128) and small ones whose P is not a whole
// number of CTA runs of the widest shape (P 12, 24), N of 16, 64 and 128, one and several groups.
// Rows: B in {1, 3} over a slot pool with the rows' slots out of order; L in {1, 5, 40} tokens per
// row; dt_max 0 (no clamp) and 0.1.
//
// Pass: every y within 2^-8 relative plus 1e-4 of its term magnitude (sum |s_n C_n| + |D x|) of
// the FP64 recursion on the represented BF16 inputs and FP32 states; every written state within
// 2e-4 relative (plus 1e-5) of the reference; the pool's other slots untouched; the call split
// into L single-token calls gives the same bytes (y and state) as the one L-token call; two runs
// are bitwise identical.
//
// The speculative verify pair: mamba2_conv with an input record and mamba2_scan with
// store_state = false over B rows of W columns leave both state pools untouched and give the
// in-place calls' y bytes; mamba2_verify_fold of every commit (0..W, per row, into the source slot
// and into another slot) leaves the bytes the in-place conv + scan over the committed columns
// leave, and touches no other slot.

#include "ninferno/ops/mamba2.h"

#include "ops/op_tester.h"
#include "core/fnv.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

struct Case {
    ops::Mamba2Geometry g;
    int batch = 1, row_tokens = 1;
};

std::vector<float> uniform(std::size_t n, std::uint32_t seed, float lo, float hi, bool bf16) {
    std::vector<float> v(n);
    fill_uniform(v, seed, lo, hi);
    if (bf16) round_to_bf16(v);
    return v;
}

std::vector<std::uint16_t> to_bits(const std::vector<float>& v) {
    std::vector<std::uint16_t> b(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) b[i] = f32_to_bf16(v[i]);
    return b;
}

// One call (or L single-token calls) of the scan; returns y (BF16 bits) and the whole state pool.
struct Result {
    std::vector<std::uint16_t> y;
    std::vector<float> states;
};

Result run(const Case& c, const std::vector<float>& xbc, const std::vector<float>& dt,
           const std::vector<float>& dt_bias, const std::vector<float>& a_log,
           const std::vector<float>& d, const std::vector<float>& states0,
           const std::vector<int>& slots, bool split, DType state_dtype = DType::FP32) {
    const auto& g   = c.g;
    const int rows  = g.heads * g.head_dim + 2 * g.groups * g.state;
    const int T     = c.batch * c.row_tokens;
    const int inner = g.heads * g.head_dim;
    const int pool  = static_cast<int>(slots.size()) + 1;
    DeviceBuffer dbias = to_device_f32(dt_bias), dalog = to_device_f32(a_log), dd = to_device_f32(d);
    std::vector<std::uint16_t> half_bits(states0.size());
    for (std::size_t i = 0; i < states0.size(); ++i) {
        const __half h = __float2half_rn(states0[i]);
        std::memcpy(&half_bits[i], &h, sizeof(h));
    }
    DeviceBuffer dstates = state_dtype == DType::BF16   ? to_device_bf16(states0)
                           : state_dtype == DType::FP16 ? to_device(half_bits)
                                                        : to_device_f32(states0);
    DeviceBuffer dslots  = to_device_i32(slots);
    Tensor tb(dbias.p, DType::FP32, {g.heads});
    Tensor ta(dalog.p, DType::FP32, {g.heads});
    Tensor td(dd.p, DType::FP32, {g.heads});
    Tensor ts(dstates.p, state_dtype, {g.state, g.head_dim, g.heads, pool});
    Tensor tslots(dslots.p, DType::I32, {c.batch});
    Result r;
    r.y.assign(static_cast<std::size_t>(inner) * T, 0);
    if (!split) {
        DeviceBuffer dx = to_device_bf16(xbc), ddt = to_device_bf16(dt), dy(sizeof(std::uint16_t) * inner * T);
        Tensor tx(dx.p, DType::BF16, {rows, T});
        Tensor tdt(ddt.p, DType::BF16, {g.heads, T});
        Tensor ty(dy.p, DType::BF16, {inner, T});
        ops::mamba2_scan(tx, tdt, tb, ta, td, ts, tslots, c.batch, g, ty, nullptr);
        cuda_synchronize();
        dy.copy_to_host(r.y.data(), r.y.size() * sizeof(std::uint16_t));
    } else {
        // Token j of every row, as a call of B columns (one per row).
        for (int j = 0; j < c.row_tokens; ++j) {
            std::vector<float> xs(static_cast<std::size_t>(rows) * c.batch), ds(static_cast<std::size_t>(g.heads) * c.batch);
            for (int b = 0; b < c.batch; ++b) {
                const int col = b * c.row_tokens + j;
                std::memcpy(&xs[static_cast<std::size_t>(b) * rows], &xbc[static_cast<std::size_t>(col) * rows], sizeof(float) * rows);
                std::memcpy(&ds[static_cast<std::size_t>(b) * g.heads], &dt[static_cast<std::size_t>(col) * g.heads], sizeof(float) * g.heads);
            }
            DeviceBuffer dx = to_device_bf16(xs), ddt = to_device_bf16(ds), dy(sizeof(std::uint16_t) * inner * c.batch);
            Tensor tx(dx.p, DType::BF16, {rows, c.batch});
            Tensor tdt(ddt.p, DType::BF16, {g.heads, c.batch});
            Tensor ty(dy.p, DType::BF16, {inner, c.batch});
            ops::mamba2_scan(tx, tdt, tb, ta, td, ts, tslots, c.batch, g, ty, nullptr);
            cuda_synchronize();
            std::vector<std::uint16_t> part(static_cast<std::size_t>(inner) * c.batch);
            dy.copy_to_host(part.data(), part.size() * sizeof(std::uint16_t));
            for (int b = 0; b < c.batch; ++b) {
                const int col = b * c.row_tokens + j;
                std::memcpy(&r.y[static_cast<std::size_t>(col) * inner], &part[static_cast<std::size_t>(b) * inner], sizeof(std::uint16_t) * inner);
            }
        }
    }
    r.states.resize(states0.size());
    if (state_dtype != DType::FP32) {
        std::vector<std::uint16_t> bits(states0.size());
        dstates.copy_to_host(bits.data(), bits.size() * sizeof(std::uint16_t));
        for (std::size_t i = 0; i < bits.size(); ++i) {
            __half h;
            std::memcpy(&h, &bits[i], sizeof(h));
            r.states[i] = state_dtype == DType::BF16 ? bf16_to_f32(bits[i]) : __half2float(h);
        }
    } else {
        dstates.copy_to_host(r.states.data(), r.states.size() * sizeof(float));
    }
    return r;
}

double relative_l2(const std::vector<double>& a, const std::vector<double>& ref) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num += (a[i] - ref[i]) * (a[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// --state-dtype bf16 / f16: the same calls over a BF16 and an FP16 state pool (single-token calls, as at decode,
// or one prefill call) against the FP32 pool from the same (BF16-representable) start. The
// recurrence is FP32 in registers; only the stored state is rounded, once per call. Pass: y and the
// final states within 2e-2 relative L2 of the FP32 arm.
int check_bf16_state(const ops::Mamba2Geometry& g, int batch, int row_tokens, bool split,
                     std::uint32_t seed) {
    const Case c{g, batch, row_tokens};
    const int rows = g.heads * g.head_dim + 2 * g.groups * g.state;
    const int T    = batch * row_tokens;
    std::vector<int> slots(batch);
    for (int b = 0; b < batch; ++b) slots[b] = batch - b;
    const std::size_t slot_size = static_cast<std::size_t>(g.state) * g.head_dim * g.heads;
    const auto xbc     = uniform(static_cast<std::size_t>(rows) * T, seed + 1, -1.0F, 1.0F, true);
    const auto dt      = uniform(static_cast<std::size_t>(g.heads) * T, seed + 2, -4.0F, 1.5F, true);
    const auto dt_bias = uniform(g.heads, seed + 3, -1.0F, 1.0F, false);
    const auto a_log   = uniform(g.heads, seed + 4, -1.0F, 2.0F, false);
    const auto d       = uniform(g.heads, seed + 5, -1.0F, 1.0F, false);
    const auto states0 = uniform(slot_size * (batch + 1), seed + 6, -1.0F, 1.0F, true);
    const Result f32 = run(c, xbc, dt, dt_bias, a_log, d, states0, slots, split);
    std::vector<double> yf(f32.y.size());
    for (std::size_t i = 0; i < yf.size(); ++i) yf[i] = bf16_to_f32(f32.y[i]);
    const std::vector<double> sf(f32.states.begin(), f32.states.end());
    int failures = 0;
    for (const DType dtype : {DType::BF16, DType::FP16}) {
        const Result half = run(c, xbc, dt, dt_bias, a_log, d, states0, slots, split, dtype);
        std::vector<double> yb(half.y.size());
        for (std::size_t i = 0; i < yb.size(); ++i) yb[i] = bf16_to_f32(half.y[i]);
        const std::vector<double> sb(half.states.begin(), half.states.end());
        const double y_err = relative_l2(yb, yf), s_err = relative_l2(sb, sf);
        std::cout << (dtype == DType::BF16 ? "MAMBA2_BF16_STATE H" : "MAMBA2_F16_STATE H") << g.heads
                  << " N" << g.state << " B" << batch << " L" << row_tokens
                  << (split ? " decode" : " prefill") << " y_rel_l2=" << y_err
                  << " state_rel_l2=" << s_err << '\n';
        if (!(y_err < 2e-2 && s_err < 2e-2)) ++failures;
    }
    return failures;
}

int check_case(const Case& c, std::uint32_t seed) {
    const auto& g   = c.g;
    const int rows  = g.heads * g.head_dim + 2 * g.groups * g.state;
    const int T     = c.batch * c.row_tokens;
    const int inner = g.heads * g.head_dim;
    const std::size_t head_state = static_cast<std::size_t>(g.state) * g.head_dim;
    const std::size_t slot_size  = head_state * g.heads;
    // A pool of B + 1 slots; row b takes slot B - b, so the rows' slots are out of order and slot 0
    // is nobody's.
    std::vector<int> slots(c.batch);
    for (int b = 0; b < c.batch; ++b) slots[b] = c.batch - b;
    const int pool = c.batch + 1;

    const auto xbc     = uniform(static_cast<std::size_t>(rows) * T, seed + 1, -1.0F, 1.0F, true);
    const auto dt      = uniform(static_cast<std::size_t>(g.heads) * T, seed + 2, -4.0F, 1.5F, true);
    const auto dt_bias = uniform(g.heads, seed + 3, -1.0F, 1.0F, false);
    const auto a_log   = uniform(g.heads, seed + 4, -1.0F, 2.0F, false);
    const auto d       = uniform(g.heads, seed + 5, -1.0F, 1.0F, false);
    const auto states0 = uniform(slot_size * pool, seed + 6, -1.0F, 1.0F, false);

    const std::string label = "scan H" + std::to_string(g.heads) + " P" + std::to_string(g.head_dim) +
                              " G" + std::to_string(g.groups) + " N" + std::to_string(g.state) +
                              " B" + std::to_string(c.batch) + " L" + std::to_string(c.row_tokens) +
                              (g.dt_max > 0.0F ? " dt_max" : "");

    // FP64 reference.
    std::vector<double> ref_y(static_cast<std::size_t>(inner) * T), mag_y(ref_y.size());
    std::vector<double> ref_s(states0.begin(), states0.end());
    for (int b = 0; b < c.batch; ++b) {
        double* s = &ref_s[static_cast<std::size_t>(slots[b]) * slot_size];
        for (int j = 0; j < c.row_tokens; ++j) {
            const int col = b * c.row_tokens + j;
            const float* column = &xbc[static_cast<std::size_t>(col) * rows];
            for (int h = 0; h < g.heads; ++h) {
                const int grp = h / (g.heads / g.groups);
                const double v = static_cast<double>(dt[static_cast<std::size_t>(col) * g.heads + h]) + dt_bias[h];
                double step = v > 20.0 ? v : std::log1p(std::exp(v));
                step = std::max(step, static_cast<double>(g.dt_min));
                if (g.dt_max > 0.0F) step = std::min(step, static_cast<double>(g.dt_max));
                const double decay = std::exp(-step * std::exp(static_cast<double>(a_log[h])));
                for (int p = 0; p < g.head_dim; ++p) {
                    const double x = column[h * g.head_dim + p];
                    double acc = 0.0, mag = std::fabs(d[h] * x);
                    for (int n = 0; n < g.state; ++n) {
                        double& e = s[static_cast<std::size_t>(h) * head_state + static_cast<std::size_t>(p) * g.state + n];
                        e = e * decay + step * column[inner + grp * g.state + n] * x;
                        const double cn = column[inner + (g.groups + grp) * g.state + n];
                        acc += e * cn;
                        mag += std::fabs(e * cn);
                    }
                    const std::size_t o = static_cast<std::size_t>(col) * inner + h * g.head_dim + p;
                    ref_y[o] = acc + d[h] * x;
                    mag_y[o] = mag;
                }
            }
        }
    }

    int failures = 0;
    const Result one = run(c, xbc, dt, dt_bias, a_log, d, states0, slots, false);
    std::int64_t bad = 0;
    double worst = 0.0;
    for (std::size_t i = 0; i < ref_y.size(); ++i) {
        const double got = bf16_to_f32(one.y[i]);
        const double err = std::fabs(got - ref_y[i]);
        const double lim = std::ldexp(std::fabs(ref_y[i]), -8) + 1e-4 * mag_y[i] + 1e-6;
        worst = std::max(worst, err / lim);
        if (!(err <= lim) && bad++ < 3) {
            std::printf("  %s: y[%zu] got %.6g ref %.6g (limit %.3g)\n", label.c_str(), i, got, ref_y[i], lim);
        }
    }
    for (std::size_t i = 0; i < ref_s.size(); ++i) {
        const double err = std::fabs(one.states[i] - ref_s[i]);
        const double lim = 2e-4 * std::fabs(ref_s[i]) + 1e-5;
        if (!(err <= lim) && bad++ < 6) {
            std::printf("  %s: state[%zu] got %.8g ref %.8g\n", label.c_str(), i, one.states[i], ref_s[i]);
        }
    }
    if (bad != 0) {
        std::printf("FAIL %s: %lld values out of bounds\n", label.c_str(), static_cast<long long>(bad));
        ++failures;
    }
    // The unused slot keeps its bytes (checked above against the untouched reference), the split
    // call and a second run give the same bytes.
    const Result again = run(c, xbc, dt, dt_bias, a_log, d, states0, slots, false);
    if (again.y != one.y || std::memcmp(again.states.data(), one.states.data(), one.states.size() * sizeof(float)) != 0) {
        std::printf("FAIL %s: two runs differ\n", label.c_str());
        ++failures;
    }
    if (c.row_tokens > 1) {
        const Result split = run(c, xbc, dt, dt_bias, a_log, d, states0, slots, true);
        if (split.y != one.y || std::memcmp(split.states.data(), one.states.data(), one.states.size() * sizeof(float)) != 0) {
            std::printf("FAIL %s: single-token calls differ from the %d-token call\n", label.c_str(), c.row_tokens);
            ++failures;
        }
    }
    if (failures == 0) std::printf("ok   %s (worst y error %.2f of limit)\n", label.c_str(), worst);
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Verify record + fold.

struct Pools {
    std::vector<std::uint16_t> conv; // BF16 [C, 3, S]
    std::vector<float> ssm;          // FP32 [N, P, H, S]
};

struct Chain {
    ops::Mamba2Geometry g;
    int channels = 0, pool = 0;
    std::vector<float> x, weight, bias, dt, dt_bias, a_log, d; // x [C, T], dt [H, T]
};

// In place: conv + scan over columns [first, first + count) of x/dt for the one row in `slot`.
// Returns y (BF16 bits) of those columns; the pools advance.
std::vector<std::uint16_t> run_in_place(const Chain& c, Pools& pools, int slot, int first, int count) {
    const auto& g   = c.g;
    const int inner = g.heads * g.head_dim;
    if (count == 0) return {};
    std::vector<float> xs(static_cast<std::size_t>(c.channels) * count), ds(static_cast<std::size_t>(g.heads) * count);
    std::memcpy(xs.data(), &c.x[static_cast<std::size_t>(first) * c.channels], sizeof(float) * xs.size());
    std::memcpy(ds.data(), &c.dt[static_cast<std::size_t>(first) * g.heads], sizeof(float) * ds.size());
    DeviceBuffer dx = to_device_bf16(xs), ddt = to_device_bf16(ds), dw = to_device_bf16(c.weight),
                 db = to_device_bf16(c.bias), dbias = to_device_f32(c.dt_bias), dalog = to_device_f32(c.a_log),
                 dd = to_device_f32(c.d), dconv = to_device(pools.conv), dssm = to_device_f32(pools.ssm),
                 dslot = to_device_i32(std::vector<int>{slot});
    DeviceBuffer dout(sizeof(std::uint16_t) * c.channels * count), dy(sizeof(std::uint16_t) * inner * count);
    Tensor tx(dx.p, DType::BF16, {c.channels, count}), tdt(ddt.p, DType::BF16, {g.heads, count});
    Tensor tw(dw.p, DType::BF16, {4, c.channels}), tb(db.p, DType::BF16, {c.channels});
    Tensor tbias(dbias.p, DType::FP32, {g.heads}), talog(dalog.p, DType::FP32, {g.heads}), td(dd.p, DType::FP32, {g.heads});
    Tensor tconv(dconv.p, DType::BF16, {c.channels, 3, c.pool});
    Tensor tssm(dssm.p, DType::FP32, {g.state, g.head_dim, g.heads, c.pool});
    Tensor tslot(dslot.p, DType::I32, {1});
    Tensor tout(dout.p, DType::BF16, {c.channels, count}), ty(dy.p, DType::BF16, {inner, count});
    ops::mamba2_conv(tx, tw, tb, tconv, tslot, 1, tout, nullptr);
    ops::mamba2_scan(tout, tdt, tbias, talog, td, tssm, tslot, 1, g, ty, nullptr);
    cuda_synchronize();
    dconv.copy_to_host(pools.conv.data(), pools.conv.size() * sizeof(std::uint16_t));
    dssm.copy_to_host(pools.ssm.data(), pools.ssm.size() * sizeof(float));
    std::vector<std::uint16_t> y(static_cast<std::size_t>(inner) * count);
    dy.copy_to_host(y.data(), y.size() * sizeof(std::uint16_t));
    return y;
}

bool same(const Pools& a, const Pools& b) {
    return a.conv == b.conv && std::memcmp(a.ssm.data(), b.ssm.data(), a.ssm.size() * sizeof(float)) == 0;
}

int check_fold(const ops::Mamba2Geometry& g, int batch, int width, std::uint32_t seed) {
    Chain c;
    c.g        = g;
    c.channels = g.heads * g.head_dim + 2 * g.groups * g.state;
    c.pool     = 2 * batch + 1; // rows' slots 1..B, spare dest slots B+1..2B, slot 0 nobody's
    const int inner = g.heads * g.head_dim;
    const int T     = batch * width;
    c.x       = uniform(static_cast<std::size_t>(c.channels) * T, seed + 1, -1.0F, 1.0F, true);
    c.weight  = uniform(static_cast<std::size_t>(4) * c.channels, seed + 2, -0.6F, 0.6F, true);
    c.bias    = uniform(c.channels, seed + 3, -0.3F, 0.3F, true);
    c.dt      = uniform(static_cast<std::size_t>(g.heads) * T, seed + 4, -4.0F, 1.5F, true);
    c.dt_bias = uniform(g.heads, seed + 5, -1.0F, 1.0F, false);
    c.a_log   = uniform(g.heads, seed + 6, -1.0F, 2.0F, false);
    c.d       = uniform(g.heads, seed + 7, -1.0F, 1.0F, false);
    Pools start;
    start.conv = to_bits(uniform(static_cast<std::size_t>(c.channels) * 3 * c.pool, seed + 8, -1.0F, 1.0F, true));
    start.ssm  = uniform(static_cast<std::size_t>(g.state) * g.head_dim * g.heads * c.pool, seed + 9, -1.0F, 1.0F, false);
    std::vector<int> slots(batch);
    for (int b = 0; b < batch; ++b) slots[b] = batch - b; // out of order

    const std::string label = "fold H" + std::to_string(g.heads) + " P" + std::to_string(g.head_dim) +
                              " N" + std::to_string(g.state) + " B" + std::to_string(batch) + " W" +
                              std::to_string(width);
    int failures = 0;

    // The verify pair over all B x W columns.
    DeviceBuffer dx = to_device_bf16(c.x), ddt = to_device_bf16(c.dt), dw = to_device_bf16(c.weight),
                 db = to_device_bf16(c.bias), dbias = to_device_f32(c.dt_bias), dalog = to_device_f32(c.a_log),
                 dd = to_device_f32(c.d), dconv = to_device(start.conv), dssm = to_device_f32(start.ssm),
                 dslots = to_device_i32(slots);
    DeviceBuffer din_rec(sizeof(std::uint16_t) * c.channels * T), dxbc_rec(sizeof(std::uint16_t) * c.channels * T),
        ddt_rec(sizeof(std::uint16_t) * g.heads * T), dy(sizeof(std::uint16_t) * inner * T);
    Tensor tx(dx.p, DType::BF16, {c.channels, T}), tdt(ddt.p, DType::BF16, {g.heads, T});
    Tensor tw(dw.p, DType::BF16, {4, c.channels}), tb(db.p, DType::BF16, {c.channels});
    Tensor tbias(dbias.p, DType::FP32, {g.heads}), talog(dalog.p, DType::FP32, {g.heads}), td(dd.p, DType::FP32, {g.heads});
    Tensor tconv(dconv.p, DType::BF16, {c.channels, 3, c.pool});
    Tensor tssm(dssm.p, DType::FP32, {g.state, g.head_dim, g.heads, c.pool});
    Tensor tslots(dslots.p, DType::I32, {batch});
    Tensor tin_rec(din_rec.p, DType::BF16, {c.channels, T}), txbc_rec(dxbc_rec.p, DType::BF16, {c.channels, T});
    Tensor tdt_rec(ddt_rec.p, DType::BF16, {g.heads, T}), ty(dy.p, DType::BF16, {inner, T});
    ops::mamba2_conv(tx, tw, tb, tconv, tslots, batch, txbc_rec, nullptr, &tin_rec);
    ops::mamba2_scan(txbc_rec, tdt, tbias, talog, td, tssm, tslots, batch, g, ty, nullptr, false, &tdt_rec);
    cuda_synchronize();
    Pools after_verify{start.conv, start.ssm};
    dconv.copy_to_host(after_verify.conv.data(), after_verify.conv.size() * sizeof(std::uint16_t));
    dssm.copy_to_host(after_verify.ssm.data(), after_verify.ssm.size() * sizeof(float));
    if (!same(after_verify, start)) {
        std::printf("FAIL %s: the verify pair wrote a state\n", label.c_str());
        ++failures;
    }
    std::vector<std::uint16_t> verify_y(static_cast<std::size_t>(inner) * T);
    dy.copy_to_host(verify_y.data(), verify_y.size() * sizeof(std::uint16_t));
    for (int b = 0; b < batch; ++b) {
        Pools ref = start;
        const auto y = run_in_place(c, ref, slots[b], b * width, width);
        if (std::memcmp(y.data(), &verify_y[static_cast<std::size_t>(b) * width * inner], y.size() * 2) != 0) {
            std::printf("FAIL %s: row %d's verify y differs from the in-place call's\n", label.c_str(), b);
            ++failures;
        }
    }

    // Folds: every commit pattern of a few kinds, into the source slot and into a spare slot.
    std::mt19937 rng(seed);
    for (int trial = 0; trial < 6; ++trial) {
        const bool spare = trial % 2 == 1;
        std::vector<ops::Mamba2FoldRow> rows(batch);
        for (int b = 0; b < batch; ++b) {
            const int commit = trial < 2 ? (trial == 0 ? 0 : width) : static_cast<int>(rng() % (width + 1));
            rows[b] = {slots[b], spare ? batch + 1 + b : slots[b], trial == 0 ? 0 : (trial == 1 ? width : commit)};
        }
        // Reference: per row, copy source -> dest, then in place over the committed columns.
        Pools ref = start;
        const std::size_t conv_slot = static_cast<std::size_t>(c.channels) * 3;
        const std::size_t ssm_slot  = static_cast<std::size_t>(g.state) * g.head_dim * g.heads;
        for (int b = 0; b < batch; ++b) {
            const auto& r = rows[b];
            if (r.dest_slot != r.source_slot) {
                std::memcpy(&ref.conv[r.dest_slot * conv_slot], &start.conv[r.source_slot * conv_slot], conv_slot * 2);
                std::memcpy(&ref.ssm[r.dest_slot * ssm_slot], &start.ssm[r.source_slot * ssm_slot], ssm_slot * 4);
            }
            run_in_place(c, ref, r.dest_slot, b * width, r.commit);
        }
        DeviceBuffer fconv = to_device(start.conv), fssm = to_device_f32(start.ssm);
        Tensor tfconv(fconv.p, DType::BF16, {c.channels, 3, c.pool});
        Tensor tfssm(fssm.p, DType::FP32, {g.state, g.head_dim, g.heads, c.pool});
        ops::mamba2_verify_fold(tin_rec, txbc_rec, tdt_rec, width, rows, tbias, talog, td, tfconv, tfssm, g, nullptr);
        cuda_synchronize();
        Pools got{start.conv, start.ssm};
        fconv.copy_to_host(got.conv.data(), got.conv.size() * sizeof(std::uint16_t));
        fssm.copy_to_host(got.ssm.data(), got.ssm.size() * sizeof(float));
        if (!same(got, ref)) {
            std::string commits;
            for (const auto& r : rows) commits += " " + std::to_string(r.commit);
            std::printf("FAIL %s: fold (%s, commits%s) differs from the in-place calls\n", label.c_str(),
                        spare ? "spare dest" : "in place", commits.c_str());
            ++failures;
        }
    }
    if (failures == 0) std::printf("ok   %s\n", label.c_str());
    return failures;
}

// NINFERNO_MAMBA2_BENCH=T: the NemotronH scan over one row of T tokens, timed, with an FNV hash of
// y and the final state (equal hashes across builds: equal bytes).
int bench_scan(int T) {
    const ops::Mamba2Geometry g{64, 64, 8, 128, 0.001F, 0.0F};
    const int rows  = g.heads * g.head_dim + 2 * g.groups * g.state;
    const int inner = g.heads * g.head_dim;
    const auto xbc = uniform(static_cast<std::size_t>(rows) * T, 7, -1.0F, 1.0F, true);
    const auto dt  = uniform(static_cast<std::size_t>(g.heads) * T, 8, -4.0F, 1.0F, true);
    const auto dtb = uniform(g.heads, 9, -1.0F, 1.0F, false);
    const auto alg = uniform(g.heads, 10, -1.0F, 1.5F, false);
    const auto dd  = uniform(g.heads, 11, 0.5F, 1.5F, false);
    const std::vector<float> st0(static_cast<std::size_t>(g.state) * g.head_dim * g.heads, 0.0F);
    DeviceBuffer dx = to_device_bf16(xbc), ddt = to_device_bf16(dt), dy(sizeof(std::uint16_t) * inner * T);
    DeviceBuffer dbias = to_device_f32(dtb), dalog = to_device_f32(alg), dD = to_device_f32(dd);
    DeviceBuffer dst = to_device_f32(st0), dslots = to_device_i32(std::vector<int>{0});
    Tensor tx(dx.p, DType::BF16, {rows, T});
    Tensor tdt(ddt.p, DType::BF16, {g.heads, T});
    Tensor ty(dy.p, DType::BF16, {inner, T});
    Tensor tb(dbias.p, DType::FP32, {g.heads});
    Tensor ta(dalog.p, DType::FP32, {g.heads});
    Tensor td(dD.p, DType::FP32, {g.heads});
    Tensor ts(dst.p, DType::FP32, {g.state, g.head_dim, g.heads, 1});
    Tensor tslots(dslots.p, DType::I32, {1});
    ops::mamba2_scan(tx, tdt, tb, ta, td, ts, tslots, 1, g, ty, nullptr, false);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int reps = T <= 16 ? 200 : 5;
    cudaEventRecord(e0);
    for (int i = 0; i < reps; ++i) ops::mamba2_scan(tx, tdt, tb, ta, td, ts, tslots, 1, g, ty, nullptr, false);
    cudaEventRecord(e1);
    cuda_synchronize();
    float ms = 0.0F;
    cudaEventElapsedTime(&ms, e0, e1);
    ops::mamba2_scan(tx, tdt, tb, ta, td, ts, tslots, 1, g, ty, nullptr, true);
    cuda_synchronize();
    std::vector<std::uint16_t> y(static_cast<std::size_t>(inner) * T);
    std::vector<float> st(st0.size());
    dy.copy_to_host(y.data(), y.size() * sizeof(std::uint16_t));
    dst.copy_to_host(st.data(), st.size() * sizeof(float));
    std::uint64_t hash = ninferno::fnv::kOffset;
    auto mix = [&](const void* data, std::size_t bytes) {
        const auto* c = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < bytes; ++i) hash = ninferno::fnv::mix(hash, c[i]);
    };
    mix(y.data(), y.size() * sizeof(std::uint16_t));
    mix(st.data(), st.size() * sizeof(float));
    std::printf("mamba2 bench T=%d scan %.3f ms hash %016llx\n", T, ms / reps,
                static_cast<unsigned long long>(hash));
    return 0;
}

} // namespace

// mamba2_gated_norm against FP64: out = w * rmsnorm_group(y * silu(z)) over groups of `group`
// features, z a slice of a wider plane (column stride ldz) at `z_offset`. Groups of 256..1024 take
// the warp form (an aligned z) or the CTA form (z_offset 1: unaligned); 40 the warp form with five
// lanes, 36 (rows not a multiple of 8) always the CTA form.
// Pass: every output within 2^-7 relative plus 1e-3 of the reference.
int check_gated_norm(int group, int groups, int T, int z_offset, std::uint32_t seed) {
    const int rows  = group * groups;
    const int ldz   = rows + 16;
    const auto y    = uniform(static_cast<std::size_t>(rows) * T, seed, -2.0F, 2.0F, true);
    const auto z    = uniform(static_cast<std::size_t>(ldz) * T, seed + 1, -3.0F, 3.0F, true);
    const auto w    = uniform(static_cast<std::size_t>(rows), seed + 2, 0.5F, 1.5F, true);
    const float eps = 1e-5F;
    DeviceBuffer dy = to_device_bf16(y), dz = to_device_bf16(z), dw = to_device_bf16(w);
    DeviceBuffer dout(sizeof(std::uint16_t) * rows * T);
    Tensor ty(dy.p, DType::BF16, {rows, T});
    Tensor tz = Tensor(dz.p, DType::BF16, {ldz, T}).slice(0, z_offset, rows);
    Tensor tw(dw.p, DType::BF16, {rows});
    Tensor tout(dout.p, DType::BF16, {rows, T});
    ops::mamba2_gated_norm(ty, tz, tw, group, eps, tout, nullptr);
    cuda_synchronize();
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(rows) * T);
    dout.copy_to_host(bits.data(), bits.size() * sizeof(std::uint16_t));
    int bad = 0;
    for (int t = 0; t < T; ++t) {
        for (int gi = 0; gi < groups; ++gi) {
            std::vector<double> v(group);
            double ss = 0.0;
            for (int i = 0; i < group; ++i) {
                const int r     = gi * group + i;
                const double zv = z[static_cast<std::size_t>(t) * ldz + z_offset + r];
                v[i] = y[static_cast<std::size_t>(t) * rows + r] * (zv / (1.0 + std::exp(-zv)));
                ss += v[i] * v[i];
            }
            const double inv = 1.0 / std::sqrt(ss / group + eps);
            for (int i = 0; i < group; ++i) {
                const int r      = gi * group + i;
                const double ref = w[r] * v[i] * inv;
                const double got = bf16_to_f32(bits[static_cast<std::size_t>(t) * rows + r]);
                if (std::fabs(got - ref) > std::fabs(ref) / 128.0 + 1e-3) {
                    if (bad++ < 3) {
                        std::cout << "FAIL gated_norm group " << group << " x" << groups << " T"
                                  << T << " z+" << z_offset << " t" << t << " r" << r << ": " << got
                                  << " vs " << ref << '\n';
                    }
                }
            }
        }
    }
    return bad != 0 ? 1 : 0;
}

int main() {
    if (cuda_unavailable()) {
        std::cout << "CUDA unavailable; skipping mamba2 test\n";
        return 77;
    }
    if (const char* bench = std::getenv("NINFERNO_MAMBA2_BENCH"))
        return bench_scan(std::atoi(bench));
    int failures = 0;
    try {
        const ops::Mamba2Geometry geometries[] = {
            {64, 64, 8, 128, 0.001F, 0.0F}, // NemotronH
            {8, 24, 2, 128, 0.0F, 0.1F},
            {6, 12, 3, 128, 0.001F, 0.0F},
            {8, 16, 2, 64, 0.0F, 0.0F},
            {4, 32, 1, 16, 0.01F, 0.0F},
        };
        std::uint32_t seed = 100;
        for (const auto& g : geometries) {
            for (const int batch : {1, 3}) {
                for (const int l : {1, 5, 40}) {
                    if (g.heads == 64 && batch == 3 && l == 40) continue; // the reference's cost
                    failures += check_case(Case{g, batch, l}, seed += 10);
                }
            }
        }
        for (const auto& g : geometries) {
            for (const int batch : {1, 3}) {
                for (const int width : {1, 3, 4}) failures += check_fold(g, batch, width, seed += 10);
            }
        }
        failures += check_bf16_state(geometries[0], 1, 64, true, seed += 10);
        failures += check_bf16_state(geometries[0], 3, 40, false, seed += 10);
        failures += check_bf16_state(geometries[1], 3, 256, true, seed += 10);
        for (const int group : {36, 40, 256, 512, 1024}) {
            for (const int T : {1, 5}) {
                for (const int z_offset : {0, 8, 1}) {
                    failures += check_gated_norm(group, 3, T, z_offset, seed += 10);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "mamba2 test threw: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cout << "mamba2: " << failures << " failures\n";
        return 1;
    }
    std::cout << "mamba2: all cases passed\n";
    return 0;
}
