// The Inkling 2 family Ops (ninferno/ops/inkling2.h) against FP64 oracles of their stated
// mathematics, evaluated from the represented (BF16 / FP32) inputs.
#include "core/device.h"
#include "ninferno/ops/inkling2.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

// One BF16 rounding of the result, FP32 arithmetic before it.
constexpr PointwiseCriterion kBf16Out{/*absolute*/ 2.0e-4, /*relative*/ 8.0e-3};

std::vector<float> bf16_values(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    fill_uniform(v, seed, lo, hi);
    round_to_bf16(v);
    return v;
}

std::vector<double> rms_normed(const std::vector<float>& x, std::size_t h, std::size_t t) {
    double sq = 0;
    for (std::size_t i = 0; i < h; ++i) { sq += double(x[t * h + i]) * x[t * h + i]; }
    const double inv = 1.0 / std::sqrt(sq / double(h) + 1e-6);
    std::vector<double> out(h);
    for (std::size_t i = 0; i < h; ++i) { out[i] = x[t * h + i] * inv; }
    return out;
}

int depth_mix_case(int hidden, int tokens, int count, int queries, std::uint32_t seed) {
    std::vector<std::vector<float>> states;
    std::vector<DeviceBuffer> device_states;
    ops::Inkling2MixStates in;
    for (int s = 0; s < count; ++s) {
        states.push_back(bf16_values(std::size_t(hidden) * tokens, seed + s, -3.0f, 3.0f));
        device_states.push_back(to_device_bf16(states.back()));
        in.data[std::size_t(s)] = device_states.back().p;
    }
    in.count   = count;
    auto query = std::vector<float>(std::size_t(queries) * hidden);
    fill_uniform(query, seed + 99, -0.5f, 0.5f);
    auto device_query = to_device_f32(query);
    std::vector<DeviceBuffer> device_out;
    ops::Inkling2MixOutputs out;
    for (int r = 0; r < queries; ++r) {
        device_out.push_back(DeviceBuffer(std::size_t(hidden) * tokens * 2));
        out.data[std::size_t(r)] = device_out.back().p;
    }
    out.count = queries;
    Tensor q(device_query.p, DType::FP32, {hidden, queries});
    ops::inkling2_depth_mix(in, q, hidden, tokens, out, nullptr);
    cuda_synchronize();
    int failures = 0;
    for (int r = 0; r < queries; ++r) {
        std::vector<double> expected(std::size_t(hidden) * tokens);
        for (int t = 0; t < tokens; ++t) {
            std::vector<std::vector<double>> normed;
            std::vector<double> score;
            for (int s = 0; s < count; ++s) {
                normed.push_back(rms_normed(states[std::size_t(s)], hidden, t));
                double dot = 0;
                for (int h = 0; h < hidden; ++h) {
                    dot += normed.back()[std::size_t(h)] * query[std::size_t(r) * hidden + h];
                }
                score.push_back(dot / std::sqrt(double(hidden)));
            }
            const double m = *std::max_element(score.begin(), score.end());
            double z       = 0;
            for (auto& s : score) { z += (s = std::exp(s - m)); }
            for (int h = 0; h < hidden; ++h) {
                double acc = 0;
                for (int s = 0; s < count; ++s) {
                    acc += score[std::size_t(s)] / z * normed[std::size_t(s)][std::size_t(h)];
                }
                expected[std::size_t(t) * hidden + h] = acc;
            }
        }
        failures += verify_pointwise("inkling2_depth_mix",
                                     from_device_bf16(device_out[std::size_t(r)], expected.size()),
                                     expected, kBf16Out);
    }
    return failures;
}

// Two calls in a row (the state carries the last three inputs from the first into the second).
int conv_case(int channels, int heads, bool dynamic, std::uint32_t seed) {
    auto weight = std::vector<float>(std::size_t(channels) * 4);
    fill_uniform(weight, seed, -0.8f, 0.8f);
    std::vector<float> state(std::size_t(channels) * 3, 0.0f);
    fill_uniform(state, seed + 1, -1.0f, 1.0f);
    auto device_weight = to_device_f32(weight);
    auto device_state  = to_device_f32(state);
    std::vector<double> history(std::size_t(channels) * 3); // u[-3..-1] per channel
    for (std::size_t i = 0; i < history.size(); ++i) { history[i] = state[i]; }
    int failures = 0;
    for (int call = 0; call < 2; ++call) {
        const int tokens = call == 0 ? 5 : 1;
        auto x        = bf16_values(std::size_t(channels) * tokens, seed + 10 + call, -2.0f, 2.0f);
        auto taps     = bf16_values(std::size_t(heads) * 4 * tokens, seed + 20 + call, -0.5f, 0.5f);
        auto device_x = to_device_bf16(x);
        auto device_taps = to_device_bf16(taps);
        DeviceBuffer device_out(std::size_t(channels) * tokens * 2);
        Tensor tx(device_x.p, DType::BF16, {channels, tokens});
        Tensor tw(device_weight.p, DType::FP32, {4, channels});
        Tensor tt = dynamic ? Tensor(device_taps.p, DType::BF16, {heads * 4, tokens}) : Tensor();
        Tensor ts(device_state.p, DType::FP32, {3, channels});
        Tensor to(device_out.p, DType::BF16, {channels, tokens});
        ops::inkling2_conv(tx, tw, tt, heads, ts, to, nullptr);
        cuda_synchronize();
        const double scale = 1.0 / std::sqrt(4.0 * (dynamic ? 2 : 1) + 1.0);
        std::vector<double> expected(std::size_t(channels) * tokens);
        for (int c = 0; c < channels; ++c) {
            std::vector<double> u(history.begin() + std::ptrdiff_t(c) * 3,
                                  history.begin() + std::ptrdiff_t(c) * 3 + 3);
            for (int t = 0; t < tokens; ++t) { u.push_back(x[std::size_t(t) * channels + c]); }
            const int head = c / (channels / heads);
            for (int t = 0; t < tokens; ++t) {
                const std::size_t i = std::size_t(t) + 3; // u index of x_t
                double y            = u[i];
                for (int k = 0; k < 4; ++k) {
                    y += weight[std::size_t(c) * 4 + k] * u[i - 3 + std::size_t(k)];
                }
                if (dynamic) {
                    for (int lag = 0; lag < 4; ++lag) {
                        y += taps[std::size_t(t) * heads * 4 + std::size_t(head) * 4 + lag] *
                             u[i - std::size_t(lag)];
                    }
                }
                expected[std::size_t(t) * channels + c] = y * scale;
            }
            for (int j = 0; j < 3; ++j) {
                history[std::size_t(c) * 3 + j] = u[u.size() - 3 + std::size_t(j)];
            }
        }
        failures +=
            verify_pointwise(dynamic ? "inkling2_conv dynamic" : "inkling2_conv",
                             from_device_bf16(device_out, expected.size()), expected, kBf16Out);
        failures +=
            verify_pointwise("inkling2_conv state", from_device_f32(device_state, history.size()),
                             history, PointwiseCriterion{1e-6, 1e-6});
    }
    return failures;
}

int control_case(int hidden, int tokens, std::uint32_t seed) {
    const std::size_t n = std::size_t(hidden) * tokens;
    auto cur = bf16_values(n, seed, -3, 3), msg = bf16_values(n, seed + 1, -3, 3);
    auto gain = bf16_values(n, seed + 2, -2, 2), forget = bf16_values(n, seed + 3, -2, 2);
    auto dc = to_device_bf16(cur), dm = to_device_bf16(msg), dg = to_device_bf16(gain),
         df = to_device_bf16(forget);
    DeviceBuffer dout(n * 2), dpair(n * 4);
    Tensor tc(dc.p, DType::BF16, {hidden, tokens}), tm(dm.p, DType::BF16, {hidden, tokens});
    Tensor tg(dg.p, DType::BF16, {hidden, tokens}), tf(df.p, DType::BF16, {hidden, tokens});
    Tensor to(dout.p, DType::BF16, {hidden, tokens}),
        tp(dpair.p, DType::BF16, {2 * hidden, tokens});
    ops::inkling2_control(tc, tm, tg, tf, to, nullptr);
    ops::inkling2_norm_pair(tc, tm, tp, nullptr);
    cuda_synchronize();
    std::vector<double> expected(n), pair(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        expected[i] =
            cur[i] - std::tanh(double(forget[i])) * cur[i] + std::tanh(double(gain[i])) * msg[i];
    }
    for (int t = 0; t < tokens; ++t) {
        const auto a = rms_normed(cur, std::size_t(hidden), std::size_t(t)),
                   b = rms_normed(msg, std::size_t(hidden), std::size_t(t));
        for (int h = 0; h < hidden; ++h) {
            pair[std::size_t(t) * 2 * hidden + h]          = a[std::size_t(h)];
            pair[std::size_t(t) * 2 * hidden + hidden + h] = b[std::size_t(h)];
        }
    }
    return verify_pointwise("inkling2_control", from_device_bf16(dout, n), expected, kBf16Out) +
           verify_pointwise("inkling2_norm_pair", from_device_bf16(dpair, 2 * n), pair, kBf16Out);
}

std::vector<double> clipped(const std::vector<double>& x, double c) {
    double sq = 0;
    for (double v : x) { sq += v * v; }
    const double f = c / std::max(std::sqrt(sq), c);
    std::vector<double> out(x);
    for (auto& v : out) { v *= f; }
    return out;
}

int prepare_case(bool full, std::uint32_t seed) {
    const int d = 128, hq = 24, hkv = 8, dr = 16, tokens = 3;
    const std::vector<int> positions = {9000, 9001, 20000};
    auto q                           = bf16_values(std::size_t(d) * hq * tokens, seed, -3, 3),
         k                           = bf16_values(std::size_t(d) * hkv * tokens, seed + 1, -3, 3);
    auto v                           = bf16_values(std::size_t(d) * hkv * tokens, seed + 2, -3, 3),
         r                           = bf16_values(std::size_t(dr) * hq * tokens, seed + 3, -3, 3);
    std::vector<float> qc(std::size_t(hq) * d), rc(std::size_t(hq) * dr);
    fill_uniform(qc, seed + 4, 0.2f, 2.0f);
    fill_uniform(rc, seed + 5, 0.2f, 2.0f);
    auto dq = to_device_bf16(q), dk = to_device_bf16(k), dv = to_device_bf16(v),
         drr = to_device_bf16(r);
    auto dqc = to_device_f32(qc), drc = to_device_f32(rc);
    auto dp = to_device_i32(std::vector<int>(positions));
    Tensor tq(dq.p, DType::BF16, {d, hq, tokens}), tk(dk.p, DType::BF16, {d, hkv, tokens});
    Tensor tv(dv.p, DType::BF16, {d, hkv, tokens}), tr(drr.p, DType::BF16, {dr, hq, tokens});
    Tensor tqc(dqc.p, DType::FP32, {d, hq}), trc(drc.p, DType::FP32, {dr, hq});
    Tensor tp(dp.p, DType::I32, {tokens});
    ops::Inkling2PrepareParams params{22.627417f, 8.0f, 1e-6f, 0.1f, 8192.0f, full};
    ops::inkling2_attention_prepare(tq, tk, tv, tr, tqc, trc, tp, params, nullptr);
    cuda_synchronize();
    std::vector<double> eq(q.size()), ek(k.size()), ev(v.size()), er(r.size());
    for (int t = 0; t < tokens; ++t) {
        const double tau =
            full ? 1.0 + 0.1 * std::log(std::max(1.0, (positions[std::size_t(t)] + 1.0) / 8192.0))
                 : 1.0;
        const auto rows = [&](const std::vector<float>& x, std::vector<double>& e, int heads, int n,
                              const std::vector<float>* scale, double c, double post, bool rms) {
            for (int h = 0; h < heads; ++h) {
                std::vector<double> row(std::size_t(n));
                const std::size_t base = (std::size_t(t) * heads + h) * n;
                for (int i = 0; i < n; ++i) {
                    row[std::size_t(i)] =
                        x[base + i] * (scale ? (*scale)[std::size_t(h) * n + i] : 1.0f);
                }
                if (rms) {
                    double sq = 0;
                    for (double y : row) { sq += y * y; }
                    for (auto& y : row) { y /= std::sqrt(sq / n + 1e-6); }
                } else {
                    row = clipped(row, c);
                    for (auto& y : row) { y *= post; }
                }
                std::copy(row.begin(), row.end(), e.begin() + std::ptrdiff_t(base));
            }
        };
        rows(q, eq, hq, d, &qc, 22.627417, tau, false);
        rows(k, ek, hkv, d, nullptr, 22.627417, 1.0, false);
        rows(v, ev, hkv, d, nullptr, 0, 1.0, true);
        rows(r, er, hq, dr, &rc, 8.0, tau, false);
    }
    return verify_pointwise("inkling2_prepare q", from_device_bf16(dq, q.size()), eq, kBf16Out) +
           verify_pointwise("inkling2_prepare k", from_device_bf16(dk, k.size()), ek, kBf16Out) +
           verify_pointwise("inkling2_prepare v", from_device_bf16(dv, v.size()), ev, kBf16Out) +
           verify_pointwise("inkling2_prepare r", from_device_bf16(drr, r.size()), er, kBf16Out);
}

// Keys 0..last appended in calls of `chunk`, then the last call's queries attend.
int attention_case(int window, int extent, std::uint32_t seed) {
    const int d = 128, hq = 24, hkv = 8, dr = 16, chunk = 3, last = 62;
    const int capacity = window ? window + chunk - 1 : last + 1;
    const int keys     = last + 1;
    auto k             = bf16_values(std::size_t(d) * hkv * keys, seed, -1.5f, 1.5f),
         v             = bf16_values(std::size_t(d) * hkv * keys, seed + 1, -2, 2);
    std::vector<float> bias(std::size_t(extent) * dr);
    fill_uniform(bias, seed + 2, -1, 1);
    DeviceBuffer kc(std::size_t(d) * capacity * hkv * 2), vc(std::size_t(d) * capacity * hkv * 2);
    Tensor tkc(kc.p, DType::BF16, {d, capacity, hkv}), tvc(vc.p, DType::BF16, {d, capacity, hkv});
    for (int first = 0; first < keys; first += chunk) {
        const int n = std::min(chunk, keys - first);
        std::vector<float> ks(k.begin() + std::ptrdiff_t(first) * d * hkv,
                              k.begin() + std::ptrdiff_t(first + n) * d * hkv);
        std::vector<float> vs(v.begin() + std::ptrdiff_t(first) * d * hkv,
                              v.begin() + std::ptrdiff_t(first + n) * d * hkv);
        std::vector<int> pos(std::size_t(n));
        for (int i = 0; i < n; ++i) { pos[std::size_t(i)] = first + i; }
        auto dks = to_device_bf16(ks), dvs = to_device_bf16(vs), dpos = to_device_i32(pos);
        Tensor tk(dks.p, DType::BF16, {d, hkv, n}), tv(dvs.p, DType::BF16, {d, hkv, n}),
            tp(dpos.p, DType::I32, {n});
        ops::inkling2_kv_append(tk, tv, tp, capacity, tkc, tvc, nullptr);
    }
    const std::vector<int> positions = {last - 2, last - 1, last};
    const int tokens                 = 3;
    auto q    = bf16_values(std::size_t(d) * hq * tokens, seed + 3, -1.5f, 1.5f),
         r    = bf16_values(std::size_t(dr) * hq * tokens, seed + 4, -2, 2);
    auto sink = bf16_values(std::size_t(hq) * tokens, seed + 5, -4, 4);
    auto dq = to_device_bf16(q), drr = to_device_bf16(r), ds = to_device_bf16(sink),
         db = to_device_f32(bias);
    auto dp = to_device_i32(std::vector<int>(positions));
    DeviceBuffer dout(std::size_t(d) * hq * tokens * 2);
    Tensor tq(dq.p, DType::BF16, {d, hq, tokens}), tr(drr.p, DType::BF16, {dr, hq, tokens});
    Tensor ts(ds.p, DType::BF16, {hq, tokens}), tb(db.p, DType::FP32, {dr, extent});
    Tensor tp(dp.p, DType::I32, {tokens}), to(dout.p, DType::BF16, {d, hq, tokens});
    ops::Inkling2AttentionParams params{extent, window, capacity, keys};
    ops::inkling2_attention(tq, tr, ts, tb, tkc, tvc, tp, params, to, nullptr);
    cuda_synchronize();
    std::vector<double> expected(std::size_t(d) * hq * tokens);
    const double sc = 1.0 / std::sqrt(double(d));
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < hq; ++h) {
            const int g = h / (hq / hkv), p = positions[std::size_t(t)];
            std::vector<double> s;
            std::vector<int> js;
            for (int j = 0; j <= p; ++j) {
                if (window && p - j >= window) { continue; }
                double dot = 0;
                for (int i = 0; i < d; ++i) {
                    dot += double(q[(std::size_t(t) * hq + h) * d + i]) *
                           k[(std::size_t(j) * hkv + g) * d + i];
                }
                if (p - j < extent) {
                    for (int i = 0; i < dr; ++i) {
                        dot += double(r[(std::size_t(t) * hq + h) * dr + i]) *
                               bias[std::size_t(p - j) * dr + i];
                    }
                }
                s.push_back(dot * sc);
                js.push_back(j);
            }
            const double sk = sink[std::size_t(t) * hq + h] * sc;
            double m        = sk;
            for (double x : s) { m = std::max(m, x); }
            double z = std::exp(sk - m);
            for (auto& x : s) { z += (x = std::exp(x - m)); }
            for (int i = 0; i < d; ++i) {
                double acc = 0;
                for (std::size_t n = 0; n < s.size(); ++n) {
                    acc += s[n] * v[(std::size_t(js[n]) * hkv + g) * d + i];
                }
                expected[(std::size_t(t) * hq + h) * d + i] = acc / z;
            }
        }
    }
    return verify_pointwise(window ? "inkling2_attention sliding" : "inkling2_attention full",
                            from_device_bf16(dout, expected.size()), expected, kBf16Out);
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += depth_mix_case(2304, 1, 1, 1, 11u);
    failures += depth_mix_case(2304, 3, 5, 3, 21u);
    failures += depth_mix_case(2304, 2, 12, 4, 31u);
    failures += conv_case(1024, 8, false, 41u);
    failures += conv_case(1024, 8, true, 51u);
    failures += conv_case(2304, 1, false, 61u);
    failures += control_case(2304, 3, 71u);
    failures += prepare_case(true, 81u);
    failures += prepare_case(false, 91u);
    failures += attention_case(0, 16, 101u);  // full layer, the bias ends below the context
    failures += attention_case(8, 8, 111u);   // sliding ring of window + chunk - 1 slots
    failures += attention_case(0, 128, 121u); // full layer, every distance biased
    std::cout << (failures ? "FAIL" : "OK") << " inkling2\n";
    return failures ? 1 : 0;
}
