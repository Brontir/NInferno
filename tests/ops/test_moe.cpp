// The generic MoE Op (ninferno/ops/moe.h) against an FP64 oracle over random geometries: expert
// count, top-k, hidden and expert widths, every expert weight format, both scorings, with and
// without renormalization, scales and biases, all three activations, both gate/up row orders, a
// shared expert with and without its gate, both epilogues, decode, streaming and prefill widths.
// Then the route contract: under the prefill profile a token's bytes do not depend on its call,
// on the default routes and on the plans a route table gives (narrower jobs included).
//
// The oracle evaluates the formula in FP64 from the exact stored-weight decode and rounds where
// the Op's contract rounds (the activation product and each expert's output to BF16; the shared
// expert as the generic projection path stores it). A token whose k-th and (k+1)-th selection
// ranks are closer than a float's resolution is a near-tie the FP32 router may break either way;
// it is counted and not compared.
#include "core/arena.h"
#include "core/device.h"
#include "core/ggml_frag.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/route_table.h"
#include "ninferno/ops/weight_input.h"
#include "ops/host_experts/host_experts.h"
#include "ops/common/native_nvfp4.h"
#include "ops/moe/moe_internal.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninferno;
namespace qw = ninferno::test::quantized_weight;

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
}

std::uint16_t to_bf16(double v) {
    const float f = static_cast<float>(v);
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffU) > 0x7f800000U) return 0x7fc0;
    u += 0x7fffU + ((u >> 16) & 1U);
    return static_cast<std::uint16_t>(u >> 16);
}

double from_bf16(std::uint16_t h) {
    const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

double round_bf16(double v) { return from_bf16(to_bf16(v)); }

// A host matrix in one format and its device copy.
struct Matrix {
    QType qtype{};
    std::int32_t n = 0, k = 0;
    qw::PackedWeight packed;
    std::vector<std::uint16_t> bf16;
    DeviceBuffer device;
    Weight weight{};

    double at(std::int32_t r, std::int32_t c) const {
        if (qtype == QType::BF16) return from_bf16(bf16[static_cast<std::size_t>(r) * k + c]);
        return qw::logical_weight_fp64(packed, r, c);
    }
};

Matrix make_matrix(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed, double sd = 0.05) {
    Matrix m;
    m.qtype = qtype;
    m.n     = n;
    m.k     = k;
    if (qtype == QType::BF16) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> normal(0.0, sd);
        m.bf16.resize(static_cast<std::size_t>(n) * k);
        for (auto& v : m.bf16) v = to_bf16(normal(rng));
        m.device = DeviceBuffer(m.bf16.size() * 2);
        m.device.copy_from_host(m.bf16.data(), m.device.bytes);
        Weight w{};
        w.qtype  = QType::BF16;
        w.layout = QuantLayout::Contiguous;
        w.qdata = w.payload = m.device.p;
        w.payload_bytes     = m.device.bytes;
        w.n = w.shape[0] = w.padded_shape[0] = n;
        w.k = w.shape[1] = w.padded_shape[1] = k;
        w.ndim   = 2;
        m.weight = w;
        return m;
    }
    qw::PatternedWeightOptions options;
    if (qtype == QType::NVFP4) {
        options.weight_scale_divisor = 16.0F;
        options.input_scale_divisor  = 1.0F;
    } else if (qtype != QType::FP8_E4M3FN_ROW_BF16 && qtype != QType::MXFP4 &&
               qtype != QType::FP8_E4M3FN_ROW_F32 && !u4_zp_group(qtype) &&
               !is_ggml_block(qtype)) {
        options.row_split_scale = qw::RowSplitScalePattern::Small;
        options.row_split_codes = qw::RowSplitCodePattern::Hashed;
    }
    m.packed = qw::make_patterned_weight(qtype, n, k, seed, options);
    m.device = DeviceBuffer(m.packed.payload.size());
    m.device.copy_from_host(m.packed.payload.data(), m.device.bytes);
    m.weight = m.packed.device_weight(m.device.p);
    return m;
}

struct Vector {
    std::vector<double> values;
    DeviceBuffer device;
    Tensor tensor;
};

Vector make_vector(std::int64_t n, std::uint32_t seed, bool bf16, double lo, double hi) {
    Vector v;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(lo, hi);
    v.values.resize(static_cast<std::size_t>(n));
    if (bf16) {
        std::vector<std::uint16_t> words(v.values.size());
        for (std::size_t i = 0; i < words.size(); ++i) {
            words[i]    = to_bf16(u(rng));
            v.values[i] = from_bf16(words[i]);
        }
        v.device = DeviceBuffer(words.size() * 2);
        v.device.copy_from_host(words.data(), v.device.bytes);
        v.tensor = Tensor(v.device.p, DType::BF16, {static_cast<std::int32_t>(n)});
    } else {
        std::vector<float> words(v.values.size());
        for (std::size_t i = 0; i < words.size(); ++i) {
            words[i]    = static_cast<float>(u(rng));
            v.values[i] = words[i];
        }
        v.device = DeviceBuffer(words.size() * 4);
        v.device.copy_from_host(words.data(), v.device.bytes);
        v.tensor = Tensor(v.device.p, DType::FP32, {static_cast<std::int32_t>(n)});
    }
    return v;
}

struct Case {
    std::string label;
    std::int32_t experts, top_k, hidden, inter, shared_inter;
    QType gate_up, down;
    ops::MoeConfig config;
    QType shared_format = QType::Q8_G32_FP16;
    ops::LinearPolicy shared_policy = ops::LinearPolicy::A16Only;
    ops::LinearPolicy expert_policy = ops::LinearPolicy::A16Only;
    bool router_bias, selection_bias, expert_scale, expert_bias, shared_gate;
    ops::MoeEpilogue epilogue;
    // Columns of the gate/up bank past H: a bank zero-padded to a block multiple (moe.h). The
    // test's padded columns hold random codes, so only a zero-widened input leaves them inert.
    std::int32_t input_pad = 0;
};

struct Layer {
    Matrix router, gate_up, down, shared_gate_up, shared_down, shared_gate;
    Vector router_bias, selection_bias, expert_scale, gate_up_bias, down_bias;
    ops::MoeWeights weights;
    bool has_router_bias = false, has_selection_bias = false, has_expert_scale = false,
         has_expert_bias = false, has_shared = false, has_shared_gate = false;
};

bool relu2(const Case& c) { return c.config.activation == ops::MoeActivation::Relu2; }

Layer make_layer(const Case& c, std::uint32_t seed) {
    Layer l;
    const auto E = c.experts, H = c.hidden, F = c.inter;
    const int sets = relu2(c) ? 1 : 2; // a non-gated expert has up rows only
    l.router  = make_matrix(QType::BF16, E, H, seed + 1, 0.08);
    l.gate_up               = make_matrix(c.gate_up, E * sets * F, H + c.input_pad, seed + 2);
    l.down    = make_matrix(c.down, E * H, F, seed + 3);
    l.weights.router  = l.router.weight;
    l.weights.gate_up = l.gate_up.weight;
    l.weights.down    = l.down.weight;
    l.weights.expert_policy = c.expert_policy;
    if ((l.has_router_bias = c.router_bias)) {
        l.router_bias = make_vector(E, seed + 4, true, -0.2, 0.2);
        l.weights.router_bias = l.router_bias.tensor;
    }
    if ((l.has_selection_bias = c.selection_bias)) {
        l.selection_bias = make_vector(E, seed + 5, false, -0.05, 0.05);
        l.weights.selection_bias = l.selection_bias.tensor;
    }
    if ((l.has_expert_scale = c.expert_scale)) {
        l.expert_scale = make_vector(E, seed + 6, false, 0.5, 1.5);
        l.weights.expert_scale = l.expert_scale.tensor;
    }
    if ((l.has_expert_bias = c.expert_bias)) {
        l.gate_up_bias = make_vector(static_cast<std::int64_t>(E) * sets * F, seed + 7, true, -0.5, 0.5);
        l.down_bias    = make_vector(static_cast<std::int64_t>(E) * H, seed + 8, false, -0.5, 0.5);
        l.weights.gate_up_bias = l.gate_up_bias.tensor;
        l.weights.down_bias    = l.down_bias.tensor;
    }
    if ((l.has_shared = c.shared_inter > 0)) {
        l.shared_gate_up = make_matrix(c.shared_format, sets * c.shared_inter, H, seed + 9);
        l.shared_down    = make_matrix(c.shared_format, H, c.shared_inter, seed + 10);
        l.weights.shared_gate_up = l.shared_gate_up.weight;
        l.weights.shared_down    = l.shared_down.weight;
        l.weights.shared_policy  = c.shared_policy;
        if ((l.has_shared_gate = c.shared_gate)) {
            l.shared_gate         = make_matrix(QType::BF16, 1, H, seed + 11, 0.1);
            l.weights.shared_gate = l.shared_gate.weight;
        }
    }
    return l;
}

double act(const ops::MoeConfig& c, double g, double u) {
    switch (c.activation) {
    case ops::MoeActivation::GeGluTanh:
        return 0.5 * g * (1.0 + std::tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))) * u;
    case ops::MoeActivation::ClampedSwiGlu: {
        const double gc = std::min(g, static_cast<double>(c.limit));
        const double uc = std::clamp(u, -static_cast<double>(c.limit), static_cast<double>(c.limit));
        return (uc + 1.0) * gc / (1.0 + std::exp(-c.alpha * gc));
    }
    case ops::MoeActivation::Relu2: // non-gated: u only
        return g > 0.0 ? g * g : 0.0;
    case ops::MoeActivation::SwiGlu:
        break;
    }
    return g / (1.0 + std::exp(-g)) * u;
}

double dot(const Matrix& m, std::int32_t row, const std::vector<double>& x) {
    double s = 0.0;
    for (std::int32_t c = 0; c < m.k; ++c) s += m.at(row, c) * x[static_cast<std::size_t>(c)];
    return s;
}

// FP64 reference of one token; returns false for a near-tie selection.

// An A8 grouped stage's activation (the generic path's R8 quantization, w4a8_quantize_block), in
// the same FP32 operations on the stage's BF16 input: per 32 values s = amax/127 and q =
// rint(v * (127/amax)) clamped to [-127, 127]; the stage computes with q*s.
// E4M3 (round to nearest even, saturating at 448) of a finite float: 3 mantissa bits from 2^-6
// up, subnormal steps of 2^-9 below.
double e4m3_rne(float v) {
    const float a = std::fabs(v);
    if (a == 0.0F) return 0.0;
    int e = 0;
    std::frexp(a, &e); // a = m * 2^e, m in [0.5, 1)
    const int exponent = std::max(e - 1, -6);
    const float step   = std::ldexp(1.0F, exponent - 3);
    const float q      = std::min(448.0F, std::nearbyint(a / step) * step);
    return v < 0.0F ? -static_cast<double>(q) : static_cast<double>(q);
}

// The MXFP8 activation of the MXFP4 banks' block-scaled MMA (launch_moe_mx8_quantize): per 32
// values the smallest power of two 2^e with amax / 2^e <= 448 (the float exponent of amax / 448,
// one up unless a power of two), codes E4M3(v * 2^-e); the stage computes with code * 2^e.
std::vector<double> quantized_mx8(const std::vector<double>& v) {
    std::vector<double> out(v.size());
    for (std::size_t base = 0; base + 32 <= v.size(); base += 32) {
        float amax = 0.0F;
        for (int i = 0; i < 32; ++i)
            amax = std::max(amax, std::fabs(static_cast<float>(v[base + i])));
        if (amax == 0.0F) {
            for (int i = 0; i < 32; ++i) out[base + i] = 0.0;
            continue;
        }
        std::uint32_t bits = 0;
        const float s      = amax * (1.0F / 448.0F);
        std::memcpy(&bits, &s, 4);
        int e             = static_cast<int>(bits >> 23) + ((bits & 0x7fffffU) != 0 ? 1 : 0);
        e                 = std::max(1, std::min(253, e));
        const float scale = std::ldexp(1.0F, e - 127);
        const float inv   = std::ldexp(1.0F, 127 - e);
        for (int i = 0; i < 32; ++i) {
            out[base + i] = e4m3_rne(static_cast<float>(v[base + i]) * inv) * scale;
        }
    }
    return out;
}

std::vector<double> quantized_a8(const std::vector<double>& v, bool mx = false) {
    if (mx) return quantized_mx8(v);
    std::vector<double> out(v.size());
    for (std::size_t base = 0; base + 32 <= v.size(); base += 32) {
        float amax = 0.0F;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(static_cast<float>(v[base + i])));
        const float inv   = amax > 0.0F ? 127.0F / amax : 0.0F;
        const float scale = amax / 127.0F;
        for (int i = 0; i < 32; ++i) {
            const float q = std::min(127.0F, std::max(-127.0F, std::nearbyint(static_cast<float>(v[base + i]) * inv)));
            out[base + i] = static_cast<double>(q) * static_cast<double>(scale);
        }
    }
    return out;
}

// Which grouped stages of a T-token call compute A8 (an R8 stage of the plan moe selects).
struct A8Stages {
    bool gate_up = false, down = false;
    bool mx = false; // both quantize as MXFP8 (--part mx8 on MXFP4 banks)
};

// Whether the grouped W4A8 of MXFP4 banks quantizes MXFP8 (the block-scaled MMA): by default on
// sm_120a builds except under --part int8; --part mx8 forces it.
bool g_mx8 = false;

// The stages a plan quantizes: its R8 stages, or both under the grouped W4A8 (MXFP8 where the
// block-scaled MMA takes MXFP4 banks), which the speed profile runs from 64 tokens.
A8Stages a8_stages(const Case& c, const ops::detail::moe::MoePlan& p) {
    if (p.w4a8) return {true, true, g_mx8 && c.gate_up == QType::MXFP4 && c.down == QType::MXFP4};
    return {p.gate_up_r8 >= 0, p.down_r8 >= 0};
}

bool reference_token(const Case& c, const Layer& l, const std::vector<double>& x,
                     std::vector<double>& r, A8Stages a8 = {}) {
    const auto E = c.experts, H = c.hidden, F = c.inter;
    // The routed experts' gate/up input (the router and the shared expert read x as it is),
    // widened with zeros to a padded bank's K.
    std::vector<double> xw = x;
    xw.resize(static_cast<std::size_t>(H + c.input_pad), 0.0);
    const std::vector<double> xe = a8.gate_up ? quantized_a8(xw, a8.mx) : xw;
    std::vector<double> logit(E), score(E), rank(E);
    for (int e = 0; e < E; ++e) {
        logit[e] = dot(l.router, e, x) + (l.has_router_bias ? l.router_bias.values[e] : 0.0);
    }
    const bool sigmoid = c.config.scoring == ops::MoeScoring::Sigmoid;
    const double m = *std::max_element(logit.begin(), logit.end());
    double z = 0.0;
    for (int e = 0; e < E; ++e) {
        score[e] = sigmoid ? 1.0 / (1.0 + std::exp(-logit[e])) : std::exp(logit[e] - m);
        z += score[e];
    }
    for (int e = 0; e < E; ++e) {
        if (!sigmoid) score[e] /= z;
        rank[e] = score[e] + (l.has_selection_bias ? l.selection_bias.values[e] : 0.0);
    }
    std::vector<int> order(E);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return rank[a] > rank[b]; });
    if (c.top_k < E) {
        const double gap = rank[order[c.top_k - 1]] - rank[order[c.top_k]];
        if (gap <= 1e-6 * std::max(1.0, std::abs(rank[order[c.top_k - 1]]))) return false;
    }
    double sum = 0.0;
    for (int k = 0; k < c.top_k; ++k) sum += score[order[k]];
    r.assign(static_cast<std::size_t>(H), 0.0);
    const bool interleaved = c.config.gate_up_layout == ops::MoeGateUpLayout::Interleaved;
    for (int k = 0; k < c.top_k; ++k) {
        const int e = order[k];
        double w    = c.config.renormalize ? score[e] / sum : score[e];
        w *= c.config.routed_scale;
        if (l.has_expert_scale) w *= l.expert_scale.values[e];
        w = static_cast<double>(static_cast<float>(w));
        std::vector<double> mid(F);
        for (int i = 0; i < F; ++i) {
            if (relu2(c)) {
                const int ur = e * F + i;
                const double u = dot(l.gate_up, ur, xe) + (l.has_expert_bias ? l.gate_up_bias.values[ur] : 0.0);
                mid[i] = round_bf16(act(c.config, u, 0.0));
                continue;
            }
            const int gr = e * 2 * F + (interleaved ? 2 * i : i);
            const int ur = e * 2 * F + (interleaved ? 2 * i + 1 : F + i);
            const double g = dot(l.gate_up, gr, xe) + (l.has_expert_bias ? l.gate_up_bias.values[gr] : 0.0);
            const double u = dot(l.gate_up, ur, xe) + (l.has_expert_bias ? l.gate_up_bias.values[ur] : 0.0);
            mid[i] = round_bf16(act(c.config, g, u));
        }
        if (a8.down) mid = quantized_a8(mid, a8.mx);
        for (int h = 0; h < H; ++h) {
            const int row = e * H + h;
            const double y = dot(l.down, row, mid) + (l.has_expert_bias ? l.down_bias.values[row] : 0.0);
            r[h] += w * round_bf16(y);
        }
    }
    if (l.has_shared) {
        const int Fs = c.shared_inter;
        std::vector<double> mid(Fs);
        for (int i = 0; i < Fs; ++i) {
            if (relu2(c)) { // the up projection stored in BF16, then relu^2
                const double u = round_bf16(dot(l.shared_gate_up, i, x));
                mid[i] = round_bf16(u > 0.0 ? u * u : 0.0);
                continue;
            }
            const double g = dot(l.shared_gate_up, i, x), u = dot(l.shared_gate_up, Fs + i, x);
            mid[i] = round_bf16(g / (1.0 + std::exp(-g)) * u);
        }
        double gate = 1.0;
        if (l.has_shared_gate) gate = 1.0 / (1.0 + std::exp(-round_bf16(dot(l.shared_gate, 0, x))));
        for (int h = 0; h < H; ++h) r[h] += gate * round_bf16(dot(l.shared_down, h, mid));
    }
    return true;
}

struct Run {
    std::vector<std::uint16_t> out;
    std::size_t peak = 0, capacity = 0;
};

Run run_op(const Case& c, const Layer& l, const std::vector<std::uint16_t>& x, std::int32_t offset,
           std::int32_t t, const std::vector<std::uint16_t>& residual,
           const ops::detail::moe::MoePlan* plan = nullptr) {
    DeviceBuffer dx(static_cast<std::size_t>(t) * c.hidden * 2);
    dx.copy_from_host(x.data() + static_cast<std::size_t>(offset) * c.hidden, dx.bytes);
    DeviceBuffer dout(dx.bytes);
    if (c.epilogue == ops::MoeEpilogue::AddResidual) {
        dout.copy_from_host(residual.data() + static_cast<std::size_t>(offset) * c.hidden, dout.bytes);
    } else {
        check(cudaMemset(dout.p, 0xff, dout.bytes), "poison");
    }
    Run run;
    run.capacity = ops::moe_workspace_bytes(l.weights, c.config, t, t);
    DeviceArena workspace(std::max<std::size_t>(run.capacity, 256));
    Tensor tx(dx.p, DType::BF16, {c.hidden, t});
    Tensor tout(dout.p, DType::BF16, {c.hidden, t});
    if (plan != nullptr) {
        ops::detail::moe::moe_with_plan(tx, l.weights, c.config, c.epilogue, tout, workspace,
                                        nullptr, *plan);
    } else {
        ops::moe(tx, l.weights, c.config, c.epilogue, tout, workspace, nullptr);
    }
    check(cudaDeviceSynchronize(), "moe");
    run.peak = workspace.peak_used();
    run.out.resize(static_cast<std::size_t>(t) * c.hidden);
    check(cudaMemcpy(run.out.data(), dout.p, dout.bytes, cudaMemcpyDeviceToHost), "download");
    return run;
}

std::vector<std::uint16_t> random_bf16(std::size_t n, std::uint32_t seed, double lo, double hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(lo, hi);
    std::vector<std::uint16_t> out(n);
    for (auto& v : out) v = to_bf16(u(rng));
    return out;
}

// One call's output (tokens [offset, offset + t) of x) against the oracle, `a8` naming the stages
// that computed A8; returns the failure count.
int compare_run(const Case& c, const Layer& l, const std::vector<std::uint16_t>& x,
                const std::vector<std::uint16_t>& residual, std::int32_t offset, std::int32_t t,
                const Run& run, A8Stages a8, const std::string& label) {
    int failures = 0;
    if (run.peak > run.capacity) {
        std::cerr << label << ": workspace " << run.peak << " above capacity " << run.capacity << '\n';
        ++failures;
    }
    double err2 = 0.0, ref2 = 0.0, worst = 0.0, scale = 0.0;
    int compared = 0, ties = 0;
    std::vector<double> xs(c.hidden), r;
    for (std::int32_t token = 0; token < t; ++token) {
        // Sample: every token of a small call, 12 spread over a large one.
        if (t > 24 && token % (t / 12) != 0) continue;
        const std::size_t base = static_cast<std::size_t>(offset + token) * c.hidden;
        for (int h = 0; h < c.hidden; ++h) xs[h] = from_bf16(x[base + h]);
        if (!reference_token(c, l, xs, r, a8)) {
            ++ties;
            continue;
        }
        ++compared;
        for (int h = 0; h < c.hidden; ++h) {
            double ref = r[h];
            if (c.epilogue == ops::MoeEpilogue::AddResidual) ref += from_bf16(residual[base + h]);
            const double got = from_bf16(run.out[static_cast<std::size_t>(token) * c.hidden + h]);
            if (!std::isfinite(got)) {
                worst = INFINITY;
                continue;
            }
            err2 += (got - ref) * (got - ref);
            ref2 += ref * ref;
            worst = std::max(worst, std::abs(got - ref));
            scale = std::max(scale, std::abs(ref));
        }
    }
    const double rel = ref2 > 0 ? std::sqrt(err2 / ref2) : std::sqrt(err2);
    // BF16 storage of the product and of each expert output, and FP32 accumulation: the
    // relative L2 criterion is a few BF16 ulps. An A8 stage answers to the same criterion against
    // the oracle of its quantized activation (the quantization error is not the kernel's).
    const bool ok = compared > 0 && std::isfinite(worst) && rel <= 1.5e-2 && worst <= 0.08 * scale + 1e-3;
    if (!ok) {
        std::cerr << label << ": rel_l2 " << rel << " worst " << worst << " (max |ref| " << scale
                  << ") compared " << compared << " near-ties " << ties << '\n';
        ++failures;
    } else {
        std::cout << label << ": rel_l2 " << rel << " compared " << compared << " near-ties " << ties << '\n';
    }
    return failures;
}

int check_case(const Case& c, std::uint32_t seed, const std::vector<std::int32_t>& widths) {
    const Layer l = make_layer(c, seed);
    const std::int32_t t_max = *std::max_element(widths.begin(), widths.end());
    const auto x        = random_bf16(static_cast<std::size_t>(t_max) * c.hidden, seed + 21, -1.0, 1.0);
    const auto residual = random_bf16(x.size(), seed + 22, -2.0, 2.0);
    int failures = 0;
    for (const std::int32_t t : widths) {
        const Run run = run_op(c, l, x, 0, t, residual);
        const auto plan = ops::detail::moe::select_plan(l.weights, c.config, t, ops::prefill_routes_active());
        const A8Stages a8       = a8_stages(c, plan);
        const std::string label = c.label + " T=" + std::to_string(t) +
                                  (a8.gate_up || a8.down ? " (" + ops::detail::moe::plan_name(plan) + ")" : "");
        failures += compare_run(c, l, x, residual, 0, t, run, a8, label);
    }
    return failures;
}

// Under the prefill profile every call of 17 tokens or more gives a token the same bytes.
int check_invariance(const Case& c, std::uint32_t seed) {
    const Layer l = make_layer(c, seed);
    const std::int32_t wide = 300;
    const auto x        = random_bf16(static_cast<std::size_t>(wide) * c.hidden, seed + 31, -1.0, 1.0);
    const auto residual = random_bf16(x.size(), seed + 32, -2.0, 2.0);
    const ops::ScopedPrefillRoutes profile(true);
    const Run reference = run_op(c, l, x, 0, wide, residual);
    int failures = 0;
    const std::pair<std::int32_t, std::int32_t> windows[] = {
        {0, 17}, {0, 64}, {0, 65}, {5, 17}, {100, 200}, {283, 17}, {1, 299}, {17, 130}};
    for (const auto [offset, t] : windows) {
        const Run run = run_op(c, l, x, offset, t, residual);
        if (std::memcmp(run.out.data(), reference.out.data() + static_cast<std::size_t>(offset) * c.hidden,
                        run.out.size() * 2) != 0) {
            std::cerr << c.label << " invariance: window " << offset << "+" << t << " differs\n";
            ++failures;
        }
    }
    if (failures == 0) std::cout << c.label << ": 8 windows byte-identical to the wide call\n";
    return failures;
}

// Under the invariant profile (--activations exact) every call, a single token included, gives a
// token the same bytes as the wide call.
int check_invariant_profile(const Case& c, std::uint32_t seed) {
    const Layer l = make_layer(c, seed);
    const std::int32_t wide = 300;
    const auto x        = random_bf16(static_cast<std::size_t>(wide) * c.hidden, seed + 31, -1.0, 1.0);
    const auto residual = random_bf16(x.size(), seed + 32, -2.0, 2.0);
    int failures = 0;
    for (const bool prefill : {false, true}) {
        const ops::ScopedPrefillRoutes profile(prefill);
        const ops::ScopedInvariantRoutes invariant(true);
        const Run reference = run_op(c, l, x, 0, wide, residual);
        const std::pair<std::int32_t, std::int32_t> windows[] = {
            {0, 1},   {7, 1},  {299, 1}, {0, 2},    {3, 4},   {0, 8},   {1, 9},
            {5, 16},  {0, 17}, {40, 33}, {100, 200}, {283, 17}, {1, 299}};
        for (const auto [offset, t] : windows) {
            const Run run = run_op(c, l, x, offset, t, residual);
            if (std::memcmp(run.out.data(),
                            reference.out.data() + static_cast<std::size_t>(offset) * c.hidden,
                            run.out.size() * 2) != 0) {
                std::cerr << c.label << " invariant profile" << (prefill ? " (prefill)" : "")
                          << ": window " << offset << "+" << t << " differs\n";
                ++failures;
            }
        }
    }
    if (failures == 0) {
        std::cout << c.label << ": invariant profile, 13 windows x 2 byte-identical to the wide call\n";
    }
    return failures;
}

// The grouped R8 route of codebook experts under an A8 expert policy (decode, T <= 4): every R8
// schedule on both stages and on each stage alone against the oracle of the quantized activations;
// a slot's bytes depend on its own activation only, so every call with R8 on both stages gives a
// token the bytes of the widest one whatever its schedule, T or position. The default route is the
// GPU class's A8 route (moe::a8_default_route: R8 on gate/up, an IQ3_XXS down of width 512..1023
// on the class's down schedule), A16 on a class without one; its bytes, too, do not depend on T or
// position. Two exceptions keep A16: a call of T >= 5 (verify, prefill: refused for a forced R8
// plan), and a bank whose K is too wide for a CTA's quantized activation (above 10922 at T = 3 or
// 4). Any R8 plan under A16Only is refused, and the default route there stays on the register MMA.
int check_r8(const Case& base, std::uint32_t seed, const std::vector<std::int32_t>& widths) {
    namespace moe = ops::detail::moe;
    Case c         = base;
    c.expert_policy = ops::LinearPolicy::AllowA8;
    const Layer l  = make_layer(c, seed);
    const std::int32_t wide = moe::kR8Tokens;
    const auto x =
        random_bf16(static_cast<std::size_t>(wide + 1) * c.hidden, seed + 21, -1.0, 1.0);
    const auto residual = random_bf16(x.size(), seed + 22, -2.0, 2.0);
    int failures        = 0;
    // The default plan at T with the given R8 stages.
    const auto forced = [&](std::int32_t t, int gate_up_r8, int down_r8) {
        auto p       = moe::select_plan(l.weights, c.config, t, false);
        p.gate_up_r8 = gate_up_r8;
        p.down_r8    = down_r8;
        return p;
    };
    const auto reference_plan = forced(wide, 0, 0);
    const Run reference       = run_op(c, l, x, 0, wide, residual, &reference_plan);
    failures += compare_run(c, l, x, residual, 0, wide, reference, {true, true},
                            c.label + " A8 T=" + std::to_string(wide) + " (" +
                                moe::plan_name(reference_plan) + ")");
    const auto same_bytes = [&](const Run& run, const Run& wide_run, std::int32_t offset,
                                const std::string& label) {
        if (std::memcmp(run.out.data(),
                        wide_run.out.data() + static_cast<std::size_t>(offset) * c.hidden,
                        run.out.size() * 2) != 0) {
            std::cerr << label << ": bytes differ from the T=" << wide << " call\n";
            ++failures;
        }
    };
    for (const std::int32_t t : widths) {
        const std::int32_t offset = wide - t;
        for (int s = 0; s < moe::r8_schedule_count(); ++s) {
            for (const auto& [gu, dn] : {std::pair{s, s}, std::pair{s, -1}, std::pair{-1, s}}) {
                const auto p            = forced(t, gu, dn);
                const Run run           = run_op(c, l, x, offset, t, residual, &p);
                const std::string label = c.label + " A8 T=" + std::to_string(t) + "+" +
                                          std::to_string(offset) + " (" + moe::plan_name(p) + ")";
                if (dn >= 0) {
                    // The R8 down stage's fused combine against the separate combine kernel.
                    moe::set_fused_combine(false);
                    const Run separate = run_op(c, l, x, offset, t, residual, &p);
                    moe::set_fused_combine(true);
                    if (std::memcmp(run.out.data(), separate.out.data(), run.out.size() * 2) != 0) {
                        std::cerr << label << ": the fused combine's bytes differ from combine's"
                                  << std::endl;
                        ++failures;
                    }
                }
                if (gu >= 0 && dn >= 0) {
                    same_bytes(run, reference, offset, label);
                    if (t != widths.front()) continue; // the oracle once per schedule
                }
                failures += compare_run(c, l, x, residual, offset, t, run, a8_stages(c, p), label);
            }
        }
    }
    // The shipped default route: the class's A8 route (A16 without one) at every T <= 4, each
    // token's bytes those of the widest call at any T and position.
    {
        // The heuristic route, not a measured table's plans: the class's shipped table may
        // carry this shape (ad106's does); it comes back afterwards.
        const auto shipped = ops::route_table_status();
        ops::clear_route_table();
        const auto route      = moe::a8_default_route();
        const Run wide_run    = run_op(c, l, x, 0, wide, residual);
        const int down        = route && c.down == QType::GGML_IQ3_XXS && c.inter >= 512 &&
                                 c.inter < 1024
                                     ? route->down
                                     : -1;
        for (std::int32_t t = 1; t <= wide; ++t) {
            const auto p = moe::select_plan(l.weights, c.config, t, false);
            const std::string label = c.label + " A8 default T=" + std::to_string(t) + " (" +
                                      moe::plan_name(p) + ")";
            const int gate_up   = route ? route->gate_up_at(t, false) : -1;
            const bool expected = gate_up >= 0 ? p.gate_up_r8 == gate_up && p.down_r8 < 0 &&
                                                     (down < 0 || p.down_schedule == down)
                                               : p.gate_up_r8 < 0 && p.down_r8 < 0;
            if (!expected) {
                std::cerr << label << ": not the class's A8 route\n";
                ++failures;
            }
            for (std::int32_t offset = 0; offset + t <= wide; offset += std::max(1, wide - t)) {
                const Run run = run_op(c, l, x, offset, t, residual);
                same_bytes(run, wide_run, offset, label + " +" + std::to_string(offset));
                if (t == 1 && offset == 0) {
                    failures +=
                        compare_run(c, l, x, residual, offset, t, run, a8_stages(c, p), label);
                }
            }
        }
        // T = 5: A16 by default, a forced R8 stage refused.
        const auto p5 = moe::select_plan(l.weights, c.config, wide + 1, false);
        if (p5.gate_up_r8 >= 0 || p5.down_r8 >= 0) {
            std::cerr << c.label << ": T=" << wide + 1 << " took " << moe::plan_name(p5) << '\n';
            ++failures;
        }
        const auto forced5 = forced(wide + 1, 0, -1);
        bool refused       = false;
        try {
            (void)run_op(c, l, x, 0, wide + 1, residual, &forced5);
        } catch (const std::invalid_argument&) { refused = true; }
        if (!refused) {
            std::cerr << c.label << ": an R8 stage ran a call of " << wide + 1 << " tokens\n";
            ++failures;
        }
        if (shipped.loaded) { (void)ops::load_route_table(shipped.path); }
    }
    // A bank too wide for a CTA's quantized activation keeps A16 at the widths it does not fit
    // (36 bytes per 32 values and slot, 48 KiB; slots: T rounded up to a power of two).
    {
        auto wide_bank = l.weights.gate_up;
        wide_bank.k    = 12288;
        const bool fits[] = {true, true, false, false};
        for (std::int32_t t = 1; t <= wide; ++t) {
            if (moe::r8_admits(l.weights, wide_bank, t) != fits[t - 1]) {
                std::cerr << c.label << ": K=12288 at T=" << t << " admitted "
                          << !fits[t - 1] << '\n';
                ++failures;
            }
        }
    }
    // A16Only: the default route computes A16, and a forced R8 stage is refused.
    {
        Layer a16 = make_layer(base, seed);
        const auto p = moe::select_plan(a16.weights, base.config, 1, false);
        if (p.gate_up_r8 >= 0 || p.down_r8 >= 0) {
            std::cerr << base.label << ": A16Only took " << moe::plan_name(p) << '\n';
            ++failures;
        }
        bool refused = false;
        try {
            (void)run_op(base, a16, x, 0, 1, residual, &reference_plan);
        } catch (const std::invalid_argument&) { refused = true; }
        if (!refused) {
            std::cerr << base.label << ": A16Only ran an R8 stage\n";
            ++failures;
        }
    }
    // An activation off 16-byte alignment is refused before a launch, A8 or not: the router's
    // register MMA reads it in 16-byte words as the gate/up stages do, so no plan could take it.
    for (const bool a8 : {true, false}) {
        const Layer layer = a8 ? make_layer(c, seed) : make_layer(base, seed);
        DeviceBuffer dx((static_cast<std::size_t>(c.hidden) + 8) * 2), dout(c.hidden * 2);
        Tensor tx(static_cast<std::uint8_t*>(dx.p) + 2, DType::BF16, {c.hidden, 1});
        Tensor tout(dout.p, DType::BF16, {c.hidden, 1});
        DeviceArena workspace(
            std::max<std::size_t>(ops::moe_workspace_bytes(layer.weights, c.config, 1, 1), 256));
        bool refused = false;
        try {
            ops::moe(tx, layer.weights, c.config, c.epilogue, tout, workspace, nullptr);
        } catch (const std::invalid_argument&) { refused = true; }
        if (!refused || cudaDeviceSynchronize() != cudaSuccess) {
            std::cerr << c.label << (a8 ? " A8" : " A16") << ": an unaligned activation was not "
                      << "refused before a launch\n";
            ++failures;
        }
    }
    if (failures == 0) std::cout << c.label << ": A8 R8 route, every schedule and width consistent\n";
    return failures;
}

const char* qname(QType q);

// The A8 route under a route table: a band of register-MMA stages keeps its measured schedules
// (the down schedule is not rewritten to the class's) and takes the class's A8 gate/up on top at
// T <= 4, A16 from T = 5; a band marked A16 ("|a16") stays A16 at every T; a band that names an R8
// stage (the loader keeps those within T = 1..4) runs it there, and the heuristic streaming
// register MMA takes T = 5 on. Each executed plan matches the oracle.
// The grouped W4A8 (MoePlan::w4a8): the prefill plan with both expert stages on the int8 tile,
// against the oracle with both stages' activations quantized per 32 values, and each token's bytes
// those of the widest call at any T and position (no K split; a slot's sums do not depend on its
// job). The widths cross a job (64 slots) and leave partial W4A8 tiles.
int check_w4a8(const Case& base, std::uint32_t seed, const std::vector<std::int32_t>& widths) {
    namespace moe           = ops::detail::moe;
    Case c                  = base;
    c.expert_policy         = ops::LinearPolicy::AllowA8;
    const Layer l           = make_layer(c, seed);
    const A8Stages a8{true, true, g_mx8 && c.gate_up == QType::MXFP4 && c.down == QType::MXFP4};
    const std::int32_t wide = widths.back();
    const auto x = random_bf16(static_cast<std::size_t>(wide) * c.hidden, seed + 31, -1.0, 1.0);
    const auto residual = random_bf16(x.size(), seed + 32, -2.0, 2.0);
    // The prefill profile, as the other invariance checks: the shared expert's projections
    // (the generic path) are token-invariant only under it.
    const ops::ScopedPrefillRoutes profile(true);
    int failures      = 0;
    const auto forced = [&](std::int32_t t) {
        auto p = moe::select_plan(l.weights, c.config, t, true);
        p.w4a8 = true;
        return p;
    };
    const auto wide_plan = forced(wide);
    const Run wide_run   = run_op(c, l, x, 0, wide, residual, &wide_plan);
    failures += compare_run(c, l, x, residual, 0, wide, wide_run, a8,
                            c.label + " W4A8 T=" + std::to_string(wide));
    for (const std::int32_t t : widths) {
        if (t == wide) continue;
        const std::int32_t offset = wide - t;
        const auto p              = forced(t);
        const Run run             = run_op(c, l, x, offset, t, residual, &p);
        const std::string label =
            c.label + " W4A8 T=" + std::to_string(t) + "+" + std::to_string(offset);
        failures += compare_run(c, l, x, residual, offset, t, run, a8, label);
        if (std::memcmp(run.out.data(),
                        wide_run.out.data() + static_cast<std::size_t>(offset) * c.hidden,
                        run.out.size() * 2) != 0) {
            std::cerr << label << ": bytes differ from the T=" << wide << " call\n";
            ++failures;
        }
    }
    // An A16 expert policy refuses the plan.
    {
        Case a16          = c;
        a16.expert_policy = ops::LinearPolicy::A16Only;
        const Layer la    = make_layer(a16, seed);
        auto p            = moe::select_plan(la.weights, a16.config, widths.front(), true);
        p.w4a8            = true;
        bool refused      = false;
        try {
            (void)run_op(a16, la, x, 0, widths.front(), residual, &p);
        } catch (const std::invalid_argument&) { refused = true; }
        if (!refused) {
            std::cerr << c.label << ": a W4A8 plan ran under an A16 expert policy\n";
            ++failures;
        }
    }
    return failures;
}

// The speed profile (a Gemma 4 prefill chunk, a verify of many rows) takes the grouped W4A8 from
// kMoeW4a8SpeedMinTokens on, on admitted banks and by default; narrower calls keep the A16 / R8
// plans. The selected wide plan matches the oracle with both stages' activations quantized.
int check_speed_w4a8(const Case& base, std::uint32_t seed) {
    namespace moe   = ops::detail::moe;
    Case c          = base;
    c.expert_policy = ops::LinearPolicy::AllowA8;
    const Layer l   = make_layer(c, seed);
    const std::int32_t wide   = moe::kMoeW4a8SpeedMinTokens + 32;
    const std::int32_t narrow = moe::kMoeW4a8SpeedMinTokens - 1;
    const auto x = random_bf16(static_cast<std::size_t>(wide) * c.hidden, seed + 31, -1.0, 1.0);
    const auto residual = random_bf16(x.size(), seed + 32, -2.0, 2.0);
    const ops::ScopedPrefillRoutes profile(false);
    int failures = 0;
    if (moe::select_plan(l.weights, c.config, narrow, false).w4a8) {
        std::cerr << c.label << ": the speed profile took W4A8 at T=" << narrow << '\n';
        ++failures;
    }
    const auto plan = moe::select_plan(l.weights, c.config, wide, false);
    if (!plan.w4a8) {
        std::cerr << c.label << ": the speed profile kept A16 at T=" << wide << '\n';
        return failures + 1;
    }
    const Run run = run_op(c, l, x, 0, wide, residual, &plan);
    failures += compare_run(c, l, x, residual, 0, wide, run, {true, true},
                            c.label + " speed-profile W4A8 T=" + std::to_string(wide));
    return failures;
}

int check_r8_table() {
    namespace moe = ops::detail::moe;
    const auto route = moe::a8_default_route();
    const std::string tuned_down = "rmma.m1.n1.w1x4";
    const std::string tuned      = "rmma.m1.n1.w4x1|" + tuned_down;
    const std::string path =
        (std::filesystem::temp_directory_path() / "ninferno_test_moe_r8_routes.json").string();
    {
        std::ofstream f(path);
        const auto entry = [](const char* format, const std::string& plan, int t_max = 16) {
            return std::string(R"({"format": ")") + format +
                   R"(", "experts": 16, "top_k": 4, "hidden": 256, "inter": 512, )" +
                   R"("bands": [[1, )" + std::to_string(t_max) + R"(, ")" + plan + R"("]]})";
        };
        f << R"({"schema": "ninferno-route-table/1", "gpu": {"class": ")" << ops::route_gpu_class()
          << R"("}, "entries": [], "moe": [)" << entry("iq3xxs", tuned) << ", "
          << entry("iq2s", tuned + "|a16") << ", "
          << entry("iq2xxs", "r8.w4r1|" + tuned_down, moe::kR8Tokens) << "]}";
    }
    const auto status = ops::load_route_table(path);
    std::filesystem::remove(path);
    if (!status.loaded) {
        std::cerr << "A8 route table: the test table did not load: " << status.message << '\n';
        return 1;
    }
    int failures = 0;
    std::uint32_t seed = 131U;
    for (const QType q : {QType::GGML_IQ3_XXS, QType::GGML_IQ2_S, QType::GGML_IQ2_XXS}) {
        Case c;
        c.label = std::string(qname(q)) + " E16 k4 (A8 table)";
        c.experts = 16; c.top_k = 4; c.hidden = 256; c.inter = 512; c.shared_inter = 0;
        c.gate_up = c.down = q;
        c.config.top_k = 4;
        c.expert_policy = ops::LinearPolicy::AllowA8;
        c.router_bias = false; c.selection_bias = false; c.expert_scale = false;
        c.expert_bias = false; c.shared_gate = false;
        c.epilogue = ops::MoeEpilogue::Store;
        const Layer l = make_layer(c, seed += 2);
        const auto x = random_bf16(static_cast<std::size_t>(16) * c.hidden, seed + 21, -1.0, 1.0);
        const std::vector<std::uint16_t> residual(x.size(), 0);
        for (std::int32_t t = 1; t <= 16; ++t) {
            const auto p = moe::select_plan(l.weights, c.config, t, false);
            const std::string name = moe::plan_name(p);
            const bool decode      = t <= moe::kR8Tokens;
            std::string expected;
            if (q == QType::GGML_IQ3_XXS) {
                // The measured down (1 x 4, which the heuristic route would take to the class's
                // down schedule) under the class's A8 gate/up at decode widths.
                const int s   = route && decode ? route->gate_up_at(t, false) : -1;
                const auto r8 = s >= 0 ? moe::r8_schedule_info(s) : moe::R8ScheduleInfo{0, 0};
                expected      = s >= 0 ? "r8.w" + std::to_string(r8.warps) + "r" +
                                        std::to_string(r8.rows) + "|" + tuned_down
                                      : tuned;
            } else if (q == QType::GGML_IQ2_S) {
                expected = tuned + "|a16";
            } else {
                // The named R8 gate/up at decode widths; above the band, the heuristic A16 plan.
                expected = decode ? "r8.w4r1|" + tuned_down
                                  : (p.gate_up_r8 < 0 && p.down_r8 < 0 ? name
                                                                       : "a register-MMA plan");
            }
            if (name != expected) {
                std::cerr << c.label << " T=" << t << ": plan " << name << ", expected " << expected
                          << '\n';
                ++failures;
                continue;
            }
            if (t == 2 || t == 5) {
                const Run run = run_op(c, l, x, 0, t, residual);
                failures += compare_run(c, l, x, residual, 0, t, run, a8_stages(c, p),
                                        c.label + " T=" + std::to_string(t) + " (" + name + ")");
            }
        }
    }
    ops::clear_route_table();
    if (failures == 0) std::cout << "A8 route table: measured bands kept, |a16 honoured\n";
    return failures;
}

// The K limit of the A8 route through select_plan: a layer of hidden 12288 (a CTA's quantized
// activation fits 2 slots, not 4) takes the class's A8 gate/up at T = 1 and 2 and A16 at T = 3
// and 4; both widths match the oracle.
int check_r8_k_limit() {
    namespace moe = ops::detail::moe;
    const auto route = moe::a8_default_route();
    Case c;
    c.label = "iq2s/iq3xxs E4 k2 H12288 F256 (A8 K limit)";
    c.experts = 4; c.top_k = 2; c.hidden = 12288; c.inter = 256; c.shared_inter = 0;
    c.gate_up = QType::GGML_IQ2_S;
    c.down    = QType::GGML_IQ3_XXS;
    c.config.top_k = 2;
    c.expert_policy = ops::LinearPolicy::AllowA8;
    c.router_bias = false; c.selection_bias = false; c.expert_scale = false;
    c.expert_bias = false; c.shared_gate = false;
    c.epilogue = ops::MoeEpilogue::Store;
    const Layer l = make_layer(c, 141U);
    const auto x = random_bf16(static_cast<std::size_t>(4) * c.hidden, 142U, -1.0, 1.0);
    const std::vector<std::uint16_t> residual(x.size(), 0);
    int failures = 0;
    for (std::int32_t t = 1; t <= moe::kR8Tokens; ++t) {
        const auto p = moe::select_plan(l.weights, c.config, t, false);
        const bool a8 = route && t <= 2 && route->gate_up_at(t, false) >= 0;
        const std::string label =
            c.label + " T=" + std::to_string(t) + " (" + moe::plan_name(p) + ")";
        if ((p.gate_up_r8 >= 0) != a8 || p.down_r8 >= 0) {
            std::cerr << label << ": expected " << (a8 ? "A8 gate/up" : "A16") << '\n';
            ++failures;
            continue;
        }
        if (t == 2 || t == 3) {
            const Run run = run_op(c, l, x, 0, t, residual);
            failures += compare_run(c, l, x, residual, 0, t, run, a8_stages(c, p), label);
        }
    }
    if (failures == 0) std::cout << c.label << ": A8 where the activation fits, A16 above\n";
    return failures;
}

// The routed experts' activation policy as the loader derives it (ops::prepare_moe_weights from
// the per-expert Uses, the Qwen3.5/3.6 MoE layout): A8 only where every gate, up and down Use
// allows it. A down bank of A16Only Uses keeps the whole layer A16 (no R8 gate/up at any decode
// width); every Use AllowA8 gives AllowA8 and the class's default route. IQ2_S experts, E 4, H 256,
// F 256, a BF16 router and shared expert.
int check_expert_policy() {
    namespace moe = ops::detail::moe;
    constexpr std::uint64_t E = 4, H = 256, F = 256;
    struct Parent {
        DeviceBuffer bytes;
        std::unique_ptr<WeightParent> parent;
    };
    const auto parent = [](QType q, std::uint64_t n, std::uint64_t k) {
        const std::array<std::uint64_t, 2> shape{n, k};
        const QuantLayout layout =
            q == QType::BF16 ? QuantLayout::Contiguous : ggml_published_layout(q);
        const auto g = weight_geometry(q, layout, shape);
        Parent p;
        p.bytes = DeviceBuffer(g.bytes);
        check(cudaMemset(p.bytes.p, 0, g.bytes), "zero");
        p.parent = std::make_unique<WeightParent>(
            WeightParent{g, static_cast<const std::byte*>(p.bytes.p), 0.0F});
        return p;
    };
    const auto rows = [](const Parent& p, std::uint64_t begin, std::uint64_t count,
                         std::uint64_t k) {
        return WeightView{{count, k}, {{p.parent.get(), begin * k, (begin + count) * k}}};
    };
    const Parent router = parent(QType::BF16, E, H), score = parent(QType::BF16, 1, H);
    const Parent shared_gu = parent(QType::BF16, 2 * F, H), shared_dn = parent(QType::BF16, H, F);
    const Parent gate_up = parent(QType::GGML_IQ2_S, E * 2 * F, H);
    const Parent down    = parent(QType::GGML_IQ2_S, E * H, F);
    std::vector<WeightView> gu_views, dn_views;
    for (std::uint64_t e = 0; e < E; ++e) {
        gu_views.push_back(rows(gate_up, 2 * e * F, F, H));     // gate
        gu_views.push_back(rows(gate_up, 2 * e * F + F, F, H)); // up
        dn_views.push_back(rows(down, e * H, H, F));
    }
    const WeightView router_v = rows(router, 0, E, H), score_v = rows(score, 0, 1, H);
    const WeightView sg = rows(shared_gu, 0, F, H), su = rows(shared_gu, F, F, H);
    const WeightView sd = rows(shared_dn, 0, H, F);
    ops::MoeConfig config;
    config.top_k = 2;
    const auto route = moe::a8_default_route();
    int failures     = 0;
    for (const auto down_policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
        std::vector<ops::WeightInput> gu_in, dn_in;
        for (const auto& v : gu_views) gu_in.push_back({v, ops::LinearPolicy::AllowA8});
        for (const auto& v : dn_views) dn_in.push_back({v, down_policy});
        const auto w = ops::prepare_moe_weights(
            {router_v, ops::LinearPolicy::A16Only}, {score_v, ops::LinearPolicy::A16Only}, gu_in,
            dn_in, {sg, ops::LinearPolicy::A16Only}, {su, ops::LinearPolicy::A16Only},
            {sd, ops::LinearPolicy::A16Only});
        const bool a8 = down_policy == ops::LinearPolicy::AllowA8;
        const std::string label =
            std::string("expert policy: gate/up AllowA8, down ") + (a8 ? "AllowA8" : "A16Only");
        if (w.expert_policy != down_policy) {
            std::cerr << label << ": the layer's expert policy is not "
                      << (a8 ? "AllowA8" : "A16Only") << '\n';
            ++failures;
        }
        for (std::int32_t t = 1; t <= moe::kR8Tokens; ++t) {
            const auto p = moe::select_plan(w, config, t, false);
            const int expected = a8 && route ? route->gate_up_at(t, false) : -1;
            if (p.gate_up_r8 != expected || p.down_r8 >= 0) {
                std::cerr << label << " T=" << t << ": " << moe::plan_name(p) << ", gate/up R8 "
                          << expected << " expected\n";
                ++failures;
            }
        }
    }
    // A Q4_0 or Q4_K frag bank under an A8 policy takes the class's frag route (its fragment-order
    // R8 codecs, moe_r8.cuh); an IQ3_S bank labelled ggml_frag_v1 (no frag codec; the grouped
    // kernel has no frag map for it either, so only the R8 admission is asked) stays on A16.
    for (const QType q : {QType::GGML_Q4_0, QType::GGML_Q4_K}) {
        const Parent bank = parent(q, E * 2 * F, H);
        const Parent dn   = parent(q, E * H, F);
        ops::MoeWeights w;
        w.router         = native_weight(router_v);
        w.gate_up        = native_weight(rows(bank, 0, E * 2 * F, H));
        w.down           = native_weight(rows(dn, 0, E * H, F));
        w.gate_up.layout = w.down.layout = QuantLayout::GgmlFrag;
        w.expert_policy  = ops::LinearPolicy::AllowA8;
        for (std::int32_t t = 1; t <= moe::kR8Tokens; ++t) {
            const auto p       = moe::select_plan(w, config, t, false);
            const int expected = route ? route->gate_up_at(t, true) : -1;
            if (p.gate_up_r8 != expected || p.down_r8 != -1 || !moe::r8_admits(w, w.gate_up, t)) {
                std::cerr << "frag " << qname(q) << " T=" << t << ": took " << moe::plan_name(p)
                          << '\n';
                ++failures;
            }
        }
    }
    {
        const Parent bank = parent(QType::GGML_IQ3_S, E * 2 * F, H);
        Weight iq3s       = native_weight(rows(bank, 0, E * 2 * F, H));
        const bool published = moe::r8_expert_supports(iq3s);
        iq3s.layout          = QuantLayout::GgmlFrag;
        if (!published || moe::r8_expert_supports(iq3s)) {
            std::cerr << "iq3s: R8 admits published " << published << ", ggml_frag_v1 "
                      << moe::r8_expert_supports(iq3s) << " (expected 1, 0)\n";
            ++failures;
        }
    }
    if (failures == 0) {
        std::cout << "expert policy: a down bank on A16Only keeps the layer A16; Q4_0 and Q4_K "
                     "frag banks take the frag route, a frag bank without frag R8 codecs stays "
                     "A16\n";
    }
    return failures;
}

// Shipped-class geometries: gpt-oss-20b-shaped (32 x 2880 top-4, MXFP4, clamped SwiGLU,
// interleaved, biases) at reduced K, and a Qwen3-30B-A3B-shaped bank (128 x 768 top-8).
Case oss_like() {
    Case c;
    c.label = "gpt-oss-like mxfp4 E32 k4";
    c.experts = 32; c.top_k = 4; c.hidden = 2880; c.inter = 256; c.shared_inter = 0;
    c.gate_up = c.down = QType::MXFP4;
    c.config.top_k = 4;
    c.config.activation = ops::MoeActivation::ClampedSwiGlu;
    c.config.gate_up_layout = ops::MoeGateUpLayout::Interleaved;
    c.router_bias = true; c.selection_bias = false; c.expert_scale = false;
    c.expert_bias = true; c.shared_gate = false;
    c.epilogue = ops::MoeEpilogue::AddResidual;
    return c;
}

Case qwen3_like() {
    Case c;
    c.label = "qwen3-moe-like nvfp4 E128 k8";
    c.experts = 128; c.top_k = 8; c.hidden = 512; c.inter = 192; c.shared_inter = 0;
    c.gate_up = c.down = QType::NVFP4;
    c.config.top_k = 8;
    c.router_bias = false; c.selection_bias = false; c.expert_scale = false;
    c.expert_bias = false; c.shared_gate = false;
    c.epilogue = ops::MoeEpilogue::AddResidual;
    return c;
}

// A Qwen3.6-35B-A3B-shaped layer as a published int4 checkpoint keeps it (Intel AutoRound,
// GPTQ-Int4): 256 experts of 2048 x 512, top-8 softmax renormalized, SwiGLU, int4 g128 experts
// with zero points, a BF16 shared expert of width 512 behind its sigmoid gate. The Qwen3.5
// engine runs such a layer on this Op (SparseMoe has no int4 route).
Case qwen36_a3b_u4(QType format, QType shared) {
    Case c;
    c.label = std::string("qwen3.6-35b-a3b-like ") + (format == QType::U4_G128_ZP_FP16 ? "u4 g128" : "u4 g32") +
              (shared == QType::BF16 ? ", bf16 shared" : ", u4 shared") + " E256 k8";
    c.experts = 256; c.top_k = 8; c.hidden = 2048; c.inter = 512; c.shared_inter = 512;
    c.gate_up = c.down = format;
    c.shared_format = shared;
    c.config.top_k = 8;
    c.router_bias = false; c.selection_bias = false; c.expert_scale = false;
    c.expert_bias = false; c.shared_gate = true;
    c.epilogue = ops::MoeEpilogue::AddResidual;
    return c;
}

int check_table_plans() {
    const std::string path =
        (std::filesystem::temp_directory_path() / "ninferno_test_moe_routes.json").string();
    {
        std::ofstream f(path);
        f << R"({"schema": "ninferno-route-table/1", "gpu": {"class": ")" << ops::route_gpu_class()
          << R"("}, "entries": [], "moe": [
  {"format": "mxfp4", "experts": 32, "top_k": 4, "hidden": 2880, "inter": 256,
   "bands": [[1, 16, "rmma.m1.n1.w4x1|rmma.m1.n1.w4x1"]],
   "prefill": "rmma.m1.n4.w4x1|rmma.m1.n4.w4x1"},
  {"format": "nvfp4", "experts": 128, "top_k": 8, "hidden": 512, "inter": 192,
   "prefill": "rmma.m1.n8.w4x1|rmma.m1.n4.w4x1"}]})";
    }
    const auto status = ops::load_route_table(path);
    std::filesystem::remove(path);
    int failures = 0;
    if (!status.loaded) {
        std::cerr << "route-table plans: the test table did not load: " << status.message << '\n';
        return 1;
    }
    const struct {
        Case c;
        std::uint32_t seed;
        std::int32_t tokens;
        const char* plan;
    } cases[] = {{oss_like(), 83U, 8, "rmma.m1.n1.w4x1|rmma.m1.n1.w4x1"},
                 {oss_like(), 83U, 300, "rmma.m1.n4.w4x1|rmma.m1.n4.w4x1"},
                 {qwen3_like(), 85U, 300, "rmma.m1.n8.w4x1|rmma.m1.n4.w4x1"}};

    for (const auto& k : cases) {
        const Layer l = make_layer(k.c, k.seed);
        const auto plan = ops::detail::moe::plan_name(
            ops::detail::moe::select_plan(l.weights, k.c.config, k.tokens, true));
        if (plan != k.plan) {
            std::cerr << k.c.label << " T=" << k.tokens << ": the table is not steering (" << plan
                      << ")\n";
            ++failures;
        }
    }
    for (const Case& base : {oss_like(), qwen3_like()}) {
        Case c = base;
        c.label += " (table)";
        const ops::ScopedPrefillRoutes profile(true);
        failures += check_case(c, 83U, {1, 8, 17, 300});
        failures += check_invariance(c, 84U);
    }
    {
        // The invariant profile ignores the table: the streaming K split at every width.
        const Layer l = make_layer(oss_like(), 83U);
        const ops::ScopedInvariantRoutes invariant(true);
        const struct {
            std::int32_t tokens;
            const char* plan;
        } expected[] = {{1, "rmma.m1.n1.w1x8|rmma.m1.n1.w4x1"},
                        {8, "rmma.m1.n1.w1x8|rmma.m1.n1.w4x1"},
                        {300, "rmma.m1.n4.w1x8|rmma.m1.n8.w4x1"}};
        for (const auto& e : expected) {
            for (const bool prefill : {false, true}) {
                const auto plan = ops::detail::moe::plan_name(
                    ops::detail::moe::select_plan(l.weights, oss_like().config, e.tokens, prefill));
                if (plan != e.plan) {
                    std::cerr << "invariant profile T=" << e.tokens << ": plan " << plan
                              << " is not the streaming split\n";
                    ++failures;
                }
            }
        }
        failures += check_invariant_profile(oss_like(), 86U);
    }
    ops::clear_route_table();
    return failures;
}

// MoE expert offload's CPU decode path (moe_host_experts): the same layer with its routed planes
// and expert biases at host addresses. Held to the oracle as moe() is; close to moe() (same
// selection, same roundings, another summation order); bytes independent of the CPU's thread
// count and instruction set; a captured graph replays the eager bytes; refusals before any launch.
struct HostLayer {
    std::vector<std::uint8_t> gate_up, down;
    std::vector<std::uint16_t> gate_up_bias_bf16;
    std::vector<float> down_bias_f32;
    ops::MoeWeights weights;
};

HostLayer host_copy(const Layer& l) {
    HostLayer h;
    h.gate_up         = l.gate_up.packed.payload;
    h.down            = l.down.packed.payload;
    h.weights         = l.weights;
    h.weights.gate_up = l.gate_up.packed.device_weight(h.gate_up.data());
    h.weights.down    = l.down.packed.device_weight(h.down.data());
    if (l.has_expert_bias) {
        for (const double v : l.gate_up_bias.values) h.gate_up_bias_bf16.push_back(to_bf16(v));
        for (const double v : l.down_bias.values) h.down_bias_f32.push_back(static_cast<float>(v));
        h.weights.gate_up_bias = Tensor(h.gate_up_bias_bf16.data(), DType::BF16,
                                        {static_cast<std::int32_t>(h.gate_up_bias_bf16.size())});
        h.weights.down_bias = Tensor(h.down_bias_f32.data(), DType::FP32,
                                     {static_cast<std::int32_t>(h.down_bias_f32.size())});
    }
    return h;
}

// One call on `stream`: eager, or captured once and replayed `replays` times (out reset to the
// residual or the poison before each replay).
std::vector<std::uint16_t> run_host_op(const Case& c, const Layer& l, const HostLayer& h,
                                       const std::vector<std::uint16_t>& x, std::int32_t t,
                                       const std::vector<std::uint16_t>& residual,
                                       ops::HostExperts* host, cudaStream_t stream, int replays) {
    DeviceBuffer dx(static_cast<std::size_t>(t) * c.hidden * 2);
    dx.copy_from_host(x.data(), dx.bytes);
    DeviceBuffer dout(dx.bytes);
    // Every copy and fill on `stream`, so the non-blocking stream orders them.
    const auto reset = [&] {
        if (c.epilogue == ops::MoeEpilogue::AddResidual) {
            check(cudaMemcpyAsync(dout.p, residual.data(), dout.bytes, cudaMemcpyHostToDevice,
                                  stream),
                  "residual");
        } else {
            check(cudaMemsetAsync(dout.p, 0xff, dout.bytes, stream), "poison");
        }
    };
    DeviceArena workspace(std::max<std::size_t>(ops::moe_workspace_bytes(l.weights, c.config, t, t), 256));
    Tensor tx(dx.p, DType::BF16, {c.hidden, t});
    Tensor tout(dout.p, DType::BF16, {c.hidden, t});
    const auto call = [&] {
        if (host != nullptr) {
            ops::moe_host_experts(tx, l.weights, h.weights, c.config, c.epilogue, tout, workspace,
                                  stream, *host);
        } else {
            ops::moe(tx, l.weights, c.config, c.epilogue, tout, workspace, stream);
        }
    };
    reset();
    if (replays == 0) {
        call();
        check(cudaStreamSynchronize(stream), "moe_host_experts");
    } else {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        call();
        check(cudaStreamEndCapture(stream, &graph), "end capture");
        check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        for (int r = 0; r < replays; ++r) {
            reset();
            check(cudaGraphLaunch(exec, stream), "graph launch");
            check(cudaStreamSynchronize(stream), "graph replay");
        }
        check(cudaGraphExecDestroy(exec), "graph exec destroy");
        check(cudaGraphDestroy(graph), "graph destroy");
    }
    std::vector<std::uint16_t> out(static_cast<std::size_t>(t) * c.hidden);
    check(cudaMemcpyAsync(out.data(), dout.p, dout.bytes, cudaMemcpyDeviceToHost, stream),
          "download");
    check(cudaStreamSynchronize(stream), "download");
    return out;
}

int check_host_experts(const Case& c, std::uint32_t seed) {
    const Layer l     = make_layer(c, seed);
    const HostLayer h = host_copy(l);
    const auto x        = random_bf16(static_cast<std::size_t>(16) * c.hidden, seed + 41, -1.0, 1.0);
    const auto residual = random_bf16(x.size(), seed + 42, -2.0, 2.0);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    ops::HostExperts fast(c.hidden, c.top_k, 0, ops::host::detect_isa());
    ops::HostExperts scalar(c.hidden, c.top_k, 1, ops::host::Isa::Scalar);
    int failures = 0;
    for (const std::int32_t t : {1, 3, 16}) {
        const std::string label = c.label + " host experts T=" + std::to_string(t);
        const auto got    = run_host_op(c, l, h, x, t, residual, &fast, stream, 0);
        const auto one    = run_host_op(c, l, h, x, t, residual, &scalar, stream, 0);
        const auto graph  = run_host_op(c, l, h, x, t, residual, &fast, stream, 3);
        const auto device = run_host_op(c, l, h, x, t, residual, nullptr, stream, 0);
        if (got != one) {
            std::cerr << label << ": bytes differ between " << ops::host::isa_name(fast.isa())
                      << " x" << fast.threads() << " and scalar x1\n";
            ++failures;
        }
        if (got != graph) {
            std::cerr << label << ": graph replay differs from the eager call\n";
            ++failures;
        }
        double err2 = 0.0, ref2 = 0.0, dev2 = 0.0, worst = 0.0, scale = 0.0;
        int compared = 0, ties = 0;
        std::vector<double> xs(c.hidden), r;
        for (std::int32_t token = 0; token < t; ++token) {
            for (int j = 0; j < c.hidden; ++j) xs[j] = from_bf16(x[static_cast<std::size_t>(token) * c.hidden + j]);
            if (!reference_token(c, l, xs, r)) {
                ++ties;
                continue;
            }
            ++compared;
            for (int j = 0; j < c.hidden; ++j) {
                const std::size_t i = static_cast<std::size_t>(token) * c.hidden + j;
                double ref = r[j];
                if (c.epilogue == ops::MoeEpilogue::AddResidual) ref += from_bf16(residual[i]);
                const double v = from_bf16(got[i]);
                if (!std::isfinite(v)) {
                    worst = INFINITY;
                    continue;
                }
                err2 += (v - ref) * (v - ref);
                ref2 += ref * ref;
                dev2 += (v - from_bf16(device[i])) * (v - from_bf16(device[i]));
                worst = std::max(worst, std::abs(v - ref));
                scale = std::max(scale, std::abs(ref));
            }
        }
        const double rel     = ref2 > 0 ? std::sqrt(err2 / ref2) : std::sqrt(err2);
        const double rel_gpu = ref2 > 0 ? std::sqrt(dev2 / ref2) : std::sqrt(dev2);
        const bool ok = compared > 0 && std::isfinite(worst) && rel <= 1.5e-2 &&
                        worst <= 0.08 * scale + 1e-3 && rel_gpu <= 1.0e-2;
        if (!ok) {
            std::cerr << label << ": rel_l2 " << rel << " vs moe() " << rel_gpu << " worst "
                      << worst << " (max |ref| " << scale << ") compared " << compared
                      << " near-ties " << ties << '\n';
            ++failures;
        } else {
            std::cout << label << ": rel_l2 " << rel << " vs moe() " << rel_gpu
                      << ", bytes equal over ISA/threads and graph replay\n";
        }
    }
    // Refusals, before any launch: 17 tokens; a format the CPU does not read.
    {
        const auto big = random_bf16(static_cast<std::size_t>(17) * c.hidden, seed + 43, -1.0, 1.0);
        bool refused = false;
        try {
            (void)run_host_op(c, l, h, big, 17, big, &fast, stream, 0);
        } catch (const std::invalid_argument&) { refused = true; }
        if (!refused) {
            std::cerr << c.label << ": the CPU expert path accepted 17 tokens\n";
            ++failures;
        }
        HostLayer wrong = h;
        wrong.weights.down.qtype = QType::Q8_G32_FP16;
        refused = false;
        try {
            (void)run_host_op(c, l, wrong, x, 1, residual, &fast, stream, 0);
        } catch (const std::invalid_argument&) { refused = true; }
        if (!refused) {
            std::cerr << c.label << ": the CPU expert path accepted a Q8 down plane\n";
            ++failures;
        }
    }
    check(cudaStreamDestroy(stream), "stream destroy");
    return failures;
}

const char* qname(QType q) {
    switch (q) {
    case QType::BF16: return "bf16";
    case QType::FP8_E4M3FN_ROW_BF16: return "fp8";
    case QType::NVFP4: return "nvfp4";
    case QType::MXFP4: return "mxfp4";
    case QType::Q4_G64_FP16: return "q4";
    case QType::Q5_G64_FP16: return "q5";
    case QType::Q6_G64_FP16: return "q6";
    case QType::Q8_G32_FP16: return "q8";
    case QType::FP8_E4M3FN_ROW_F32: return "fp8r32";
    case QType::U4_G128_ZP_FP16: return "u4zp";
    case QType::U4_G32_ZP_FP16: return "u4zp32";
    case QType::U4_G64_ZP_FP16: return "u4zp64";
    case QType::GGML_Q4_K: return "q4k";
    case QType::GGML_Q5_K: return "q5k";
    case QType::GGML_Q6_K: return "q6k";
    case QType::GGML_Q3_K: return "q3k";
    case QType::GGML_Q2_K: return "q2k";
    case QType::GGML_IQ4_XS: return "iq4xs";
    case QType::GGML_IQ3_XXS: return "iq3xxs";
    case QType::GGML_IQ3_S: return "iq3s";
    case QType::GGML_Q4_0: return "q40";
    case QType::GGML_Q4_1: return "q41";
    case QType::GGML_Q5_0: return "q50";
    case QType::GGML_Q5_1: return "q51";
    case QType::GGML_IQ4_NL: return "iq4nl";
    case QType::GGML_IQ2_XXS: return "iq2xxs";
    case QType::GGML_IQ2_XS: return "iq2xs";
    case QType::GGML_IQ2_S: return "iq2s";
    case QType::GGML_IQ1_S: return "iq1s";
    case QType::GGML_IQ1_M: return "iq1m";
    case QType::GGML_TQ1_0: return "tq10";
    case QType::GGML_TQ2_0: return "tq20";
    case QType::GGML_Q1_0: return "q10";
    case QType::GGML_Q2_0: return "q20";
    default: return "?";
    }
}

} // namespace

// The load-time tuner (ops::tune_moe): its plans are in place after it runs and moe() takes them
// (moe()'s bytes are its selected plan's), the cache gives the same plans back to a process that
// forgot them without timing again, and NINFERNO_MOE_TUNE=off, like the default, tunes nothing.
int check_tuner() {
    namespace mo = ops::detail::moe;
    int failures   = 0;
    const auto path = std::filesystem::temp_directory_path() / "ninferno_test_moe_tune.json";
    std::filesystem::remove(path);
    ::setenv("NINFERNO_MOE_TUNE_CACHE", path.c_str(), 1);
    ::setenv("NINFERNO_MOE_TUNE", "on", 1);
    Case c;
    c.label   = "tuner q4_k E32 k4";
    c.experts = 32; c.top_k = 4; c.hidden = 512; c.inter = 256; c.shared_inter = 0;
    c.gate_up = c.down = QType::GGML_Q4_K;
    c.config.top_k = 4;
    c.router_bias = false; c.selection_bias = false; c.expert_scale = false;
    c.expert_bias = false; c.shared_gate = false;
    c.epilogue = ops::MoeEpilogue::Store;
    const Layer l = make_layer(c, 4242U);
    constexpr std::int32_t kWidths = 8;
    const auto fail = [&](const std::string& what) {
        std::cerr << "tuner: " << what << '\n';
        ++failures;
    };
    mo::clear_tuned_plans();
    const auto first = ops::tune_moe(l.weights, c.config, kWidths, nullptr);
    if (!first.ran || first.cached || !std::filesystem::exists(path)) {
        fail("the first tune did not time and write its cache: " + first.summary);
    }
    const auto x = random_bf16(static_cast<std::size_t>(kWidths) * c.hidden, 99U, -1.0, 1.0);
    std::vector<std::string> names;
    for (std::int32_t t = 1; t <= kWidths; ++t) {
        const auto plan = mo::select_plan(l.weights, c.config, t, false);
        names.push_back(mo::plan_name(plan));
        const auto chosen = run_op(c, l, x, 0, t, {});
        const auto named  = run_op(c, l, x, 0, t, {}, &plan);
        if (chosen.out != named.out) {
            fail("T=" + std::to_string(t) + ": moe() is not its selected plan " + names.back());
        }
    }
    mo::clear_tuned_plans();
    const auto second = ops::tune_moe(l.weights, c.config, kWidths, nullptr);
    if (!second.ran || !second.cached) { fail("the second tune did not read the cache: " + second.summary); }
    for (std::int32_t t = 1; t <= kWidths; ++t) {
        const auto name = mo::plan_name(mo::select_plan(l.weights, c.config, t, false));
        if (name != names[static_cast<std::size_t>(t - 1)]) {
            fail("T=" + std::to_string(t) + ": the cached plan " + name + " is not the tuned " +
                 names[static_cast<std::size_t>(t - 1)]);
        }
    }
    mo::clear_tuned_plans();
    ::setenv("NINFERNO_MOE_TUNE", "off", 1);
    const auto off = ops::tune_moe(l.weights, c.config, kWidths, nullptr);
    if (off.ran) { fail("NINFERNO_MOE_TUNE=off tuned"); }
    ::unsetenv("NINFERNO_MOE_TUNE");
    if (ops::tune_moe(l.weights, c.config, kWidths, nullptr).ran) { fail("the default tuned"); }
    ::unsetenv("NINFERNO_MOE_TUNE_CACHE");
    std::filesystem::remove(path);
    mo::clear_tuned_plans();
    std::cout << "tuner: " << first.summary << '\n';
    return failures;
}

// The fused selection and plan of a decode call (at most 32 assignments: one warp's plan) against
// the separate selection and the CTA plan: the same ids, weights, slots and job list, bit for bit.
// Ties included (logits from a few levels), narrow jobs included (an expert's run cut in several).
int check_warp_plan() {
    namespace mo = ops::detail::moe;
    int failures = 0;
    std::mt19937 rng(20261001U);
    constexpr int kMaxA = 32, kMaxT = 16;
    float* logits = nullptr;
    int* ints     = nullptr;
    float* w      = nullptr;
    float* bias   = nullptr;
    // ids, slot_of, packed_token, job_expert, job_begin, job_count, job_total; two copies.
    constexpr int kInts = 6 * kMaxA + 1;
    check(cudaMalloc(&logits, sizeof(float) * kMaxT * 512), "malloc logits");
    check(cudaMalloc(&ints, sizeof(int) * 2 * kInts), "malloc ints");
    check(cudaMalloc(&w, sizeof(float) * 2 * kMaxA), "malloc weights");
    check(cudaMalloc(&bias, sizeof(float) * 512), "malloc bias");
    {
        std::normal_distribution<float> normal(0.0F, 0.5F);
        std::vector<float> host(512);
        for (auto& v : host) v = normal(rng);
        check(cudaMemcpy(bias, host.data(), sizeof(float) * host.size(), cudaMemcpyHostToDevice),
              "copy bias");
    }
    int cases = 0;
    for (const int experts : {8, 64, 128, 256, 512}) {
        for (const int top_k : {1, 2, 4, 6, 8}) {
            if (top_k > experts) continue;
            for (int tokens = 1; tokens <= kMaxT && tokens * top_k <= kMaxA; ++tokens) {
                for (const int columns : {mo::kJobColumns, 2, 1}) {
                    for (const int scoring : {0, 1, 2})
                        for (const bool ties : {false, true}) {
                            std::vector<float> host(static_cast<std::size_t>(tokens) * experts);
                            std::uniform_int_distribution<int> level(0, 3);
                            std::normal_distribution<float> normal(0.0F, 2.0F);
                            for (auto& v : host)
                                v = ties ? 0.5F * static_cast<float>(level(rng)) : normal(rng);
                            check(cudaMemcpy(logits, host.data(), sizeof(float) * host.size(),
                                             cudaMemcpyHostToDevice),
                                  "copy logits");
                            check(cudaMemset(ints, 0xff, sizeof(int) * 2 * kInts), "clear ints");
                            std::vector<int> out[2];
                            for (int arm = 0; arm < 2; ++arm) {
                                int* base = ints + arm * kInts;
                                mo::SelectArgs s;
                                s.logits  = logits;
                                s.experts = experts;
                                s.top_k   = top_k;
                                s.tokens  = tokens;
                                s.ids     = base;
                                s.weights = w + arm * kMaxA;
                                // softmax; sigmoid with a selection bias and a routed scale
                                // (Nemotron); sigmoid without them.
                                s.sigmoid        = scoring != 0;
                                s.selection_bias = scoring == 1 ? bias : nullptr;
                                s.routed_scale   = scoring == 1 ? 2.5F : 1.0F;
                                mo::PlanArgs p;
                                p.ids          = base;
                                p.assignments  = tokens * top_k;
                                p.top_k        = top_k;
                                p.experts      = experts;
                                p.job_columns  = columns;
                                p.slot_of      = base + kMaxA;
                                p.packed_token = base + 2 * kMaxA;
                                p.job_expert   = base + 3 * kMaxA;
                                p.job_begin    = base + 4 * kMaxA;
                                p.job_count    = base + 5 * kMaxA;
                                p.job_total    = base + 6 * kMaxA;
                                if (arm == 0) {
                                    mo::launch_select(s, nullptr);
                                    mo::launch_plan(p, nullptr);
                                } else {
                                    mo::launch_select_plan(s, p, nullptr);
                                }
                                out[arm].resize(kInts + kMaxA);
                                check(cudaMemcpy(out[arm].data(), base, sizeof(int) * kInts,
                                                 cudaMemcpyDeviceToHost),
                                      "copy plan");
                                check(cudaMemcpy(out[arm].data() + kInts, w + arm * kMaxA,
                                                 sizeof(float) * kMaxA, cudaMemcpyDeviceToHost),
                                      "copy weights");
                            }
                            ++cases;
                            if (out[0] != out[1]) {
                                ++failures;
                                std::cout << "FAIL warp plan E" << experts << " k" << top_k << " T"
                                          << tokens << " columns " << columns << " scoring "
                                          << scoring << (ties ? " ties" : "") << '\n';
                            }
                        }
                }
            }
        }
    }
    check(cudaFree(logits), "free");
    check(cudaFree(ints), "free");
    check(cudaFree(w), "free");
    check(cudaFree(bias), "free");
    std::cout << "warp plan: " << cases << " cases, " << failures << " failures\n";
    return failures;
}

// --part random0|random1|random2|layers runs one share of the cases, no argument all of them:
// ctest runs the four shares apart (tests/ops/tests.cmake), 125 s in one process on the RTX 5090.
// Every random case is still drawn in each share, so the generator's sequence never moves.
int main(int argc, char** argv) {
    const std::string part = argc == 3 && std::string_view(argv[1]) == "--part" ? argv[2] : "";
    if (argc != 1 && part != "random0" && part != "random1" && part != "random2" &&
        part != "layers" && part != "mx8" && part != "int8") {
        std::cerr << "usage: ninferno_moe_test [--part random0|random1|random2|layers|mx8|int8]\n";
        return 2;
    }
    // int8: random0 with the MXFP4 banks on the int8 W4A8, which sm_120a builds otherwise leave
    // for the block-scaled MMA.
    if (part == "int8") ops::detail::moe::set_mx8(false);
#ifndef NINFERNO_WITHOUT_NVFP4
    g_mx8 = part != "int8";
#endif
    const std::string share = part == "int8" ? "random0" : part;
    if (part == "mx8") {
        // MXFP4 banks on the block-scaled MMA: sm_120a family builds only.
#ifdef NINFERNO_WITHOUT_NVFP4
        std::cout << "no block-scaled MMA in this build; skipping mx8\n";
        return 77;
#else
        if (!cuda_available()) return 77;
        ops::detail::moe::set_mx8(true);
        g_mx8        = true;
        int failures = 0;
        try {
            failures += check_w4a8(oss_like(), 111U, {17, 70, 130});
            failures += check_w4a8(oss_like(), 118U, {33, 200, 300});
        } catch (const std::exception& e) {
            std::cerr << "mx8: " << e.what() << '\n';
            ++failures;
        }
        if (failures != 0) {
            std::cerr << "moe mx8: " << failures << " failure(s)\n";
            return 1;
        }
        std::cout << "moe mx8: OK\n";
        return 0;
#endif
    }
    const auto runs = [&](std::string_view s) { return share.empty() || share == s; };
    if (!cuda_available()) {
        std::cout << "CUDA unavailable; skipping moe test\n";
        return 77;
    }
    int failures = 0;
    std::mt19937 rng(20260925U);
    try {
        if (runs("layers")) failures += check_warp_plan();
        // 1. Random geometries in every expert format.
        const QType formats[] = {QType::MXFP4, QType::NVFP4, QType::BF16, QType::FP8_E4M3FN_ROW_BF16,
                                 QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q6_G64_FP16,
                                 QType::Q8_G32_FP16, QType::FP8_E4M3FN_ROW_F32,
                                 QType::U4_G128_ZP_FP16, QType::U4_G32_ZP_FP16,
                                 QType::U4_G64_ZP_FP16, QType::GGML_Q4_K, QType::GGML_Q5_K,
                                 QType::GGML_Q6_K, QType::GGML_Q3_K, QType::GGML_Q2_K,
                                 QType::GGML_IQ4_XS, QType::GGML_IQ3_XXS, QType::GGML_IQ3_S,
                                 QType::GGML_Q4_0, QType::GGML_Q4_1, QType::GGML_Q5_0,
                                 QType::GGML_Q5_1, QType::GGML_IQ4_NL, QType::GGML_IQ2_XXS,
                                 QType::GGML_IQ2_XS, QType::GGML_IQ2_S, QType::GGML_IQ1_S,
                                 QType::GGML_IQ1_M, QType::GGML_TQ1_0, QType::GGML_TQ2_0,
                                 QType::GGML_Q1_0, QType::GGML_Q2_0};
        int index = 0;
        for (const QType q : formats) {
            for (int rep = 0; rep < 3; ++rep, ++index) {
                Case c;
                const bool nv = q == QType::NVFP4;
                c.experts = nv ? 2 * static_cast<std::int32_t>(1 + rng() % 8)
                               : static_cast<std::int32_t>(1 + rng() % 40);
                c.top_k  = static_cast<std::int32_t>(1 + rng() % std::min(c.experts, 8));
                // NVFP4: K % 64 and whole 128-row tiles per plane (E even, H % 64, 2F % 64).
                // U4: K (H for gate/up, F for down) in whole groups (32, 64 or 128 values).
                // ggml formats: whole 256-value superblocks; the 32-value block formats' F any
                // whole blocks (a down K that ends in a part 128-value step).
                const auto u4 = static_cast<std::int32_t>(u4_zp_group(q));
                const bool kq = is_ggml_block(q);
                const bool b32 = kq && ggml_block_values(q) == 32;
                c.hidden = nv   ? 64 * static_cast<std::int32_t>(1 + rng() % 5)
                           : u4 ? u4 * static_cast<std::int32_t>(1 + rng() % (384 / u4))
                           : kq ? 256 * static_cast<std::int32_t>(1 + rng() % 2)
                                : 32 * static_cast<std::int32_t>(1 + rng() % 10);
                c.inter  = nv   ? 64 * static_cast<std::int32_t>(1 + rng() % 4)
                           : u4 ? u4 * static_cast<std::int32_t>(1 + rng() % (384 / u4))
                           : b32 ? 32 * static_cast<std::int32_t>(1 + rng() % 16)
                           : kq ? 256 * static_cast<std::int32_t>(1 + rng() % 2)
                                : 32 * static_cast<std::int32_t>(1 + rng() % 8);
                c.shared_inter = rep == 1 ? 64 * static_cast<std::int32_t>(1 + rng() % 3) : 0;
                c.gate_up = q;
                c.down    = q;
                c.config.top_k    = c.top_k;
                c.config.scoring  = rep == 2 ? ops::MoeScoring::Sigmoid : ops::MoeScoring::Softmax;
                c.config.renormalize = rep != 2 || (rng() & 1U);
                c.config.routed_scale = rep == 2 ? 2.5F : 1.0F;
                c.config.activation = static_cast<ops::MoeActivation>((index / 3 + rep) % 4);
                if (c.shared_inter > 0) c.config.activation = ops::MoeActivation::SwiGlu;
                c.config.gate_up_layout = (index % 2) ? ops::MoeGateUpLayout::Interleaved
                                                      : ops::MoeGateUpLayout::Concatenated;
                c.router_bias    = (index % 3) == 0;
                c.selection_bias = rep == 2;
                c.expert_scale   = (index % 4) == 1;
                c.expert_bias    = (index % 2) == 0;
                c.shared_gate    = c.shared_inter > 0 && (rng() & 1U);
                c.epilogue = (index % 2) ? ops::MoeEpilogue::AddResidual : ops::MoeEpilogue::Store;
                c.label = std::string(qname(q)) + " E" + std::to_string(c.experts) + " k" +
                          std::to_string(c.top_k) + " H" + std::to_string(c.hidden) + " F" +
                          std::to_string(c.inter) + (c.shared_inter ? " shared" + std::to_string(c.shared_inter) : "") +
                          " act" + std::to_string(static_cast<int>(c.config.activation)) +
                          (c.config.scoring == ops::MoeScoring::Sigmoid ? " sigmoid" : " softmax") +
                          (c.config.gate_up_layout == ops::MoeGateUpLayout::Interleaved ? " interleaved" : "");
                if (runs("random" + std::to_string(index / 3 % 3))) {
                    failures += check_case(c, 1000U + static_cast<std::uint32_t>(index),
                                           {1, 3, 16, 40, 130});
                }
            }
        }
        if (runs("layers")) {
            // 2. Shipped-class geometries (oss_like, qwen3_like) on the default routes.
            // 1b. A gate/up bank zero-padded past H (Nemotron-H's padded GGUF: H 2688, K 2816): the
            //     Op widens x with zeros, at decode (R8 and A16) and prefill widths.
            for (const QType q : {QType::GGML_IQ3_XXS, QType::GGML_Q4_K, QType::BF16}) {
                for (const auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
                    Case c;
                    c.experts      = 16;
                    c.top_k        = 4;
                    c.hidden       = 320;
                    c.input_pad    = 192;
                    c.inter        = 256;
                    c.shared_inter = 64;
                    c.gate_up = c.down    = q;
                    c.expert_policy       = policy;
                    c.config.top_k        = c.top_k;
                    c.config.scoring      = ops::MoeScoring::Sigmoid;
                    c.config.routed_scale = 2.5F;
                    c.config.activation   = ops::MoeActivation::Relu2;
                    c.router_bias         = false;
                    c.selection_bias      = true;
                    c.expert_scale        = false;
                    c.expert_bias         = false;
                    c.shared_gate         = false;
                    c.epilogue            = ops::MoeEpilogue::AddResidual;
                    c.label               = std::string("padded K ") + qname(q) +
                              (policy == ops::LinearPolicy::AllowA8 ? " a8" : " a16") +
                              " H320 K512";
                    failures += check_case(c, 90U, {1, 2, 4, 16, 40, 130});
                }
            }
            {
                const Case c = oss_like();
                failures += check_case(c, 77U, {1, 4, 17, 200});
                failures += check_invariance(c, 78U);
                {
                    const ops::ScopedInvariantRoutes invariant(true);
                    failures += check_case(c, 77U, {1, 9, 40, 300});
                }
                failures += check_invariant_profile(c, 78U);
            }
            {
                const Case c = qwen3_like();
                failures += check_case(c, 79U, {1, 8, 64, 300});
                failures += check_invariance(c, 80U);
                {
                    const ops::ScopedInvariantRoutes invariant(true);
                    failures += check_case(c, 79U, {1, 9, 40, 300});
                }
                failures += check_invariant_profile(c, 80U);
            }
            {
                Case c;
                c.label          = "shared-expert q4/q6 E64 k6";
                c.experts        = 64;
                c.top_k          = 6;
                c.hidden         = 256;
                c.inter          = 96;
                c.shared_inter   = 128;
                c.gate_up        = QType::Q4_G64_FP16;
                c.down           = QType::Q6_G64_FP16;
                c.config.top_k   = 6;
                c.router_bias    = false;
                c.selection_bias = false;
                c.expert_scale   = true;
                c.expert_bias    = false;
                c.shared_gate    = true;
                c.epilogue       = ops::MoeEpilogue::Store;
                failures += check_case(c, 81U, {1, 2, 33});
                failures += check_invariance(c, 82U);
                failures += check_invariant_profile(c, 82U);
            }
            // 2a. Nemotron-H's layer class: relu^2 experts (Q4_0 / Q5_0, F of whole 32-value blocks
            //     but not whole 128-value steps), sigmoid routing with a selection bias, routed
            //     scale 2.5, and a relu^2 Q8 shared expert under A16 and under A8 (R8 at decode,
            //     W4A8 from 17 tokens).
            for (const auto& [format, policy] :
                 {std::pair{QType::GGML_Q4_0, ops::LinearPolicy::A16Only},
                  std::pair{QType::GGML_Q5_0, ops::LinearPolicy::AllowA8}}) {
                Case c;
                c.label = std::string("nemotron-like relu2 ") + qname(format) +
                          (policy == ops::LinearPolicy::AllowA8 ? " shared A8" : " shared A16");
                c.experts      = 32;
                c.top_k        = 6;
                c.hidden       = 256;
                c.inter        = 224;
                c.shared_inter = 384;
                c.gate_up = c.down    = format;
                c.config.top_k        = 6;
                c.config.scoring      = ops::MoeScoring::Sigmoid;
                c.config.routed_scale = 2.5F;
                c.config.activation   = ops::MoeActivation::Relu2;
                c.shared_policy       = policy;
                c.router_bias         = false;
                c.selection_bias      = true;
                c.expert_scale        = false;
                c.expert_bias         = false;
                c.shared_gate         = false;
                c.epilogue            = ops::MoeEpilogue::Store;
                failures += check_case(c, 95U, {1, 2, 4, 17, 40});
                failures += check_invariance(c, 96U);
            }
            // 2b. The int4 layer of a published Qwen3.6-35B-A3B quantization (decode, streaming and
            //     prefill widths; the prefill profile's chunk invariance; the invariant profile).
            for (const auto& [format, shared] :
                 {std::pair{QType::U4_G128_ZP_FP16, QType::BF16},
                  std::pair{QType::U4_G32_ZP_FP16, QType::U4_G32_ZP_FP16}}) {
                const Case c = qwen36_a3b_u4(format, shared);
                failures += check_case(c, 91U, {1, 4, 17, 64});
                failures += check_invariance(c, 92U);
                failures += check_invariant_profile(c, 92U);
            }
            // 2c. Experts of an R8 format under an A8 expert policy: the grouped R8 route at decode
            //     and decode widths (every schedule, both stages and each alone; the default
            //     route), in every codebook, K-quant and 32-value ggml format (activations, row
            //     orders, biases, epilogues varied) and in the Qwen3.6-35B-A3B UD-IQ3_XXS mixture
            //     (IQ2_S gate/up, IQ3_XXS down).
            {
                const QType codebooks[] = {
                    QType::GGML_IQ2_XXS, QType::GGML_IQ2_XS, QType::GGML_IQ2_S, QType::GGML_IQ3_XXS,
                    QType::GGML_IQ3_S, QType::GGML_Q3_K, QType::GGML_Q4_K, QType::GGML_Q5_K,
                    QType::GGML_Q6_K, QType::GGML_Q4_0, QType::GGML_Q4_1, QType::GGML_Q5_0,
                    QType::GGML_Q5_1, QType::GGML_IQ4_NL, QType::GGML_IQ4_XS, QType::MXFP4,
                    // The native formats (moe_r8_native.cu, moe_w4a8_native.cu).
                    QType::Q8_G32_FP16, QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q6_G64_FP16,
                    QType::U4_G128_ZP_FP16, QType::U4_G32_ZP_FP16, QType::U4_G64_ZP_FP16};
                int index = 0;
                for (const QType q : codebooks) {
                    for (int rep = 0; rep < 2; ++rep, ++index) {
                        Case c;
                        c.experts      = 24;
                        c.top_k        = 4;
                        c.hidden       = 256 * (1 + rep);
                        c.inter        = 256 * (2 - rep);
                        c.shared_inter = 0;
                        c.gate_up = c.down      = q;
                        c.config.top_k          = 4;
                        c.config.activation     = static_cast<ops::MoeActivation>(index % 4);
                        c.config.gate_up_layout = rep ? ops::MoeGateUpLayout::Interleaved
                                                      : ops::MoeGateUpLayout::Concatenated;
                        c.router_bias           = rep == 0;
                        c.selection_bias        = false;
                        c.expert_scale          = rep == 1;
                        c.expert_bias           = rep == 1;
                        c.shared_gate           = false;
                        c.epilogue = rep ? ops::MoeEpilogue::AddResidual : ops::MoeEpilogue::Store;
                        c.label = std::string(qname(q)) + " E24 k4 H" + std::to_string(c.hidden) +
                                  " F" + std::to_string(c.inter) + " act" +
                                  std::to_string(static_cast<int>(c.config.activation)) +
                                  (rep ? " interleaved" : "");
                        failures += check_r8(c, 97U + static_cast<std::uint32_t>(index), {1, 3});
                        failures +=
                            check_invariant_profile(c, 140U + static_cast<std::uint32_t>(index));
                        Case a8          = c;
                        a8.expert_policy = ops::LinearPolicy::AllowA8;
                        a8.label += " A8 policy";
                        failures +=
                            check_case(a8, 99U + static_cast<std::uint32_t>(index), {1, 4, 16, 64});
                    }
                }
                Case c;
                c.label          = "qwen3.6-35b-a3b-like ud-iq3_xxs (iq2s/iq3xxs) E256 k8";
                c.experts        = 256;
                c.top_k          = 8;
                c.hidden         = 2048;
                c.inter          = 512;
                c.shared_inter   = 512;
                c.gate_up        = QType::GGML_IQ2_S;
                c.down           = QType::GGML_IQ3_XXS;
                c.config.top_k   = 8;
                c.router_bias    = false;
                c.selection_bias = false;
                c.expert_scale   = false;
                c.expert_bias    = false;
                c.shared_gate    = true;
                c.epilogue       = ops::MoeEpilogue::AddResidual;
                failures += check_r8(c, 93U, {1, 4});
                failures += check_invariant_profile(c, 93U);
                // gpt-oss-20b-shaped: MXFP4 with H 2880, clamped SwiGLU, interleaved, biases.
                failures += check_r8(oss_like(), 94U, {1, 4});
                // NVFP4 experts (per-tensor divisor 16) take R8 without the native NVFP4 kernels.
                if (!ops::detail::kNativeNvfp4) failures += check_r8(qwen3_like(), 95U, {1, 4});
                failures += check_r8_table();
                failures += check_r8_k_limit();
                failures += check_expert_policy();
            }
            // 2d. The grouped W4A8 of a wide prefill call (MoePlan::w4a8), one layer per codec
            // family:
            //     MXFP4 with H 2880 (K of whole 32-value blocks only: the padded tail) and clamped
            //     SwiGLU, the Qwen3.6-35B-A3B GGUF mixtures (Q4_K / Q6_K and IQ2_S / IQ3_XXS),
            //     relu^2 Q5_0 with F 224 (the Up stage and a padded down K), IQ4_NL interleaved
            //     with biases.
            {
                failures += check_w4a8(oss_like(), 111U, {17, 70, 130});
                Case q;
                q.label          = "qwen3.6-35b-a3b-like q4_k/q6_k E256 k8";
                q.experts        = 256;
                q.top_k          = 8;
                q.hidden         = 2048;
                q.inter          = 512;
                q.shared_inter   = 512;
                q.gate_up        = QType::GGML_Q4_K;
                q.down           = QType::GGML_Q6_K;
                q.config.top_k   = 8;
                q.router_bias    = false;
                q.selection_bias = false;
                q.expert_scale   = false;
                q.expert_bias    = false;
                q.shared_gate    = true;
                q.epilogue       = ops::MoeEpilogue::AddResidual;
                failures += check_w4a8(q, 112U, {17, 40, 96});
                failures += check_speed_w4a8(q, 117U);
                Case iq    = q;
                iq.label   = "qwen3.6-35b-a3b-like iq2_s/iq3_xxs E256 k8";
                iq.gate_up = QType::GGML_IQ2_S;
                iq.down    = QType::GGML_IQ3_XXS;
                failures += check_w4a8(iq, 113U, {17, 96});
                // The 1-bit i-quants (UD-IQ1 mixes): 8g +- 1 int8 codes, a scale per 32 values
                // (IQ1_S) or per 16 (IQ1_M).
                Case i1    = q;
                i1.label   = "iq1_s/iq1_m E256 k8";
                i1.gate_up = QType::GGML_IQ1_S;
                i1.down    = QType::GGML_IQ1_M;
                failures += check_w4a8(i1, 119U, {17, 96});
                // The low-bit K-quants of the Q2_K mixes: Q2_K (a scale and a min per 16 values)
                // gate/up, Q3_K down.
                Case k2    = q;
                k2.label   = "q2_k/q3_k E256 k8";
                k2.gate_up = QType::GGML_Q2_K;
                k2.down    = QType::GGML_Q3_K;
                failures += check_w4a8(k2, 118U, {17, 96});
                Case n;
                n.label        = "nemotron-like relu2 q5_0 F224";
                n.experts      = 32;
                n.top_k        = 6;
                n.hidden       = 256;
                n.inter        = 224;
                n.shared_inter = 0;
                n.gate_up = n.down    = QType::GGML_Q5_0;
                n.config.top_k        = 6;
                n.config.scoring      = ops::MoeScoring::Sigmoid;
                n.config.routed_scale = 2.5F;
                n.config.activation   = ops::MoeActivation::Relu2;
                n.router_bias         = false;
                n.selection_bias      = true;
                n.expert_scale        = false;
                n.expert_bias         = false;
                n.shared_gate         = false;
                n.epilogue            = ops::MoeEpilogue::Store;
                failures += check_w4a8(n, 114U, {17, 40, 150});
                Case i4;
                i4.label        = "iq4_nl interleaved biased E16 k2";
                i4.experts      = 16;
                i4.top_k        = 2;
                i4.hidden       = 384;
                i4.inter        = 160;
                i4.shared_inter = 0;
                i4.gate_up = i4.down     = QType::GGML_IQ4_NL;
                i4.config.top_k          = 2;
                i4.config.gate_up_layout = ops::MoeGateUpLayout::Interleaved;
                i4.router_bias           = true;
                i4.selection_bias        = false;
                i4.expert_scale          = true;
                i4.expert_bias           = true;
                i4.shared_gate           = false;
                i4.epilogue              = ops::MoeEpilogue::Store;
                failures += check_w4a8(i4, 115U, {17, 90});
                // NVFP4 experts take the grouped W4A8 where the native NVFP4 kernels are absent.
                if (ops::detail::generic::w4a8_supports(QType::NVFP4)) {
                    failures += check_w4a8(qwen3_like(), 116U, {17, 96});
                }
            }
            // 3. Route-table plans (routes/<class>.json "moe"): a table that steers both layers to
            //    other schedules, narrower jobs included, must fit moe_workspace_bytes, match the
            //    oracle and keep the prefill profile's chunk invariance.
            failures += check_table_plans();
            failures += check_tuner();
            // 3b. The CPU expert path (MoE expert offload's decode): gpt-oss- and Qwen3-shaped
            //     layers, and a GeGLU MXFP4 layer with sigmoid scoring, selection bias and expert
            //     scales.
            {
                failures += check_host_experts(oss_like(), 91U);
                failures += check_host_experts(qwen3_like(), 92U);
                Case c;
                c.label        = "geglu mxfp4 E16 k2 sigmoid";
                c.experts      = 16;
                c.top_k        = 2;
                c.hidden       = 256;
                c.inter        = 96;
                c.shared_inter = 0;
                c.gate_up = c.down    = QType::MXFP4;
                c.config.top_k        = 2;
                c.config.scoring      = ops::MoeScoring::Sigmoid;
                c.config.routed_scale = 2.5F;
                c.config.activation   = ops::MoeActivation::GeGluTanh;
                c.router_bias         = true;
                c.selection_bias      = true;
                c.expert_scale        = true;
                c.expert_bias         = false;
                c.shared_gate         = false;
                c.epilogue            = ops::MoeEpilogue::Store;
                failures += check_host_experts(c, 93U);
            }
            // 4. Refusals.
            {
                Case c;
                c.label        = "refusal";
                c.experts      = 8;
                c.top_k        = 2;
                c.hidden       = 64;
                c.inter        = 32;
                c.shared_inter = 0;
                c.gate_up = c.down = QType::BF16;
                c.config.top_k     = 9;
                c.router_bias = c.selection_bias = c.expert_scale = c.expert_bias = c.shared_gate =
                    false;
                c.epilogue    = ops::MoeEpilogue::Store;
                const Layer l = make_layer(c, 5U);
                bool refused  = false;
                try {
                    (void)ops::moe_workspace_bytes(l.weights, c.config, 1, 1);
                } catch (const std::invalid_argument&) { refused = true; }
                if (!refused) {
                    std::cerr << "top_k above E was accepted\n";
                    ++failures;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "moe test: unexpected exception: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "moe: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "moe: all cases passed\n";
    return 0;
}
