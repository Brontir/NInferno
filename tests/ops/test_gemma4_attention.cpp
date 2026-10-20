#include "core/device.h"
#include "ninferno/ops/gemma4_attention.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/gemma4/launch.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

// Rows the call must not read hold a BF16 NaN: never-written ring slots, rows that fell out of
// the window, and full-layer slots past the frontier. A kernel that stages one of them into the
// value product turns its output into NaN, whatever the mask says.
constexpr std::uint16_t kPoison       = 0x7fc0U;
constexpr std::uint16_t kOutputCanary = 0x7fc1U;

// Criterion for the BF16 profile, derived rather than fitted. u = 2^-8 is the BF16 unit
// roundoff. The Op rounds twice: the probabilities entering the value product (the normalizer
// is formed from the unrounded ones) and the output.
//
//  * Probability rounding perturbs out[d] by sum_j dp_j v_j / sum_j p_j with |dp_j| <= u p_j,
//    hence by at most u * max|V|. Output rounding adds at most u * |out|. Pointwise:
//        |got - ideal| <= kSlack * u * (max|V| + |ideal|),
//    where kSlack = 17/16 covers FP32 accumulation and the approximate exponential.
//  * Normwise, each rounding is an independent relative perturbation with rms at most u/sqrt(3),
//    so their quadrature sum stays below 0.82 u; the bound is rel_l2 <= u.
// Non-finite output anywhere fails.
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

constexpr Geometry kSliding26{"26B sliding D256 16q/8kv W1024", 256, 16, 8, 1024};
constexpr Geometry kFull26{"26B full D512 16q/2kv", 512, 16, 2, 0};
constexpr Geometry kSliding31{"31B sliding D256 32q/16kv W1024", 256, 32, 16, 1024};
constexpr Geometry kFull31{"31B full D512 32q/4kv", 512, 32, 4, 0};
// Gemma 4 12B's full layer (one KV head, group 16) and an unshipped sliding geometry (group 4,
// a 512-key window): the head counts and the window are runtime.
constexpr Geometry kFull12{"12B full D512 16q/1kv", 512, 16, 1, 0};
constexpr Geometry kSlidingOdd{"D256 8q/2kv W512", 256, 8, 2, 512};
constexpr const Geometry* kGeometries[] = {&kSliding26, &kFull26, &kSliding31, &kFull31};
constexpr const Geometry* kOpenGeometries[] = {&kFull12, &kSlidingOdd};
// gpt-oss-20b's layers: a 64-wide head, 64 query / 8 KV heads, a 128-key window or full, with
// per-head attention sinks.
constexpr Geometry kOssSliding{"gpt-oss sliding D64 64q/8kv W128", 64, 64, 8, 128};
constexpr Geometry kOssFull{"gpt-oss full D64 64q/8kv", 64, 64, 8, 0};
constexpr const Geometry* kOssGeometries[] = {&kOssSliding, &kOssFull};

// GQA groups that are not powers of two (the runtime-group instance): Llama 3.2 3B and Phi-4-mini
// (D128 24q/8kv), SmolLM2 360M (D64 15q/5kv), Granite 4.x 3B (D64 40q/8kv), Qwen3 14B (D128
// 40q/8kv), a sliding group-5 layer (OLMo 3 32B-like, D128 40q/8kv W4096 cut to W512 here),
// Seed-OSS 36B (D128 80q/8kv, group 10), and the D256/D512 widths at groups 3, 6 and 12.
constexpr Geometry kGroup3D128{"group 3 D128 24q/8kv", 128, 24, 8, 0};
constexpr Geometry kGroup3D64{"group 3 D64 15q/5kv", 64, 15, 5, 0};
constexpr Geometry kGroup5D64{"group 5 D64 40q/8kv", 64, 40, 8, 0};
constexpr Geometry kGroup5D128{"group 5 D128 40q/8kv", 128, 40, 8, 0};
constexpr Geometry kGroup5Sliding{"group 5 sliding D128 40q/8kv W512", 128, 40, 8, 512};
constexpr Geometry kGroup10D128{"group 10 D128 80q/8kv", 128, 80, 8, 0};
constexpr Geometry kGroup3Sliding256{"group 3 sliding D256 12q/4kv W1024", 256, 12, 4, 1024};
constexpr Geometry kGroup6D512{"group 6 D512 12q/2kv", 512, 12, 2, 0};
constexpr Geometry kGroup12D128{"group 12 D128 12q/1kv", 128, 12, 1, 0};
constexpr Geometry kGroup7D64{"group 7 D64 14q/2kv", 64, 14, 2, 0};
// Group 1 at head widths 64/128 (SmolLM2 1.7B, OLMo 2/3): a row block of 64 tokens spans two
// 32-key tiles, full and sliding.
constexpr Geometry kGroup1D128{"group 1 D128 16q/16kv", 128, 16, 16, 0};
constexpr Geometry kGroup1D64{"group 1 D64 32q/32kv", 64, 32, 32, 0};
constexpr Geometry kGroup1Sliding{"group 1 sliding D128 16q/16kv W512", 128, 16, 16, 512};
constexpr const Geometry* kGroupGeometries[] = {&kGroup3D128,       &kGroup3D64,    &kGroup5D64,
                                                &kGroup5D128,       &kGroup5Sliding,
                                                &kGroup10D128,      &kGroup3Sliding256,
                                                &kGroup6D512,       &kGroup12D128};

// DFlash drafter layers: one D128 geometry per model, sliding (W2048) or full.
constexpr Geometry kDraftSliding26{"26B draft sliding D128 32q/8kv W2048", 128, 32, 8, 2048};
constexpr Geometry kDraftFull26{"26B draft full D128 32q/8kv", 128, 32, 8, 0};
constexpr Geometry kDraftSliding31{"31B draft sliding D128 64q/8kv W2048", 128, 64, 8, 2048};
constexpr Geometry kDraftFull31{"31B draft full D128 64q/8kv", 128, 64, 8, 0};
constexpr const Geometry* kDraftGeometries[] = {&kDraftSliding26, &kDraftFull26, &kDraftSliding31,
                                                &kDraftFull31};
const float kDraftScale                      = 1.0F / std::sqrt(128.0F);

using Mask                 = ops::Gemma4AttentionMask;
constexpr Mask kCausal     = Mask::Causal;
constexpr Mask kBlock      = Mask::Block;
constexpr int kDraftTokens = 16; // one DFlash block per round

// One call: T query columns at positions base..base+T-1.
struct CallSpec {
    const Geometry* geometry;
    int tokens;
    int base;
    int capacity_slack          = 0; // extra slots beyond the minimum the contract allows
    std::uint32_t visible_slack = 0; // envelope headroom beyond base + T
    int tokens_slack            = 0; // envelope headroom beyond T
    float scale                 = 1.0F;
    int hot_token               = -1; // plant window-boundary keys for this query column
    std::uint32_t seed          = 1;
    Mask mask                   = kCausal;
    bool hot_last               = false; // plant a dominant key at the call's last position
    bool sinks                  = false; // per-query-head sink logits (gpt-oss)
};

// Logical (represented) values of one call, independent of storage.
struct Logical {
    CallSpec spec;
    int D = 0, Hq = 0, Hkv = 0, G = 0, T = 0, W = 0, p0 = 0, capacity = 0;
    ops::Gemma4AttentionEnvelope envelope{};
    int pos_lo = 0;       // first position with generated K/V
    std::vector<float> q; // [T][Hq][D]
    std::vector<float> k; // [pos - pos_lo][Hkv][D]
    std::vector<float> v; // [pos - pos_lo][Hkv][D]
    std::vector<float> sinks; // [Hq] when the spec asks for sinks
    double v_max = 0.0;

    [[nodiscard]] int last() const { return p0 + T - 1; }

    [[nodiscard]] const float* key(int pos, int head) const {
        return k.data() + (static_cast<std::size_t>(pos - pos_lo) * Hkv + head) * D;
    }

    [[nodiscard]] float* key(int pos, int head) {
        return k.data() + (static_cast<std::size_t>(pos - pos_lo) * Hkv + head) * D;
    }

    [[nodiscard]] const float* value(int pos, int head) const {
        return v.data() + (static_cast<std::size_t>(pos - pos_lo) * Hkv + head) * D;
    }

    [[nodiscard]] const float* query(int token, int head) const {
        return q.data() + (static_cast<std::size_t>(token) * Hq + head) * D;
    }
};

float bf16_round(float value) { return bf16_to_f32(f32_to_bf16(value)); }

Logical make_logical(const CallSpec& spec) {
    const Geometry& g = *spec.geometry;
    Logical data;
    data.spec     = spec;
    data.D        = g.head_dim;
    data.Hq       = g.q_heads;
    data.Hkv      = g.kv_heads;
    data.G        = g.q_heads / g.kv_heads;
    data.T        = spec.tokens;
    data.W        = g.window;
    data.p0       = spec.base;
    const int top = spec.base + spec.tokens; // p_last + 1
    data.envelope = {static_cast<std::uint32_t>(top) + spec.visible_slack,
                     spec.tokens + spec.tokens_slack};
    if (g.window > 0) {
        data.capacity = g.window + data.envelope.max_tokens - 1 + spec.capacity_slack;
        // Every position the pre-append ring still holds.
        data.pos_lo = std::max(0, spec.base - data.capacity);
    } else {
        data.capacity = static_cast<int>(data.envelope.max_visible_keys) + spec.capacity_slack;
        data.pos_lo   = 0;
    }

    std::mt19937 rng(spec.seed * 7919U + static_cast<std::uint32_t>(g.head_dim * 31 + g.q_heads));
    // Scores scale*q.k have a standard deviation near 2 for any scale: realistic, neither flat
    // nor one-hot. V carries a per-(head, feature) bias so outputs are not averages near zero.
    const float q_amp = 6.0F / std::sqrt(static_cast<float>(data.D)) / spec.scale;
    const bool hot    = spec.hot_token >= 0 || spec.hot_last;
    const float q_off = hot ? q_amp / 3.0F : 0.0F;
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);

    data.q.resize(static_cast<std::size_t>(data.T) * data.Hq * data.D);
    for (float& value : data.q) { value = bf16_round(q_off + q_amp * unit(rng)); }

    const int positions = top - data.pos_lo;
    data.k.resize(static_cast<std::size_t>(positions) * data.Hkv * data.D);
    data.v.resize(data.k.size());
    std::vector<float> bias(static_cast<std::size_t>(data.Hkv) * data.D);
    for (float& value : bias) { value = 0.5F * unit(rng); }
    for (std::size_t i = 0; i < data.k.size(); ++i) {
        data.k[i] = bf16_round(unit(rng));
        data.v[i] = bf16_round(bias[i % bias.size()] + unit(rng));
    }
    if (spec.hot_last) {
        // The key at P_last dominates every column that sees it: all of them under Block, only
        // the last one under Causal. However long the context, the two masks then disagree by
        // far more than the criterion on every other column.
        const float h_in = 0.75F / spec.scale;
        for (int head = 0; head < data.Hkv; ++head) {
            for (int d = 0; d < data.D; ++d) { data.key(data.last(), head)[d] = bf16_round(h_in); }
        }
    }
    if (spec.hot_token >= 0) {
        // With every query leaning along +1, a key whose features are all +h scores about
        // h * sum(q) above the crowd. For the chosen column p*: the key at p* - W + 1 is the
        // oldest it may see and the hotter key at p* - W is the first it must not see. Any
        // off-by-one in the window flips which of them dominates column p*.
        const int p_star = data.p0 + spec.hot_token;
        const float h_in = 0.75F / spec.scale;
        const int oldest = p_star - data.W + 1;
        for (int head = 0; head < data.Hkv; ++head) {
            for (int d = 0; d < data.D; ++d) {
                data.key(oldest, head)[d]     = bf16_round(h_in);
                data.key(oldest - 1, head)[d] = bf16_round(1.1667F * h_in);
            }
        }
    }
    for (const float value : data.v) { data.v_max = std::max(data.v_max, std::abs(double(value))); }
    if (spec.sinks) {
        // Sinks in score units around the typical score: some heads' sink dominates, some vanish.
        data.sinks.resize(static_cast<std::size_t>(data.Hq));
        for (float& value : data.sinks) { value = 4.0F * unit(rng); }
    }
    return data;
}

// Mutations of the formula used only by the control arms.
struct Mutation {
    int window_delta = 0;      // visible distance W + delta instead of W
    bool gqa_modulo  = false;  // KV head h % Hkv instead of h / G
    bool no_causal   = false;  // every column sees every key up to the last position of the call
                               // (on a full layer this is the Block formula)
    bool force_causal = false; // the Causal formula even where the call asked for Block
};

// The sole oracle: the documented formula evaluated naively in FP64 from the represented values,
// under the call's mask. It knows nothing about slots, rings, tiles, splits or the kernel's
// roundings.
std::vector<double> oracle(const Logical& data, const std::vector<int>& columns,
                           const Mutation& mutation = {}) {
    std::vector<double> out(columns.size() * data.Hq * data.D);
    const int items = static_cast<int>(columns.size()) * data.Hq;
    std::atomic<int> next{0};
    const auto worker = [&]() {
        std::vector<double> weights;
        std::vector<double> sum(static_cast<std::size_t>(data.D));
        for (int item = next++; item < items; item = next++) {
            const int column = item / data.Hq;
            const int head   = item % data.Hq;
            const int token  = columns[static_cast<std::size_t>(column)];
            const int p      = data.p0 + token;
            const int kvh    = mutation.gqa_modulo ? head % data.Hkv : head / data.G;
            const int window = data.W > 0 ? data.W + mutation.window_delta : 0;
            const int j_lo   = window > 0 ? std::max(0, p - window + 1) : 0;
            const bool block =
                (data.spec.mask == kBlock || mutation.no_causal) && !mutation.force_causal;
            const int j_hi = block ? data.last() : p;
            const float* q = data.query(token, head);
            weights.assign(static_cast<std::size_t>(j_hi - j_lo + 1), 0.0);
            double maximum = -std::numeric_limits<double>::infinity();
            for (int j = j_lo; j <= j_hi; ++j) {
                const float* key = data.key(j, kvh);
                double dot       = 0.0;
                for (int d = 0; d < data.D; ++d) { dot += double(q[d]) * double(key[d]); }
                const double score                          = double(data.spec.scale) * dot;
                weights[static_cast<std::size_t>(j - j_lo)] = score;
                maximum                                     = std::max(maximum, score);
            }
            const bool sink = !data.sinks.empty();
            if (sink) { maximum = std::max(maximum, double(data.sinks[static_cast<std::size_t>(head)])); }
            double denominator = sink ? std::exp(double(data.sinks[static_cast<std::size_t>(head)]) - maximum) : 0.0;
            for (double& w : weights) {
                w = std::exp(w - maximum);
                denominator += w;
            }
            std::fill(sum.begin(), sum.end(), 0.0);
            for (int j = j_lo; j <= j_hi; ++j) {
                const double w     = weights[static_cast<std::size_t>(j - j_lo)];
                const float* value = data.value(j, kvh);
                for (int d = 0; d < data.D; ++d) {
                    sum[static_cast<std::size_t>(d)] += w * value[d];
                }
            }
            double* dst = out.data() + static_cast<std::size_t>(item) * data.D;
            for (int d = 0; d < data.D; ++d) {
                dst[d] = sum[static_cast<std::size_t>(d)] / denominator;
            }
        }
    };
    const unsigned threads = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (unsigned i = 0; i < threads; ++i) { pool.emplace_back(worker); }
    for (std::thread& thread : pool) { thread.join(); }
    return out;
}

struct Distance {
    double max_ratio        = 0.0; // worst |got - ideal| / pointwise limit
    double rel_l2           = 0.0;
    std::int64_t bad        = 0; // pointwise violations
    std::int64_t non_finite = 0;

    [[nodiscard]] bool passes() const { return non_finite == 0 && bad == 0 && rel_l2 <= kRelL2Max; }

    [[nodiscard]] double worst_ratio() const { return std::max(max_ratio, rel_l2 / kRelL2Max); }
};

Distance distance(const std::vector<double>& got, const std::vector<double>& ideal, double v_max) {
    Distance result;
    long double error_sq = 0.0L;
    long double ideal_sq = 0.0L;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ideal[i])) {
            ++result.non_finite;
            continue;
        }
        const double error = std::abs(got[i] - ideal[i]);
        const double limit = kSlack * kUnit * (v_max + std::abs(ideal[i]));
        result.max_ratio   = std::max(result.max_ratio, error / limit);
        result.bad += error > limit ? 1 : 0;
        error_sq += static_cast<long double>(error) * error;
        ideal_sq += static_cast<long double>(ideal[i]) * ideal[i];
    }
    result.rel_l2 = std::sqrt(static_cast<double>(error_sq)) /
                    std::max(std::sqrt(static_cast<double>(ideal_sq)), 1.0e-300);
    return result;
}

std::vector<int> checked_columns(int tokens) {
    // Every column up to T=300. Beyond that every third column (a row block never spans fewer
    // than four consecutive columns, so every block is sampled) plus the tail.
    std::vector<int> columns;
    for (int t = 0; t < tokens; ++t) {
        if (tokens <= 300 || t % 3 == 0 || t >= tokens - 8) { columns.push_back(t); }
    }
    return columns;
}

std::vector<std::uint16_t> to_bits(const float* values, std::size_t count) {
    std::vector<std::uint16_t> bits(count);
    for (std::size_t i = 0; i < count; ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

// Host images of the storage: the ring or slot planes before the call's own rows are appended
// (poisoned where the call must not read), the rows the call appends, and the planes after.
struct Storage {
    std::vector<std::uint16_t> k_before, v_before; // [Hkv][C][D]
    std::vector<std::uint16_t> k_after, v_after;
    std::vector<std::uint16_t> k_new, v_new; // [T][Hkv][D]
    std::vector<std::uint16_t> q;            // [T][Hq][D]
    std::vector<int> positions;
    std::vector<float> sinks;
};

Storage make_storage(const Logical& data) {
    Storage s;
    const std::size_t plane = static_cast<std::size_t>(data.Hkv) * data.capacity * data.D;
    s.k_before.assign(plane, kPoison);
    s.v_before.assign(plane, kPoison);
    const int oldest_visible = data.W > 0 ? std::max(0, data.p0 - data.W + 1) : 0;
    const auto slot_offset   = [&](int pos, int head) {
        return (static_cast<std::size_t>(head) * data.capacity + pos % data.capacity) * data.D;
    };
    // History before the call: positions [pos_lo, p0). Rows older than the oldest key the call
    // may see stay poisoned; they are dead for this call.
    for (int pos = std::max(data.pos_lo, oldest_visible); pos < data.p0; ++pos) {
        for (int head = 0; head < data.Hkv; ++head) {
            const auto kb = to_bits(data.key(pos, head), data.D);
            const auto vb = to_bits(data.value(pos, head), data.D);
            std::copy(kb.begin(), kb.end(), s.k_before.begin() + slot_offset(pos, head));
            std::copy(vb.begin(), vb.end(), s.v_before.begin() + slot_offset(pos, head));
        }
    }
    s.k_after = s.k_before;
    s.v_after = s.v_before;
    for (int t = 0; t < data.T; ++t) {
        const int pos = data.p0 + t;
        s.positions.push_back(pos);
        for (int head = 0; head < data.Hkv; ++head) {
            const auto kb = to_bits(data.key(pos, head), data.D);
            const auto vb = to_bits(data.value(pos, head), data.D);
            s.k_new.insert(s.k_new.end(), kb.begin(), kb.end());
            s.v_new.insert(s.v_new.end(), vb.begin(), vb.end());
            std::copy(kb.begin(), kb.end(), s.k_after.begin() + slot_offset(pos, head));
            std::copy(vb.begin(), vb.end(), s.v_after.begin() + slot_offset(pos, head));
        }
    }
    s.q     = to_bits(data.q.data(), data.q.size());
    s.sinks = data.sinks;
    return s;
}

// Device buffers of one call site; the envelope, capacity and shapes stay fixed across replays.
struct Device {
    explicit Device(const Logical& shape)
        : T(shape.T), D(shape.D), Hq(shape.Hq), Hkv(shape.Hkv), capacity(shape.capacity),
          window(shape.W), cache_k(static_cast<std::size_t>(Hkv) * capacity * D * 2),
          cache_v(static_cast<std::size_t>(Hkv) * capacity * D * 2),
          k_new(static_cast<std::size_t>(T) * Hkv * D * 2),
          v_new(static_cast<std::size_t>(T) * Hkv * D * 2),
          q(static_cast<std::size_t>(T) * Hq * D * 2), positions(static_cast<std::size_t>(T) * 4),
          out(static_cast<std::size_t>(T) * Hq * D * 2), sinks(static_cast<std::size_t>(Hq) * 4),
          has_sinks(!shape.sinks.empty()),
          workspace(std::max<std::size_t>(
              1, ops::gemma4_attention_workspace_bytes(D, Hq, Hkv, shape.envelope))),
          envelope(shape.envelope), mask(shape.spec.mask) {}

    void upload(const Storage& s) {
        cache_k.copy_from_host(s.k_before.data(), cache_k.bytes());
        cache_v.copy_from_host(s.v_before.data(), cache_v.bytes());
        k_new.copy_from_host(s.k_new.data(), k_new.bytes());
        v_new.copy_from_host(s.v_new.data(), v_new.bytes());
        q.copy_from_host(s.q.data(), q.bytes());
        positions.copy_from_host(s.positions.data(), positions.bytes());
        if (!s.sinks.empty()) { sinks.copy_from_host(s.sinks.data(), sinks.bytes()); }
        const std::vector<std::uint16_t> canary(out.bytes() / 2, kOutputCanary);
        out.copy_from_host(canary.data(), out.bytes());
    }

    // Append the call's rows, then attend. With append == false only the attention runs, over
    // whatever the planes already hold.
    void call(cudaStream_t stream, float scale, bool append = true) {
        Tensor k_tensor(k_new.data(), DType::BF16, {D, Hkv, T});
        Tensor v_tensor(v_new.data(), DType::BF16, {D, Hkv, T});
        Tensor q_tensor(q.data(), DType::BF16, {D, Hq, T});
        Tensor p_tensor(positions.data(), DType::I32, {T});
        Tensor o_tensor(out.data(), DType::BF16, {D, Hq, T});
        const ops::Gemma4KvLayer layer{
            .k      = Tensor(cache_k.data(), DType::BF16, {D, capacity, Hkv}),
            .v      = Tensor(cache_v.data(), DType::BF16, {D, capacity, Hkv}),
            .window = window,
        };
        if (append) { ops::gemma4_kv_append(k_tensor, v_tensor, p_tensor, layer, stream); }
        if (has_sinks) {
            Tensor s_tensor(sinks.data(), DType::FP32, {Hq});
            ops::gemma4_attention(q_tensor, p_tensor, layer, scale, envelope, workspace, o_tensor,
                                  stream, mask, &s_tensor);
        } else if (mask == kCausal) {
            // The pre-existing signature: the mask defaults to Causal.
            ops::gemma4_attention(q_tensor, p_tensor, layer, scale, envelope, workspace, o_tensor,
                                  stream);
        } else {
            ops::gemma4_attention(q_tensor, p_tensor, layer, scale, envelope, workspace, o_tensor,
                                  stream, mask);
        }
    }

    int T, D, Hq, Hkv, capacity, window;
    GuardedDeviceBuffer cache_k, cache_v, k_new, v_new, q, positions, out, sinks;
    bool has_sinks = false;
    WorkspaceArena workspace;
    ops::Gemma4AttentionEnvelope envelope;
    Mask mask;
};

std::string describe(const Logical& data) {
    return std::string(data.spec.geometry->label) + (data.spec.mask == kBlock ? " Block" : "") +
           " T=" + std::to_string(data.T) + " base=" + std::to_string(data.p0) +
           " C=" + std::to_string(data.capacity) +
           (data.spec.scale != 1.0F ? " scale=" + std::to_string(data.spec.scale) : "") +
           (data.spec.hot_token >= 0 ? " boundary@" + std::to_string(data.spec.hot_token) : "") +
           (data.spec.hot_last ? " hot@P_last" : "") + (data.spec.sinks ? " sinks" : "");
}

double g_worst_ratio = 0.0;

// Output of the checked columns only, promoted to FP64.
std::vector<double> checked_output(const std::vector<double>& out_all, const Logical& data,
                                   const std::vector<int>& columns) {
    std::vector<double> got;
    const std::size_t per_column = static_cast<std::size_t>(data.Hq) * data.D;
    for (const int column : columns) {
        got.insert(got.end(), out_all.begin() + column * per_column,
                   out_all.begin() + (column + 1) * per_column);
    }
    return got;
}

// Checks one completed call: exact cache image, guards, full overwrite, and the oracle.
int verify(const std::string& label, const Logical& data, const Storage& s, Device& dev) {
    int failures     = 0;
    const auto k_now = from_device<std::uint16_t>(dev.cache_k.data(), s.k_after.size());
    const auto v_now = from_device<std::uint16_t>(dev.cache_v.data(), s.v_after.size());
    if (k_now != s.k_after || v_now != s.v_after) {
        std::cerr << label << ": cache planes differ from the exact append image\n";
        ++failures;
    }
    for (GuardedDeviceBuffer* buffer :
         {&dev.cache_k, &dev.cache_v, &dev.out, &dev.q, &dev.k_new, &dev.v_new, &dev.positions}) {
        failures += buffer->verify_guards(label);
    }
    const auto out_all      = from_device_bf16(dev.out.data(), dev.out.bytes() / 2);
    std::int64_t non_finite = 0;
    for (const double value : out_all) { non_finite += std::isfinite(value) ? 0 : 1; }
    if (non_finite != 0) {
        std::cerr << label << ": " << non_finite << " output values are non-finite or unwritten\n";
        ++failures;
    }
    const auto columns = checked_columns(data.T);
    const Distance check =
        distance(checked_output(out_all, data, columns), oracle(data, columns), data.v_max);
    g_worst_ratio = std::max(g_worst_ratio, check.worst_ratio());
    std::printf("  %-66s worst=%.3f of limit (pointwise %.3f, rel_l2 %.2e)%s\n", label.c_str(),
                check.worst_ratio(), check.max_ratio, check.rel_l2, check.passes() ? "" : "  FAIL");
    if (!check.passes()) {
        std::cerr << label << ": " << check.bad << " pointwise violations, rel_l2=" << check.rel_l2
                  << ", non_finite=" << check.non_finite << '\n';
        ++failures;
    }
    return failures;
}

// True when the slot of position P_last + 1 holds NaN in both planes of every KV head after the
// call's own append: a stale row (a rejected draft's, in the drafter) right behind the block that
// the call must never read.
bool stale_after_last_is_poison(const Logical& data, const Storage& s) {
    const int slot = (data.last() + 1) % data.capacity;
    for (int head = 0; head < data.Hkv; ++head) {
        const std::size_t row = (static_cast<std::size_t>(head) * data.capacity + slot) * data.D;
        for (int d = 0; d < data.D; ++d) {
            if (s.k_after[row + d] != kPoison || s.v_after[row + d] != kPoison) { return false; }
        }
    }
    return true;
}

// stale_row: the spec must leave a NaN row at P_last + 1 (asserted, not assumed).
int run_call(const CallSpec& spec, bool stale_row = false) {
    const Logical data    = make_logical(spec);
    const Storage storage = make_storage(data);
    std::string label     = describe(data);
    if (stale_row) {
        if (!stale_after_last_is_poison(data, storage)) {
            std::cerr << label << ": the setup leaves no NaN row at P_last + 1\n";
            return 1;
        }
        label += " NaN@P+1";
    }
    Device dev(data);
    dev.upload(storage);
    dev.call(nullptr, spec.scale);
    cuda_synchronize();
    return verify(label, data, storage, dev);
}

// Sensitivity control for the stale-row checks: over the post-append image of `spec`, attend with
// every position one further (P+1 .. P_last+1) and no append. The key at P_last + 1 is then
// visible and holds NaN, so the output must turn non-finite: a kernel that read one row past the
// block would be caught by the NaN@P+1 calls.
int run_overread_control(const CallSpec& spec) {
    const Logical data = make_logical(spec);
    const Storage base = make_storage(data);
    if (!stale_after_last_is_poison(data, base) ||
        data.envelope.max_visible_keys < static_cast<std::uint32_t>(data.last() + 2)) {
        std::cerr << "control stale row: the setup cannot shift the call by one position\n";
        return 1;
    }
    Storage shifted  = base;
    shifted.k_before = base.k_after;
    shifted.v_before = base.v_after;
    for (int& position : shifted.positions) { ++position; }
    Device dev(data);
    dev.upload(shifted);
    dev.call(nullptr, spec.scale, /*append=*/false);
    cuda_synchronize();
    const auto out          = from_device_bf16(dev.out.data(), dev.out.bytes() / 2);
    std::int64_t non_finite = 0;
    for (const double value : out) { non_finite += std::isfinite(value) ? 0 : 1; }
    std::printf("  control %-15s on %-58s non-finite outputs: %lld of %zu\n", "stale row read",
                describe(data).c_str(), static_cast<long long>(non_finite), out.size());
    if (non_finite == 0) {
        std::cerr << "control stale row read: a visible NaN row left the output finite; the "
                     "suite cannot see a read past the block\n";
        return 1;
    }
    return 0;
}

// A control arm must fail: the kernel's output, judged against the mutated formula with the same
// criterion, has to violate it. Otherwise the suite could not tell the mistake from the truth.
int run_control(const char* name, const CallSpec& spec, const Mutation& mutation) {
    const Logical data    = make_logical(spec);
    const Storage storage = make_storage(data);
    Device dev(data);
    dev.upload(storage);
    dev.call(nullptr, spec.scale);
    cuda_synchronize();
    const auto columns = checked_columns(data.T);
    const auto got =
        checked_output(from_device_bf16(dev.out.data(), dev.out.bytes() / 2), data, columns);
    const Distance truth   = distance(got, oracle(data, columns), data.v_max);
    const Distance mutated = distance(got, oracle(data, columns, mutation), data.v_max);
    std::printf("  control %-15s on %-58s mutated: worst=%7.2f of limit (pointwise %.2f, "
                "rel_l2 %.2e) | truth %.3f\n",
                name, describe(data).c_str(), mutated.worst_ratio(), mutated.max_ratio,
                mutated.rel_l2, truth.worst_ratio());
    int failures = 0;
    if (!truth.passes()) {
        std::cerr << "control " << name << ": the kernel fails its own oracle on this data\n";
        ++failures;
    }
    if (mutated.passes()) {
        std::cerr
            << "control " << name
            << ": the mutated formula passes the criterion; the suite cannot see this error\n";
        ++failures;
    }
    return failures;
}

// A prefill chunk's size is set by where a turn was cut, so a column must not depend on which call
// it arrived in: the same T columns attended in one call and again as consecutive calls cut at
// `cuts` (append, then attend, per part, over the same planes) must be byte-identical. Cuts off
// the row-block grid (16 tokens at G=2, 4 at G=8) are the ones that moved a block's key tiles.
int run_split_invariance(const CallSpec& spec, const std::vector<int>& cuts) {
    const Logical data    = make_logical(spec);
    const Storage storage = make_storage(data);
    Device dev(data);
    dev.upload(storage);
    dev.call(nullptr, spec.scale);
    cuda_synchronize();
    const auto whole = from_device<std::uint16_t>(dev.out.data(), dev.out.bytes() / 2);

    dev.upload(storage);
    cuda_synchronize();
    const ops::Gemma4KvLayer layer{
        .k      = Tensor(dev.cache_k.data(), DType::BF16, {data.D, data.capacity, data.Hkv}),
        .v      = Tensor(dev.cache_v.data(), DType::BF16, {data.D, data.capacity, data.Hkv}),
        .window = data.W,
    };
    std::vector<int> bounds = cuts;
    bounds.insert(bounds.begin(), 0);
    bounds.push_back(data.T);
    for (std::size_t part = 0; part + 1 < bounds.size(); ++part) {
        const int t0 = bounds[part], count = bounds[part + 1] - bounds[part];
        const auto at = [](void* base, std::size_t elements, std::size_t bytes_each) {
            return static_cast<char*>(base) + elements * bytes_each;
        };
        Tensor k(at(dev.k_new.data(), std::size_t(t0) * data.Hkv * data.D, 2), DType::BF16,
                 {data.D, data.Hkv, count});
        Tensor v(at(dev.v_new.data(), std::size_t(t0) * data.Hkv * data.D, 2), DType::BF16,
                 {data.D, data.Hkv, count});
        Tensor q(at(dev.q.data(), std::size_t(t0) * data.Hq * data.D, 2), DType::BF16,
                 {data.D, data.Hq, count});
        Tensor p(at(dev.positions.data(), std::size_t(t0), 4), DType::I32, {count});
        Tensor o(at(dev.out.data(), std::size_t(t0) * data.Hq * data.D, 2), DType::BF16,
                 {data.D, data.Hq, count});
        ops::gemma4_kv_append(k, v, p, layer, nullptr);
        Tensor sinks(dev.sinks.data(), DType::FP32, {data.Hq});
        ops::gemma4_attention(q, p, layer, spec.scale, dev.envelope, dev.workspace, o, nullptr,
                              Mask::Causal, dev.has_sinks ? &sinks : nullptr);
    }
    cuda_synchronize();
    const auto parts    = from_device<std::uint16_t>(dev.out.data(), dev.out.bytes() / 2);
    std::int64_t differ = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) { differ += parts[i] != whole[i] ? 1 : 0; }
    std::string cut_list;
    for (const int cut : cuts) { cut_list += (cut_list.empty() ? "" : ",") + std::to_string(cut); }
    std::printf("  %-50s cut at %-12s %lld of %zu values differ (must be 0)\n",
                describe(data).c_str(), cut_list.c_str(), static_cast<long long>(differ),
                parts.size());
    return differ != 0 ? 1 : 0;
}

// The prefill tile of a 512-wide group-8 layer (32 rows on 128-column warp slices, or 64 rows on
// 256-column slices, per device and T) must not change a byte: the same call under each.
int run_slice_width_invariance(const CallSpec& spec) {
    const Logical data    = make_logical(spec);
    const Storage storage = make_storage(data);
    Device dev(data);
    const auto output = [&](ops::detail::Gemma4PrefillSlices slices) {
        ops::detail::gemma4_attention_set_prefill_slices(slices);
        dev.upload(storage);
        dev.call(nullptr, spec.scale);
        cuda_synchronize();
        ops::detail::gemma4_attention_set_prefill_slices(ops::detail::Gemma4PrefillSlices::Auto);
        return from_device<std::uint16_t>(dev.out.data(), dev.out.bytes() / 2);
    };
    const auto narrow   = output(ops::detail::Gemma4PrefillSlices::Narrow);
    const auto wide     = output(ops::detail::Gemma4PrefillSlices::Wide);
    std::int64_t differ = 0;
    for (std::size_t i = 0; i < wide.size(); ++i) { differ += wide[i] != narrow[i] ? 1 : 0; }
    std::printf("  %-50s narrow vs wide slices: %lld of %zu values differ (must be 0)\n",
                describe(data).c_str(), static_cast<long long>(differ), wide.size());
    return differ != 0 ? 1 : 0;
}

// A chained prefill call (spans of key tiles handed between CTAs) must give the plain launch's
// bytes: the same call with the chain off and on.
int run_chain_invariance(const CallSpec& spec) {
    using ops::detail::Gemma4PrefillChain;
    const Logical data    = make_logical(spec);
    const Storage storage = make_storage(data);
    Device dev(data);
    const auto output = [&](Gemma4PrefillChain chain) {
        ops::detail::gemma4_attention_set_prefill_chain(chain);
        dev.upload(storage);
        dev.call(nullptr, spec.scale);
        cuda_synchronize();
        ops::detail::gemma4_attention_set_prefill_chain(Gemma4PrefillChain::Auto);
        return from_device<std::uint16_t>(dev.out.data(), dev.out.bytes() / 2);
    };
    const auto plain    = output(Gemma4PrefillChain::Off);
    const auto chained  = output(Gemma4PrefillChain::On);
    std::int64_t differ = 0;
    for (std::size_t i = 0; i < plain.size(); ++i) { differ += plain[i] != chained[i] ? 1 : 0; }
    std::printf("  %-50s plain vs chained: %lld of %zu values differ (must be 0)\n",
                describe(data).c_str(), static_cast<long long>(differ), plain.size());
    return differ != 0 ? 1 : 0;
}

// The invariant route profile (--activations exact), decode: a call's bytes do not depend on the
// envelope, which a batched call takes from its longest row. The same call under envelopes with
// more visible keys (and so larger capacities) must be byte-identical and agree with the oracle.
// Without the profile the split count follows the envelope: the control count is printed only.
int run_envelope_invariance(const CallSpec& spec, const std::vector<std::uint32_t>& slacks) {
    int failures = 0;
    const auto output = [&](std::uint32_t slack, bool invariant, bool check) {
        CallSpec s            = spec;
        s.visible_slack       = slack;
        const Logical data    = make_logical(s);
        const Storage storage = make_storage(data);
        const ops::ScopedInvariantRoutes routes(invariant);
        Device dev(data);
        dev.upload(storage);
        dev.call(nullptr, s.scale);
        cuda_synchronize();
        if (check) { failures += verify(describe(data) + " invariant", data, storage, dev); }
        return from_device<std::uint16_t>(dev.out.data(), dev.out.bytes() / 2);
    };
    const auto reference = output(slacks.front(), true, true);
    std::int64_t differ = 0, control = 0;
    const auto speed = output(slacks.front(), false, false);
    for (std::size_t i = 1; i < slacks.size(); ++i) {
        const auto got = output(slacks[i], true, true);
        const auto fast = output(slacks[i], false, false);
        for (std::size_t j = 0; j < got.size(); ++j) {
            differ += got[j] != reference[j] ? 1 : 0;
            control += fast[j] != speed[j] ? 1 : 0;
        }
    }
    // Neither profile's bytes may follow the envelope's key capacity (a session's max_context, or
    // a batch's longest row): the decode split count comes from geometry, window and T.
    std::printf("  %-50s T=%d base %d envelopes +%u..+%u: %lld values differ (must be 0; speed "
                "profile: %lld, must be 0)\n",
                spec.geometry->label, spec.tokens, spec.base, slacks.front(), slacks.back(),
                static_cast<long long>(differ), static_cast<long long>(control));
    return failures + (differ != 0 ? 1 : 0) + (control != 0 ? 1 : 0);
}

// Capture append + attention once, then replay with new positions, histories and inputs.
// `proto` fixes geometry, T, capacity slack, scale and mask; its base is ignored.
int run_graph(const CallSpec& proto, const std::vector<int>& bases, std::uint32_t visible_max) {
    int failures = 0;
    // The first spec fixes every shape and the envelope of the captured call site.
    CallSpec shape_spec  = proto;
    shape_spec.base      = bases.front();
    shape_spec.hot_token = -1;
    shape_spec.visible_slack =
        visible_max - static_cast<std::uint32_t>(bases.front() + proto.tokens);
    const Logical shape = make_logical(shape_spec);
    Device dev(shape);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    dev.upload(make_storage(shape));
    cudaGraph_t graph      = nullptr;
    cudaGraphExec_t replay = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    dev.call(stream, proto.scale);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&replay, graph, nullptr, nullptr, 0));

    std::uint32_t seed = 101;
    for (const int base : bases) {
        CallSpec spec      = shape_spec;
        spec.base          = base;
        spec.seed          = seed++;
        spec.visible_slack = visible_max - static_cast<std::uint32_t>(base + proto.tokens);
        const Logical data = make_logical(spec);
        if (data.capacity != dev.capacity) { throw std::logic_error("graph replay reshaped"); }
        const Storage storage = make_storage(data);
        dev.upload(storage);
        // Pageable cudaMemcpy may return before its DMA lands, and the replay stream does not
        // order after the legacy stream: without this the output canary can arrive after the
        // kernel has written the output.
        cuda_synchronize();
        CUDA_CHECK(cudaGraphLaunch(replay, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        failures += verify("graph replay " + describe(data), data, storage, dev);
    }
    CUDA_CHECK(cudaGraphExecDestroy(replay));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return failures;
}

int expect_invalid(const char* label, const std::function<void()>& body) {
    try {
        body();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": threw the wrong exception: " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": accepted an unsupported configuration\n";
    return 1;
}

int run_rejections() {
    int failures = 0;
    // Dummy non-null storage; validation must reject before any launch could touch it.
    DeviceBuffer scratch(1 << 20);
    WorkspaceArena workspace(1 << 20);
    const auto attempt = [&](int D, int Hq, int Hkv, int window, Mask mask = kCausal) {
        Tensor q(scratch.p, DType::BF16, {D, Hq, 1});
        Tensor positions(scratch.p, DType::I32, {1});
        Tensor out(scratch.p, DType::BF16, {D, Hq, 1});
        const ops::Gemma4KvLayer layer{Tensor(scratch.p, DType::BF16, {D, 4, Hkv}),
                                       Tensor(scratch.p, DType::BF16, {D, 4, Hkv}), window};
        ops::gemma4_attention(q, positions, layer, 1.0F, {4, 1}, workspace, out, nullptr, mask);
    };
    const auto append = [&](int D, int Hkv, int window) {
        Tensor k(scratch.p, DType::BF16, {D, Hkv, 1});
        Tensor positions(scratch.p, DType::I32, {1});
        const ops::Gemma4KvLayer layer{Tensor(scratch.p, DType::BF16, {D, 4, Hkv}),
                                       Tensor(scratch.p, DType::BF16, {D, 4, Hkv}), window};
        ops::gemma4_kv_append(k, k, positions, layer, nullptr);
    };
    failures += expect_invalid("D192", [&] { attempt(192, 16, 8, 1024); });
    failures += expect_invalid("D512 12q/8kv", [&] { attempt(512, 12, 8, 0); });
    failures += expect_invalid("D256 group 17", [&] { attempt(256, 17, 1, 1024); });
    failures += expect_invalid("D128 group 32", [&] { attempt(128, 64, 2, 0); });
    failures += expect_invalid("a negative window", [&] { attempt(256, 16, 8, -1); });
    failures +=
        expect_invalid("Block on the D512 full layer", [&] { attempt(512, 16, 2, 0, kBlock); });
    failures +=
        expect_invalid("Block on the 31B D512 full layer", [&] { attempt(512, 32, 4, 0, kBlock); });
    failures += expect_invalid("an unknown mask value",
                               [&] { attempt(128, 32, 8, 0, static_cast<Mask>(7)); });
    failures += expect_invalid("append D192", [&] { append(192, 4, 0); });
    failures += expect_invalid("append a negative window", [&] { append(128, 8, -1); });
    failures += expect_invalid("workspace D256 group 17", [&] {
        (void)ops::gemma4_attention_workspace_bytes(256, 34, 2, {16, 1});
    });
    failures += expect_invalid("workspace D384", [&] {
        (void)ops::gemma4_attention_workspace_bytes(384, 32, 4, {16, 1});
    });
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("  rejections: head widths other than 64/128/256/512, GQA groups above 16, "
                "negative windows and Block outside the D128/D256 layers throw "
                "invalid_argument\n");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        std::printf(
            "oracle agreement (limit = 1.0; pointwise %.4g*(max|V|+|ideal|), rel_l2 %.4g)\n",
            kSlack * kUnit, kRelL2Max);
        // Every geometry over the decode/prefill route boundaries (16|17) and interior widths,
        // from an empty history and as a continuation. Sliding continuations start past the ring
        // capacity, so the ring has wrapped; the slack varies so the ring is not always minimal.
        std::uint32_t seed = 1;
        for (const Geometry* geometry : kGeometries) {
            const bool sliding = geometry->window > 0;
            for (const int tokens : {1, 3, 16, 17, 64, 300}) {
                failures += run_call({geometry, tokens, 0, 0, 0, 0, 1.0F, -1, seed++});
                const int base = sliding ? (tokens >= 300 ? 2500 : 5000) : 3001;
                failures += run_call({geometry, tokens, base, (tokens % 2) ? 0 : 64,
                                      static_cast<std::uint32_t>(tokens % 3) * 97, tokens % 2, 1.0F,
                                      -1, seed++});
            }
            failures += run_call({geometry, 2048, 0, 0, 0, 0, 1.0F, -1, seed++});
        }
        // Prefill continuations at scale != 1 (the scale is an argument, not a constant).
        failures += run_call({&kFull26, 2048, 1000, 0, 0, 0, 0.125F, -1, seed++});
        failures += run_call({&kSliding31, 33, 1500, 5, 0, 0, 0.5F, -1, seed++});
        failures += run_call({&kSliding26, 2048, 3000, 0, 0, 0, 1.0F, -1, seed++});
        // Window boundary: the hottest key sits exactly at distance W (excluded) and the next
        // one at W-1 (included), for the decode route, the route boundary and the prefill route.
        failures += run_call({&kSliding26, 1, 5000, 64, 0, 0, 1.0F, 0, seed++});
        failures += run_call({&kSliding31, 16, 3000, 0, 0, 0, 1.0F, 9, seed++});
        failures += run_call({&kSliding26, 17, 1100, 0, 0, 0, 1.0F, 10, seed++});
        failures += run_call({&kSliding31, 64, 5000, 1, 0, 0, 1.0F, 20, seed++});

        std::printf("prefill split invariance (parts of T >= 17 on the prefill route)\n");
        for (const Geometry* geometry : kGeometries) {
            // Off the row grid (8, 1000 - 17 from a grid base), on it (64), several cuts.
            failures += run_split_invariance({geometry, 1100, 2560, 0, 0, 0, 1.0F, -1, seed++},
                                             {1000});
            failures += run_split_invariance({geometry, 300, 2560, 0, 0, 0, 1.0F, -1, seed++},
                                             {64, 81, 250});
            failures += run_split_invariance({geometry, 400, 5000, 0, 0, 0, 1.0F, -1, seed++},
                                             {27, 380});
        }

        std::printf("prefill tile of the 512-wide group-8 layers (wide slices forced: oracle; "
                    "narrow vs wide: bytes)\n");
        {
            using ops::detail::Gemma4PrefillSlices;
            ops::detail::gemma4_attention_set_prefill_slices(Gemma4PrefillSlices::Wide);
            failures += run_call({&kFull31, 1100, 2560, 0, 0, 0, 1.0F, -1, seed++});
            failures += run_call({&kFull26, 300, 0, 0, 0, 0, 0.125F, -1, seed++});
            failures += run_call({&kFull31, 37, 1000, 3, 0, 0, 1.0F, -1, seed++});
            ops::detail::gemma4_attention_set_prefill_slices(Gemma4PrefillSlices::Auto);
            for (const Geometry* geometry : {&kFull31, &kFull26}) {
                // On the row grid, off it (a call starting 3 tokens into a block), short and long.
                failures += run_slice_width_invariance({geometry, 1100, 2560, 0, 0, 0, 1.0F, -1,
                                                        seed++});
                failures += run_slice_width_invariance({geometry, 300, 2563, 0, 0, 0, 1.0F, -1,
                                                        seed++});
                failures += run_slice_width_invariance({geometry, 17, 0, 0, 0, 0, 1.0F, -1,
                                                        seed++});
                failures += run_slice_width_invariance({geometry, 2048, 0, 0, 0, 0, 0.125F, -1,
                                                        seed++});
            }
        }

        std::printf("chained prefill of the full layers (plain vs chained: bytes)\n");
        for (const Geometry* geometry : {&kFull26, &kFull31, &kFull12, &kGroup6D512}) {
            // One span (short keys), several, a call off the row grid, both prefill tiles.
            failures += run_chain_invariance({geometry, 300, 0, 0, 0, 0, 1.0F, -1, seed++});
            failures += run_chain_invariance({geometry, 1100, 2560, 0, 0, 0, 1.0F, -1, seed++});
            failures += run_chain_invariance({geometry, 300, 9003, 0, 0, 0, 0.125F, -1, seed++});
            failures += run_chain_invariance({geometry, 17, 20000, 0, 0, 0, 1.0F, -1, seed++});
        }
        // Chained calls against the oracle.
        ops::detail::gemma4_attention_set_prefill_chain(ops::detail::Gemma4PrefillChain::On);
        failures += run_call({&kFull26, 1100, 6000, 0, 0, 0, 1.0F, -1, seed++});
        failures += run_call({&kFull31, 600, 12000, 3, 0, 0, 1.0F, -1, seed++});
        ops::detail::gemma4_attention_set_prefill_chain(ops::detail::Gemma4PrefillChain::Auto);

        std::printf("invariant route profile: prefill cut into calls of 1..16 tokens\n");
        {
            const ops::ScopedPrefillRoutes prefill(true);
            const ops::ScopedInvariantRoutes invariant(true);
            for (const Geometry* geometry :
                 {&kFull26, &kSliding26, &kDraftFull26, &kOssFull, &kOssSliding, &kGroup3D128}) {
                failures += run_split_invariance({geometry, 300, 2561, 0, 0, 0, 0.125F, -1, seed++},
                                                 {1, 2, 18, 30, 31, 250, 299});
                failures += run_split_invariance({geometry, 40, 0, 0, 0, 0, 0.125F, -1, seed++},
                                                 {1, 5, 16, 17, 33});
            }
        }
        // Sliding layers past the window: a tile boundary lies between a row block's first and
        // second token, so calls ending on a block's first token (positions 16k at group 2) must
        // fold the same tiles as calls running through the block (Gemma 4 prefill chunks that
        // leave a 17-token tail, and decode steps).
        std::printf("sliding layers: calls ending on a row block's first token\n");
        {
            const ops::ScopedPrefillRoutes prefill(true);
            const ops::ScopedInvariantRoutes invariant(true);
            for (const Geometry* geometry : {&kSliding26, &kSliding31, &kSlidingOdd}) {
                for (const float scale : {1.0F, 0.125F, 0.0625F}) {
                    failures += run_split_invariance({geometry, 1100, 0, 0, 0, 0, scale, -1, seed++},
                                                     {1057, 1083});
                    failures += run_split_invariance(
                        {geometry, 300, 2560, 0, 0, 0, scale, -1, seed++}, {17, 33, 34, 81, 97});
                    failures += run_split_invariance(
                        {geometry, 64, 3000, 0, 0, 0, scale, -1, seed++}, {8, 9, 24, 25, 40, 41});
                }
            }
        }
        std::printf("decode under wider envelopes, both route profiles\n");
        for (const Geometry* geometry :
             {&kFull26, &kSliding26, &kDraftFull26, &kOssFull, &kOssSliding, &kGroup3D128}) {
            for (const int tokens : {1, 4, 16}) {
                for (const int base : {300, 3001}) {
                    failures += run_envelope_invariance(
                        {geometry, tokens, base, 0, 0, 0, 0.125F, -1, seed++}, {0, 7, 900, 6000});
                }
            }
        }

        // A verify call is T decode steps: under the invariant profile (without the prefill one)
        // its columns must carry the bytes of single-token calls at their positions, on full
        // layers past the keys of one decode split and on sliding ones past the window (the decode
        // kernel's key chunks and tiles followed the call: gpt-oss's verify drifted from its
        // decode).
        std::printf("invariant decode route: a verify call cut into decode steps\n");
        {
            const ops::ScopedInvariantRoutes invariant(true);
            for (const Geometry* geometry :
                 {&kFull26, &kSliding26, &kDraftFull26, &kOssFull, &kOssSliding, &kGroup3D128}) {
                for (const int tokens : {8, 16}) {
                    std::vector<int> steps;
                    for (int t = 1; t < tokens; ++t) { steps.push_back(t); }
                    CallSpec spec{geometry, tokens, 2961, 0, 0, 0, 0.125F, -1, seed++};
                    spec.sinks = geometry == &kOssFull || geometry == &kOssSliding;
                    failures += run_split_invariance(spec, steps);
                }
            }
        }

        std::printf("DFlash drafter layers (D128, scale 1/sqrt(128))\n");
        // Every drafter geometry and mask over the route boundary and interior widths, empty and
        // continued. The drafter itself runs T=16 only; the wider calls prove the shared routes
        // stay correct at this width.
        for (const Geometry* geometry : kDraftGeometries) {
            const bool sliding = geometry->window > 0;
            for (const Mask mask : {kCausal, kBlock}) {
                for (const int tokens : {1, 3, 16, 17, 64}) {
                    failures +=
                        run_call({geometry, tokens, 0, 0, 0, 0, kDraftScale, -1, seed++, mask});
                    failures +=
                        run_call({geometry, tokens, sliding ? 5000 : 3001, (tokens % 2) ? 0 : 64,
                                  static_cast<std::uint32_t>(tokens % 3) * 97, tokens % 2,
                                  kDraftScale, -1, seed++, mask});
                }
            }
        }
        // Context chunks of 2048 committed tokens go through gemma4_kv_append; these calls also
        // append 2048 rows (into a wrapped ring for the continuation).
        failures += run_call({&kDraftSliding26, 2048, 0, 0, 0, 0, kDraftScale, -1, seed++});
        failures += run_call({&kDraftSliding31, 2048, 5000, 0, 0, 0, kDraftScale, -1, seed++});

        // The drafter's round: T=16 at positions P..P+15, context from nothing to 32k. A NaN row
        // (a stale rejected draft) sits right after the block and must not reach the output.
        for (const Geometry* geometry : {&kDraftFull26, &kDraftFull31}) {
            for (const int context : {0, 100, 2047, 5000, 32752}) {
                for (const Mask mask : {kBlock, kCausal}) {
                    if (mask == kCausal && context != 100 && context != 5000) { continue; }
                    failures += run_call({geometry, kDraftTokens, context, 0, kDraftTokens, 0,
                                          kDraftScale, -1, seed++, mask},
                                         /*stale_row=*/true);
                }
            }
        }
        // Block with one dominant key at P_last: every column must follow it.
        failures += run_call({&kDraftFull31, kDraftTokens, 20000, 0, 1, 0, kDraftScale, -1, seed++,
                              kBlock, /*hot_last=*/true},
                             true);

        // Sliding ring wrap at W=2048. Capacities: the minimum W + 16 - 1, W + 16 + a 512 chunk,
        // and W + 2048 - 1 (a ring that also takes 2048-row context appends). Bases put the block
        // across the ring end, at slot 0, ending on the last slot, and far past many wraps.
        for (const Geometry* geometry : {&kDraftSliding26, &kDraftSliding31}) {
            for (const int slack : {0, 513, 2032}) {
                const int capacity = geometry->window + kDraftTokens - 1 + slack;
                for (const int base :
                     {3 * capacity - 5, 2 * capacity, 2 * capacity - kDraftTokens, 100003}) {
                    // Causal (z-lab's Gemma 4 drafters) and Block (bidirectional inside the block,
                    // Muse Glimmer's assistant): the same ring, the same oldest key per column.
                    for (const Mask mask : {kCausal, kBlock}) {
                        failures += run_call({geometry, kDraftTokens, base, slack, 0, 0,
                                              kDraftScale, -1, seed++, mask},
                                             /*stale_row=*/slack > 0);
                    }
                }
            }
        }
        // Window boundary at exactly W=2048: the hottest key at distance W (excluded), the next at
        // W-1 (included), on the decode route (first, middle and last column), for T=1, and on the
        // prefill route.
        failures += run_call({&kDraftSliding26, 16, 5000, 513, 0, 0, kDraftScale, 0, seed++});
        failures += run_call({&kDraftSliding26, 16, 7000, 0, 0, 0, kDraftScale, 15, seed++});
        failures += run_call({&kDraftSliding31, 16, 3000, 2032, 0, 0, kDraftScale, 9, seed++});
        failures += run_call({&kDraftSliding31, 1, 4096, 0, 0, 0, kDraftScale, 0, seed++});
        failures += run_call({&kDraftSliding26, 17, 2100, 0, 0, 0, kDraftScale, 10, seed++});
        // The same boundary under Block: each column keeps its own oldest key while it sees the
        // whole block.
        failures +=
            run_call({&kDraftSliding26, 16, 5000, 513, 0, 0, kDraftScale, 0, seed++, kBlock});
        failures +=
            run_call({&kDraftSliding31, 16, 3000, 2032, 0, 0, kDraftScale, 9, seed++, kBlock});
        failures += run_call({&kDraftSliding26, 16, 7000, 0, 0, 0, kDraftScale, 15, seed++, kBlock});

        std::printf("image soft tokens on the target's sliding layers (D256, Block)\n");
        // One prefill call per image: its soft tokens (up to 280) see one another and every
        // earlier key inside the window. Empty and continued contexts, a ring that wraps, and the
        // window boundary under Block.
        for (const Geometry* geometry : {&kSliding26, &kSliding31}) {
            for (const int tokens : {1, 70, 256, 280}) {
                for (const int base : {0, 300, 5000}) {
                    failures += run_call({geometry, tokens, base, 0, 0, 0, 1.0F, -1, seed++, kBlock});
                }
            }
            failures += run_call({geometry, 280, 3 * (geometry->window + 279) - 5, 0, 0, 0, 1.0F,
                                  -1, seed++, kBlock});
            failures += run_call({geometry, 280, 4000, 0, 0, 0, 1.0F, 0, seed++, kBlock});
            failures += run_call({geometry, 280, 4000, 0, 0, 0, 1.0F, 279, seed++, kBlock});
            failures += run_call({geometry, 64, 9000, 0, 0, 0, 1.0F, -1, seed++, kBlock,
                                  /*hot_last=*/true});
        }

        std::printf("graph capture, replayed with new positions inside one envelope\n");
        failures += run_graph({&kFull26, 1, 0}, {5, 1999, 777}, 2048);
        failures += run_graph({&kSliding26, 4, 0, 7}, {100, 5000, 2047}, 6000);
        failures += run_graph({&kFull31, 64, 0}, {0, 3000}, 3064);
        failures += run_graph({&kSliding31, 16, 0}, {4000, 20}, 4016);
        failures += run_graph({&kDraftFull26, 16, 0, 0, 0, 0, kDraftScale, -1, 1, kBlock},
                              {100, 30000, 2047, 0}, 32768);
        failures += run_graph({&kDraftFull31, 16, 0, 0, 0, 0, kDraftScale, -1, 1, kBlock},
                              {5000, 16, 32752}, 32768);
        failures += run_graph({&kDraftFull31, 16, 0, 0, 0, 0, kDraftScale}, {700, 12}, 8192);
        failures += run_graph({&kDraftSliding26, 16, 0, 513, 0, 0, kDraftScale},
                              {5000, 20, 3 * 2576 - 5}, 100000);
        failures += run_graph({&kDraftSliding31, 16, 0, 2032, 0, 0, kDraftScale},
                              {4000, 3 * 4095 - 5, 16}, 100000);

        std::printf("control arms (each must exceed the limit)\n");
        const CallSpec boundary_decode{&kSliding26, 1, 5000, 64, 0, 0, 1.0F, 0, 900};
        const CallSpec boundary_prefill{&kSliding31, 64, 5000, 1, 0, 0, 1.0F, 20, 901};
        failures += run_control("window W+1", boundary_decode, {.window_delta = 1});
        failures += run_control("window W-1", boundary_decode, {.window_delta = -1});
        failures += run_control("window W+1", boundary_prefill, {.window_delta = 1});
        failures += run_control("window W-1", boundary_prefill, {.window_delta = -1});
        failures += run_control("GQA h % Hkv", {&kFull26, 17, 700, 0, 0, 0, 1.0F, -1, 902},
                                {.gqa_modulo = true});
        failures += run_control("GQA h % Hkv", {&kSliding31, 1, 5000, 0, 0, 0, 1.0F, -1, 903},
                                {.gqa_modulo = true});
        failures += run_control("no causal mask", {&kFull31, 300, 0, 0, 0, 0, 1.0F, -1, 904},
                                {.no_causal = true});
        failures += run_control("no causal mask", {&kSliding26, 16, 3000, 0, 0, 0, 1.0F, -1, 905},
                                {.no_causal = true});
        // Drafter: a causal kernel judged by the Block formula, and a Block kernel judged by the
        // causal one, at short context and (with a dominant key at P_last) at long context.
        failures += run_control("causal as Block",
                                {&kDraftFull26, 16, 0, 0, 16, 0, kDraftScale, -1, 906, kCausal},
                                {.no_causal = true});
        failures += run_control("causal as Block",
                                {&kDraftFull31, 16, 100, 0, 16, 0, kDraftScale, -1, 907, kCausal},
                                {.no_causal = true});
        failures +=
            run_control("causal as Block",
                        {&kDraftFull26, 16, 32752, 0, 16, 0, kDraftScale, -1, 908, kCausal, true},
                        {.no_causal = true});
        failures += run_control("Block as causal",
                                {&kDraftFull26, 16, 100, 0, 16, 0, kDraftScale, -1, 909, kBlock},
                                {.force_causal = true});
        failures +=
            run_control("Block as causal",
                        {&kDraftFull31, 16, 20000, 0, 16, 0, kDraftScale, -1, 910, kBlock, true},
                        {.force_causal = true});
        failures += run_control("GQA h % Hkv",
                                {&kDraftFull31, 16, 200, 0, 0, 0, kDraftScale, -1, 911, kBlock},
                                {.gqa_modulo = true});
        const CallSpec draft_boundary{&kDraftSliding26, 16, 5000, 513, 0, 0, kDraftScale, 0, 912};
        const CallSpec draft_boundary_last{&kDraftSliding31, 16, 7000, 0, 0, 0,
                                           kDraftScale,      15, 913};
        failures += run_control("window W+1", draft_boundary, {.window_delta = 1});
        failures += run_control("window W-1", draft_boundary, {.window_delta = -1});
        failures += run_control("window W+1", draft_boundary_last, {.window_delta = 1});
        failures += run_control("window W-1", draft_boundary_last, {.window_delta = -1});
        // A read of the NaN row right after the block must be visible to the suite.
        failures +=
            run_overread_control({&kDraftFull26, 16, 3000, 0, 16, 0, kDraftScale, -1, 914, kBlock});
        failures +=
            run_overread_control({&kDraftFull31, 16, 100, 0, 16, 0, kDraftScale, -1, 915, kCausal});
        failures += run_overread_control(
            {&kDraftSliding26, 16, 5000, 513, 1, 0, kDraftScale, -1, 916, kCausal});

        std::printf("geometries outside the shipped models (runtime head counts and window)\n");
        // Own seed range, so the shipped geometries above keep their data.
        std::uint32_t open_seed = 100000;
        for (const Geometry* geometry : kOpenGeometries) {
            const bool sliding = geometry->window > 0;
            for (const int tokens : {1, 3, 16, 17, 64, 300}) {
                failures += run_call({geometry, tokens, 0, 0, 0, 0, 1.0F, -1, open_seed++});
                const int base = sliding ? (tokens >= 300 ? 2500 : 5000) : 3001;
                failures += run_call({geometry, tokens, base, (tokens % 2) ? 0 : 64,
                                      static_cast<std::uint32_t>(tokens % 3) * 97, tokens % 2, 1.0F,
                                      -1, open_seed++});
            }
            failures += run_call({geometry, 2048, 0, 0, 0, 0, 1.0F, -1, open_seed++});
            failures += run_split_invariance({geometry, 1100, 2560, 0, 0, 0, 1.0F, -1, open_seed++},
                                             {1000});
            failures += run_split_invariance({geometry, 300, 2560, 0, 0, 0, 1.0F, -1, open_seed++},
                                             {64, 81, 250});
        }
        failures += run_graph({&kFull12, 1, 0}, {5, 1999, 777}, 2048);
        failures += run_graph({&kSlidingOdd, 4, 0, 7}, {100, 5000, 1023}, 6000);

        std::printf("gpt-oss layers: D64, attention sinks, scale 1/8\n");
        std::uint32_t oss_seed = 200000;
        const float oss_scale  = 0.125F;
        const auto oss = [&](const Geometry* g, int tokens, int base, bool sinks, int slack = 0) {
            CallSpec spec{g, tokens, base, slack, 0, 0, oss_scale, -1, oss_seed++};
            spec.sinks = sinks;
            return spec;
        };
        for (const Geometry* geometry : kOssGeometries) {
            const bool sliding = geometry->window > 0;
            for (const int tokens : {1, 3, 16, 17, 64, 300}) {
                failures += run_call(oss(geometry, tokens, 0, true));
                failures += run_call(oss(geometry, tokens, sliding ? 5000 : 3001, true,
                                         (tokens % 2) ? 0 : 64));
            }
            failures += run_call(oss(geometry, 17, 700, false));
            failures += run_call(oss(geometry, 2048, 0, true));
            failures += run_split_invariance(oss(geometry, 1100, 2560, true), {1000});
            failures += run_split_invariance(oss(geometry, 300, 2560, true), {64, 81, 250});
        }

        std::printf("GQA groups that are not powers of two (runtime-group instance)\n");
        std::uint32_t group_seed = 300000;
        for (const Geometry* geometry : kGroupGeometries) {
            const bool sliding = geometry->window > 0;
            for (const int tokens : {1, 3, 16, 17, 64, 300}) {
                failures += run_call({geometry, tokens, 0, 0, 0, 0, 0.125F, -1, group_seed++});
                const int base = sliding ? (tokens >= 300 ? 2500 : 5000) : 3001;
                failures += run_call({geometry, tokens, base, (tokens % 2) ? 0 : 64,
                                      static_cast<std::uint32_t>(tokens % 3) * 97, tokens % 2,
                                      0.125F, -1, group_seed++});
            }
            failures += run_call({geometry, 2048, 0, 0, 0, 0, 0.125F, -1, group_seed++});
            // Off the row grid (the block holds Br / G tokens, not a power of two), on it, cuts.
            failures += run_split_invariance(
                {geometry, 1100, 2560, 0, 0, 0, 0.125F, -1, group_seed++}, {1000});
            failures += run_split_invariance(
                {geometry, 300, 2561, 0, 0, 0, 0.125F, -1, group_seed++}, {21, 64, 81, 250});
        }
        // A model's prefill from position 0 in chunks of 64 or 128 tokens: calls that end inside a
        // row block of Br / G tokens (Llama 3.2 3B, SmolLM2 360M, Qwen2.5 0.5B).
        for (const Geometry* geometry : {&kGroup3D128, &kGroup3D64, &kGroup7D64, &kGroup5D128,
                                         &kGroup1D128, &kGroup1D64, &kGroup1Sliding}) {
            failures += run_split_invariance({geometry, 700, 0, 0, 0, 0, 0.125F, -1, group_seed++},
                                             {128, 256, 384, 512, 640});
            failures += run_split_invariance({geometry, 300, 0, 0, 0, 0, 0.125F, -1, group_seed++},
                                             {64, 128, 192, 256});
            failures += run_split_invariance({geometry, 300, 2561, 0, 0, 0, 0.125F, -1, group_seed++},
                                             {21, 40, 81, 250});
        }
        {
            CallSpec sinks{&kGroup5D64, 17, 700, 0, 0, 0, 0.125F, -1, group_seed++};
            sinks.sinks = true;
            failures += run_call(sinks);
            sinks.tokens = 1;
            sinks.seed   = group_seed++;
            failures += run_call(sinks);
        }
        failures += run_call({&kGroup5Sliding, 1, 5000, 64, 0, 0, 0.125F, 0, group_seed++});
        failures += run_call({&kGroup5Sliding, 64, 5000, 1, 0, 0, 0.125F, 20, group_seed++});
        failures += run_graph({&kGroup3D128, 1, 0}, {5, 1999, 777}, 2048);
        failures += run_graph({&kGroup5Sliding, 4, 0, 7}, {100, 5000, 1023}, 6000);
        failures += run_control("GQA h % Hkv", {&kGroup3D128, 17, 700, 0, 0, 0, 0.125F, -1, 950},
                                {.gqa_modulo = true});
        failures += run_control("GQA h % Hkv", {&kGroup5D64, 1, 3000, 0, 0, 0, 0.125F, -1, 951},
                                {.gqa_modulo = true});

        failures += run_rejections();
    } catch (const std::exception& error) {
        std::cerr << "gemma4_attention: " << error.what() << '\n';
        return 1;
    }
    std::printf("worst agreement over all checked calls: %.3f of the limit\n", g_worst_ratio);
    if (failures != 0) {
        std::cerr << "gemma4_attention: " << failures << " checks failed\n";
        return 1;
    }
    std::cout << "gemma4_attention: all checks passed\n";
    return 0;
}
