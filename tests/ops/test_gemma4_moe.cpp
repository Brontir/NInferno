#include "core/weight.h"
#include "core/weight_view.h"
#include "ninferno/ops/gemma4_moe.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/weight_input.h"

#include "ops/gemma4_moe/gemma4_moe_internal.h"
#include "ops/oracle.h"
#ifndef NINFERNO_WITHOUT_NVFP4
#include "ops/linear/nvfp4/nvfp4_format.h" // the validator is built with the NVFP4 family only
#endif
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <numbers>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;
namespace g4 = ninferno::ops::detail::gemma4;

namespace {

constexpr int kHidden     = 2816;
constexpr int kExperts    = 128;
constexpr int kTopK       = 8;
constexpr int kInter      = 704;
constexpr int kGateUpRows = 2 * kInter;
constexpr float kEps      = 1.0e-6F;
// Distinct token columns the oracle evaluates; every case draws its columns
// from this pool.
constexpr int kPool = 160;
// A column whose 8th and 9th logits are closer than this is a near-tie and is
// left out of the exact selection check. Both routers carry far more precision
// than this (FP32 dot, or a split BF16 head and tail on the tensor-core
// router).
constexpr double kNearTie = 1.0e-3;
// Graph replays of each PDL route with a NaN-filled workspace (check_replayed_against_decode).
constexpr int kPdlReplays = 10000;

// Routes that keep the activation represented: BF16 inputs, FP32 intermediate,
// BF16 output. The weights decode to exactly the oracle's values, so what
// remains is the BF16 rounding of the output (half an ulp is at most 2^-8 of
// the value, so 4e-3 of the largest reference bounds any element) and FP32
// accumulation, which averages over a vector to about 2^-9/sqrt(3) = 1.1e-3.
constexpr ReductionCriterion kA16Tolerance{
    /*relative_l2*/ 2.5e-3,
    /*gross_absolute*/ 0.0,
    /*gross_relative_to_max_reference*/ 4.0e-3,
};

// The represented prefill route rounds two more planes to BF16 than decode
// does: the GeGLU product that feeds down, and every expert's output before the
// rank-ordered sum. Three independent roundings of about 2^-9/sqrt(3) each give
// sqrt(3) x 1.1e-3 = 1.9e-3 over a vector; measured 2.87e-3 (the product's
// rounding is amplified through down's cancellation). One element can carry the
// expert-output roundings of all eight terms, whose magnitudes may sum to more
// than the result, so the gross term allows two ulps of the largest reference
// rather than half of one.
constexpr ReductionCriterion kA16PrefillTolerance{
    /*relative_l2*/ 4.0e-3,
    /*gross_absolute*/ 0.0,
    /*gross_relative_to_max_reference*/ 8.0e-3,
};

// The four-bit prefill route encodes both GEMM inputs in e2m1 with a block
// scale. A dot product keeps the per-element relative error of e2m1 rather than
// averaging it away (the signal carries the same cancellation as the error), so
// the output error is a large fraction of e2m1's own relative step and both
// terms of the bound have to be relative. Measured 0.155-0.159 over every T and
// both divisor policies, worst element 0.16 of the largest reference; the
// closest control (un-renormalized p) sits at 0.49, 2.7 times this bound.
constexpr ReductionCriterion kA4Tolerance{
    /*relative_l2*/ 1.8e-1,
    /*gross_absolute*/ 0.0,
    /*gross_relative_to_max_reference*/ 2.5e-1,
};

// ---- deterministic generators -------------------------------------------------------------------
std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

double decode_e4m3(std::uint8_t word) {
    const int e            = (word >> 3) & 15;
    const int m            = word & 7;
    const double magnitude = e == 0 ? std::ldexp(static_cast<double>(m) / 8.0, -6)
                                    : std::ldexp(1.0 + static_cast<double>(m) / 8.0, e - 7);
    return (word & 0x80) != 0 ? -magnitude : magnitude;
}

constexpr double kE2m1[16] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

template <class Function>
void parallel_for(int count, Function&& function) {
    const int threads =
        std::max(1, std::min(count, static_cast<int>(std::thread::hardware_concurrency())));
    std::vector<std::thread> workers;
    for (int thread = 0; thread < threads; ++thread) {
        workers.emplace_back([&, thread] {
            for (int index = thread; index < count; index += threads) { function(index); }
        });
    }
    for (auto& worker : workers) { worker.join(); }
}

// One stacked NVFP4 plane as the artifact stores it: codes, the swizzled
// K16M128x4 scale plane at the next 256-byte boundary, and one FP32 divisor per
// expert after it.
struct HostPlane {
    int rows                   = 0;
    int k                      = 0;
    int rows_per_expert        = 0;
    std::size_t scale_offset   = 0;
    std::size_t divisor_offset = 0;
    std::vector<std::uint8_t> payload;
    std::vector<float> divisors;

    std::size_t scale_at(int row, int group) const {
        return scale_offset + (static_cast<std::size_t>(row / 128) * (k / 64) + group / 4) * 512 +
               static_cast<std::size_t>(row % 32) * 16 +
               static_cast<std::size_t>((row % 128) / 32) * 4 + static_cast<std::size_t>(group % 4);
    }

    // The exact represented dot product of one row with a double vector.
    double dot(std::int64_t row, const double* x) const {
        const std::uint8_t* codes = payload.data() + static_cast<std::size_t>(row) * k / 2;
        double acc                = 0.0;
        for (int group = 0; group < k / 16; ++group) {
            double part = 0.0;
            for (int i = 0; i < 8; ++i) {
                const std::uint8_t byte = codes[group * 8 + i];
                part += kE2m1[byte & 15] * x[group * 16 + 2 * i];
                part += kE2m1[byte >> 4] * x[group * 16 + 2 * i + 1];
            }
            acc += decode_e4m3(payload[scale_at(static_cast<int>(row), group)]) * part;
        }
        return acc / static_cast<double>(divisors[static_cast<std::size_t>(row / rows_per_expert)]);
    }
};

// Codes are uniform nibbles; block scales span e4m3 exponents 12..15 (32 to
// 448), the range a checkpoint's scales occupy once the divisor has put the
// tensor maximum at 448; each expert's divisor puts its own maximum weight at
// `amax[e]`.
HostPlane make_plane(int rows, int k, const std::vector<double>& amax, std::uint64_t seed) {
    HostPlane plane;
    plane.rows                   = rows;
    plane.k                      = k;
    plane.rows_per_expert        = rows / kExperts;
    const std::size_t code_bytes = static_cast<std::size_t>(rows) * k / 2;
    plane.scale_offset           = (code_bytes + 255) / 256 * 256;
    plane.divisor_offset         = plane.scale_offset + static_cast<std::size_t>(rows) * k / 16;
    plane.payload.assign(plane.divisor_offset + kExperts * sizeof(float), 0);
    parallel_for(rows / 128, [&](int tile) {
        std::uint64_t state = mix64(seed ^ (static_cast<std::uint64_t>(tile) << 20));
        for (int row = tile * 128; row < tile * 128 + 128; ++row) {
            std::uint8_t* codes = plane.payload.data() + static_cast<std::size_t>(row) * k / 2;
            for (int byte = 0; byte < k / 2; byte += 8) {
                state = mix64(state);
                std::memcpy(codes + byte, &state, 8);
            }
            for (int group = 0; group < k / 16; ++group) {
                state              = mix64(state);
                const int exponent = 12 + static_cast<int>(state & 3);
                int mantissa       = static_cast<int>((state >> 2) & 7);
                if (exponent == 15 && mantissa == 7) { mantissa = 6; }
                plane.payload[plane.scale_at(row, group)] =
                    static_cast<std::uint8_t>((exponent << 3) | mantissa);
            }
        }
    });
    plane.divisors.resize(kExperts);
    for (int expert = 0; expert < kExperts; ++expert) {
        plane.divisors[expert] = static_cast<float>(6.0 * 448.0 / amax[expert]);
    }
    std::memcpy(plane.payload.data() + plane.divisor_offset, plane.divisors.data(),
                kExperts * sizeof(float));
    return plane;
}

// The expert the fixture kills.
constexpr int kDeadExpert = 37;

// Zero codes and block scales of one expert, divisor 1: what ninferno-convert keeps for an expert
// whose calibration saw nothing (amax 0).
void kill_expert(HostPlane& plane) {
    const int first = kDeadExpert * plane.rows_per_expert;
    for (int row = first; row < first + plane.rows_per_expert; ++row) {
        std::memset(plane.payload.data() + static_cast<std::size_t>(row) * plane.k / 2, 0,
                    static_cast<std::size_t>(plane.k) / 2);
        for (int group = 0; group < plane.k / 16; ++group) {
            plane.payload[plane.scale_at(row, group)] = 0;
        }
    }
    plane.divisors[kDeadExpert] = 1.0F;
    std::memcpy(plane.payload.data() + plane.divisor_offset + kDeadExpert * sizeof(float),
                &plane.divisors[kDeadExpert], sizeof(float));
}

Weight device_plane_weight(const HostPlane& plane, void* device) {
    Weight weight{};
    auto* base           = static_cast<std::uint8_t*>(device);
    weight.payload       = base;
    weight.payload_bytes = plane.payload.size();
    weight.qtype         = QType::NVFP4;
    weight.group_size    = 16;
    weight.group         = 16;
    weight.layout        = QuantLayout::BlockScaleK16M128x4;
    weight.scale_dtype   = DType::FP8_E4M3FN;
    weight.ndim          = 2;
    weight.n = weight.shape[0] = weight.padded_shape[0] = plane.rows;
    weight.k = weight.shape[1] = weight.padded_shape[1] = plane.k;
    weight.qdata                                        = base;
    weight.scales                                       = base + plane.scale_offset;
    weight.weight_divisors                              = base + plane.divisor_offset;
    weight.weight_divisor_rows                          = plane.rows_per_expert;
    weight.weight_scale_divisor                         = plane.divisors[0];
    weight.input_scale_divisor                          = 1.0F;
    return weight;
}

Weight device_router_weight(void* device) {
    Weight weight{};
    weight.payload       = device;
    weight.payload_bytes = static_cast<std::uint64_t>(kExperts) * kHidden * 2;
    weight.qtype         = QType::BF16;
    weight.qdata         = device;
    weight.layout        = QuantLayout::Contiguous;
    weight.ndim          = 2;
    weight.n = weight.shape[0] = weight.padded_shape[0] = kExperts;
    weight.k = weight.shape[1] = weight.padded_shape[1] = kHidden;
    return weight;
}

enum Control { IgnoreExpertScale, UnnormalizedProbability, SwappedGateUp, OneDivisor, kControls };

const char* kControlNames[kControls] = {"ignore expert_scale", "un-renormalized p",
                                        "gelu on up (swapped)", "one divisor for all experts"};

struct PoolToken {
    std::vector<float> router_input; // represented BF16 values
    std::vector<float> expert_input;
    std::array<int, kTopK> selected{};
    double tie_gap = 0.0;
    std::vector<double> reference;
    std::array<std::vector<double>, kControls> controls;
    std::array<double, kTopK> activation_max{};
    // For a token routed to the dead expert: the output with that expert left out and the other
    // seven renormalized, a plausible wrong handling of a dead expert.
    std::vector<double> dead_dropped;
};

class Fixture {
public:
    Fixture() {
        std::mt19937 rng(20260922U);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        std::uniform_real_distribution<float> uniform(0.0F, 1.0F);

        router_.resize(static_cast<std::size_t>(kExperts) * kHidden);
        for (auto& value : router_) { value = 2.0F * normal(rng); }
        round_to_bf16(router_);
        router_scale_.resize(kHidden);
        for (auto& value : router_scale_) { value = 0.5F + uniform(rng); }
        expert_scale_.resize(kExperts);
        for (auto& value : expert_scale_) { value = 0.5F + 1.5F * uniform(rng); }

        // Every expert its own weight divisor, over a fourfold range, for both
        // planes.
        std::vector<double> gate_amax(kExperts), down_amax(kExperts);
        for (int expert = 0; expert < kExperts; ++expert) {
            gate_amax[expert] = 0.05 + 0.15 * std::fmod(expert * 0.6180339887, 1.0);
            down_amax[expert] = 0.10 + 0.30 * std::fmod(expert * 0.4142135623 + 0.3, 1.0);
        }
        gate_up_ = make_plane(kExperts * kGateUpRows, kHidden, gate_amax, 0x6a7e0001ULL);
        down_    = make_plane(kExperts * kHidden, kInter, down_amax, 0xd0e70002ULL);
        // A dead expert, as ninferno-convert writes one calibration never saw: zero codes
        // and block scales under divisor 1. Its router row is a scaled copy of a live expert's,
        // so tokens are routed to it; its contribution must be exactly zero and every output
        // finite, on every route.
        kill_expert(gate_up_);
        kill_expert(down_);
        for (int h = 0; h < kHidden; ++h) {
            router_[static_cast<std::size_t>(kDeadExpert) * kHidden + h] =
                1.5F * router_[static_cast<std::size_t>(kDeadExpert + 1) * kHidden + h];
        }
        round_to_bf16(router_);

        std::vector<float> channel_gain(kHidden);
        for (auto& value : channel_gain) { value = 0.5F + 1.5F * uniform(rng); }
        pool_.resize(kPool);
        for (auto& token : pool_) {
            token.router_input.resize(kHidden);
            token.expert_input.resize(kHidden);
            const float router_gain = 0.25F + 4.0F * uniform(rng);
            for (int h = 0; h < kHidden; ++h) {
                token.router_input[h] = router_gain * normal(rng) * channel_gain[h];
                token.expert_input[h] = normal(rng) * channel_gain[h];
            }
            round_to_bf16(token.router_input);
            round_to_bf16(token.expert_input);
        }
        parallel_for(kPool, [&](int index) { evaluate(pool_[index]); });

        // Activation divisors as a checkpoint calibrates them: 448*6 over the
        // largest input seen, with a per-expert margin, so no expert saturates on
        // this data and every expert differs.
        double input_max = 0.0;
        for (const auto& token : pool_) {
            for (float value : token.expert_input) {
                input_max = std::max(input_max, std::abs(double(value)));
            }
        }
        std::vector<double> middle_max(kExperts, 0.0);
        double middle_all = 0.0;
        for (const auto& token : pool_) {
            for (int rank = 0; rank < kTopK; ++rank) {
                middle_max[token.selected[rank]] =
                    std::max(middle_max[token.selected[rank]], token.activation_max[rank]);
                middle_all = std::max(middle_all, token.activation_max[rank]);
            }
        }
        gate_up_input_divisors_.resize(kExperts);
        down_input_divisors_.resize(kExperts);
        for (int expert = 0; expert < kExperts; ++expert) {
            const double margin             = 1.0 + 0.5 * std::fmod(expert * 0.7548776662, 1.0);
            gate_up_input_divisors_[expert] = static_cast<float>(2688.0 / (input_max * margin));
            const double seen = middle_max[expert] > 0.0 ? middle_max[expert] : middle_all;
            down_input_divisors_[expert] = static_cast<float>(2688.0 / (seen * margin));
        }

        const std::vector<std::uint16_t> router_bits = bits(router_);
        router_device_                               = to_device(router_bits);
        router_scale_device_                         = to_device(router_scale_);
        expert_scale_device_                         = to_device(expert_scale_);
        gate_up_device_                              = to_device(gate_up_.payload);
        down_device_                                 = to_device(down_.payload);
        gate_up_inputs_device_                       = to_device(gate_up_input_divisors_);
        down_inputs_device_                          = to_device(down_input_divisors_);
    }

    ops::Gemma4MoeWeights weights() const {
        ops::Gemma4MoeWeights w;
        w.router                 = device_router_weight(router_device_.p);
        w.router_scale           = Tensor(router_scale_device_.p, DType::FP32, {kHidden});
        w.expert_scale           = Tensor(expert_scale_device_.p, DType::FP32, {kExperts});
        w.gate_up                = device_plane_weight(gate_up_, gate_up_device_.p);
        w.down                   = device_plane_weight(down_, down_device_.p);
        w.gate_up_input_divisors = Tensor(gate_up_inputs_device_.p, DType::FP32, {kExperts});
        w.down_input_divisors    = Tensor(down_inputs_device_.p, DType::FP32, {kExperts});
        return w;
    }

    const PoolToken& token(int index) const { return pool_[index]; }

    int verify_persistent() const {
        int failures = 0;
        failures +=
            verify_exact("router preservation",
                         from_device<std::uint16_t>(router_device_, router_.size()), bits(router_));
        failures += verify_exact(
            "gate_up preservation",
            from_device<std::uint8_t>(gate_up_device_, gate_up_.payload.size()), gate_up_.payload);
        failures += verify_exact("down preservation",
                                 from_device<std::uint8_t>(down_device_, down_.payload.size()),
                                 down_.payload);
        return failures;
    }

    // Distinct experts selected over the pool.
    int distinct_experts() const {
        std::set<int> seen;
        for (const auto& token : pool_) {
            seen.insert(token.selected.begin(), token.selected.end());
        }
        return static_cast<int>(seen.size());
    }

    double mean_top8_mass() const { return top8_mass_sum_ / kPool; }

    // Pool tokens that route to the dead expert.
    int dead_routed() const { return static_cast<int>(dead_tokens().size()); }

    std::vector<int> dead_tokens() const {
        std::vector<int> out;
        for (int index = 0; index < kPool; ++index) {
            if (!pool_[index].dead_dropped.empty()) { out.push_back(index); }
        }
        return out;
    }

private:
    static std::vector<std::uint16_t> bits(const std::vector<float>& values) {
        std::vector<std::uint16_t> out(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
        return out;
    }

    void evaluate(PoolToken& token) {
        // Router, exactly.
        double square = 0.0;
        for (float value : token.router_input) { square += double(value) * double(value); }
        const double inverse =
            1.0 / std::sqrt(square / kHidden + double(kEps)) / std::sqrt(double(kHidden));
        std::vector<double> r(kHidden);
        for (int h = 0; h < kHidden; ++h) {
            r[h] = double(token.router_input[h]) * inverse * double(router_scale_[h]);
        }
        std::vector<double> logits(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            double s = 0.0;
            for (int h = 0; h < kHidden; ++h) {
                s += double(router_[static_cast<std::size_t>(e) * kHidden + h]) * r[h];
            }
            logits[e] = s;
        }
        std::vector<int> ranked(kExperts);
        std::iota(ranked.begin(), ranked.end(), 0);
        std::sort(ranked.begin(), ranked.end(), [&](int a, int b) {
            return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
        });
        for (int k = 0; k < kTopK; ++k) { token.selected[k] = ranked[k]; }
        token.tie_gap    = logits[ranked[kTopK - 1]] - logits[ranked[kTopK]];
        const double top = logits[ranked[0]];
        double all       = 0.0;
        for (double s : logits) { all += std::exp(s - top); }
        double selected_mass = 0.0;
        std::array<double, kTopK> p{};
        for (int k = 0; k < kTopK; ++k) {
            p[k] = std::exp(logits[ranked[k]] - top) / all;
            selected_mass += p[k];
        }
        {
            static std::mutex mutex;
            std::lock_guard<std::mutex> lock(mutex);
            top8_mass_sum_ += selected_mass;
        }
        std::array<double, kTopK> weight{}, weight_no_scale{}, weight_unnormalized{};
        std::array<double, kTopK> weight_dead_dropped{};
        double dead_mass = 0.0;
        for (int k = 0; k < kTopK; ++k) { dead_mass += ranked[k] == kDeadExpert ? p[k] : 0.0; }
        for (int k = 0; k < kTopK; ++k) {
            weight[k]              = p[k] / selected_mass * double(expert_scale_[ranked[k]]);
            weight_no_scale[k]     = p[k] / selected_mass;
            weight_unnormalized[k] = p[k] * double(expert_scale_[ranked[k]]);
            weight_dead_dropped[k] =
                ranked[k] == kDeadExpert
                    ? 0.0
                    : p[k] / (selected_mass - dead_mass) * double(expert_scale_[ranked[k]]);
        }
        token.dead_dropped.assign(dead_mass > 0.0 ? kHidden : 0, 0.0);

        const std::vector<double> x(token.expert_input.begin(), token.expert_input.end());
        token.reference.assign(kHidden, 0.0);
        for (auto& control : token.controls) { control.assign(kHidden, 0.0); }
        std::vector<double> act(kInter), swapped(kInter), one_divisor(kInter);
        for (int k = 0; k < kTopK; ++k) {
            const int e = ranked[k];
            // The one-divisor control reads every expert with expert 0's divisor.
            const double gate_ratio = double(gate_up_.divisors[e]) / double(gate_up_.divisors[0]);
            const double down_ratio = double(down_.divisors[e]) / double(down_.divisors[0]);
            double act_max          = 0.0;
            for (int j = 0; j < kInter; ++j) {
                const double gate =
                    gate_up_.dot(static_cast<std::int64_t>(e) * kGateUpRows + j, x.data());
                const double up =
                    gate_up_.dot(static_cast<std::int64_t>(e) * kGateUpRows + kInter + j, x.data());
                act[j]         = gelu_tanh_oracle(gate) * up;
                swapped[j]     = gelu_tanh_oracle(up) * gate;
                one_divisor[j] = gelu_tanh_oracle(gate * gate_ratio) * (up * gate_ratio);
                act_max        = std::max(act_max, std::abs(act[j]));
            }
            token.activation_max[k] = act_max;
            for (int h = 0; h < kHidden; ++h) {
                const std::int64_t row = static_cast<std::int64_t>(e) * kHidden + h;
                const double y         = down_.dot(row, act.data());
                token.reference[h] += weight[k] * y;
                if (!token.dead_dropped.empty()) {
                    token.dead_dropped[h] += weight_dead_dropped[k] * y;
                }
                token.controls[IgnoreExpertScale][h] += weight_no_scale[k] * y;
                token.controls[UnnormalizedProbability][h] += weight_unnormalized[k] * y;
                token.controls[SwappedGateUp][h] += weight[k] * down_.dot(row, swapped.data());
                token.controls[OneDivisor][h] +=
                    weight[k] * down_.dot(row, one_divisor.data()) * down_ratio;
            }
        }
    }

    std::vector<float> router_, router_scale_, expert_scale_;
    HostPlane gate_up_, down_;
    std::vector<PoolToken> pool_;
    std::vector<float> gate_up_input_divisors_, down_input_divisors_;
    double top8_mass_sum_ = 0.0;
    DeviceBuffer router_device_, router_scale_device_, expert_scale_device_, gate_up_device_,
        down_device_, gate_up_inputs_device_, down_inputs_device_;
};

struct CaseResult {
    ReductionStats stats;
    int failures = 0;
};

const char* route_name(g4::Route route, g4::A4Input input) {
    switch (route) {
    case g4::Route::Decode:
        return "decode";
    case g4::Route::SmallT:
        return "small-T";
    case g4::Route::PrefillA16:
        return "prefill-A16";
    case g4::Route::PrefillA4:
        return input == g4::A4Input::PerExpert ? "prefill-A4/expert-divisor"
                                               : "prefill-A4/token-divisor";
    }
    return "?";
}

// Runs one case. `forced` names a route; otherwise the public entry chooses
// one.
int run_case(const Fixture& fixture, int tokens, bool forced, g4::Route route, g4::A4Input input,
             bool graph, bool narrow = false, g4::DecodeOptions decode = g4::kDefaultDecode) {
    const g4::Route effective = forced ? route : g4::default_route(tokens);
    std::string variant;
    if (effective == g4::Route::Decode) {
        variant = std::string(decode.rank_select ? "/rank-select" : "") +
                  (decode.loads_first ? "/loads-first" : "") +
                  (decode.staged_down ? "/staged-down" : "");
    }
    const std::string label = std::string("gemma4_moe T=") + std::to_string(tokens) + " " +
                              route_name(effective, input) + (narrow ? "/narrow" : "") + variant +
                              (forced ? " (forced)" : "") + (graph ? " graph" : "");
    std::vector<int> column_token(tokens);
    std::vector<std::uint16_t> router_bits(static_cast<std::size_t>(tokens) * kHidden);
    std::vector<std::uint16_t> expert_bits(router_bits.size());
    std::vector<double> reference(router_bits.size());
    std::array<std::vector<double>, kControls> controls;
    for (auto& control : controls) { control.resize(router_bits.size()); }
    for (int t = 0; t < tokens; ++t) {
        const int index        = (t * 37 + tokens * 11) % kPool;
        column_token[t]        = index;
        const PoolToken& token = fixture.token(index);
        for (int h = 0; h < kHidden; ++h) {
            const std::size_t at = static_cast<std::size_t>(t) * kHidden + h;
            router_bits[at]      = f32_to_bf16(token.router_input[h]);
            expert_bits[at]      = f32_to_bf16(token.expert_input[h]);
            reference[at]        = token.reference[h];
            for (int c = 0; c < kControls; ++c) { controls[c][at] = token.controls[c][h]; }
        }
    }
    DeviceBuffer router_input = to_device(router_bits);
    DeviceBuffer expert_input = to_device(expert_bits);
    GuardedDeviceBuffer output(router_bits.size() * 2);
    Tensor router_tensor(router_input.p, DType::BF16, {kHidden, tokens});
    Tensor expert_tensor(expert_input.p, DType::BF16, {kHidden, tokens});
    Tensor out(output.data(), DType::BF16, {kHidden, tokens});
    const ops::Gemma4MoeWeights weights = fixture.weights();
    const std::size_t workspace_bytes   = forced ? g4::route_workspace_bytes(route, tokens)
                                                 : ops::gemma4_moe_workspace_bytes(tokens, tokens);
    WorkspaceArena workspace(workspace_bytes);

    auto launch = [&](cudaStream_t stream) {
        if (forced) {
            g4::gemma4_moe_with_route(router_tensor, expert_tensor, weights, kEps, out, workspace,
                                      stream, route, input, narrow, decode);
        } else {
            ops::gemma4_moe(router_tensor, expert_tensor, weights, kEps, out, workspace, stream);
        }
    };
    int failures = 0;
    launch(nullptr);
    cuda_synchronize();
    const std::vector<std::uint16_t> eager =
        from_device<std::uint16_t>(output.data(), router_bits.size());
    if (graph) {
        cudaStream_t stream  = nullptr;
        cudaGraph_t captured = nullptr;
        cudaGraphExec_t exec = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        output.fill(0);
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
        launch(stream);
        cuda_check(cudaStreamEndCapture(stream, &captured), "end capture");
        cuda_check(cudaGraphInstantiate(&exec, captured, nullptr, nullptr, 0), "instantiate");
        cuda_check(cudaGraphLaunch(exec, stream), "graph launch");
        cuda_check(cudaGraphLaunch(exec, stream), "graph replay");
        cuda_check(cudaStreamSynchronize(stream), "graph sync");
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(captured);
        cudaStreamDestroy(stream);
        const auto replayed = from_device<std::uint16_t>(output.data(), router_bits.size());
        failures += verify_exact((label + " replay equals eager").c_str(), replayed, eager);
    }

    std::vector<double> got(eager.size());
    for (std::size_t i = 0; i < eager.size(); ++i) { got[i] = bf16_to_f32(eager[i]); }
    const ReductionCriterion& criterion = effective == g4::Route::PrefillA4 ? kA4Tolerance
                                          : effective == g4::Route::PrefillA16
                                              ? kA16PrefillTolerance
                                              : kA16Tolerance;
    failures += verify_reduction(label, got, reference, criterion);
    const ReductionStats stats = compute_reduction_stats(got.data(), reference.data(),
                                                         static_cast<std::int64_t>(got.size()));
    std::printf("CASE %-58s rel_l2=%.4e (limit %.1e, ratio %.3f) max_abs=%.4e "
                "gross_ratio=%.3f\n",
                label.c_str(), stats.relative_l2, criterion.relative_l2,
                stats.relative_l2 / criterion.relative_l2, stats.maximum_absolute_error,
                stats.maximum_absolute_error / gross_error_limit(stats, criterion));

    // Every control has to fail the same bound; its distance from the oracle is
    // reported.
    for (int c = 0; c < kControls; ++c) {
        const ReductionStats control = compute_reduction_stats(
            got.data(), controls[c].data(), static_cast<std::int64_t>(got.size()));
        const bool passes =
            reduction_passes(control, static_cast<std::int64_t>(got.size()), criterion);
        std::printf("  CONTROL %-30s rel_l2 vs output=%.4e (%.1fx limit) %s\n", kControlNames[c],
                    control.relative_l2, control.relative_l2 / criterion.relative_l2,
                    passes ? "PASSED - the bound cannot see this defect" : "fails as it must");
        if (passes) { ++failures; }
    }

    // The selection the call made is the first allocation of its workspace.
    const int capacity = effective == g4::Route::Decode ? 1 : std::min(tokens, g4::kSliceMax);
    const auto ids = from_device<int>(workspace.base(), static_cast<std::size_t>(kTopK) * capacity);
    int checked = 0, excluded = 0, mismatched = 0;
    const int first = tokens - capacity; // the last slice is what the workspace still holds
    for (int t = first; t < tokens; ++t) {
        const PoolToken& token = fixture.token(column_token[t]);
        if (token.tie_gap < kNearTie) {
            ++excluded;
            continue;
        }
        ++checked;
        std::array<int, kTopK> got_ids{};
        for (int k = 0; k < kTopK; ++k) {
            got_ids[k] = ids[static_cast<std::size_t>(t - first) * kTopK + k];
        }
        std::array<int, kTopK> want = token.selected;
        std::sort(got_ids.begin(), got_ids.end());
        std::sort(want.begin(), want.end());
        if (got_ids != want) { ++mismatched; }
    }
    std::printf("  SELECTION checked=%d near-tie-excluded=%d mismatched=%d\n", checked, excluded,
                mismatched);
    if (mismatched != 0 || checked == 0) {
        std::cerr << label << ": selected expert set differs from the oracle\n";
        ++failures;
    }

    failures += output.verify_guards(label);
    failures +=
        verify_exact((label + " router input preserved").c_str(),
                     from_device<std::uint16_t>(router_input, router_bits.size()), router_bits);
    failures +=
        verify_exact((label + " expert input preserved").c_str(),
                     from_device<std::uint16_t>(expert_input, expert_bits.size()), expert_bits);
    if (workspace.used() != 0 || workspace.peak_used() > workspace_bytes) {
        std::cerr << label << ": workspace was not released or overran its query\n";
        ++failures;
    }
    return failures;
}

// One call of `route` over `tokens` columns against as many decode calls, one column each. The
// small-T route computes every (token, expert) product, the selection and the rank-order sum as the
// decode route does, so its output must agree bit for bit. The control is the A16 prefill route,
// as exact but differently ordered: it must differ somewhere, or the comparison sees nothing.
int check_decode_parity(const Fixture& fixture, int tokens, g4::Route route) {
    const bool control = route != g4::Route::SmallT;
    std::vector<std::uint16_t> router_bits(static_cast<std::size_t>(tokens) * kHidden);
    std::vector<std::uint16_t> expert_bits(router_bits.size());
    for (int t = 0; t < tokens; ++t) {
        const PoolToken& token = fixture.token((t * 37 + tokens * 11) % kPool);
        for (int h = 0; h < kHidden; ++h) {
            const std::size_t at = static_cast<std::size_t>(t) * kHidden + h;
            router_bits[at]      = f32_to_bf16(token.router_input[h]);
            expert_bits[at]      = f32_to_bf16(token.expert_input[h]);
        }
    }
    DeviceBuffer router_input = to_device(router_bits);
    DeviceBuffer expert_input = to_device(expert_bits);
    DeviceBuffer small_out(router_bits.size() * 2);
    DeviceBuffer decode_out(router_bits.size() * 2);
    const ops::Gemma4MoeWeights weights = fixture.weights();
    WorkspaceArena small_workspace(g4::route_workspace_bytes(route, tokens));
    WorkspaceArena decode_workspace(g4::route_workspace_bytes(g4::Route::Decode, 1));
    Tensor router_tensor(router_input.p, DType::BF16, {kHidden, tokens});
    Tensor expert_tensor(expert_input.p, DType::BF16, {kHidden, tokens});
    Tensor small_tensor(small_out.p, DType::BF16, {kHidden, tokens});
    g4::gemma4_moe_with_route(router_tensor, expert_tensor, weights, kEps, small_tensor,
                              small_workspace, nullptr, route);
    for (int t = 0; t < tokens; ++t) {
        const std::size_t offset = static_cast<std::size_t>(t) * kHidden;
        Tensor router_column(static_cast<std::uint16_t*>(router_input.p) + offset, DType::BF16,
                             {kHidden, 1});
        Tensor expert_column(static_cast<std::uint16_t*>(expert_input.p) + offset, DType::BF16,
                             {kHidden, 1});
        Tensor out_column(static_cast<std::uint16_t*>(decode_out.p) + offset, DType::BF16,
                          {kHidden, 1});
        g4::gemma4_moe_with_route(router_column, expert_column, weights, kEps, out_column,
                                  decode_workspace, nullptr, g4::Route::Decode);
    }
    cuda_synchronize();
    const auto got      = from_device<std::uint16_t>(small_out, router_bits.size());
    const auto want     = from_device<std::uint16_t>(decode_out, router_bits.size());
    std::int64_t differ = 0;
    for (std::size_t i = 0; i < got.size(); ++i) { differ += got[i] != want[i] ? 1 : 0; }
    std::printf("PARITY T=%d %s against %d decode calls: %lld of %zu values differ (%s)\n", tokens,
                route_name(route, g4::A4Input::PerExpert), tokens, static_cast<long long>(differ),
                got.size(), control ? "control, must be > 0" : "must be 0");
    return (control ? differ == 0 : differ != 0) ? 1 : 0;
}

// A prefill chunk's size is set by where a turn was cut, so a token's output must not depend on
// how many tokens share the call: the leading T columns of the widest call, recomputed as a T-token
// call through the public entry, must be byte-identical for every prefill-route T. The router's K
// split was once chosen from T, which moved every logit by an ulp and every route weight with it.
// Both prefill tiers are held to it; the A16 tier is also checked to differ from the A4 one, so a
// request for BF16 experts cannot silently fall back to four-bit activations.
// With `invariant` the calls run under the invariant route profile (--activations exact): every
// width, a single token included, takes the A16 prefill route and must give the widest call's
// bytes.
int check_prefill_token_invariance(const Fixture& fixture, ops::Gemma4MoeActivations activations,
                                   bool invariant = false) {
    const ops::ScopedInvariantRoutes routes(invariant);
    const char* tier = invariant ? "invariant profile"
                       : activations == ops::Gemma4MoeActivations::A16Only ? "A16"
                                                                           : "A4";
    constexpr int kWidest = 1024;
    std::vector<std::uint16_t> router_bits(static_cast<std::size_t>(kWidest) * kHidden);
    std::vector<std::uint16_t> expert_bits(router_bits.size());
    for (int t = 0; t < kWidest; ++t) {
        const PoolToken& token = fixture.token((t * 29 + 3) % kPool);
        for (int h = 0; h < kHidden; ++h) {
            const std::size_t at = static_cast<std::size_t>(t) * kHidden + h;
            router_bits[at]      = f32_to_bf16(token.router_input[h]);
            expert_bits[at]      = f32_to_bf16(token.expert_input[h]);
        }
    }
    DeviceBuffer router_input = to_device(router_bits);
    DeviceBuffer expert_input = to_device(expert_bits);
    const ops::Gemma4MoeWeights weights = fixture.weights();
    const auto run = [&](int tokens, ops::Gemma4MoeActivations choice, int offset = 0) {
        DeviceBuffer out(static_cast<std::size_t>(tokens) * kHidden * 2);
        WorkspaceArena workspace(ops::gemma4_moe_workspace_bytes(tokens, tokens));
        const std::size_t skip = static_cast<std::size_t>(offset) * kHidden;
        Tensor router_tensor(static_cast<std::uint16_t*>(router_input.p) + skip, DType::BF16,
                             {kHidden, tokens});
        Tensor expert_tensor(static_cast<std::uint16_t*>(expert_input.p) + skip, DType::BF16,
                             {kHidden, tokens});
        Tensor out_tensor(out.p, DType::BF16, {kHidden, tokens});
        ops::gemma4_moe(router_tensor, expert_tensor, weights, kEps, out_tensor, workspace, nullptr,
                        choice);
        cuda_synchronize();
        return from_device<std::uint16_t>(out, static_cast<std::size_t>(tokens) * kHidden);
    };
    const auto widest = run(kWidest, activations);
    int failures      = 0;
    std::vector<int> widths{g4::kPrefillMin, 31, 33, 64, 100, 160, 256, 300, 513, 1000};
    if (invariant) { widths.insert(widths.begin(), {1, 2, 3, 4, 8, 15, 16}); }
    for (const int tokens : widths) {
        const auto got      = run(tokens, activations);
        std::int64_t differ = 0;
        for (std::size_t i = 0; i < got.size(); ++i) { differ += got[i] != widest[i] ? 1 : 0; }
        std::printf("TOKEN INVARIANCE %s T=%d against the leading columns of T=%d: %lld of %zu "
                    "values differ (must be 0)\n",
                    tier, tokens, kWidest, static_cast<long long>(differ), got.size());
        failures += differ != 0 ? 1 : 0;
    }
    // A later chunk's tokens lead their own call but sit anywhere inside a wider one.
    std::vector<std::pair<int, int>> windows{{300, 300}, {600, 424}, {33, 100}, {60, 17}};
    if (invariant) { windows.insert(windows.end(), {{7, 1}, {1023, 1}, {60, 4}, {33, 16}}); }
    for (const auto [offset, tokens] : windows) {
        const auto got      = run(tokens, activations, offset);
        std::int64_t differ = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            differ += got[i] != widest[static_cast<std::size_t>(offset) * kHidden + i] ? 1 : 0;
        }
        std::printf("TOKEN INVARIANCE %s columns [%d, %d) as their own call against the same "
                    "columns of T=%d: %lld of %zu values differ (must be 0)\n",
                    tier, offset, offset + tokens, kWidest, static_cast<long long>(differ),
                    got.size());
        failures += differ != 0 ? 1 : 0;
    }
    if (activations == ops::Gemma4MoeActivations::A16Only && !invariant) {
        const auto four_bit = run(kWidest, ops::Gemma4MoeActivations::Recorded);
        std::int64_t differ = 0;
        for (std::size_t i = 0; i < four_bit.size(); ++i) {
            differ += four_bit[i] != widest[i] ? 1 : 0;
        }
        std::printf("TIERS T=%d A16 against A4 through the public entry: %lld of %zu values differ "
                    "(control, must be > 0)\n",
                    kWidest, static_cast<long long>(differ), four_bit.size());
        failures += differ == 0 ? 1 : 0;
    }
    return failures;
}

// The public entry as a graph that first fills its workspace with 0xFF (NaN in every float format,
// -1 in every index), replayed against decode calls. The routes chain their kernels with
// programmatic dependent launch: a consumer starts while its producer still runs, and must not read
// what the producer writes before griddepcontrol.wait. A load hoisted above the wait reads the
// workspace as it was - here NaN, in the engine whatever the arena held before - instead of this
// call's value; replaying a warm workspace, as the cases above do, hands it last call's identical
// value and hides the defect. A race that loses only now and then needs many replays, each
// compared bit for bit.
int check_replayed_against_decode(const Fixture& fixture, int tokens, int replays) {
    std::vector<std::uint16_t> router_bits(static_cast<std::size_t>(tokens) * kHidden);
    std::vector<std::uint16_t> expert_bits(router_bits.size());
    for (int t = 0; t < tokens; ++t) {
        const PoolToken& token = fixture.token((t * 53 + tokens * 7) % kPool);
        for (int h = 0; h < kHidden; ++h) {
            const std::size_t at = static_cast<std::size_t>(t) * kHidden + h;
            router_bits[at]      = f32_to_bf16(token.router_input[h]);
            expert_bits[at]      = f32_to_bf16(token.expert_input[h]);
        }
    }
    DeviceBuffer router_input = to_device(router_bits);
    DeviceBuffer expert_input = to_device(expert_bits);
    DeviceBuffer out(router_bits.size() * 2);
    DeviceBuffer decode_out(router_bits.size() * 2);
    const ops::Gemma4MoeWeights weights = fixture.weights();
    WorkspaceArena workspace(ops::gemma4_moe_workspace_bytes(tokens, tokens));
    WorkspaceArena decode_workspace(g4::route_workspace_bytes(g4::Route::Decode, 1));
    for (int t = 0; t < tokens; ++t) {
        const std::size_t offset = static_cast<std::size_t>(t) * kHidden;
        Tensor router_column(static_cast<std::uint16_t*>(router_input.p) + offset, DType::BF16,
                             {kHidden, 1});
        Tensor expert_column(static_cast<std::uint16_t*>(expert_input.p) + offset, DType::BF16,
                             {kHidden, 1});
        Tensor out_column(static_cast<std::uint16_t*>(decode_out.p) + offset, DType::BF16,
                          {kHidden, 1});
        g4::gemma4_moe_with_route(router_column, expert_column, weights, kEps, out_column,
                                  decode_workspace, nullptr, g4::Route::Decode);
    }
    cuda_synchronize();
    const auto want = from_device<std::uint16_t>(decode_out, router_bits.size());

    Tensor router_tensor(router_input.p, DType::BF16, {kHidden, tokens});
    Tensor expert_tensor(expert_input.p, DType::BF16, {kHidden, tokens});
    Tensor out_tensor(out.p, DType::BF16, {kHidden, tokens});
    cudaStream_t stream  = nullptr;
    cudaGraph_t captured = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
    cuda_check(cudaMemsetAsync(workspace.base(), 0xff, workspace.capacity(), stream), "poison");
    ops::gemma4_moe(router_tensor, expert_tensor, weights, kEps, out_tensor, workspace, stream);
    cuda_check(cudaStreamEndCapture(stream, &captured), "end capture");
    cuda_check(cudaGraphInstantiate(&exec, captured, nullptr, nullptr, 0), "instantiate");
    int wrong_replays = 0;
    const auto start  = std::chrono::steady_clock::now();
    for (int replay = 0; replay < replays; ++replay) {
        cuda_check(cudaMemsetAsync(out.p, 0xff, out.bytes, stream), "poison out");
        cuda_check(cudaGraphLaunch(exec, stream), "graph launch");
        cuda_check(cudaStreamSynchronize(stream), "graph sync");
        const auto got = from_device<std::uint16_t>(out, router_bits.size());
        wrong_replays += got != want ? 1 : 0;
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(captured);
    cudaStreamDestroy(stream);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("REPLAY T=%d %s, workspace NaN-filled in the graph: %d of %d replays differ from "
                "%d decode calls (must be 0), %.1f s\n",
                tokens, route_name(g4::default_route(tokens), g4::A4Input::PerExpert),
                wrong_replays, replays, tokens, seconds);
    return wrong_replays != 0 ? 1 : 0;
}

template <class Mutate>
int expect_rejected(const Fixture& fixture, const char* what, Mutate&& mutate) {
    const int tokens = 2;
    DeviceBuffer input(static_cast<std::size_t>(kHidden) * tokens * 2);
    input.fill(0);
    DeviceBuffer output(static_cast<std::size_t>(kHidden) * tokens * 2);
    Tensor router_input(input.p, DType::BF16, {kHidden, tokens});
    Tensor expert_input(input.p, DType::BF16, {kHidden, tokens});
    Tensor out(output.p, DType::BF16, {kHidden, tokens});
    ops::Gemma4MoeWeights weights = fixture.weights();
    mutate(router_input, expert_input, weights, out);
    WorkspaceArena workspace(ops::gemma4_moe_workspace_bytes(1, 64));
    try {
        ops::gemma4_moe(router_input, expert_input, weights, kEps, out, workspace, nullptr);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "gemma4_moe accepted " << what << '\n';
    return 1;
}

int run_rejections(const Fixture& fixture) {
    int failures = 0;
    failures += expect_rejected(fixture, "a Q4 gate_up", [](auto&, auto&, auto& w, auto&) {
        w.gate_up.qtype = QType::Q4_G64_FP16;
    });
    failures +=
        expect_rejected(fixture, "a single-divisor down plane", [](auto&, auto&, auto& w, auto&) {
            w.down.weight_divisor_rows = w.down.n;
        });
    failures += expect_rejected(
        fixture, "gate/up with a divisor per half",
        [](auto&, auto&, auto& w, auto&) { w.gate_up.weight_divisor_rows = kInter + 64; });
    failures += expect_rejected(fixture, "a 256-expert gate_up", [](auto&, auto&, auto& w, auto&) {
        w.gate_up.n = w.gate_up.shape[0] = w.gate_up.padded_shape[0] = 2 * kExperts * kGateUpRows;
    });
    failures += expect_rejected(fixture, "a 2048-wide down", [](auto&, auto&, auto& w, auto&) {
        w.down.n = w.down.shape[0] = w.down.padded_shape[0] = kExperts * 2048;
    });
    failures += expect_rejected(fixture, "a 129-row router", [](auto&, auto&, auto& w, auto&) {
        w.router.n = w.router.shape[0] = w.router.padded_shape[0] = kExperts + 1;
    });
    failures += expect_rejected(fixture, "an FP32 router",
                                [](auto&, auto&, auto& w, auto&) { w.router.qtype = QType::FP32; });
    failures += expect_rejected(fixture, "a short expert_scale",
                                [](auto&, auto&, auto& w, auto&) { w.expert_scale.ne[0] = 64; });
    failures +=
        expect_rejected(fixture, "a missing down_input_divisors",
                        [](auto&, auto&, auto& w, auto&) { w.down_input_divisors.data = nullptr; });
    failures += expect_rejected(fixture, "token counts that differ",
                                [](auto&, auto& x, auto&, auto&) { x.ne[1] = 1; });
    failures +=
        expect_rejected(fixture, "a hidden size of 2048", [](auto& r, auto& x, auto&, auto& o) {
            r.ne[0] = x.ne[0] = o.ne[0] = 2048;
        });
    failures += expect_rejected(fixture, "an FP32 output",
                                [](auto&, auto&, auto&, auto& o) { o.dtype = DType::FP32; });
    failures += expect_rejected(fixture, "an output aliasing the input",
                                [](auto& r, auto&, auto&, auto& o) { o.data = r.data; });
    std::printf("REJECTIONS %s\n", failures == 0 ? "all rejected" : "FAILED");
    return failures;
}

// The public entry chooses its route in one function. Pin what it resolves to
// on both sides of every frontier, and that the public workspace query sizes
// exactly that route's layout, so a change to a constant cannot silently move a
// boundary.
// prepare_gemma4_moe_weights on full-size parents laid out as the converter writes them: every
// expert a consecutive row range, one divisor per expert. Host-only; nothing is dereferenced.
int check_prepare_weights() {
    int failures      = 0;
    const auto expect = [&](bool condition, const char* what) {
        if (!condition) {
            std::cerr << "prepare_gemma4_moe_weights: " << what << '\n';
            ++failures;
        }
    };
    const auto rejects = [&](const char* what, auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return; }
        std::cerr << "prepare_gemma4_moe_weights accepted " << what << '\n';
        ++failures;
    };
    const std::array<std::uint64_t, 2> gate_shape{kExperts * kGateUpRows, kHidden};
    const std::array<std::uint64_t, 2> down_shape{kExperts * kHidden, kInter};
    const std::array<std::uint64_t, 2> router_shape{kExperts, kHidden};
    const auto gate_geometry =
        weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, gate_shape, kExperts);
    const auto down_geometry =
        weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, down_shape, kExperts);
    const auto single_geometry =
        weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, gate_shape, 1);
    const auto router_geometry =
        weight_geometry(QType::BF16, QuantLayout::Contiguous, router_shape);
    std::vector<std::byte> gate_bytes(gate_geometry.bytes), down_bytes(down_geometry.bytes),
        router_bytes(router_geometry.bytes);
    const WeightParent gate_parent{gate_geometry, gate_bytes.data(), 8.0F};
    const WeightParent single_parent{single_geometry, gate_bytes.data(), 8.0F};
    const WeightParent down_parent{down_geometry, down_bytes.data(), 4.0F};
    const WeightParent router_parent{router_geometry, router_bytes.data(), 0.0F};
    const WeightView router_view{{kExperts, kHidden},
                                 {{&router_parent, 0, std::uint64_t{kExperts} * kHidden}}};

    const std::uint64_t gate_elements = std::uint64_t{kGateUpRows} * kHidden;
    const std::uint64_t down_elements = std::uint64_t{kHidden} * kInter;
    std::vector<WeightView> gate_views, down_views, single_views, swapped_views;
    for (std::uint64_t e = 0; e < kExperts; ++e) {
        gate_views.push_back(
            {{kGateUpRows, kHidden}, {{&gate_parent, e * gate_elements, (e + 1) * gate_elements}}});
        single_views.push_back({{kGateUpRows, kHidden},
                                {{&single_parent, e * gate_elements, (e + 1) * gate_elements}}});
        down_views.push_back(
            {{kHidden, kInter}, {{&down_parent, e * down_elements, (e + 1) * down_elements}}});
    }
    swapped_views = gate_views;
    std::swap(swapped_views[5], swapped_views[6]);
    const auto inputs = [](const std::vector<WeightView>& views, float base,
                           ops::LinearPolicy policy = ops::LinearPolicy::AllowA4) {
        std::vector<ops::WeightInput> out;
        for (std::size_t e = 0; e < views.size(); ++e) {
            out.push_back({views[e], policy, base + static_cast<float>(e)});
        }
        return out;
    };
    const auto gate_inputs = inputs(gate_views, 100.0F);
    const auto down_inputs = inputs(down_views, 500.0F);
    const Tensor router_scale(reinterpret_cast<void*>(0x1000), DType::FP32, {kHidden});
    const Tensor expert_scale(reinterpret_cast<void*>(0x2000), DType::FP32, {kExperts});
    const Tensor gate_divisors(reinterpret_cast<void*>(0x3000), DType::FP32, {kExperts});
    const Tensor down_divisors(reinterpret_cast<void*>(0x4000), DType::FP32, {kExperts});
    const ops::WeightInput router{router_view, ops::LinearPolicy::A16Only};

    const ops::Gemma4MoeWeights weights = ops::prepare_gemma4_moe_weights(
        router, router_scale, expert_scale, gate_inputs, down_inputs, gate_divisors, down_divisors);
    expect(weights.router.qtype == QType::BF16 && weights.router.qdata == router_bytes.data() &&
               weights.router.n == kExperts && weights.router.k == kHidden,
           "router is not the BF16 parent");
    expect(weights.gate_up.payload == gate_bytes.data() &&
               weights.gate_up.n == kExperts * kGateUpRows && weights.gate_up.k == kHidden &&
               weights.gate_up.weight_divisor_rows == kGateUpRows &&
               weights.gate_up.weight_divisors == gate_bytes.data() + gate_geometry.divisor_offset,
           "gate_up is not the whole parent with a divisor per expert");
    expect(weights.down.payload == down_bytes.data() && weights.down.n == kExperts * kHidden &&
               weights.down.k == kInter && weights.down.weight_divisor_rows == kHidden &&
               weights.down.weight_divisors == down_bytes.data() + down_geometry.divisor_offset,
           "down is not the whole parent with a divisor per expert");
    expect(weights.router_scale.data == router_scale.data &&
               weights.expert_scale.data == expert_scale.data &&
               weights.gate_up_input_divisors.data == gate_divisors.data &&
               weights.down_input_divisors.data == down_divisors.data,
           "caller tensors were not passed through");
#ifndef NINFERNO_WITHOUT_NVFP4
    try {
        (void)ops::detail::validate_nvfp4_weight(weights.gate_up, "gemma4 gate_up", true);
        (void)ops::detail::validate_nvfp4_weight(weights.down, "gemma4 down", true);
    } catch (const std::exception& error) { expect(false, error.what()); }
#endif
    const ops::Gemma4MoeInputDivisors divisors =
        ops::gemma4_moe_input_divisors(gate_inputs, down_inputs);
    expect(divisors.gate_up.size() == kExperts && divisors.down.size() == kExperts &&
               divisors.gate_up[0] == 100.0F && divisors.gate_up[127] == 227.0F &&
               divisors.down[0] == 500.0F && divisors.down[127] == 627.0F,
           "per-expert activation divisors were not read in expert order");

    const auto prepare = [&](const std::vector<ops::WeightInput>& gate,
                             const std::vector<ops::WeightInput>& down) {
        (void)ops::prepare_gemma4_moe_weights(router, router_scale, expert_scale, gate, down,
                                              gate_divisors, down_divisors);
    };
    rejects("experts out of row order", [&] { prepare(inputs(swapped_views, 1.0F), down_inputs); });
    rejects("a parent with one divisor for all experts",
            [&] { prepare(inputs(single_views, 1.0F), down_inputs); });
    rejects("an A16-only expert", [&] {
        std::vector<ops::WeightInput> gate;
        for (std::size_t e = 0; e < gate_views.size(); ++e) {
            gate.push_back({gate_views[e],
                            e == 7 ? ops::LinearPolicy::A16Only : ops::LinearPolicy::AllowA4,
                            1.0F});
        }
        prepare(gate, down_inputs);
    });
    rejects("127 experts", [&] {
        auto gate = inputs(gate_views, 1.0F);
        gate.pop_back();
        prepare(gate, down_inputs);
    });
    rejects("gate/up and down exchanged", [&] { prepare(down_inputs, gate_inputs); });
    rejects("an expert without an activation divisor", [&] {
        std::vector<ops::WeightInput> down;
        for (std::size_t e = 0; e < down_views.size(); ++e) {
            down.push_back({down_views[e], ops::LinearPolicy::AllowA4,
                            e == 3 ? std::optional<float>{} : std::optional<float>{1.0F}});
        }
        (void)ops::gemma4_moe_input_divisors(gate_inputs, down);
    });
    std::printf("PREPARE %s\n",
                failures == 0 ? "weights prepared, six malformed banks refused" : "FAILED");
    return failures;
}

int check_route_boundaries() {
    struct Case {
        int tokens;
        g4::Route route;
        ops::Gemma4MoeActivations activations = ops::Gemma4MoeActivations::Recorded;
    };

    constexpr auto kA16 = ops::Gemma4MoeActivations::A16Only;
    const Case cases[]  = {
        {1, g4::Route::Decode},
        {2, g4::Route::SmallT},
        {g4::kSmallTMax - 1, g4::Route::SmallT},
        {g4::kSmallTMax, g4::Route::SmallT},
        {g4::kSmallTMax + 1, g4::Route::PrefillA4},
        {g4::kSliceMax, g4::Route::PrefillA4},
        {g4::kSliceMax + 1, g4::Route::PrefillA4},
        // BF16 experts change the prefill tier only: decode and small-T are A16 already.
        {1, g4::Route::Decode, kA16},
        {g4::kSmallTMax, g4::Route::SmallT, kA16},
        {g4::kSmallTMax + 1, g4::Route::PrefillA16, kA16},
        {g4::kSliceMax + 1, g4::Route::PrefillA16, kA16},
    };
    int failures = 0;
    for (const Case& c : cases) {
        const g4::Route resolved       = g4::default_route(c.tokens, c.activations);
        const std::size_t public_bytes = ops::gemma4_moe_workspace_bytes(c.tokens, c.tokens);
        // The public capacity also covers the invariant route profile's A16 prefill route.
        const std::size_t route_bytes =
            std::max(g4::route_workspace_bytes(resolved, c.tokens),
                     g4::route_workspace_bytes(g4::Route::PrefillA16, c.tokens));
        std::printf("ROUTE T=%-5d %s -> %s (workspace %zu bytes)\n", c.tokens,
                    c.activations == kA16 ? "A16" : "rec",
                    route_name(resolved, g4::A4Input::PerExpert), public_bytes);
        if (resolved != c.route || public_bytes != route_bytes) {
            std::cerr << "route boundary at T=" << c.tokens << " resolved unexpectedly\n";
            ++failures;
        }
    }
    return failures;
}

// The dead expert on every route, over the pool tokens that select it: each output must be
// finite and meet the route's bound against the oracle, where the dead expert keeps its routing
// weight and contributes exactly zero. The control is the same tokens with the dead expert left
// out and the other seven renormalized; it has to fail the bound, or these tokens cannot see how
// the dead expert is treated.
int check_dead_expert(const Fixture& fixture) {
    const std::vector<int> dead = fixture.dead_tokens();
    if (dead.empty()) {
        std::cerr << "dead expert: no pool token selects it\n";
        return 1;
    }

    struct DeadCase {
        int tokens;
        bool forced;
        g4::Route route;
    };

    const DeadCase cases[] = {{1, false, g4::Route::Decode},
                              {static_cast<int>(dead.size()), false, g4::Route::SmallT},
                              {33, true, g4::Route::PrefillA16},
                              {33, true, g4::Route::PrefillA4}};
    int failures           = 0;
    for (const DeadCase& c : cases) {
        std::vector<std::uint16_t> router_bits(static_cast<std::size_t>(c.tokens) * kHidden);
        std::vector<std::uint16_t> expert_bits(router_bits.size());
        std::vector<double> reference(router_bits.size()), dropped(router_bits.size());
        for (int t = 0; t < c.tokens; ++t) {
            const PoolToken& token = fixture.token(dead[static_cast<std::size_t>(t) % dead.size()]);
            for (int h = 0; h < kHidden; ++h) {
                const std::size_t at = static_cast<std::size_t>(t) * kHidden + h;
                router_bits[at]      = f32_to_bf16(token.router_input[h]);
                expert_bits[at]      = f32_to_bf16(token.expert_input[h]);
                reference[at]        = token.reference[h];
                dropped[at]          = token.dead_dropped[h];
            }
        }
        DeviceBuffer router_input = to_device(router_bits);
        DeviceBuffer expert_input = to_device(expert_bits);
        GuardedDeviceBuffer output(router_bits.size() * 2);
        Tensor router_tensor(router_input.p, DType::BF16, {kHidden, c.tokens});
        Tensor expert_tensor(expert_input.p, DType::BF16, {kHidden, c.tokens});
        Tensor out(output.data(), DType::BF16, {kHidden, c.tokens});
        const ops::Gemma4MoeWeights weights = fixture.weights();
        WorkspaceArena workspace(c.forced ? g4::route_workspace_bytes(c.route, c.tokens)
                                          : ops::gemma4_moe_workspace_bytes(c.tokens, c.tokens));
        if (c.forced) {
            g4::gemma4_moe_with_route(router_tensor, expert_tensor, weights, kEps, out, workspace,
                                      nullptr, c.route, g4::A4Input::PerExpert, false,
                                      g4::kDefaultDecode);
        } else {
            ops::gemma4_moe(router_tensor, expert_tensor, weights, kEps, out, workspace, nullptr);
        }
        cuda_synchronize();
        const auto bits = from_device<std::uint16_t>(output.data(), router_bits.size());
        std::vector<double> got(bits.size());
        int non_finite = 0;
        for (std::size_t i = 0; i < bits.size(); ++i) {
            got[i] = bf16_to_f32(bits[i]);
            non_finite += std::isfinite(got[i]) ? 0 : 1;
        }
        const g4::Route effective           = c.forced ? c.route : g4::default_route(c.tokens);
        const ReductionCriterion& criterion = effective == g4::Route::PrefillA4 ? kA4Tolerance
                                              : effective == g4::Route::PrefillA16
                                                  ? kA16PrefillTolerance
                                                  : kA16Tolerance;
        const std::string label = std::string("dead expert T=") + std::to_string(c.tokens) + " " +
                                  route_name(effective, g4::A4Input::PerExpert);
        failures += verify_reduction(label, got, reference, criterion);
        failures += non_finite != 0 ? 1 : 0;
        failures += output.verify_guards(label);
        const ReductionStats stats   = compute_reduction_stats(got.data(), reference.data(),
                                                               static_cast<std::int64_t>(got.size()));
        const ReductionStats control = compute_reduction_stats(
            got.data(), dropped.data(), static_cast<std::int64_t>(got.size()));
        const bool control_passes =
            reduction_passes(control, static_cast<std::int64_t>(got.size()), criterion);
        std::printf("DEAD %-40s non-finite=%d rel_l2=%.4e (limit %.1e); CONTROL dead expert "
                    "dropped, others renormalized: rel_l2=%.4e (%.1fx limit) %s\n",
                    label.c_str(), non_finite, stats.relative_l2, criterion.relative_l2,
                    control.relative_l2, control.relative_l2 / criterion.relative_l2,
                    control_passes ? "PASSED - the bound cannot see this" : "fails as it must");
        failures += control_passes ? 1 : 0;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
#ifdef NINFERNO_WITHOUT_NVFP4
    // The op is the sm_120a NVFP4 kernels: a build without the family has none to test.
    std::cout << "SKIP: the Gemma 4 NVFP4 MoE kernels are not in this build\n";
    return 77;
#else
    const Fixture fixture;
    std::printf("FIXTURE pool=%d distinct experts selected=%d mean top-8 "
                "probability mass=%.3f\n",
                kPool, fixture.distinct_experts(), fixture.mean_top8_mass());

    // Every case below checks outputs against the oracle; with the dead expert among the selected
    // ones that includes its exactly-zero contribution and finite outputs.
    std::printf("DEAD EXPERT %d: selected by %d of %d pool tokens (must be > 0)\n", kDeadExpert,
                fixture.dead_routed(), kPool);
    int failures = fixture.dead_routed() > 0 ? 0 : 1;
    failures += check_dead_expert(fixture);
    failures += check_route_boundaries();
    // The public entry over the required token counts; decode and small-T also
    // through a graph.
    for (const int tokens : {1, 2, 7, 16, 33, 128, 512, 2048}) {
        failures += run_case(fixture, tokens, false, g4::Route::Decode, g4::A4Input::PerExpert,
                             tokens == 1 || tokens == 7 || tokens == 33);
    }
    // Every decode variant, alone and together; together also through a graph.
    for (const g4::DecodeOptions decode :
         {g4::DecodeOptions{true, false, false}, g4::DecodeOptions{false, true, false},
          g4::DecodeOptions{false, false, true}, g4::DecodeOptions{true, true, true}}) {
        const bool all = decode.rank_select && decode.loads_first && decode.staged_down;
        failures += run_case(fixture, 1, true, g4::Route::Decode, g4::A4Input::PerExpert, all,
                             false, decode);
    }
    // Small-T against the decode route, column by column: at the frontier (the verify width) and at
    // capacity (where experts collect more tokens than one staging round holds); and the control
    // that shows the comparison can fail.
    static_assert(g4::kSmallTMax >= 16, "the verify pass of speculative decoding runs at T=16");
    for (const int tokens : {2, g4::kSmallTMax, g4::kSmallTCapacity}) {
        failures += check_decode_parity(fixture, tokens, g4::Route::SmallT);
    }
    failures += check_decode_parity(fixture, 16, g4::Route::PrefillA16);
    // The same comparison through a replayed graph whose workspace is cleared first: decode (T=1),
    // the verify width of a four-token draft (T=5) and the frontier.
    for (const int tokens : {1, 5, g4::kSmallTMax}) {
        failures += check_replayed_against_decode(fixture, tokens, kPdlReplays);
    }
    // Every other route at the boundaries it could be chosen at.
    failures += run_case(fixture, 1, true, g4::Route::SmallT, g4::A4Input::PerExpert, false);
    failures += run_case(fixture, 32, true, g4::Route::SmallT, g4::A4Input::PerExpert, true);
    failures += run_case(fixture, 64, true, g4::Route::SmallT, g4::A4Input::PerExpert, false);
    failures += run_case(fixture, g4::kSmallTCapacity, true, g4::Route::SmallT,
                         g4::A4Input::PerExpert, true);
    for (const int tokens : {33, 512}) {
        failures +=
            run_case(fixture, tokens, true, g4::Route::PrefillA16, g4::A4Input::PerExpert, false);
        failures +=
            run_case(fixture, tokens, true, g4::Route::PrefillA4, g4::A4Input::PerExpert, false);
        failures +=
            run_case(fixture, tokens, true, g4::Route::PrefillA4, g4::A4Input::PerToken, false);
        failures += run_case(fixture, tokens, true, g4::Route::PrefillA4, g4::A4Input::PerExpert,
                             false, true);
    }
    failures += check_prefill_token_invariance(fixture, ops::Gemma4MoeActivations::Recorded);
    failures += check_prefill_token_invariance(fixture, ops::Gemma4MoeActivations::A16Only);
    // The invariant route profile: the A16 prefill route at every width, against the oracle at
    // one and four tokens, and every width against the widest call.
    for (const int tokens : {1, 4}) {
        failures +=
            run_case(fixture, tokens, true, g4::Route::PrefillA16, g4::A4Input::PerExpert, false);
    }
    failures += check_prefill_token_invariance(fixture, ops::Gemma4MoeActivations::A16Only, true);
    failures += check_prefill_token_invariance(fixture, ops::Gemma4MoeActivations::Recorded, true);
    failures += run_rejections(fixture);
    failures += check_prepare_weights();
    failures += fixture.verify_persistent();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gemma4_moe correctness\n";
    return failures == 0 ? 0 : 1;
#endif
}
