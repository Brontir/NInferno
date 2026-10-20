// MoE routing against an FP64 reference, on the device selection routines the kernels use:
// the Qwen3.5 sparse-MoE warp top-8 with its renormalised softmax and shared
// gate (sparse_moe_select_top8_warp: decode, small-T and prefill routes), and the Gemma 4 warp
// top-8 with its per-expert gain (select_top8 + top8_weight: the decode and small-T routes).
//
// Logit families: U[-18, -1]; U[-1e4, -1e3]; a mixture across six decades; every logit equal;
// exact ties straddling rank 8 (the lower id must win); ties inside the top 8; values drawn from
// five levels only, so ties sit everywhere in the merge network; +0.0 against -0.0.
//
// Pass: the ids and their order equal the reference exactly on every row; every weight is
// finite and within 1e-6 relative (1e-38 absolute, for weights FP32 cannot resolve) of the FP64
// softmax over the selected eight; the renormalised weights sum to 1 within 1e-6; two runs are
// bitwise identical. Weights below 1e-5 are counted and must occur (the U[-1e4, -1e3] rows).
//
// Controls that must fail: the reference with the opposite tie rule (higher id first) and the
// reference that normalises over all experts instead of the selected eight.
//
// NaN logits (a numerically failed router GEMM) rank as -inf: both routines select eight distinct
// in-range experts, the finite ones first and then the lowest ids, never the Gemma network's
// padding.
//
// Not covered here: a router bias (neither model has one: Qwen3.5 scores are the router GEMM,
// Gemma 4 applies router.scale before the GEMM and per_expert_scale after the softmax), and the
// Gemma decode variant that ranks by counting in shared memory (block_select_top8, file-local in
// gemma4_moe_simt.cu, reached only through ninferno_gemma4_moe_test's random-logit cases).

#include "ops/gemma4_moe/gemma4_moe_common.cuh"
#include "ops/sparse_moe/sparse_moe_route.cuh"

#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

namespace d  = ninferno::ops::detail;
namespace g4 = ninferno::ops::detail::gemma4;

constexpr int kTopK         = d::kSparseMoeTopK;
constexpr int kQwenExperts  = d::kSparseMoeExperts; // + one shared-expert gate score
constexpr int kGemmaExperts = g4::kExperts;

__global__ void qwen_select_kernel(const float* scores, int* ids, float* alpha, float* shared,
                                   float* logits) {
    __shared__ float row[kQwenExperts + 1];
    __shared__ int selected[kTopK];
    __shared__ float weight[kTopK];
    __shared__ float selected_logit[kTopK];
    __shared__ float shared_scale;
    const int r = static_cast<int>(blockIdx.x);
    for (int i = static_cast<int>(threadIdx.x); i <= kQwenExperts; i += 32) {
        row[i] = scores[static_cast<std::int64_t>(r) * (kQwenExperts + 1) + i];
    }
    __syncwarp();
    d::sparse_moe_select_top8_warp(row, selected, weight, &shared_scale, selected_logit);
    __syncwarp();
    const int lane = static_cast<int>(threadIdx.x);
    if (lane < kTopK) {
        ids[r * kTopK + lane]    = selected[lane];
        alpha[r * kTopK + lane]  = weight[lane];
        logits[r * kTopK + lane] = selected_logit[lane];
    }
    if (lane == 0) { shared[r] = shared_scale; }
}

__global__ void gemma_select_kernel(const float* logits, const float* expert_scale, int* ids,
                                    float* weights) {
    const int r    = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    float logit[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        logit[item] = logits[static_cast<std::int64_t>(r) * kGemmaExperts + lane + 32 * item];
    }
    const g4::Top8 top = g4::select_top8(logit);
    if (lane < kTopK) {
        ids[r * kTopK + lane]     = top.id[lane];
        weights[r * kTopK + lane] = g4::top8_weight(top, lane, expert_scale);
    }
}

struct Reference {
    std::vector<int> ids;
    std::vector<double> weights; // before any per-expert gain
};

// Rank by value, an exact tie to the lower id (or, for the control, the higher one); weights are
// the softmax over the selected eight (or, for the control, over every expert).
Reference reference(const float* row, int experts, bool lower_id_wins = true,
                    bool over_all = false) {
    std::vector<int> order(static_cast<std::size_t>(experts));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (row[a] != row[b]) { return row[a] > row[b]; }
        return lower_id_wins ? a < b : a > b;
    });
    Reference out;
    out.ids.assign(order.begin(), order.begin() + kTopK);
    const double top   = row[out.ids[0]];
    double denominator = 0.0;
    for (int k = 0; k < (over_all ? experts : kTopK); ++k) {
        denominator += std::exp(static_cast<double>(row[order[static_cast<std::size_t>(k)]]) - top);
    }
    for (int k = 0; k < kTopK; ++k) {
        out.weights.push_back(std::exp(static_cast<double>(row[out.ids[k]]) - top) / denominator);
    }
    return out;
}

bool weight_close(double got, double want) {
    return std::isfinite(got) && std::abs(got - want) <= 1e-6 * std::abs(want) + 1e-38;
}

// Rows of logits for `experts` experts, one family after another; `names` labels each row.
std::vector<float> make_rows(int experts, std::vector<std::string>& names) {
    std::mt19937 rng(20260925U);
    std::vector<float> rows;
    const auto add = [&](const std::string& name, const std::function<float(int)>& value) {
        for (int e = 0; e < experts; ++e) { rows.push_back(value(e)); }
        names.push_back(name);
    };
    const auto uniform = [&](double lo, double hi) {
        return static_cast<float>(std::uniform_real_distribution<double>(lo, hi)(rng));
    };
    for (int i = 0; i < 8; ++i) {
        add("U[-18,-1]", [&](int) { return uniform(-18.0, -1.0); });
    }
    for (int i = 0; i < 4; ++i) {
        add("U[-1e4,-1e3]", [&](int) { return uniform(-1.0e4, -1.0e3); });
    }
    for (int i = 0; i < 4; ++i) {
        add("mixed decades", [&](int) {
            switch (rng() % 4) {
            case 0:
                return uniform(-1.0e4, -1.0e3);
            case 1:
                return uniform(-18.0, -1.0);
            case 2:
                return uniform(-1.0, 1.0);
            default:
                return uniform(0.0, 5.0);
            }
        });
    }
    for (const float level : {-3.5F, 0.0F, -1.0e4F, 7.25F}) {
        add("all equal " + std::to_string(level), [&](int) { return level; });
    }
    // Seven clear winners, then five equal logits competing for rank 8 at ids that sit in
    // different lanes and items of the warp; the lowest of them must win.
    const std::array<std::array<int, 5>, 4> boundary{{{experts - 1, 13, experts / 2 + 7, 77, 40},
                                                      {31, 32, 63, 64, 0},
                                                      {experts - 32, experts - 33, 95, 96, 1},
                                                      {5, experts - 1, 6, 90, 33}}};
    for (const auto& tied : boundary) {
        std::vector<float> row(static_cast<std::size_t>(experts));
        for (int e = 0; e < experts; ++e) { row[e] = uniform(-18.0, -9.0); }
        int placed = 0;
        for (int e = experts - 3; placed < 7; e -= 5) {
            if (std::find(tied.begin(), tied.end(), e) == tied.end()) {
                row[e] = -1.0F - 0.5F * static_cast<float>(placed++);
            }
        }
        for (const int e : tied) { row[e] = -6.0F; }
        add("tie at rank 8", [&](int e) { return row[e]; });
    }
    // Ties inside the top eight: four pairs, ordered by id within each pair.
    {
        std::vector<float> row(static_cast<std::size_t>(experts));
        for (int e = 0; e < experts; ++e) { row[e] = uniform(-18.0, -9.0); }
        const int pairs[4][2] = {{experts - 1, 3}, {64, 65}, {31, 96}, {0, experts / 2}};
        for (int p = 0; p < 4; ++p) {
            row[pairs[p][0]] = row[pairs[p][1]] = -1.0F - static_cast<float>(p);
        }
        add("ties inside top 8", [&](int e) { return row[e]; });
    }
    for (int i = 0; i < 24; ++i) {
        add("five levels", [&](int) { return -1.0F - 0.5F * static_cast<float>(rng() % 5); });
    }
    add("+0 against -0", [&](int e) { return (e % 3 == 0) ? -0.0F : (e % 3 == 1 ? 0.0F : -2.0F); });
    return rows;
}

struct Tally {
    int failures      = 0;
    int rows          = 0;
    int tiny_weights  = 0; // weights below 1e-5 compared
    int tie_rule_diff = 0; // rows where the opposite tie rule selects differently
    int over_all_diff = 0; // rows where normalising over every expert gives other weights
};

void compare_row(const std::string& label, int row, const int* ids, const double* weights,
                 const Reference& want, const Reference& other_tie, const Reference& over_all,
                 Tally& tally) {
    ++tally.rows;
    bool ok = true;
    for (int k = 0; k < kTopK; ++k) {
        if (ids[k] != want.ids[static_cast<std::size_t>(k)]) { ok = false; }
        if (!weight_close(weights[k], want.weights[static_cast<std::size_t>(k)])) { ok = false; }
        if (want.weights[static_cast<std::size_t>(k)] < 1e-5) { ++tally.tiny_weights; }
    }
    double sum = 0.0;
    for (int k = 0; k < kTopK; ++k) { sum += weights[k]; }
    if (!(std::abs(sum - 1.0) <= 1e-6)) { ok = false; }
    if (!ok) {
        ++tally.failures;
        std::cerr << label << " row " << row << ": ids";
        for (int k = 0; k < kTopK; ++k) { std::cerr << ' ' << ids[k]; }
        std::cerr << " want";
        for (int k = 0; k < kTopK; ++k) {
            std::cerr << ' ' << want.ids[static_cast<std::size_t>(k)];
        }
        std::cerr << "; weight sum " << sum << '\n';
    }
    bool differs = false;
    for (int k = 0; k < kTopK; ++k) {
        differs = differs || ids[k] != other_tie.ids[static_cast<std::size_t>(k)];
    }
    tally.tie_rule_diff += differs ? 1 : 0;
    bool weights_differ = false;
    for (int k = 0; k < kTopK; ++k) {
        weights_differ = weights_differ ||
                         !weight_close(weights[k], over_all.weights[static_cast<std::size_t>(k)]);
    }
    tally.over_all_diff += weights_differ ? 1 : 0;
}

int report(const char* router, const Tally& tally, bool deterministic) {
    std::printf("%s: %d rows, %d mismatched, %d weights below 1e-5 compared, runs %s\n", router,
                tally.rows, tally.failures, tally.tiny_weights,
                deterministic ? "bitwise identical" : "DIFFER");
    std::printf("  CONTROL higher id wins ties: %d rows select differently (must be > 0)\n",
                tally.tie_rule_diff);
    std::printf("  CONTROL softmax over all experts: %d rows weigh differently (must be > 0)\n",
                tally.over_all_diff);
    return tally.failures + (deterministic ? 0 : 1) + (tally.tie_rule_diff > 0 ? 0 : 1) +
           (tally.over_all_diff > 0 ? 0 : 1) + (tally.tiny_weights > 0 ? 0 : 1);
}

int check_qwen() {
    std::vector<std::string> names;
    std::vector<float> logits = make_rows(kQwenExperts, names);
    const int rows            = static_cast<int>(names.size());
    // One more score a row: the shared-expert gate, whose sigmoid the routine also returns.
    std::vector<float> scores;
    std::mt19937 rng(7U);
    for (int r = 0; r < rows; ++r) {
        scores.insert(scores.end(), logits.begin() + static_cast<std::ptrdiff_t>(r) * kQwenExperts,
                      logits.begin() + static_cast<std::ptrdiff_t>(r + 1) * kQwenExperts);
        scores.push_back(std::uniform_real_distribution<float>(-12.0F, 12.0F)(rng));
    }
    DeviceBuffer device_scores = to_device(scores);
    DeviceBuffer ids(static_cast<std::size_t>(rows) * kTopK * sizeof(int));
    DeviceBuffer alpha(static_cast<std::size_t>(rows) * kTopK * sizeof(float));
    DeviceBuffer shared(static_cast<std::size_t>(rows) * sizeof(float));
    DeviceBuffer selected(static_cast<std::size_t>(rows) * kTopK * sizeof(float));
    const auto run = [&] {
        ids.fill(0xff);
        alpha.fill(0xff);
        qwen_select_kernel<<<rows, 32>>>(static_cast<const float*>(device_scores.p),
                                         static_cast<int*>(ids.p), static_cast<float*>(alpha.p),
                                         static_cast<float*>(shared.p),
                                         static_cast<float*>(selected.p));
        cuda_check(cudaGetLastError(), "qwen select launch");
        cuda_synchronize();
        return std::pair{from_device<int>(ids, static_cast<std::size_t>(rows) * kTopK),
                         from_device<float>(alpha, static_cast<std::size_t>(rows) * kTopK)};
    };
    const auto [got_ids, got_alpha]     = run();
    const auto [again_ids, again_alpha] = run();
    const bool deterministic =
        got_ids == again_ids &&
        std::memcmp(got_alpha.data(), again_alpha.data(), got_alpha.size() * sizeof(float)) == 0;
    const auto got_shared = from_device<float>(shared, static_cast<std::size_t>(rows));
    Tally tally;
    for (int r = 0; r < rows; ++r) {
        const float* row = logits.data() + static_cast<std::size_t>(r) * kQwenExperts;
        std::vector<double> weights(kTopK);
        for (int k = 0; k < kTopK; ++k) {
            weights[k] = got_alpha[static_cast<std::size_t>(r) * kTopK + k];
        }
        compare_row("qwen " + names[static_cast<std::size_t>(r)], r, got_ids.data() + r * kTopK,
                    weights.data(), reference(row, kQwenExperts),
                    reference(row, kQwenExperts, false), reference(row, kQwenExperts, true, true),
                    tally);
        const double gate = scores[static_cast<std::size_t>(r) * (kQwenExperts + 1) + kQwenExperts];
        const double want = 1.0 / (1.0 + std::exp(-gate));
        if (!(std::abs(got_shared[static_cast<std::size_t>(r)] - want) <= 1e-6 * want)) {
            std::cerr << "qwen row " << r << ": shared gate "
                      << got_shared[static_cast<std::size_t>(r)] << " want " << want << '\n';
            ++tally.failures;
        }
    }
    return report("qwen sparse_moe_select_top8_warp (256 experts)", tally, deterministic);
}

int check_gemma() {
    std::vector<std::string> names;
    std::vector<float> logits = make_rows(kGemmaExperts, names);
    const int rows            = static_cast<int>(names.size());
    std::vector<float> expert_scale(kGemmaExperts);
    std::mt19937 rng(11U);
    for (float& value : expert_scale) {
        value = std::uniform_real_distribution<float>(0.5F, 2.0F)(rng);
    }
    DeviceBuffer device_logits = to_device(logits);
    DeviceBuffer device_scale  = to_device(expert_scale);
    DeviceBuffer ids(static_cast<std::size_t>(rows) * kTopK * sizeof(int));
    DeviceBuffer weights(static_cast<std::size_t>(rows) * kTopK * sizeof(float));
    const auto run = [&] {
        ids.fill(0xff);
        weights.fill(0xff);
        gemma_select_kernel<<<rows, 32>>>(static_cast<const float*>(device_logits.p),
                                          static_cast<const float*>(device_scale.p),
                                          static_cast<int*>(ids.p), static_cast<float*>(weights.p));
        cuda_check(cudaGetLastError(), "gemma select launch");
        cuda_synchronize();
        return std::pair{from_device<int>(ids, static_cast<std::size_t>(rows) * kTopK),
                         from_device<float>(weights, static_cast<std::size_t>(rows) * kTopK)};
    };
    const auto [got_ids, got_weights]     = run();
    const auto [again_ids, again_weights] = run();
    const bool deterministic =
        got_ids == again_ids && std::memcmp(got_weights.data(), again_weights.data(),
                                            got_weights.size() * sizeof(float)) == 0;
    Tally tally;
    for (int r = 0; r < rows; ++r) {
        const float* row = logits.data() + static_cast<std::size_t>(r) * kGemmaExperts;
        // The routine returns p_k / sum_top p * expert_scale[id_k]; the gain is divided out
        // here (by the id the routine chose) so the renormalised softmax is what is compared.
        std::vector<double> renormalised(kTopK);
        for (int k = 0; k < kTopK; ++k) {
            const int id = got_ids[static_cast<std::size_t>(r) * kTopK + k];
            renormalised[k] =
                id >= 0 && id < kGemmaExperts
                    ? got_weights[static_cast<std::size_t>(r) * kTopK + k] /
                          static_cast<double>(expert_scale[static_cast<std::size_t>(id)])
                    : NAN;
        }
        compare_row("gemma " + names[static_cast<std::size_t>(r)], r, got_ids.data() + r * kTopK,
                    renormalised.data(), reference(row, kGemmaExperts),
                    reference(row, kGemmaExperts, false), reference(row, kGemmaExperts, true, true),
                    tally);
    }
    return report("gemma4 select_top8 + top8_weight (128 experts)", tally, deterministic);
}

int check_nan() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    int failures    = 0;
    for (int pattern = 0; pattern < 2; ++pattern) {
        for (const bool gemma : {false, true}) {
            const int experts = gemma ? kGemmaExperts : kQwenExperts;
            const int stride  = gemma ? kGemmaExperts : kQwenExperts + 1;
            std::vector<float> row(static_cast<std::size_t>(stride), nan);
            if (!gemma) { row[static_cast<std::size_t>(kQwenExperts)] = 0.0F; } // shared gate
            if (pattern == 1) {
                for (int e = 0; e < experts; e += 21) {
                    row[static_cast<std::size_t>(e)] = -0.5F * e;
                }
            }
            std::vector<float> ranked(row.begin(), row.begin() + experts);
            for (float& value : ranked) {
                if (std::isnan(value)) { value = -std::numeric_limits<float>::infinity(); }
            }
            const std::vector<int> want = reference(ranked.data(), experts).ids;
            DeviceBuffer device_row     = to_device(row);
            DeviceBuffer ids(kTopK * sizeof(int));
            DeviceBuffer weights(kTopK * sizeof(float));
            ids.fill(0xff);
            if (gemma) {
                const std::vector<float> scale(static_cast<std::size_t>(kGemmaExperts), 1.0F);
                DeviceBuffer device_scale = to_device(scale);
                gemma_select_kernel<<<1, 32>>>(static_cast<const float*>(device_row.p),
                                               static_cast<const float*>(device_scale.p),
                                               static_cast<int*>(ids.p),
                                               static_cast<float*>(weights.p));
                cuda_check(cudaGetLastError(), "gemma NaN select launch");
                cuda_synchronize();
            } else {
                DeviceBuffer shared(sizeof(float));
                DeviceBuffer logits(kTopK * sizeof(float));
                qwen_select_kernel<<<1, 32>>>(
                    static_cast<const float*>(device_row.p), static_cast<int*>(ids.p),
                    static_cast<float*>(weights.p), static_cast<float*>(shared.p),
                    static_cast<float*>(logits.p));
                cuda_check(cudaGetLastError(), "qwen NaN select launch");
                cuda_synchronize();
            }
            const std::vector<int> got = from_device<int>(ids, kTopK);
            if (got != want) {
                std::cerr << (gemma ? "gemma" : "qwen") << " NaN pattern " << pattern << ": ids";
                for (const int id : got) { std::cerr << ' ' << id; }
                std::cerr << " want";
                for (const int id : want) { std::cerr << ' ' << id; }
                std::cerr << '\n';
                ++failures;
            }
        }
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " NaN router logits select in-range experts\n";
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
        failures += check_qwen();
        failures += check_gemma();
        failures += check_nan();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " moe routing\n";
    return failures == 0 ? 0 : 1;
}
