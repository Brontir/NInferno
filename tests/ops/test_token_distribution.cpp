#include "ninferno/ops/token_distribution.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

constexpr ReductionCriterion kLogprobCriterion{
    /*relative_l2=*/2.0e-5,
    /*gross_absolute=*/2.0e-4,
    /*gross_relative_to_max_reference=*/0.0,
};

using Logits = std::vector<std::uint16_t>;

Logits make_random(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t columns,
                   float span) {
    Logits logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t c = 0; c < columns; ++c) {
        const std::size_t base = static_cast<std::size_t>(c) * physical_rows;
        for (std::int32_t r = 0; r < valid_rows; ++r) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(r) * 1664525u +
                                        static_cast<std::uint32_t>(c + 1) * 1013904223u;
            const float unit = static_cast<float>(mixed % 6144u) / 6144.0f;
            logits[base + r] = f32_to_bf16(-span + 2.0f * span * unit);
        }
        // Rows past valid_rows would win every selection if they took part.
        for (std::int32_t r = valid_rows; r < physical_rows; ++r)
            logits[base + r] = f32_to_bf16(96.0f);
    }
    return logits;
}

// One sharp row per column over a flat tail: most of K falls outside the window.
Logits make_peaked(std::int32_t rows, std::int32_t columns) {
    Logits logits(static_cast<std::size_t>(rows) * columns, f32_to_bf16(-20.0f));
    for (std::int32_t c = 0; c < columns; ++c) {
        const std::size_t base              = static_cast<std::size_t>(c) * rows;
        logits[base + (c * 31) % rows]      = f32_to_bf16(25.0f);
        logits[base + (c * 31 + 7) % rows]  = f32_to_bf16(9.0f);
        logits[base + (c * 31 + 11) % rows] = f32_to_bf16(-4.5f);
    }
    return logits;
}

// A few distinct leading rows over more than the candidate buffer of rows tied below them: the
// leaders must always be listed, only the tied rows are chosen arbitrarily.
Logits make_leaders_over_ties(std::int32_t rows, std::int32_t columns, std::int32_t leaders) {
    Logits logits(static_cast<std::size_t>(rows) * columns, f32_to_bf16(5.0f));
    for (std::int32_t c = 0; c < columns; ++c) {
        const std::size_t base = static_cast<std::size_t>(c) * rows;
        for (std::int32_t i = 0; i < leaders; ++i) {
            // Spread over the column, late rows included, so the atomic order cannot favour them.
            logits[base + (rows - 1 - i * 397 - c * 13) % rows] = f32_to_bf16(10.0f + i);
        }
    }
    return logits;
}

// The same near zero, where BF16 values lie closer than a float window's last bisection step: two
// rows just above more than the buffer of rows tied at 1e-4 must be listed, below three leaders.
Logits make_near_zero_ties(std::int32_t rows, std::int32_t columns) {
    Logits logits(static_cast<std::size_t>(rows) * columns, f32_to_bf16(1.0e-4f));
    for (std::int32_t c = 0; c < columns; ++c) {
        const std::size_t base = static_cast<std::size_t>(c) * rows;
        for (std::int32_t i = 0; i < 3; ++i) {
            logits[base + (rows - 1 - i * 397 - c * 13) % rows] = f32_to_bf16(10.0f + i);
        }
        logits[base + (rows - 7 - c * 11) % rows]   = f32_to_bf16(1.5e-4f);
        logits[base + (rows - 900 - c * 17) % rows] = f32_to_bf16(1.5e-4f);
    }
    return logits;
}

double lse_of(const Logits& logits, std::size_t base, std::int32_t valid_rows) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::int32_t r = 0; r < valid_rows; ++r) {
        maximum = std::max(maximum, static_cast<double>(bf16_to_f32(logits[base + r])));
    }
    double sum = 0.0;
    for (std::int32_t r = 0; r < valid_rows; ++r) {
        sum += std::exp(static_cast<double>(bf16_to_f32(logits[base + r])) - maximum);
    }
    return maximum + std::log(sum);
}

// The listed ids and their log-probabilities; -1 / -inf past the window.
void topk_oracle(const Logits& logits, std::int32_t physical_rows, std::int32_t valid_rows,
                 std::int32_t columns, std::int32_t k, std::vector<std::int32_t>& ids,
                 std::vector<double>& logprobs) {
    ids.assign(static_cast<std::size_t>(k) * columns, -1);
    logprobs.assign(ids.size(), -std::numeric_limits<double>::infinity());
    for (std::int32_t c = 0; c < columns; ++c) {
        const std::size_t base = static_cast<std::size_t>(c) * physical_rows;
        std::vector<std::int32_t> order(static_cast<std::size_t>(valid_rows));
        std::iota(order.begin(), order.end(), 0);
        const auto value = [&](std::int32_t r) { return bf16_to_f32(logits[base + r]); };
        std::stable_sort(order.begin(), order.end(),
                         [&](std::int32_t a, std::int32_t b) { return value(a) > value(b); });
        const double lse  = lse_of(logits, base, valid_rows);
        const float floor = value(order[0]) - ops::kTokenTopkWindow;
        for (std::int32_t i = 0; i < k && i < valid_rows; ++i) {
            if (value(order[i]) < floor) break;
            ids[static_cast<std::size_t>(c) * k + i]      = order[i];
            logprobs[static_cast<std::size_t>(c) * k + i] = value(order[i]) - lse;
        }
    }
}

// -inf slots must match exactly; the finite rest is compared as a reduction.
int verify_logprobs(const std::string& label, const std::vector<float>& got,
                    const std::vector<double>& expected) {
    std::vector<double> got_finite, expected_finite;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const bool g = std::isinf(got[i]) && got[i] < 0, e = std::isinf(expected[i]);
        if (g != e) {
            std::cerr << label << ": slot " << i << " got " << got[i] << ", expected "
                      << expected[i] << '\n';
            return 1;
        }
        if (!e) {
            got_finite.push_back(got[i]);
            expected_finite.push_back(expected[i]);
        }
    }
    return expected_finite.empty()
               ? 0
               : verify_reduction(label, got_finite, expected_finite, kLogprobCriterion);
}

int run_case(const std::string& label, std::int32_t physical_rows, std::int32_t valid_rows,
             std::int32_t columns, std::int32_t k, const Logits& logits, bool exact_ids = true) {
    std::vector<std::int32_t> expected_ids;
    std::vector<double> expected_logprobs;
    topk_oracle(logits, physical_rows, valid_rows, columns, k, expected_ids, expected_logprobs);
    const std::size_t slots = static_cast<std::size_t>(k) * columns;

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_ids(slots * sizeof(std::int32_t));
    GuardedDeviceBuffer device_logprobs(slots * sizeof(float));
    GuardedDeviceBuffer device_gathered(slots * sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    device_ids.fill(0xcd);
    device_logprobs.fill(0xcd);
    device_gathered.fill(0xcd);
    Tensor tl(device_logits.data(), DType::BF16, {physical_rows, columns});
    Tensor ti(device_ids.data(), DType::I32, {k, columns});
    Tensor tp(device_logprobs.data(), DType::FP32, {k, columns});
    Tensor tg(device_gathered.data(), DType::FP32, {k, columns});
    ops::logprob_topk(tl, valid_rows, ti, tp, nullptr);
    // The listed ids read back through gather_logprobs give the same log-probabilities.
    ops::gather_logprobs(tl, valid_rows, ti, tg, nullptr);
    cuda_synchronize();

    const auto ids = from_device<std::int32_t>(device_ids.data(), slots);
    int failures   = 0;
    if (exact_ids) {
        failures += verify_exact((label + " ids").c_str(), ids, expected_ids);
    } else {
        // Ties past the candidate buffer: any of the tied rows, each listed once, at its value;
        // a row strictly above the K-th value is always listed, in its place.
        for (std::int32_t c = 0; c < columns; ++c) {
            const double kth = expected_logprobs[static_cast<std::size_t>(c) * k + k - 1];
            for (std::int32_t i = 0; i < k; ++i) {
                const std::size_t slot = static_cast<std::size_t>(c) * k + i;
                if (expected_logprobs[slot] > kth && ids[slot] != expected_ids[slot]) {
                    std::cerr << label << ": column " << c << " slot " << i << " lists row "
                              << ids[slot] << ", expected the leader " << expected_ids[slot]
                              << '\n';
                    ++failures;
                }
            }
            std::vector<std::int32_t> listed(ids.begin() + c * k, ids.begin() + (c + 1) * k);
            std::sort(listed.begin(), listed.end());
            if (std::adjacent_find(listed.begin(), listed.end()) != listed.end()) {
                std::cerr << label << ": column " << c << " lists a row twice\n";
                ++failures;
            }
        }
    }
    failures += verify_logprobs(
        label + " logprobs", from_device<float>(device_logprobs.data(), slots), expected_logprobs);
    failures += verify_logprobs(
        label + " gathered", from_device<float>(device_gathered.data(), slots), expected_logprobs);
    failures +=
        verify_exact((label + " preserves logits").c_str(),
                     from_device<std::uint16_t>(device_logits.data(), logits.size()), logits);
    failures += device_logits.verify_guards(label + " logits guards");
    failures += device_ids.verify_guards(label + " ids guards");
    failures += device_logprobs.verify_guards(label + " logprobs guards");
    failures += device_gathered.verify_guards(label + " gathered guards");
    return failures;
}

template <class Function>
int expect_invalid(const char* label, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int run_validation_cases() {
    DeviceBuffer logits_data(8 * 3 * sizeof(std::uint16_t));
    DeviceBuffer ids_data(4 * 3 * sizeof(std::int32_t));
    DeviceBuffer out_data(4 * 3 * sizeof(float));
    Tensor logits(logits_data.p, DType::BF16, {8, 3});
    Tensor ids(ids_data.p, DType::I32, {4, 3});
    Tensor out(out_data.p, DType::FP32, {4, 3});
    int failures = 0;
    failures += expect_invalid("logprob_topk rejects valid_rows=0",
                               [&] { ops::logprob_topk(logits, 0, ids, out, nullptr); });
    failures += expect_invalid("logprob_topk rejects valid_rows>physical_rows",
                               [&] { ops::logprob_topk(logits, 9, ids, out, nullptr); });
    failures += expect_invalid("logprob_topk rejects K past kTokenTopkMax", [&] {
        DeviceBuffer wide_ids(65 * 3 * sizeof(std::int32_t)), wide_out(65 * 3 * sizeof(float));
        Tensor wi(wide_ids.p, DType::I32, {65, 3});
        Tensor wo(wide_out.p, DType::FP32, {65, 3});
        ops::logprob_topk(logits, 8, wi, wo, nullptr);
    });
    failures += expect_invalid("logprob_topk rejects a column mismatch", [&] {
        Tensor short_out(out_data.p, DType::FP32, {4, 2});
        ops::logprob_topk(logits, 8, ids, short_out, nullptr);
    });
    failures += expect_invalid("logprob_topk rejects ids dtype", [&] {
        Tensor wrong(ids_data.p, DType::FP32, {4, 3});
        ops::logprob_topk(logits, 8, wrong, out, nullptr);
    });
    failures += expect_invalid("gather_logprobs rejects an aliased output", [&] {
        Tensor alias(ids_data.p, DType::FP32, {4, 3});
        ops::gather_logprobs(logits, 8, ids, alias, nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    // A full vocabulary whose window holds most rows: the threshold narrows by bisection.
    failures += run_case("topk full vocabulary K=32", 248320, 248077, 3, 32,
                         make_random(248320, 248077, 3, 24.0f));
    failures += run_case("topk full vocabulary K=64", 151936, 151669, 2, 64,
                         make_random(151936, 151669, 2, 8.0f));
    failures += run_case("topk non-aligned rows K=1, C=1025", 523, 509, 1025, 1,
                         make_random(523, 509, 1025, 40.0f));
    failures += run_case("topk peaked K=8", 4099, 4099, 37, 8, make_peaked(4099, 37));
    // Ties: the lowest rows first.
    failures +=
        run_case("topk uniform 257 rows K=16", 263, 257, 9, 16, make_random(263, 257, 9, 0.0f));
    // More tied rows than the candidate buffer: any of them, once each.
    failures += run_case("topk uniform 5000 rows K=8", 5000, 5000, 4, 8,
                         make_random(5000, 5000, 4, 0.0f), false);
    failures += run_case("topk 3 leaders over 5000 tied rows K=8", 5003, 5003, 6, 8,
                         make_leaders_over_ties(5003, 6, 3), false);
    failures += run_case("topk 5 leaders over 5000 rows tied near zero K=8", 5003, 5003, 6, 8,
                         make_near_zero_ties(5003, 6), false);
    failures += run_case("topk one valid row", 13, 1, 7, 4, make_random(13, 1, 7, 3.0f));
    failures += run_validation_cases();
    std::cout << (failures ? "FAIL" : "OK") << " token_distribution\n";
    return failures ? 1 : 0;
}
