// Public-contract qualification for sample().
//
// The deterministic branch is checked exactly against an independent CPU
// argmax.  The stochastic branch is checked against one FP64 mathematical
// distribution oracle built from the BF16 values represented at the public
// input.  The test never reproduces the device RNG algorithm or uses another
// production path as a golden.
#include "ninferno/ops/sampling.h"
#include "ninferno/types.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

struct Candidate {
    double adjusted = 0.0;
    int token       = 0;
};

struct Distribution {
    std::vector<int> tokens;
    std::vector<double> probabilities;
};

struct RunResult {
    std::vector<int> tokens;
    std::vector<std::vector<int>> counts;
    int integrity_failures = 0;
};

bool same_config(const ops::SamplingConfig& a, const ops::SamplingConfig& b) {
    return a.temperature == b.temperature && a.top_k == b.top_k && a.top_p == b.top_p &&
           a.min_p == b.min_p && a.presence_penalty == b.presence_penalty &&
           a.frequency_penalty == b.frequency_penalty && a.seed == b.seed &&
           a.token_counts == b.token_counts;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<float> repeat_column(const std::vector<float>& column, int columns) {
    std::vector<float> logits(column.size() * static_cast<std::size_t>(columns));
    for (int t = 0; t < columns; ++t) {
        std::copy(column.begin(), column.end(),
                  logits.begin() + static_cast<std::ptrdiff_t>(t) * column.size());
    }
    return logits;
}

std::vector<int> greedy_oracle(const std::vector<float>& logits, int physical_rows,
                               int token_domain, int columns, const ops::SamplingConfig& config,
                               const std::vector<int>& counts) {
    std::vector<int> expected(static_cast<std::size_t>(columns));
    for (int t = 0; t < columns; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * physical_rows;
        int best               = 0;
        for (int token = 1; token < token_domain; ++token) {
            const auto adjusted = [&](int candidate) {
                const int count = counts[static_cast<std::size_t>(candidate)];
                return logits[base + candidate] - config.presence_penalty * (count > 0) -
                       config.frequency_penalty * count;
            };
            if (adjusted(token) > adjusted(best)) { best = token; }
        }
        expected[static_cast<std::size_t>(t)] = best;
    }
    return expected;
}

Distribution distribution_oracle(const std::vector<float>& column, int token_domain,
                                 const ops::SamplingConfig& config,
                                 const std::vector<int>* counts = nullptr) {
    std::vector<Candidate> candidates(static_cast<std::size_t>(token_domain));
    for (int token = 0; token < token_domain; ++token) {
        const int count = counts == nullptr ? 0 : (*counts)[static_cast<std::size_t>(token)];
        double adjusted = static_cast<double>(column[static_cast<std::size_t>(token)]);
        if (count > 0) { adjusted -= static_cast<double>(config.presence_penalty); }
        adjusted -= static_cast<double>(config.frequency_penalty) * static_cast<double>(count);
        candidates[static_cast<std::size_t>(token)] = {adjusted, token};
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.adjusted != b.adjusted) { return a.adjusted > b.adjusted; }
        return a.token < b.token;
    });

    // The multi-block route's candidate cap (kMaximumTopK); the tests' launches take that route.
    int cap = ninferno::kMaximumTopK;
    if (config.top_k > 0 && config.top_k < cap) { cap = config.top_k; }
    cap = std::min(cap, token_domain);
    candidates.resize(static_cast<std::size_t>(cap));

    std::vector<double> weights(static_cast<std::size_t>(cap));
    const double max_scaled = candidates.front().adjusted / config.temperature;
    double total_weight     = 0.0;
    for (int rank = 0; rank < cap; ++rank) {
        const double weight = std::exp(
            candidates[static_cast<std::size_t>(rank)].adjusted / config.temperature - max_scaled);
        weights[static_cast<std::size_t>(rank)] = weight;
        total_weight += weight;
    }

    const bool use_min_p     = config.min_p > 0.0f;
    const bool use_top_p     = config.top_p < 1.0f;
    const double min_weight  = static_cast<double>(config.min_p) * weights.front();
    const double top_p_limit = static_cast<double>(config.top_p) * total_weight;
    double cumulative        = 0.0;
    int support              = 0;
    for (int rank = 0; rank < cap; ++rank) {
        if (use_min_p && weights[static_cast<std::size_t>(rank)] < min_weight) { break; }
        cumulative += weights[static_cast<std::size_t>(rank)];
        support = rank + 1;
        if (use_top_p && cumulative >= top_p_limit) { break; }
    }
    support = std::max(support, 1);

    Distribution out;
    out.tokens.reserve(static_cast<std::size_t>(support));
    out.probabilities.reserve(static_cast<std::size_t>(support));
    double kept_weight = 0.0;
    for (int rank = 0; rank < support; ++rank) {
        kept_weight += weights[static_cast<std::size_t>(rank)];
    }
    for (int rank = 0; rank < support; ++rank) {
        out.tokens.push_back(candidates[static_cast<std::size_t>(rank)].token);
        out.probabilities.push_back(weights[static_cast<std::size_t>(rank)] / kept_weight);
    }
    return out;
}

RunResult run_batch(const std::vector<float>& logits, int physical_rows, int token_domain,
                    std::vector<ops::SamplingConfig> configs,
                    const std::vector<int>& logical_positions, int purpose,
                    const std::vector<std::vector<int>>& initial_counts = {}) {
    const int batch = static_cast<int>(configs.size());
    if (batch <= 0 || logical_positions.size() != configs.size() ||
        logits.size() != static_cast<std::size_t>(physical_rows) * configs.size() ||
        (!initial_counts.empty() && initial_counts.size() != configs.size())) {
        throw std::invalid_argument("invalid sample batch fixture");
    }
    const std::vector<std::uint16_t> input_bits = bf16_bits(logits);
    DeviceBuffer device_logits                  = to_device(input_bits);
    GuardedDeviceBuffer device_out(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    const std::vector<int> output_sentinel(static_cast<std::size_t>(batch), -777777);
    device_out.copy_from_host(output_sentinel.data(),
                              output_sentinel.size() * sizeof(std::int32_t));

    std::vector<std::unique_ptr<GuardedDeviceBuffer>> device_counts;
    if (!initial_counts.empty()) {
        device_counts.reserve(configs.size());
        for (std::size_t row = 0; row < configs.size(); ++row) {
            if (initial_counts[row].size() != static_cast<std::size_t>(token_domain)) {
                throw std::invalid_argument("invalid sample token-count fixture");
            }
            auto counts = std::make_unique<GuardedDeviceBuffer>(initial_counts[row].size() *
                                                                sizeof(std::int32_t));
            counts->copy_from_host(initial_counts[row].data(), counts->bytes());
            configs[row].token_counts = static_cast<std::int32_t*>(counts->data());
            device_counts.push_back(std::move(counts));
        }
    }
    const std::vector<ops::SamplingConfig> expected_configs = configs;
    DeviceBuffer device_configs                             = to_device(configs);
    DeviceBuffer device_positions                           = to_device(logical_positions);

    Tensor logits_tensor(device_logits.p, DType::BF16, {physical_rows, batch});
    Tensor out_tensor(device_out.data(), DType::I32, {batch});
    Tensor positions_tensor(device_positions.p, DType::I32, {batch});
    const std::size_t workspace_bytes =
        ops::sampling_workspace_capacity_bytes(token_domain, batch, batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::sample(logits_tensor, out_tensor, token_domain,
                static_cast<const ops::SamplingConfig*>(device_configs.p), positions_tensor,
                purpose, workspace, nullptr);
    cuda_synchronize();

    RunResult result;
    result.tokens = from_device<int>(device_out.data(), static_cast<std::size_t>(batch));
    result.integrity_failures += device_out.verify_guards("sample output");
    result.integrity_failures +=
        verify_exact("sample read-only logits",
                     from_device<std::uint16_t>(device_logits, input_bits.size()), input_bits);

    const std::vector<ops::SamplingConfig> actual_configs =
        from_device<ops::SamplingConfig>(device_configs, configs.size());
    for (std::size_t row = 0; row < configs.size(); ++row) {
        if (!same_config(actual_configs[row], expected_configs[row])) {
            std::cerr << "sample modified SamplingConfig row " << row << '\n';
            ++result.integrity_failures;
        }
    }
    result.integrity_failures += verify_exact(
        "sample read-only logical positions",
        from_device<std::int32_t>(device_positions, configs.size()), logical_positions);

    result.counts.reserve(device_counts.size());
    for (std::size_t row = 0; row < device_counts.size(); ++row) {
        result.counts.push_back(
            from_device<int>(device_counts[row]->data(), static_cast<std::size_t>(token_domain)));
        result.integrity_failures += device_counts[row]->verify_guards("sample token_counts");
    }
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "sample workspace query/execution high-water mismatch\n";
        ++result.integrity_failures;
    }
    return result;
}

RunResult run_homogeneous_batch(const std::vector<float>& logits, int physical_rows,
                                int token_domain, int batch, ops::SamplingConfig config,
                                int first_position, int purpose,
                                const std::vector<int>* initial_counts = nullptr) {
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch), config);
    std::vector<int> positions(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        positions[static_cast<std::size_t>(row)] = first_position + row;
    }
    std::vector<std::vector<int>> row_counts;
    if (initial_counts != nullptr) {
        row_counts.assign(static_cast<std::size_t>(batch), *initial_counts);
    }
    return run_batch(logits, physical_rows, token_domain, std::move(configs), positions, purpose,
                     row_counts);
}

RunResult run_repeated(const std::vector<float>& column, int token_domain, int total, int batch,
                       ops::SamplingConfig config, int position, int purpose) {
    if (batch <= 0 || total <= 0 || total % batch != 0) {
        throw std::invalid_argument("repeated sample count must be a positive batch multiple");
    }
    const int physical_rows                     = static_cast<int>(column.size());
    const std::vector<float> logits             = repeat_column(column, batch);
    const std::vector<std::uint16_t> input_bits = bf16_bits(logits);
    DeviceBuffer device_logits                  = to_device(input_bits);
    GuardedDeviceBuffer collected(static_cast<std::size_t>(total) * sizeof(std::int32_t));
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch), config);
    const std::vector<ops::SamplingConfig> expected_configs = configs;
    DeviceBuffer device_configs                             = to_device(configs);
    std::vector<int> positions(static_cast<std::size_t>(total));
    for (int i = 0; i < total; ++i) { positions[static_cast<std::size_t>(i)] = position + i; }
    DeviceBuffer device_positions = to_device(positions);

    Tensor logits_tensor(device_logits.p, DType::BF16, {physical_rows, batch});
    const std::size_t workspace_bytes =
        ops::sampling_workspace_capacity_bytes(token_domain, batch, batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    for (int produced = 0; produced < total; produced += batch) {
        auto* out = static_cast<std::int32_t*>(collected.data()) + produced;
        auto* pos = static_cast<std::int32_t*>(device_positions.p) + produced;
        Tensor out_tensor(out, DType::I32, {batch});
        Tensor positions_tensor(pos, DType::I32, {batch});
        ops::sample(logits_tensor, out_tensor, token_domain,
                    static_cast<const ops::SamplingConfig*>(device_configs.p), positions_tensor,
                    purpose, workspace, nullptr);
    }
    cuda_synchronize();

    RunResult result;
    result.tokens = from_device<int>(collected.data(), static_cast<std::size_t>(total));
    result.integrity_failures += collected.verify_guards("sample repeated output");
    result.integrity_failures +=
        verify_exact("sample repeated read-only logits",
                     from_device<std::uint16_t>(device_logits, input_bits.size()), input_bits);
    const std::vector<ops::SamplingConfig> actual_configs =
        from_device<ops::SamplingConfig>(device_configs, configs.size());
    for (std::size_t row = 0; row < configs.size(); ++row) {
        if (!same_config(actual_configs[row], expected_configs[row])) {
            std::cerr << "sample repeated modified SamplingConfig row " << row << '\n';
            ++result.integrity_failures;
        }
    }
    result.integrity_failures +=
        verify_exact("sample repeated read-only logical positions",
                     from_device<std::int32_t>(device_positions, positions.size()), positions);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "sample repeated workspace query/execution high-water mismatch\n";
        ++result.integrity_failures;
    }
    return result;
}

int verify_distribution(const char* label, const std::vector<int>& samples,
                        const Distribution& expected) {
    std::vector<int> observed(expected.tokens.size(), 0);
    for (int token : samples) {
        const auto it = std::find(expected.tokens.begin(), expected.tokens.end(), token);
        if (it == expected.tokens.end()) {
            std::cerr << label << ": sampled token " << token << " outside oracle support\n";
            return 1;
        }
        ++observed[static_cast<std::size_t>(it - expected.tokens.begin())];
    }

    const double n              = static_cast<double>(samples.size());
    double max_standardized_gap = 0.0;
    for (std::size_t i = 0; i < expected.tokens.size(); ++i) {
        const double probability = expected.probabilities[i];
        const double frequency   = static_cast<double>(observed[i]) / n;
        const double sigma       = std::sqrt(probability * (1.0 - probability) / n);
        const double limit       = 7.0 * sigma + 2.0 / n;
        const double gap         = std::abs(frequency - probability);
        if (gap > limit) {
            std::cerr << label << ": token=" << expected.tokens[i] << " frequency=" << frequency
                      << " oracle=" << probability << " gap=" << gap << " limit=" << limit << '\n';
            return 1;
        }
        if (sigma > 0.0) { max_standardized_gap = std::max(max_standardized_gap, gap / sigma); }
    }
    std::cout << "    " << label << " FP64 distribution match (max z=" << max_standardized_gap
              << ")\n";
    return 0;
}

// physical_rows == token_domain: the logits of a head whose rows end inside a 128-row tile
// (nvidia/Muse-Glimmer-30B-NVFP4's 202048-row NVFP4 head), no rows past the token domain.
int greedy_contract(int physical_rows, int token_domain) {
    constexpr int batch = 8;
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * batch, -9.0f);
    for (int row = 0; row < batch; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * physical_rows;
        int first              = (17 + 7919 * row) % token_domain;
        int second             = token_domain - 1 - ((31 + 65537 * row) % token_domain);
        if (second == first) { second = (first + 1) % token_domain; }
        if (second < first) { std::swap(first, second); }
        logits[base + first]  = row == 0 ? -0.0f : 16.0f + row;
        logits[base + second] = row == 0 ? 0.0f : 16.0f + row;
        if (token_domain < physical_rows) {
            logits[base + token_domain]      = 100.0f;
            logits[base + physical_rows - 1] = 200.0f;
        }
    }
    round_to_bf16(logits);

    std::vector<int> counts(static_cast<std::size_t>(token_domain), 0);
    counts[17] = 9;
    ops::SamplingConfig config;
    config.temperature       = 0.0f;
    config.top_k             = 1;
    config.top_p             = 0.01f;
    config.min_p             = 0.99f;
    config.presence_penalty  = 100.0f;
    config.frequency_penalty = 100.0f;
    config.seed              = 12345;

    const RunResult result = run_homogeneous_batch(logits, physical_rows, token_domain, batch,
                                                   config, 77, ops::kSamplePurposeDecode, &counts);
    int failures           = result.integrity_failures;
    failures +=
        verify_exact("sample greedy mathematical result", result.tokens,
                     greedy_oracle(logits, physical_rows, token_domain, batch, config, counts));
    for (std::size_t row = 0; row < result.counts.size(); ++row) {
        std::vector<int> expected_counts = counts;
        ++expected_counts[static_cast<std::size_t>(result.tokens[row])];
        failures += verify_exact("sample greedy increments selected token", result.counts[row],
                                 expected_counts);
    }
    return failures;
}

// A row of NaN logits (a numerically failed forward) still selects an id of its domain on every
// route: NaN ranks as -inf, so greedy takes id 0 and sampling one of the domain, and the token
// count it increments stays in its buffer. Row 1 keeps one finite logit among NaNs, which wins.
int nan_row_contract(int physical_rows, int token_domain) {
    constexpr int batch = 4;
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * batch,
                              std::numeric_limits<float>::quiet_NaN());
    logits[static_cast<std::size_t>(physical_rows) + token_domain / 2] = -3.0f;
    round_to_bf16(logits);
    int failures = 0;
    for (const float temperature : {0.0f, 0.8f}) {
        ops::SamplingConfig config;
        config.temperature = temperature;
        config.seed        = 7;
        const std::vector<int> counts(static_cast<std::size_t>(token_domain), 0);
        const RunResult result =
            run_homogeneous_batch(logits, physical_rows, token_domain, batch, config, 77,
                                  ops::kSamplePurposeDecode, &counts);
        failures += result.integrity_failures;
        for (int row = 0; row < batch; ++row) {
            const int token    = result.tokens[static_cast<std::size_t>(row)];
            const int expected = row == 1 ? token_domain / 2 : 0;
            if (token >= 0 && token < token_domain && (temperature > 0.0f || token == expected))
                continue;
            std::cerr << "sample NaN row domain=" << token_domain << " T=" << temperature << " row "
                      << row << ": token " << token << "\n";
            ++failures;
        }
    }
    return failures;
}

int deterministic_stochastic_contract() {
    std::vector<float> column = {5.0f, 4.5f, 4.0f, 3.0f, -1.0f};
    round_to_bf16(column);
    int failures = 0;

    struct Case {
        const char* label;
        float presence;
        float frequency;
        std::vector<int> counts;
    };

    const Case cases[] = {
        {"sample positive-temperature presence penalty", 1.0f, 0.0f, {1, 0, 0, 0, 0}},
        {"sample positive-temperature frequency penalty", 0.0f, 0.5f, {2, 0, 0, 0, 0}},
        {"sample adjusted-logit tie break", 0.5f, 0.0f, {1, 0, 0, 0, 0}},
    };
    for (const Case& test_case : cases) {
        ops::SamplingConfig config;
        config.temperature       = 0.8f;
        config.top_k             = 1;
        config.presence_penalty  = test_case.presence;
        config.frequency_penalty = test_case.frequency;
        config.seed              = 9981;
        const Distribution oracle =
            distribution_oracle(column, static_cast<int>(column.size()), config, &test_case.counts);
        RunResult result = run_homogeneous_batch(column, static_cast<int>(column.size()),
                                                 static_cast<int>(column.size()), 1, config, 11,
                                                 ops::kSamplePurposeDecode, &test_case.counts);

        std::vector<int> expected_counts = test_case.counts;
        ++expected_counts[static_cast<std::size_t>(oracle.tokens.front())];
        failures += result.integrity_failures;
        failures += verify_exact(test_case.label, result.tokens, {oracle.tokens.front()});
        failures += verify_exact("sample increments only selected token", result.counts.front(),
                                 expected_counts);
    }
    return failures;
}

int heterogeneous_batch_contract() {
    constexpr int physical_rows = 260;
    constexpr int token_domain  = 257;
    constexpr int batch         = 4;
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * batch, -8.0f);
    const auto set = [&](int row, int token, float value) {
        logits[static_cast<std::size_t>(row) * physical_rows + token] = value;
    };
    set(0, 5, 4.0f);
    set(0, 7, 3.0f);
    set(1, 11, 4.0f);
    set(1, 12, 3.0f);
    set(2, 13, 5.0f);
    set(2, 17, 4.5f);
    set(3, 19, 2.0f);
    set(3, 23, 2.0f);
    for (int row = 0; row < batch; ++row) { set(row, physical_rows - 1, 100.0f); }
    round_to_bf16(logits);

    std::vector<ops::SamplingConfig> configs(batch);
    configs[0].temperature      = 0.0f;
    configs[1].temperature      = 0.7f;
    configs[1].top_k            = 1;
    configs[1].seed             = 101;
    configs[2].temperature      = 0.7f;
    configs[2].top_k            = 1;
    configs[2].presence_penalty = 1.0f;
    configs[2].seed             = 202;
    configs[3].temperature      = 0.0f;

    std::vector<std::vector<int>> counts(batch, std::vector<int>(token_domain, 0));
    counts[0][5]           = 9;
    counts[2][13]          = 1;
    const RunResult result = run_batch(logits, physical_rows, token_domain, std::move(configs),
                                       {7, 103, 999, 41}, ops::kSamplePurposeDecode, counts);

    int failures = result.integrity_failures;
    failures += verify_exact("sample heterogeneous batch tokens", result.tokens, {5, 11, 17, 19});
    std::vector<std::vector<int>> expected = counts;
    for (int row = 0; row < batch; ++row) {
        ++expected[static_cast<std::size_t>(row)]
                  [static_cast<std::size_t>(result.tokens[static_cast<std::size_t>(row)])];
    }
    for (int row = 0; row < batch; ++row) {
        failures += verify_exact("sample heterogeneous batch isolated token counts",
                                 result.counts[static_cast<std::size_t>(row)],
                                 expected[static_cast<std::size_t>(row)]);
    }
    return failures;
}

int filtered_distribution_contract() {
    std::vector<float> column = {3.0f, 2.7f,  2.7f,  2.1f,  1.5f,  0.7f,
                                 0.1f, -0.4f, -1.0f, -2.0f, -3.0f, -4.0f};
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 0.75f;
    config.top_k       = 6;
    config.top_p       = 0.86f;
    config.min_p       = 0.12f;
    config.seed        = 20260726;

    const Distribution oracle =
        distribution_oracle(column, static_cast<int>(column.size()), config);
    if (oracle.tokens.size() < 2 || oracle.tokens.size() >= 6) {
        std::cerr << "filtered distribution fixture did not exercise both filters\n";
        return 1;
    }

    constexpr int samples  = 16384;
    const RunResult result = run_repeated(column, static_cast<int>(column.size()), samples, 8,
                                          config, 400, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample top-k/top-p/min-p", result.tokens, oracle);
}

int capped_distribution_contract() {
    // The widest public top_k over a few more candidates than it keeps.
    constexpr int columns = kMaximumTopK + 8;
    std::vector<float> column(columns, 0.0f);
    for (int token = 0; token < columns; ++token) {
        column[static_cast<std::size_t>(token)] = 2.0f - 0.05f * token;
    }
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 1.1f;
    config.top_k       = kMaximumTopK;
    config.seed        = 884422;

    const Distribution oracle =
        distribution_oracle(column, static_cast<int>(column.size()), config);
    if (oracle.tokens.size() != static_cast<std::size_t>(kMaximumTopK)) {
        std::cerr << "top-k cap oracle fixture has unexpected support\n";
        return 1;
    }

    constexpr int samples  = 16384;
    const RunResult result = run_repeated(column, static_cast<int>(column.size()), samples, 8,
                                          config, 900, ops::kSamplePurposePrefill);
    return result.integrity_failures +
           verify_distribution("sample top-k public cap", result.tokens, oracle);
}

// More lanes than one multi-block launch takes (sixteen): sample() runs them in slices, so every
// lane keeps the whole top-kMaximumTopK domain (the single-block fallback would keep 20).
int sliced_lanes_distribution_contract() {
    constexpr int vocab   = 1024;
    constexpr int columns = kMaximumTopK + 8;
    std::vector<float> column(vocab, -30.0f);
    for (int token = 0; token < columns; ++token) {
        column[static_cast<std::size_t>(token)] = 2.0f - 0.05f * token;
    }
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 1.1f;
    config.top_k       = kMaximumTopK;
    config.seed        = 551177;

    const Distribution oracle = distribution_oracle(column, vocab, config);
    if (oracle.tokens.size() != static_cast<std::size_t>(kMaximumTopK)) {
        std::cerr << "sliced lanes oracle fixture has unexpected support\n";
        return 1;
    }
    constexpr int lanes   = 20;
    constexpr int samples = 16380; // a multiple of the lanes
    const RunResult result =
        run_repeated(column, vocab, samples, lanes, config, 1300, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample twenty lanes keep the top-64 domain", result.tokens, oracle);
}

int real_shape_distribution_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    std::vector<float> column(physical_rows, -20.0f);
    const int ids[]      = {17, 7919, 65537, 200003};
    const float logits[] = {3.0f, 2.0f, 1.0f, 0.0f};
    for (int i = 0; i < 4; ++i) { column[ids[i]] = logits[i]; }
    column[token_domain]      = 100.0f;
    column[physical_rows - 1] = 200.0f;
    round_to_bf16(column);

    ops::SamplingConfig config;
    config.temperature        = 1.0f;
    config.top_k              = 4;
    config.seed               = 7654321;
    const Distribution oracle = distribution_oracle(column, token_domain, config);

    constexpr int samples = 4096;
    RunResult result =
        run_repeated(column, token_domain, samples, 8, config, 2000, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample real token-domain B=8", result.tokens, oracle);
}

int rng_key_contract() {
    std::vector<float> column = {0.0f, 0.0f};
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 1.0f;
    config.top_k       = 2;
    config.seed        = 424242;

    constexpr int samples = 128;
    const RunResult baseline =
        run_repeated(column, 2, samples, 8, config, 100, ops::kSamplePurposeDecode);
    const RunResult repeat =
        run_repeated(column, 2, samples, 8, config, 100, ops::kSamplePurposeDecode);
    const RunResult rechunked =
        run_repeated(column, 2, samples, 1, config, 100, ops::kSamplePurposeDecode);
    const RunResult shifted =
        run_repeated(column, 2, samples, 8, config, 101, ops::kSamplePurposeDecode);
    const RunResult other_purpose =
        run_repeated(column, 2, samples, 8, config, 100, ops::kSamplePurposePrefill);
    ops::SamplingConfig other_seed_config = config;
    ++other_seed_config.seed;
    const RunResult other_seed =
        run_repeated(column, 2, samples, 8, other_seed_config, 100, ops::kSamplePurposeDecode);

    int failures = baseline.integrity_failures + repeat.integrity_failures +
                   rechunked.integrity_failures + shifted.integrity_failures +
                   other_purpose.integrity_failures + other_seed.integrity_failures;
    failures += verify_exact("sample identical counter key is reproducible", repeat.tokens,
                             baseline.tokens);
    failures += verify_exact("sample RNG does not depend on compact row", rechunked.tokens,
                             baseline.tokens);
    failures += verify_exact("sample position selects the corresponding counter",
                             std::vector<int>(shifted.tokens.begin(), shifted.tokens.end() - 1),
                             std::vector<int>(baseline.tokens.begin() + 1, baseline.tokens.end()));
    if (other_purpose.tokens == baseline.tokens) {
        std::cerr << "sample purpose did not separate the counter stream\n";
        ++failures;
    }
    if (other_seed.tokens == baseline.tokens) {
        std::cerr << "sample seed did not separate the counter stream\n";
        ++failures;
    }
    return failures;
}

int workspace_route_boundary_contract() {
    constexpr int token_domain = 257;
    constexpr int batch        = 8;
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * batch, 0.0f);
    const RunResult result =
        run_homogeneous_batch(logits, token_domain, token_domain, batch, ops::SamplingConfig{}, 0,
                              ops::kSamplePurposeDecode);
    int failures = result.integrity_failures;
    failures +=
        verify_exact("sample workspace route boundary", result.tokens, std::vector<int>(batch, 0));
    return failures;
}

int increment_counts_contract() {
    const std::vector<std::int32_t> ids{1, 3, 1, 7};
    const std::vector<std::int32_t> initial{0, 2, 0, 4, 0, 0, 0, 1};
    const std::vector<std::int32_t> expected{0, 4, 0, 5, 0, 0, 0, 2};
    DeviceBuffer device_ids = to_device(ids);
    GuardedDeviceBuffer device_counts(initial.size() * sizeof(std::int32_t));
    device_counts.copy_from_host(initial.data(), initial.size() * sizeof(std::int32_t));
    Tensor token_ids(device_ids.p, DType::I32, {static_cast<std::int32_t>(ids.size())});
    Tensor counts(device_counts.data(), DType::I32, {static_cast<std::int32_t>(initial.size())});
    ops::increment_token_counts(token_ids, counts, nullptr);
    cuda_synchronize();

    int failures =
        verify_exact("increment token counts",
                     from_device<std::int32_t>(device_counts.data(), initial.size()), expected);
    failures += verify_exact("increment token counts read-only ids",
                             from_device<std::int32_t>(device_ids, ids.size()), ids);
    failures += device_counts.verify_guards("increment token counts guards");
    return failures;
}

// ---- Allowed-token masks ----------------------------------------------------------------------

// Device-mapped pinned host words: the storage the structured-output runtime uses for masks and
// the release counter.
class MappedWords {
public:
    explicit MappedWords(std::size_t count) : count_(count) {
        cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host_), count * sizeof(std::uint32_t),
                                 cudaHostAllocMapped),
                   "cudaHostAlloc mapped");
        std::fill(host_, host_ + count, 0u);
        cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_), host_, 0),
                   "cudaHostGetDevicePointer");
    }
    ~MappedWords() { cudaFreeHost(host_); }
    MappedWords(const MappedWords&)            = delete;
    MappedWords& operator=(const MappedWords&) = delete;

    [[nodiscard]] std::uint32_t* host() const noexcept { return host_; }
    [[nodiscard]] std::uint32_t* device() const noexcept { return device_; }

    void store_release(std::uint32_t value) const noexcept {
        __atomic_store_n(host_, value, __ATOMIC_RELEASE);
    }

private:
    std::size_t count_     = 0;
    std::uint32_t* host_   = nullptr;
    std::uint32_t* device_ = nullptr;
};

int mask_words(int token_domain) { return (((token_domain + 31) / 32) + 7) & ~7; }

void set_allowed(std::uint32_t* words, int token, bool allowed) {
    if (allowed) {
        words[token >> 5] |= 1u << (token & 31);
    } else {
        words[token >> 5] &= ~(1u << (token & 31));
    }
}

std::vector<bool> allowed_set(const std::uint32_t* words, int token_domain) {
    std::vector<bool> out(static_cast<std::size_t>(token_domain));
    for (int token = 0; token < token_domain; ++token) {
        out[static_cast<std::size_t>(token)] = ((words[token >> 5] >> (token & 31)) & 1u) != 0u;
    }
    return out;
}

int masked_argmax(const std::vector<float>& logits, std::size_t base, int token_domain,
                  const std::vector<bool>& allowed) {
    int best = -1;
    for (int token = 0; token < token_domain; ++token) {
        if (!allowed[static_cast<std::size_t>(token)]) { continue; }
        if (best < 0 || logits[base + token] > logits[base + best]) { best = token; }
    }
    return best;
}

Distribution masked_distribution_oracle(const std::vector<float>& column, int token_domain,
                                        const ops::SamplingConfig& config,
                                        const std::vector<bool>& allowed) {
    std::vector<int> tokens;
    for (int token = 0; token < token_domain; ++token) {
        if (allowed[static_cast<std::size_t>(token)]) { tokens.push_back(token); }
    }
    std::vector<float> compact;
    compact.reserve(tokens.size());
    for (int token : tokens) { compact.push_back(column[static_cast<std::size_t>(token)]); }
    Distribution compact_oracle =
        distribution_oracle(compact, static_cast<int>(compact.size()), config);
    for (int& token : compact_oracle.tokens) { token = tokens[static_cast<std::size_t>(token)]; }
    return compact_oracle;
}

// Greedy and stochastic rows under per-row masks, including a null-mask row in the same call,
// at the registered token domain (multi-block route) and at a single-block domain.
int masked_greedy_contract(int physical_rows, int token_domain) {
    constexpr int batch = 4;
    const int words     = mask_words(token_domain);
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * batch, -9.0f);
    std::uint32_t lcg = 12345;
    for (float& value : logits) {
        lcg   = lcg * 1664525u + 1013904223u;
        value = static_cast<float>(lcg >> 9) / static_cast<float>(1u << 23) * 8.0f - 4.0f;
    }
    for (int row = 0; row < batch; ++row) {
        const std::size_t base           = static_cast<std::size_t>(row) * physical_rows;
        logits[base + physical_rows - 1] = 200.0f; // outside the token domain
    }
    round_to_bf16(logits);

    MappedWords masks(static_cast<std::size_t>(words) * batch);
    MappedWords release(1);
    std::vector<ops::SamplingConfig> configs(batch);
    std::vector<int> expected(batch);
    for (int row = 0; row < batch; ++row) {
        std::uint32_t* row_mask = masks.host() + static_cast<std::size_t>(row) * words;
        // Row r admits every (7+r)-th token plus one: the unmasked argmax is excluded unless it
        // happens to be admitted.
        for (int token = 0; token < token_domain; token += 7 + row) {
            set_allowed(row_mask, token, true);
        }
        set_allowed(row_mask, token_domain - 1, true);
        const std::size_t base = static_cast<std::size_t>(row) * physical_rows;
        if (row == 2) {
            configs[row].token_mask = nullptr; // unconstrained row in a masked call
            std::vector<bool> all(static_cast<std::size_t>(token_domain), true);
            expected[row] = masked_argmax(logits, base, token_domain, all);
        } else {
            configs[row].token_mask         = masks.device() + static_cast<std::size_t>(row) * words;
            configs[row].token_mask_release = release.device();
            configs[row].token_mask_epoch   = 1;
            configs[row].token_mask_words   = words;
            expected[row] = masked_argmax(logits, base, token_domain,
                                          allowed_set(row_mask, token_domain));
        }
        configs[row].temperature = 0.0f;
    }
    configs[3].temperature = 0.7f; // top_k=1 stochastic row: support is the masked argmax
    configs[3].top_k       = 1;
    configs[3].seed        = 77;
    release.store_release(1);

    std::vector<int> positions{3, 4, 5, 6};
    const RunResult result = run_batch(logits, physical_rows, token_domain, configs, positions,
                                       ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_exact(("sample masked greedy domain=" + std::to_string(token_domain)).c_str(),
                        result.tokens, expected);
}

// A masked positive-temperature row matches the FP64 oracle restricted to allowed tokens: the
// top-k cut is taken among allowed tokens and the support is renormalized over them.
int masked_distribution_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    const int words             = mask_words(token_domain);
    std::vector<float> column(physical_rows, -20.0f);
    const int ids[]     = {17, 7919, 65537, 131071, 200003, 248076};
    const float value[] = {6.0f, 3.0f, 2.5f, 2.0f, 1.5f, 1.0f};
    for (int i = 0; i < 6; ++i) { column[static_cast<std::size_t>(ids[i])] = value[i]; }
    round_to_bf16(column);

    MappedWords masks(static_cast<std::size_t>(words));
    MappedWords release(1);
    // The best token (17) is disallowed; so are two filler tokens, which leaves four allowed
    // tokens plus the -20 background.
    for (int token = 0; token < token_domain; ++token) { set_allowed(masks.host(), token, true); }
    set_allowed(masks.host(), 17, false);
    set_allowed(masks.host(), 65537, false);
    release.store_release(5);

    ops::SamplingConfig config;
    config.temperature        = 1.0f;
    config.top_k              = 3;
    config.seed               = 99173;
    config.token_mask         = masks.device();
    config.token_mask_release = release.device();
    config.token_mask_epoch   = 5;
    config.token_mask_words   = words;
    const Distribution oracle = masked_distribution_oracle(
        column, token_domain, config, allowed_set(masks.host(), token_domain));
    constexpr int samples = 4096;
    const RunResult result =
        run_repeated(column, token_domain, samples, 8, config, 5000, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample masked distribution", result.tokens, oracle);
}

// All-allowed masks leave greedy and stochastic results bit-identical to the unmasked route.
int all_allowed_mask_identity_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    const int words             = mask_words(token_domain);
    std::vector<float> column(physical_rows, 0.0f);
    std::uint32_t lcg = 4242;
    for (float& value : column) {
        lcg   = lcg * 1664525u + 1013904223u;
        value = static_cast<float>(lcg >> 9) / static_cast<float>(1u << 23) * 6.0f - 3.0f;
    }
    round_to_bf16(column);
    MappedWords masks(static_cast<std::size_t>(words));
    MappedWords release(1);
    std::fill(masks.host(), masks.host() + words, 0xffffffffu);
    release.store_release(9);

    int failures = 0;
    for (float temperature : {0.0f, 0.9f}) {
        ops::SamplingConfig plain;
        plain.temperature = temperature;
        plain.top_k       = 20;
        plain.top_p       = 0.95f;
        plain.seed        = 31337;
        ops::SamplingConfig masked      = plain;
        masked.token_mask               = masks.device();
        masked.token_mask_release       = release.device();
        masked.token_mask_epoch         = 9;
        masked.token_mask_words         = words;
        const RunResult unmasked_result = run_repeated(column, token_domain, 256, 8, plain, 70,
                                                       ops::kSamplePurposeDecode);
        const RunResult masked_result   = run_repeated(column, token_domain, 256, 8, masked, 70,
                                                       ops::kSamplePurposeDecode);
        failures += unmasked_result.integrity_failures + masked_result.integrity_failures;
        failures += verify_exact("sample all-allowed mask is the unmasked result",
                                 masked_result.tokens, unmasked_result.tokens);
    }
    return failures;
}

// The overlap protocol: the host writes the mask and releases its epoch after the launch. The
// kernel must wait and then observe the late mask, not the stale contents present at launch.
int late_release_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    const int words             = mask_words(token_domain);
    std::vector<float> column(physical_rows, -5.0f);
    column[100]    = 9.0f;
    column[200]    = 8.0f;
    column[300]    = 7.0f;
    const auto bits = bf16_bits(column);
    DeviceBuffer device_logits = to_device(bits);
    MappedWords masks(static_cast<std::size_t>(words));
    MappedWords release(1);
    MappedWords fault(1);
    set_allowed(masks.host(), 100, true); // stale contents: admits the unmasked argmax
    release.store_release(41);

    ops::SamplingConfig config;
    config.token_mask         = masks.device();
    config.token_mask_release = release.device();
    config.token_mask_fault   = fault.device();
    config.token_mask_epoch   = 42;
    config.token_mask_words   = words;
    DeviceBuffer device_config    = to_device(std::vector<ops::SamplingConfig>{config});
    DeviceBuffer device_positions = to_device(std::vector<int>{1});
    GuardedDeviceBuffer device_out(sizeof(std::int32_t));
    Tensor logits_tensor(device_logits.p, DType::BF16, {physical_rows, 1});
    Tensor out_tensor(device_out.data(), DType::I32, {1});
    Tensor positions_tensor(device_positions.p, DType::I32, {1});
    WorkspaceArena workspace(
        std::max<std::size_t>(256, ops::sampling_workspace_capacity_bytes(token_domain, 1, 1)));
    ops::sample(logits_tensor, out_tensor, token_domain,
                static_cast<const ops::SamplingConfig*>(device_config.p), positions_tensor,
                ops::kSamplePurposeDecode, workspace, nullptr);
    (void)cudaStreamQuery(nullptr); // submit the launch before the host writes the mask
    // Give the launch time to reach the wait; the release happens strictly afterwards.
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
    while (std::chrono::steady_clock::now() < until) {}
    std::fill(masks.host(), masks.host() + words, 0u);
    set_allowed(masks.host(), 300, true);
    release.store_release(42);
    cuda_synchronize();
    int failures = verify_exact("sample late-released mask", from_device<int>(device_out.data(), 1),
                                std::vector<int>{300});
    failures += verify_exact("sample late release raised no fault",
                             std::vector<std::uint32_t>{fault.host()[0]},
                             std::vector<std::uint32_t>{0u});
    return failures;
}

// A mask that is never released cannot hang the device: the bounded wait raises the fault flag.
int unreleased_mask_fault_contract() {
    constexpr int token_domain = 248077;
    const int words            = mask_words(token_domain);
    std::vector<float> column(248320, 0.0f);
    column[5] = 3.0f;
    const auto bits = bf16_bits(column);
    DeviceBuffer device_logits = to_device(bits);
    MappedWords masks(static_cast<std::size_t>(words));
    MappedWords release(1);
    MappedWords fault(1);
    std::fill(masks.host(), masks.host() + words, 0xffffffffu);
    ops::SamplingConfig config;
    config.token_mask         = masks.device();
    config.token_mask_release = release.device();
    config.token_mask_fault   = fault.device();
    config.token_mask_epoch   = 1;
    config.token_mask_words   = words;
    DeviceBuffer device_config    = to_device(std::vector<ops::SamplingConfig>{config});
    DeviceBuffer device_positions = to_device(std::vector<int>{1});
    GuardedDeviceBuffer device_out(sizeof(std::int32_t));
    Tensor logits_tensor(device_logits.p, DType::BF16, {248320, 1});
    Tensor out_tensor(device_out.data(), DType::I32, {1});
    Tensor positions_tensor(device_positions.p, DType::I32, {1});
    WorkspaceArena workspace(
        std::max<std::size_t>(256, ops::sampling_workspace_capacity_bytes(token_domain, 1, 1)));
    const auto started = std::chrono::steady_clock::now();
    ops::sample(logits_tensor, out_tensor, token_domain,
                static_cast<const ops::SamplingConfig*>(device_config.p), positions_tensor,
                ops::kSamplePurposeDecode, workspace, nullptr);
    cuda_synchronize();
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "    unreleased mask wait returned after " << seconds << " s\n";
    return verify_exact("sample unreleased mask raises the fault flag",
                        std::vector<std::uint32_t>{fault.host()[0]},
                        std::vector<std::uint32_t>{1u});
}

// A host that keeps computing masks for longer than the silent-host bound is waited for: each
// advance of the progress counter renews the bound, and the late mask is applied without a fault.
int progressing_host_wait_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    const int words             = mask_words(token_domain);
    std::vector<float> column(physical_rows, -5.0f);
    column[100]                = 9.0f;
    column[300]                = 7.0f;
    const auto bits            = bf16_bits(column);
    DeviceBuffer device_logits = to_device(bits);
    MappedWords masks(static_cast<std::size_t>(words));
    MappedWords release(1);
    MappedWords fault(1);
    MappedWords progress(1);
    set_allowed(masks.host(), 100, true); // stale contents: admits the unmasked argmax
    release.store_release(6);

    ops::SamplingConfig config;
    config.token_mask             = masks.device();
    config.token_mask_release     = release.device();
    config.token_mask_fault       = fault.device();
    config.token_mask_progress    = progress.device();
    config.token_mask_epoch       = 7;
    config.token_mask_words       = words;
    DeviceBuffer device_config    = to_device(std::vector<ops::SamplingConfig>{config});
    DeviceBuffer device_positions = to_device(std::vector<int>{1});
    GuardedDeviceBuffer device_out(sizeof(std::int32_t));
    Tensor logits_tensor(device_logits.p, DType::BF16, {physical_rows, 1});
    Tensor out_tensor(device_out.data(), DType::I32, {1});
    Tensor positions_tensor(device_positions.p, DType::I32, {1});
    WorkspaceArena workspace(
        std::max<std::size_t>(256, ops::sampling_workspace_capacity_bytes(token_domain, 1, 1)));
    ops::sample(logits_tensor, out_tensor, token_domain,
                static_cast<const ops::SamplingConfig*>(device_config.p), positions_tensor,
                ops::kSamplePurposeDecode, workspace, nullptr);
    (void)cudaStreamQuery(nullptr);
    // Twice the silent-host bound, with an advance every 200 ms.
    const auto wait  = std::chrono::nanoseconds(2 * ops::kTokenMaskReleaseWaitNs);
    const auto until = std::chrono::steady_clock::now() + wait;
    while (std::chrono::steady_clock::now() < until) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        __atomic_fetch_add(progress.host(), 1u, __ATOMIC_RELAXED);
    }
    std::fill(masks.host(), masks.host() + words, 0u);
    set_allowed(masks.host(), 300, true);
    release.store_release(7);
    cuda_synchronize();
    int failures = verify_exact("sample mask released by a progressing host",
                                from_device<int>(device_out.data(), 1), std::vector<int>{300});
    failures +=
        verify_exact("progressing host raised no fault",
                     std::vector<std::uint32_t>{fault.host()[0]}, std::vector<std::uint32_t>{0u});
    return failures;
}

// Diagnostic, run only with NINFERNO_SAMPLING_MASK_BENCH=1: device time per sample() call for one
// 248077-token row, unmasked vs masked with the mask in mapped host memory vs device memory.
void mask_route_timing() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    const int words             = mask_words(token_domain);
    std::vector<float> column(physical_rows, 0.0f);
    std::uint32_t lcg = 777;
    for (float& value : column) {
        lcg   = lcg * 1664525u + 1013904223u;
        value = static_cast<float>(lcg >> 9) / static_cast<float>(1u << 23) * 6.0f - 3.0f;
    }
    const auto bits            = bf16_bits(column);
    DeviceBuffer device_logits = to_device(bits);
    MappedWords masks(static_cast<std::size_t>(words));
    MappedWords release(1);
    for (int token = 0; token < token_domain; token += 3) { set_allowed(masks.host(), token, true); }
    release.store_release(1);
    std::vector<std::uint32_t> device_words(masks.host(), masks.host() + words);
    DeviceBuffer device_mask    = to_device(device_words);
    DeviceBuffer device_release = to_device(std::vector<std::uint32_t>{1u});
    DeviceBuffer device_positions = to_device(std::vector<int>{1});
    GuardedDeviceBuffer device_out(sizeof(std::int32_t));
    Tensor logits_tensor(device_logits.p, DType::BF16, {physical_rows, 1});
    Tensor out_tensor(device_out.data(), DType::I32, {1});
    Tensor positions_tensor(device_positions.p, DType::I32, {1});
    WorkspaceArena workspace(
        std::max<std::size_t>(256, ops::sampling_workspace_capacity_bytes(token_domain, 1, 1)));
    for (float temperature : {0.0f, 0.7f}) {
        for (int route = 0; route < 4; ++route) {
            ops::SamplingConfig config;
            config.temperature = temperature;
            config.top_k       = 20;
            config.seed        = 5;
            if (route > 0) {
                config.token_mask = route == 1 ? masks.device()
                                               : static_cast<std::uint32_t*>(device_mask.p);
                config.token_mask_release =
                    route == 3 ? static_cast<const std::uint32_t*>(device_release.p)
                               : release.device();
                config.token_mask_epoch   = 1;
                config.token_mask_words   = words;
            }
            DeviceBuffer device_config = to_device(std::vector<ops::SamplingConfig>{config});
            const auto launch          = [&] {
                ops::sample(logits_tensor, out_tensor, token_domain,
                            static_cast<const ops::SamplingConfig*>(device_config.p),
                            positions_tensor, ops::kSamplePurposeDecode, workspace, nullptr);
            };
            for (int i = 0; i < 20; ++i) { launch(); }
            cuda_synchronize();
            cudaEvent_t start = nullptr;
            cudaEvent_t stop  = nullptr;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            constexpr int calls = 200;
            cudaEventRecord(start, nullptr);
            for (int i = 0; i < calls; ++i) { launch(); }
            cudaEventRecord(stop, nullptr);
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            const char* names[] = {"unmasked", "mask and release in mapped host memory",
                                   "mask in device memory, release mapped",
                                   "mask and release in device memory"};
            std::cout << "    timing temperature=" << temperature << " " << names[route] << ": "
                      << 1000.0f * ms / calls << " us per call\n";
        }
    }
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (const char* bench = std::getenv("NINFERNO_SAMPLING_MASK_BENCH"); bench && bench[0] == '1') {
        mask_route_timing();
        return 0;
    }

    int failures            = 0;
    const std::size_t at_16 = ops::sampling_workspace_capacity_bytes(257, 16, 16);
    if (ops::sampling_workspace_capacity_bytes(256, 1, 16) != 0 || at_16 == 0 ||
        ops::sampling_workspace_capacity_bytes(257, 17, 17) != at_16 || // seventeen: two slices
        ops::sampling_workspace_capacity_bytes(257, 1, 17) != at_16) {
        std::cerr << "sampling workspace route boundary contract failed\n";
        ++failures;
    }
    try {
        (void)ops::sampling_workspace_capacity_bytes(257, 0, 16);
        std::cerr << "sampling workspace accepted an invalid lane interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += greedy_contract(248320, 248077);
    failures += greedy_contract(202048, 202048);
    failures += nan_row_contract(248320, 248077);
    failures += nan_row_contract(260, 200);
    failures += deterministic_stochastic_contract();
    failures += heterogeneous_batch_contract();
    failures += filtered_distribution_contract();
    failures += capped_distribution_contract();
    failures += sliced_lanes_distribution_contract();
    failures += real_shape_distribution_contract();
    failures += rng_key_contract();
    failures += workspace_route_boundary_contract();
    failures += increment_counts_contract();
    failures += masked_greedy_contract(248320, 248077);
    failures += masked_greedy_contract(260, 200);
    failures += masked_distribution_contract();
    failures += all_allowed_mask_identity_contract();
    failures += late_release_contract();
    failures += unreleased_mask_fault_contract();
    failures += progressing_host_wait_contract();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " sample public contract\n";
    return failures == 0 ? 0 : 1;
}
