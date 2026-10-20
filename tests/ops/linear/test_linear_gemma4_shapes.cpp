// Gemma 4 26B-A4B and 31B projection shapes: registered-route conformance against the FP64 oracle,
// isolated K segments, and control arms that must fail. One source builds one executable per
// weight format (NINFERNO_GEMMA4_FORMAT_NVFP4, _Q8 or _Q6).
#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/weight.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::test;
using namespace ninferno::test::linear;

struct Problem {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
};

// The shared NVFP4 fixture draws codes and scales from row*13 and row*5 modulo 16 and 8, so every
// 128-row tile repeats the first: a kernel that read another row tile, or wrote its output there,
// would still match the oracle. This fixture keeps that layout, divisors and oracle decode and
// replaces the payload with hashed codes and nonzero hashed scales.
quantized_weight::PackedWeight make_hashed_nvfp4_weight(std::int32_t n, std::int32_t k,
                                                        std::uint32_t seed) {
    quantized_weight::PackedWeight packed = make_nvfp4_weight(n, k, seed);
    std::uint64_t state = 0x9e3779b97f4a7c15ULL * (static_cast<std::uint64_t>(seed) + 1U);
    const auto next     = [&state] {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545f4914f6cdd1dULL;
    };
    for (std::uint64_t byte = 0; byte < packed.code_plane_bytes; ++byte) {
        packed.payload[static_cast<std::size_t>(byte)] = static_cast<std::uint8_t>(next() >> 56);
    }
    // E4M3 0.0078125 .. 0.125: the fixture's own range without its zero word.
    constexpr std::uint8_t kScaleWords[]{0x04U, 0x08U, 0x0bU, 0x10U, 0x13U, 0x18U, 0x1dU, 0x20U};
    for (std::uint64_t byte = 0; byte < packed.scale_plane_bytes; ++byte) {
        packed.payload[static_cast<std::size_t>(packed.scale_plane_offset + byte)] =
            kScaleWords[next() >> 61];
    }
    return packed;
}

// ----------------------------------------------------------------------------------------------
// Isolated cases and control arms.
//
// The conformance runs above judge every public call with the path's named criterion. They can
// only fail if a mistake moves the output past that criterion, so each format also runs cases a
// correct kernel must pass and mutilated weights it must fail, judged by the same rule:
//   * segment cases feed activations that are zero outside the first or the last K block, so the
//     result is exactly that block's contribution; a kernel that skips or misreads the last block
//     of a row (the ragged phase of K=2112 above all) returns zero or the wrong block and fails.
//     A block is 64 values on the A16 path and 256 on the A4 path, whose activation rounding
//     needs that many terms to stay inside its criterion;
//   * controls corrupt the device copy of the weight while the oracle keeps the true one: the
//     scale plane shifted by one stored unit, the scales of the last 64 values cleared, or the
//     first two row tiles swapped. A check that passes one of them has not looked at what it
//     claims to.
// ----------------------------------------------------------------------------------------------

enum class Corruption : std::uint8_t {
    None,
    ScaleShift,    // every stored scale moves one unit along K
    LastBlockZero, // the scales of the last 64-value K block are cleared
    RowSwap,       // the first two row tiles trade places
};

const char* corruption_name(Corruption corruption) {
    switch (corruption) {
    case Corruption::None:
        return "none";
    case Corruption::ScaleShift:
        return "scale-shift";
    case Corruption::LastBlockZero:
        return "last-block-zero";
    case Corruption::RowSwap:
        return "row-swap";
    }
    return "?";
}

struct Segment {
    std::int32_t begin = -1; // -1: the full activation
    std::int32_t count = 0;
};

class HostProblem {
public:
    HostProblem(WeightGenerator generator, const Problem& problem)
        : problem_(problem), packed_(generator(problem.n, problem.k, problem.seed)) {
        for (std::int32_t index = 0; index < 48; ++index) {
            rows_.push_back(
                static_cast<std::int32_t>((static_cast<std::int64_t>(problem.n - 1) * index) / 47));
        }
        // The first two row tiles are what RowSwap exchanges; sample inside both.
        for (std::int32_t row : {1, 31, 127, 128, 129, 200, 255}) rows_.push_back(row);
        std::sort(rows_.begin(), rows_.end());
        rows_.erase(std::unique(rows_.begin(), rows_.end()), rows_.end());
        oracle_weight_ = quantized_weight::materialize_rows_fp32(packed_, rows_);
    }

    const Problem& problem() const { return problem_; }

    const quantized_weight::PackedWeight& packed() const { return packed_; }

    const std::vector<std::int32_t>& rows() const { return rows_; }

    const std::vector<float>& oracle_weight() const { return oracle_weight_; }

private:
    Problem problem_;
    quantized_weight::PackedWeight packed_;
    std::vector<std::int32_t> rows_;
    std::vector<float> oracle_weight_;
};

void device_move(DeviceBuffer& buffer, std::size_t destination, std::size_t source,
                 std::size_t bytes) {
    DeviceBuffer scratch(bytes);
    auto* base = static_cast<std::uint8_t*>(buffer.p);
    cuda_check(cudaMemcpy(scratch.p, base + source, bytes, cudaMemcpyDeviceToDevice), "stage");
    cuda_check(cudaMemcpy(base + destination, scratch.p, bytes, cudaMemcpyDeviceToDevice), "move");
}

void corrupt(DeviceBuffer& payload, const quantized_weight::PackedWeight& packed,
             Corruption corruption) {
    const Weight& w  = packed.weight;
    auto* base       = static_cast<std::uint8_t*>(payload.p);
    const bool nvfp4 = w.qtype == QType::NVFP4;
    // NVFP4 stores scales in [128-row tile][64-value K block] units of 512 bytes; the row-split
    // formats store one FP16 scale per row group.
    const std::size_t scale_offset = packed.scale_plane_offset;
    const std::size_t scale_bytes  = packed.scale_plane_bytes;
    switch (corruption) {
    case Corruption::None:
        return;
    case Corruption::ScaleShift: {
        const std::size_t unit = nvfp4 ? 512 : 2;
        device_move(payload, scale_offset + unit, scale_offset, scale_bytes - unit);
        return;
    }
    case Corruption::LastBlockZero: {
        if (nvfp4) {
            const std::int32_t blocks = w.k / 64;
            for (std::int32_t tile = 0; tile < w.n / 128; ++tile) {
                const std::size_t at =
                    scale_offset + (static_cast<std::size_t>(tile) * blocks + blocks - 1) * 512;
                cuda_check(cudaMemset(base + at, 0, 512), "clear last block");
            }
        } else {
            // The stored row covers padded K; the block to clear ends at the logical K.
            const std::int32_t groups  = w.padded_shape[1] / w.group;
            const std::int32_t logical = w.k / w.group;
            const std::int32_t cleared = 64 / w.group; // groups spanning the last 64 values
            std::vector<std::uint8_t> plane(packed.payload.begin() + scale_offset,
                                            packed.payload.begin() + scale_offset + scale_bytes);
            for (std::int32_t row = 0; row < w.n; ++row) {
                const std::size_t at =
                    (static_cast<std::size_t>(row) * groups + logical - cleared) * 2;
                std::fill_n(plane.begin() + at, static_cast<std::size_t>(cleared) * 2, 0);
            }
            payload.copy_from_host(plane.data(), plane.size(), scale_offset);
        }
        return;
    }
    case Corruption::RowSwap: {
        // NVFP4: two 128-row tiles of codes and scales. Row-split: the first two row groups of
        // 128 rows in every plane, whose row pitch is the plane size over N.
        const auto swap_rows = [&](std::size_t plane_offset, std::size_t plane_bytes) {
            const std::size_t tile = plane_bytes / static_cast<std::size_t>(w.n) * 128;
            DeviceBuffer first(tile);
            cuda_check(cudaMemcpy(first.p, base + plane_offset, tile, cudaMemcpyDeviceToDevice),
                       "swap stage");
            device_move(payload, plane_offset, plane_offset + tile, tile);
            cuda_check(
                cudaMemcpy(base + plane_offset + tile, first.p, tile, cudaMemcpyDeviceToDevice),
                "swap back");
        };
        swap_rows(0, packed.code_plane_bytes);
        if (packed.high_plane_bytes != 0)
            swap_rows(packed.high_plane_offset, packed.high_plane_bytes);
        swap_rows(scale_offset, scale_bytes);
        return;
    }
    }
}

struct CaseSpec {
    ActivationCompute compute;
    ops::LinearPolicy policy;
    std::int32_t t;
    Segment segment;
    Corruption corruption;
    // Judge the result by the A16 criterion whatever activation it was fed. A clean case then
    // passes only under A16Only: an AllowA4 result is W4A4 and must be caught as wrong, the
    // control that the A16 criterion tells BF16 activations from FP4-quantized ones.
    bool judge_a16 = false;
};

// Returns one failure when the outcome differs from the expectation: a clean case must pass the
// named criterion, a corrupted one must not.
int run_case(const std::string& label, const HostProblem& host, const CaseSpec& spec) {
    const Problem& problem = host.problem();
    const std::int32_t n = problem.n, k = problem.k, t = spec.t;
    std::vector<std::uint16_t> activation =
        make_linear_activation(k, t, problem.seed + 17U, spec.compute);
    if (spec.segment.begin >= 0) {
        for (std::int32_t token = 0; token < t; ++token) {
            for (std::int32_t column = 0; column < k; ++column) {
                if (column < spec.segment.begin ||
                    column >= spec.segment.begin + spec.segment.count) {
                    activation[static_cast<std::size_t>(token) * k + column] = 0;
                }
            }
        }
    }

    DeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), device_x.bytes);
    const auto& packed = host.packed();
    DeviceBuffer payload(packed.payload.size());
    payload.copy_from_host(packed.payload.data(), payload.bytes);
    corrupt(payload, packed, spec.corruption);
    const Weight weight = packed.device_weight(payload.p);
    DeviceBuffer device_out(static_cast<std::size_t>(n) * t * sizeof(std::uint16_t));
    cuda_check(cudaMemset(device_out.p, 0xff, device_out.bytes), "poison output");

    Tensor x(device_x.p, DType::BF16, {k, t});
    Tensor out(device_out.p, DType::BF16, {n, t});
    const std::size_t capacity =
        ops::linear_workspace_capacity_bytes(weight.qtype, n, k, spec.policy, t, t);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::linear(x, weight, out, spec.policy, workspace, nullptr);
    cuda_check(cudaDeviceSynchronize(), "linear");

    std::vector<std::int32_t> columns;
    if (t <= 48) {
        for (std::int32_t column = 0; column < t; ++column) columns.push_back(column);
    } else {
        for (std::int32_t index = 0; index < 24; ++index)
            columns.push_back(
                static_cast<std::int32_t>((static_cast<std::int64_t>(t - 1) * index) / 23));
    }
    const std::vector<std::int32_t>& rows = host.rows();
    std::vector<float> oracle_x(columns.size() * static_cast<std::size_t>(k));
    for (std::size_t c = 0; c < columns.size(); ++c) {
        for (std::int32_t column = 0; column < k; ++column) {
            oracle_x[c * k + column] =
                bf16_to_f32(activation[static_cast<std::size_t>(columns[c]) * k + column]);
        }
    }
    std::vector<double> reference(rows.size() * columns.size());
    cpu_linear_gemm_fp64(host.oracle_weight().data(), oracle_x.data(), reference.data(),
                         static_cast<std::int32_t>(rows.size()), k,
                         static_cast<std::int32_t>(columns.size()));

    std::vector<std::uint16_t> column_bits(static_cast<std::size_t>(n));
    std::vector<double> actual;
    actual.reserve(reference.size());
    for (const std::int32_t column : columns) {
        device_out.copy_to_host(column_bits.data(), column_bits.size() * sizeof(std::uint16_t),
                                static_cast<std::size_t>(column) * n * sizeof(std::uint16_t));
        for (const std::int32_t row : rows) actual.push_back(bf16_to_f32(column_bits[row]));
    }

    const ReductionCriterion criterion =
        linear_criterion(spec.judge_a16 ? ActivationCompute::A16 : spec.compute);
    const ReductionStats stats = compute_reduction_stats(actual.data(), reference.data(),
                                                         static_cast<std::int64_t>(actual.size()));
    const bool passes =
        reduction_passes(stats, static_cast<std::int64_t>(actual.size()), criterion);
    const bool expect_pass = spec.corruption == Corruption::None &&
                             (!spec.judge_a16 || spec.policy == ops::LinearPolicy::A16Only);
    const double gross = gross_error_limit(stats, criterion);
    std::printf("GEMMA4_CASE %s%s corruption=%s expect=%s result=%s rel_l2=%.6g rel_l2_limit=%.6g "
                "rel_l2_ratio=%.4g gross_ratio=%.4g non_finite=%d\n",
                label.c_str(), spec.judge_a16 ? " judged=A16" : "",
                corruption_name(spec.corruption), expect_pass ? "pass" : "fail",
                passes ? "pass" : "fail", stats.relative_l2, criterion.relative_l2,
                stats.relative_l2 / criterion.relative_l2,
                gross > 0 ? stats.maximum_absolute_error / gross : 0.0,
                stats.first_non_finite >= 0 ? 1 : 0);
    if (passes == expect_pass) return 0;
    std::cerr << label << ": expected the criterion to " << (expect_pass ? "pass" : "fail") << " ("
              << corruption_name(spec.corruption) << ")\n";
    return 1;
}

std::string case_label(const char* format, const Problem& problem, const char* path, std::int32_t t,
                       const Segment& segment) {
    std::string label = std::string(format) + " [" + std::to_string(problem.n) + "," +
                        std::to_string(problem.k) + "] " + path + " T=" + std::to_string(t);
    if (segment.begin >= 0)
        label += " segment=" + std::to_string(segment.begin) + "+" + std::to_string(segment.count);
    return label;
}

// Segment cases, then controls, for one problem and one path over the listed Ts.
int run_isolation(const char* format, const HostProblem& host, ActivationCompute compute,
                  ops::LinearPolicy policy, const char* path, std::int32_t segment_width,
                  std::initializer_list<int> ts, std::initializer_list<Corruption> controls) {
    int failures           = 0;
    const Problem& problem = host.problem();
    const Segment last{problem.k - segment_width, segment_width};
    const Segment first{0, segment_width};
    for (const int t : ts) {
        for (const Segment& segment : {last, first}) {
            failures += run_case(case_label(format, problem, path, t, segment), host,
                                 {compute, policy, t, segment, Corruption::None});
        }
        for (const Corruption corruption : controls) {
            // The cleared last block only shows on activations that live in it.
            const Segment segment = corruption == Corruption::LastBlockZero ? last : Segment{};
            failures += run_case(case_label(format, problem, path, t, segment), host,
                                 {compute, policy, t, segment, corruption});
        }
    }
    return failures;
}

std::vector<Invocation> policy_calls(ops::LinearPolicy policy, std::initializer_list<int> ts,
                                     std::initializer_list<int> graph_ts) {
    std::vector<Invocation> calls;
    for (const int t : ts) calls.push_back({t, CallForm::Policy, policy});
    for (const int t : graph_ts) calls.push_back({t, CallForm::Policy, policy, true});
    return calls;
}

#if defined(NINFERNO_GEMMA4_FORMAT_NVFP4)
constexpr const char* kFormat = "NVFP4";
constexpr Problem kProblems[]{
    {8192, 2816, 901U},  {1024, 2816, 903U},  {2816, 4096, 905U},  {2816, 8192, 907U},
    {2816, 2112, 909U},  {16384, 5376, 911U}, {18432, 5376, 913U}, {2048, 5376, 915U},
    {43008, 5376, 917U}, {5376, 8192, 919U},  {5376, 16384, 921U}, {5376, 21504, 923U},
};

int run_format() {
    int failures = 0;
    // A16 covers the decode route, every token bucket of the tensor-core route (2..8, 9..16,
    // 17..32, 33..64), the chunk edge and chunked prefill. A4 covers the A16 region below each
    // crossover, every MMA band edge and the TMA route, including partial last tiles on both sides
    // of each boundary.
    auto a16 = policy_calls(ops::LinearPolicy::A16Only,
                            {1,  2,  3,  4,  5,  6,  7,  8,  9,   10,  11,  12,   13,  14, 15,
                             16, 17, 18, 20, 21, 23, 24, 25, 27,  28,  29,  31,   32,  33, 40,
                             47, 48, 56, 63, 64, 65, 96, 97, 127, 128, 129, 1000, 2048},
                            {1, 2, 7, 16, 33, 64});
    a16.push_back({1, CallForm::A16Convenience});
    a16.push_back({33, CallForm::A16Convenience});
    a16.push_back({9, CallForm::Policy, ops::LinearPolicy::AllowA8});
    const auto a4 = policy_calls(
        ops::LinearPolicy::AllowA4,
        {1,   2,   3,    4,    5,    6,    7,    8,    9,    10,   11,   12,   13,  16,  17,
         19,  20,  24,   31,   32,   33,   48,   49,   63,   64,   65,   96,   97,  127, 128,
         129, 160, 191,  192,  193,  255,  256,  257,  383,  384,  385,  511,  512, 513, 767,
         768, 769, 1000, 1023, 1024, 1025, 1535, 1536, 1537, 2048, 2049, 3072, 4096},
        {5, 33, 257, 1024});
    for (const Problem& problem : kProblems) {
        failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_hashed_nvfp4_weight,
                              {problem.n, problem.k, problem.seed, Comparison::Sampled, true, a16});
        failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_hashed_nvfp4_weight,
                              {problem.n, problem.k, problem.seed, Comparison::Sampled, false, a4});
        failures += verify_workspace_envelopes(QType::NVFP4, problem.n, problem.k);

        const HostProblem host(make_hashed_nvfp4_weight, problem);
        failures +=
            run_isolation(kFormat, host, ActivationCompute::A16, ops::LinearPolicy::A16Only, "A16",
                          64, {1, 5, 16, 32, 33, 64},
                          {Corruption::ScaleShift, Corruption::LastBlockZero, Corruption::RowSwap});
        failures +=
            run_isolation(kFormat, host, ActivationCompute::A4, ops::LinearPolicy::AllowA4, "A4",
                          256, {64, 300, 2048},
                          {Corruption::ScaleShift, Corruption::LastBlockZero, Corruption::RowSwap});
        // One activation, both policies, the A16 criterion. T=64 is past every shape's A4
        // crossover, so AllowA4 serves it W4A4 and must fail where A16Only passes.
        const CaseSpec judged[]{
            {ActivationCompute::A4, ops::LinearPolicy::A16Only, 16, Segment{}, Corruption::None,
             true},
            {ActivationCompute::A4, ops::LinearPolicy::A16Only, 64, Segment{}, Corruption::None,
             true},
            {ActivationCompute::A4, ops::LinearPolicy::AllowA4, 64, Segment{}, Corruption::None,
             true},
        };
        for (const CaseSpec& spec : judged) {
            const char* path = spec.policy == ops::LinearPolicy::A16Only ? "A16" : "A4";
            failures +=
                run_case(case_label(kFormat, problem, path, spec.t, spec.segment), host, spec);
        }
    }
    // Every output of the two smallest problems, including the ragged-K one, on both paths.
    const std::vector<Invocation> full_a16 =
        policy_calls(ops::LinearPolicy::A16Only, {1, 2, 5, 31, 32, 33, 65}, {});
    const std::vector<Invocation> full_a4 =
        policy_calls(ops::LinearPolicy::AllowA4, {1, 9, 33, 129, 300, 1025}, {});
    for (const Problem& problem : {kProblems[1], kProblems[4]}) {
        failures +=
            run_shape("NVFP4_A16", ActivationCompute::A16, make_hashed_nvfp4_weight,
                      {problem.n, problem.k, problem.seed + 1U, Comparison::Full, false, full_a16});
        failures +=
            run_shape("NVFP4_A4", ActivationCompute::A4, make_hashed_nvfp4_weight,
                      {problem.n, problem.k, problem.seed + 1U, Comparison::Full, false, full_a4});
    }
    return failures;
}
#elif defined(NINFERNO_GEMMA4_FORMAT_Q8)
constexpr const char* kFormat = "Q8";

// [4224,2816] is the 26B-A4B MoE-side projection; the rest are the attention and dense-MLP
// projections a BF16 checkpoint converts to Q8 (q+k+v and q+k packed into one parent).
constexpr Problem kProblems[]{
    {4224, 2816, 931U}, {8192, 2816, 933U},  {9216, 2816, 935U},  {2816, 4096, 937U},
    {2816, 8192, 939U}, {2816, 2112, 941U},  {16384, 5376, 943U}, {18432, 5376, 945U},
    {5376, 8192, 947U}, {5376, 16384, 949U},
};

// Google's Gemma 4 MTP drafters, every projection in Q8: 26B-A4B pre_projection, query, attention
// output, packed gate/up, down, post_projection and head; the 31B-only ones; and the shortlisted
// drafter heads.
constexpr Problem kDrafterProblems[]{
    {1024, 5632, 951U},  {4096, 1024, 953U},  {8192, 1024, 955U},   {1024, 4096, 957U},
    {1024, 8192, 959U},  {16384, 1024, 961U}, {2816, 1024, 963U},   {262144, 1024, 965U},
    {1024, 10752, 967U}, {1024, 16384, 969U}, {5376, 1024, 971U},   {32768, 1024, 973U},
    {65536, 1024, 975U}, {98304, 1024, 977U}, {131072, 1024, 979U},
};

// z-lab's DFlash drafters, every projection in Q8. 26B-A4B: fc over six target layers, the
// context K/V of all five layers in one plane, block q+k+v, gate/up and down; 31B: fc, the
// context K/V plane (which is also the block q+k+v), gate/up and down. The attention outputs
// [2816,4096] and [5376,8192] are the kProblems entries above.
constexpr Problem kDflashProblems[]{
    {2816, 16896, 981U}, {10240, 2816, 983U}, {6144, 2816, 985U},
    {11264, 2816, 987U}, {2816, 5632, 989U},  {5376, 32256, 991U},
    {10240, 5376, 993U}, {21504, 5376, 995U}, {5376, 10752, 997U},
};

int run_format() {
    int failures = 0;
    // Every k-split capacity and its edges, the tiled bands and their edges, chunked large T.
    auto calls = policy_calls(
        ops::LinearPolicy::A16Only,
        {1,   2,   3,   4,   5,   6,   7,    8,    9,    13,   14,   15,   16,   17,  23,  24,  25,
         31,  32,  33,  40,  41,  47,  48,   49,   56,   57,   63,   64,   65,   72,  73,  95,  96,
         97,  127, 128, 129, 160, 161, 191,  192,  193,  255,  256,  257,  320,  321, 383, 384, 385,
         511, 512, 513, 767, 768, 769, 1000, 1024, 1025, 1536, 1537, 2048, 2049, 4096},
        {1, 16, 64, 128});
    for (const int t : {1, 64}) calls.push_back({t, CallForm::A16Convenience});
    calls.push_back({17, CallForm::Policy, ops::LinearPolicy::AllowA8});
    calls.push_back({64, CallForm::Policy, ops::LinearPolicy::AllowA4});
    for (const Problem& problem : kProblems) {
        failures +=
            run_shape("Q8_A16", ActivationCompute::A16, make_q8_g32_fp16_weight,
                      {problem.n, problem.k, problem.seed, Comparison::Sampled, true, calls});
        const HostProblem host(make_q8_g32_fp16_weight, problem);
        failures +=
            run_isolation(kFormat, host, ActivationCompute::A16, ops::LinearPolicy::A16Only, "A16",
                          64, {1, 5, 48, 65, 1000, 2048},
                          {Corruption::ScaleShift, Corruption::LastBlockZero, Corruption::RowSwap});
    }
    // Every output of the smallest problem and of the one whose K is not a whole number of
    // 128-value rows (K=2112 is stored padded to 2176).
    const std::vector<Invocation> full =
        policy_calls(ops::LinearPolicy::A16Only, {1, 5, 25, 33, 65, 257}, {});
    for (const Problem& problem : {kProblems[0], kProblems[5]}) {
        failures +=
            run_shape("Q8_A16", ActivationCompute::A16, make_q8_g32_fp16_weight,
                      {problem.n, problem.k, problem.seed + 1U, Comparison::Full, false, full});
    }

    // The MTP drafters run one draft token per call and verify up to nine: every T through 16
    // crosses a capacity edge of the split-K GEMV or the K-split, and 17/33/100 take the MMA.
    const auto drafter_calls = policy_calls(
        ops::LinearPolicy::A16Only,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 33, 100}, {1, 2, 5, 9});
    for (const Problem& problem : kDrafterProblems) {
        failures += run_shape(
            "Q8_A16", ActivationCompute::A16, make_q8_g32_fp16_weight,
            {problem.n, problem.k, problem.seed, Comparison::Sampled, true, drafter_calls});
        const HostProblem host(make_q8_g32_fp16_weight, problem);
        failures += run_isolation(
            kFormat, host, ActivationCompute::A16, ops::LinearPolicy::A16Only, "A16", 64,
            {1, 2, 5, 9}, {Corruption::ScaleShift, Corruption::LastBlockZero, Corruption::RowSwap});
    }
    // DFlash runs its block at T=16 and its fc / context K/V at the accepted count, T=1..17, plus
    // prefill chunks up to 2048: every T through 17, every capacity edge through 65 and the tiled
    // bands up to 4096.
    const auto dflash_calls = policy_calls(
        ops::LinearPolicy::A16Only,
        {1,   2,   3,   4,   5,   6,   7,   8,   9,    10,   11,   12,   13,   14,   15,   16,
         17,  18,  24,  25,  32,  33,  48,  49,  64,   65,   96,   97,   128,  129,  192,  193,
         256, 257, 384, 385, 512, 513, 768, 769, 1000, 1024, 1025, 1536, 1537, 2048, 2049, 4096},
        {1, 16, 17, 2048});
    for (const Problem& problem : kDflashProblems) {
        failures += run_shape(
            "Q8_A16", ActivationCompute::A16, make_q8_g32_fp16_weight,
            {problem.n, problem.k, problem.seed, Comparison::Sampled, true, dflash_calls});
        const HostProblem host(make_q8_g32_fp16_weight, problem);
        failures +=
            run_isolation(kFormat, host, ActivationCompute::A16, ops::LinearPolicy::A16Only, "A16",
                          64, {1, 16, 17, 2048},
                          {Corruption::ScaleShift, Corruption::LastBlockZero, Corruption::RowSwap});
    }
    // Every output of the DFlash block problem at its hot T.
    failures += run_shape("Q8_A16", ActivationCompute::A16, make_q8_g32_fp16_weight,
                          {kDflashProblems[2].n, kDflashProblems[2].k, kDflashProblems[2].seed + 1U,
                           Comparison::Full, false,
                           policy_calls(ops::LinearPolicy::A16Only, {1, 16, 17}, {})});

    // Every output of the thin drafter problems, where the split-K GEMV spreads a row over warps.
    const std::vector<Invocation> drafter_full =
        policy_calls(ops::LinearPolicy::A16Only, {1, 2, 3, 6, 8, 9, 16}, {});
    for (const Problem& problem : {kDrafterProblems[0], kDrafterProblems[9]}) {
        failures += run_shape(
            "Q8_A16", ActivationCompute::A16, make_q8_g32_fp16_weight,
            {problem.n, problem.k, problem.seed + 1U, Comparison::Full, false, drafter_full});
    }
    return failures;
}
#elif defined(NINFERNO_GEMMA4_FORMAT_Q6)
constexpr const char* kFormat = "Q6";

int run_format() {
    int failures = 0;
    for (const Problem& problem : {Problem{262144, 2816, 941U}, Problem{262144, 5376, 943U}}) {
        const auto calls = policy_calls(ops::LinearPolicy::A16Only,
                                        {1,  2,  3,  4,  5,  6,   7,   8,   9,   16,  17,   24,
                                         25, 32, 33, 40, 41, 48,  49,  56,  57,  64,  65,   72,
                                         73, 80, 81, 96, 97, 112, 113, 127, 128, 129, 1000, 2048},
                                        {1, 7, 33});
        failures +=
            run_shape("Q6_A16", ActivationCompute::A16, make_q6_g64_fp16_weight,
                      {problem.n, problem.k, problem.seed, Comparison::Sampled, false, calls});
        const HostProblem host(make_q6_g64_fp16_weight, problem);
        failures +=
            run_isolation(kFormat, host, ActivationCompute::A16, ops::LinearPolicy::A16Only, "A16",
                          64, {1, 7, 33, 129},
                          {Corruption::ScaleShift, Corruption::LastBlockZero, Corruption::RowSwap});
    }
    return failures;
}
#else
#    error "select a weight format with NINFERNO_GEMMA4_FORMAT_*"
#endif

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_format();
        std::cout << (failures == 0 ? "OK" : "FAIL") << ' ' << kFormat
                  << " Gemma 4 Linear shapes\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << kFormat << " Gemma 4 Linear shapes: " << error.what() << '\n';
        return 1;
    }
}
