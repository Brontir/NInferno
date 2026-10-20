// The DFlash feature projection split by tap (models/speculation/dflash.h,
// split_projection_columns): for BF16, a GgmlBlocks format (Q4_K) and two GgmlPlanes formats (Q6_K,
// Q5_0), a [N, taps*K] matrix with random codes and varied scales is split in place into its
// column blocks. Each slice's projection of its tap's rows equals the whole matrix's projection of
// the input with only that tap's rows set, and the running sum the dense Program keeps (the first
// slice stored, each later one added by linear_add) equals the whole projection. A format or shape
// that does not split is refused and left as it was. GPU.
#include "core/weight_view.h"
#include "models/speculation/dflash.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/linear_add.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;
using ninferno::models::speculation::Projection;
using ninferno::models::speculation::split_projection_columns;

namespace {

constexpr std::int32_t kRows = 256;

QuantLayout layout_of(QType q) {
    if (q == QType::BF16) { return QuantLayout::Contiguous; }
    return q == QType::GGML_Q4_K ? QuantLayout::GgmlBlocks : QuantLayout::GgmlPlanes;
}

// A random FP16 scale in [2^-8, 2^-7): finite, positive, different from block to block.
std::uint16_t random_scale(std::mt19937& rng) {
    return static_cast<std::uint16_t>(0x1c00U | (rng() & 0x3ffU));
}

// A [kRows, k] matrix of `q` in its converter layout: random codes, varied finite scales.
struct Matrix {
    WeightParent parent;
    DeviceBuffer buffer;
    Weight weight;
};

Matrix make_matrix(QType q, std::int32_t k, std::uint32_t seed) {
    const std::array<std::uint64_t, 2> shape{kRows, static_cast<std::uint64_t>(k)};
    Matrix m;
    m.parent.geometry = weight_geometry(q, layout_of(q), shape);
    const WeightGeometry& g = m.parent.geometry;
    std::vector<std::uint8_t> host(g.bytes);
    std::mt19937 rng(seed);
    for (auto& b : host) { b = static_cast<std::uint8_t>(rng()); }
    const auto put16 = [&](std::uint64_t at, std::uint16_t word) {
        host[at]     = static_cast<std::uint8_t>(word & 0xffU);
        host[at + 1] = static_cast<std::uint8_t>(word >> 8);
    };
    if (q == QType::BF16) {
        // Magnitudes near 2^-7, either sign.
        for (std::uint64_t i = 1; i < g.code_bytes; i += 2) { host[i] = 0x3c ^ (host[i] & 0x81); }
    } else if (q == QType::GGML_Q4_K) {
        // Each 144-byte superblock starts with its binary16 d and dmin.
        for (std::uint64_t row = 0; row < kRows; ++row) {
            for (std::uint64_t block = 0; block < g.code_bytes_per_row / 144; ++block) {
                const std::uint64_t at = row * g.code_bytes_per_row + block * 144;
                put16(at, random_scale(rng));
                put16(at + 2, random_scale(rng));
            }
        }
    } else {
        for (std::uint64_t at = g.scale_offset; at + 1 < g.scale_offset + g.scale_bytes; at += 2) {
            put16(at, random_scale(rng));
        }
    }
    m.buffer      = to_device(host);
    m.parent.data = static_cast<const std::byte*>(m.buffer.p);
    const WeightView view{.shape = {shape[0], shape[1]}, .parts = {{&m.parent, 0, g.elements}}};
    m.weight = native_weight(view);
    return m;
}

// out = the projection of x; with `add`, out += it.
std::vector<double> project(const Tensor& x, const Weight& w, DeviceBuffer& out, bool add,
                            DeviceBuffer& scratch) {
    Tensor y(out.p, DType::BF16, {w.n, x.ne[1]});
    WorkspaceArena workspace(DeviceSpan{scratch.p, scratch.bytes});
    if (add) {
        ops::linear_add(x, w, y, ops::LinearPolicy::A16Only, workspace, nullptr);
    } else {
        ops::linear(x, w, y, ops::LinearPolicy::A16Only, workspace, nullptr);
    }
    cuda_synchronize();
    return from_device_bf16(out, static_cast<std::size_t>(w.n) * x.ne[1]);
}

// |got - want| within `share` of want's largest magnitude, at every element.
int compare(const std::string& what, const std::vector<double>& got,
            const std::vector<double>& want, double share) {
    double peak = 0.0, worst = 0.0;
    for (const double v : want) { peak = std::max(peak, std::abs(v)); }
    for (std::size_t i = 0; i < want.size(); ++i) {
        worst = std::max(worst, std::abs(got[i] - want[i]));
    }
    if (!(peak > 0.0) || !(worst <= share * peak)) {
        std::cerr << "FAIL: " << what << ": worst error " << worst << " against a peak of " << peak
                  << '\n';
        return 1;
    }
    return 0;
}

int test_split(QType q, const char* name, std::int32_t taps, std::int32_t k, std::int32_t tokens) {
    const std::string label =
        std::string(name) + " taps " + std::to_string(taps) + " T " + std::to_string(tokens);
    const std::int32_t full_k = taps * k;
    Matrix m                  = make_matrix(q, full_k, 7U + static_cast<std::uint32_t>(q));
    Projection whole{.weight = m.weight, .policy = ops::LinearPolicy::A16Only};

    std::vector<float> values(static_cast<std::size_t>(full_k) * tokens);
    fill_uniform(values, 11U, -1.0F, 1.0F);
    round_to_bf16(values);
    // Tap i's rows [i*k, (i+1)*k) of every column, alone and inside an otherwise zero input.
    const auto tap_rows = [&](std::int32_t tap, bool embedded) {
        const std::int32_t height = embedded ? full_k : k;
        const std::int32_t first  = embedded ? tap * k : 0;
        std::vector<float> out(static_cast<std::size_t>(height) * tokens, 0.0F);
        for (std::int32_t t = 0; t < tokens; ++t) {
            for (std::int32_t r = 0; r < k; ++r) {
                out[static_cast<std::size_t>(t) * height + first + r] =
                    values[static_cast<std::size_t>(t) * full_k + tap * k + r];
            }
        }
        return out;
    };

    DeviceBuffer scratch(std::size_t{64} << 20);
    DeviceBuffer out(static_cast<std::size_t>(kRows) * tokens * 2);
    DeviceBuffer x_full = to_device_bf16(values);
    const auto want     = project(Tensor(x_full.p, DType::BF16, {full_k, tokens}), whole.weight,
                                  out, false, scratch);
    std::vector<std::vector<double>> want_tap;
    for (std::int32_t tap = 0; tap < taps; ++tap) {
        DeviceBuffer x = to_device_bf16(tap_rows(tap, true));
        want_tap.push_back(
            project(Tensor(x.p, DType::BF16, {full_k, tokens}), whole.weight, out, false, scratch));
    }

    std::string why;
    const std::vector<Projection> slices =
        split_projection_columns(whole, static_cast<std::uint64_t>(taps), nullptr, &why);
    if (slices.size() != static_cast<std::size_t>(taps) || whole.weight.payload != nullptr) {
        std::cerr << "FAIL: " << label << ": not split (" << why << ")\n";
        return 1;
    }
    int failures = 0;
    std::vector<DeviceBuffer> x_tap;
    for (std::int32_t tap = 0; tap < taps; ++tap) {
        x_tap.push_back(to_device_bf16(tap_rows(tap, false)));
        const auto got = project(Tensor(x_tap.back().p, DType::BF16, {k, tokens}),
                                 slices[static_cast<std::size_t>(tap)].weight, out, false, scratch);
        failures += compare(label + ": slice " + std::to_string(tap), got,
                            want_tap[static_cast<std::size_t>(tap)], 1.0 / 64.0);
    }
    // The Program's sum: the first tap stored, each later one added, in order.
    std::vector<double> sum;
    for (std::int32_t tap = 0; tap < taps; ++tap) {
        sum = project(Tensor(x_tap[static_cast<std::size_t>(tap)].p, DType::BF16, {k, tokens}),
                      slices[static_cast<std::size_t>(tap)].weight, out, tap != 0, scratch);
    }
    failures += compare(label + ": the sum over taps", sum, want, 1.0 / 32.0);
    return failures;
}

// A format or shape that does not split: no slices, the matrix untouched.
int test_refused(QType q, const char* name, std::int32_t taps, std::int32_t k) {
    Matrix m = make_matrix(q, taps * k, 3U);
    Projection whole{.weight = m.weight, .policy = ops::LinearPolicy::A16Only};
    const auto before = from_device<std::uint8_t>(m.buffer, m.buffer.bytes);
    std::string why;
    const auto slices =
        split_projection_columns(whole, static_cast<std::uint64_t>(taps), nullptr, &why);
    const auto after = from_device<std::uint8_t>(m.buffer, m.buffer.bytes);
    if (!slices.empty() || why.empty() || whole.weight.payload == nullptr || before != after) {
        std::cerr << "FAIL: " << name << " should be refused and left whole\n";
        return 1;
    }
    std::cout << name << ": refused (" << why << ")\n";
    return 0;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        int failures = 0;
        for (const std::int32_t tokens : {5, 67}) {
            failures += test_split(QType::BF16, "BF16", 3, 256, tokens);
            failures += test_split(QType::GGML_Q4_K, "Q4_K", 3, 256, tokens);
            failures += test_split(QType::GGML_Q6_K, "Q6_K", 4, 256, tokens);
            failures += test_split(QType::GGML_Q5_0, "Q5_0", 3, 128, tokens);
        }
        // A tap's columns are not whole Q4_K superblocks.
        failures += test_refused(QType::GGML_Q4_K, "Q4_K at 128 columns a tap", 2, 128);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " dflash feature projection split\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "dflash split test: " << error.what() << '\n';
        return 1;
    }
}
