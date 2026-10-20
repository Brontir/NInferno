// MoE expert offload's CPU decode path (src/ops/host_experts): the canonical row dots against an
// FP64 oracle, byte equality across instruction sets and thread counts, the worker pool, and the
// two layer computations against an FP64 formula. CPU only.
#include "ops/host_experts/host_kernels.h"
#include "ops/host_experts/host_moe.h"
#include "ops/host_experts/host_pool.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::ops::host;
namespace qw = ninferno::test::quantized_weight;
using ninferno::ops::MoeActivation;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

std::vector<float> random_inputs(int n, std::uint32_t seed, float scale = 1.0F) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(static_cast<std::size_t>(n));
    for (auto& x : v) { x = d(rng); }
    return v;
}

std::uint16_t to_bf16(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    b += 0x7fffU + ((b >> 16) & 1U);
    return static_cast<std::uint16_t>(b >> 16);
}

float from_bf16(std::uint16_t h) {
    const std::uint32_t b = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

std::vector<Isa> isas() {
    std::vector<Isa> out{Isa::Scalar};
    if (isa_available(Isa::Avx2)) { out.push_back(Isa::Avx2); }
    if (isa_available(Isa::Avx512)) { out.push_back(Isa::Avx512); }
    return out;
}

void test_tables() {
    check(e2m1_values()[7] == 6.0F && e2m1_values()[15] == -6.0F && std::signbit(e2m1_values()[8]),
          "E2M1 table");
    check(e8m0_values()[127] == 1.0F && e8m0_values()[128] == 2.0F && std::isnan(e8m0_values()[255]) &&
              e8m0_values()[0] == std::ldexp(1.0F, -127),
          "E8M0 table");
    check(e4m3_values()[0x38] == 1.0F && e4m3_values()[0x7e] == 448.0F && std::isnan(e4m3_values()[0x7f]) &&
              e4m3_values()[0x01] == std::ldexp(1.0F, -9) && e4m3_values()[0xb8] == -1.0F,
          "E4M3 table");
    for (int j = 0; j < 64; ++j) {
        std::vector<float> x(64, 0.0F), p(64);
        x[static_cast<std::size_t>(j)] = 1.0F;
        permute_input(x.data(), 64, p.data());
        check(p[static_cast<std::size_t>(permuted_index(j))] == 1.0F, "permuted_index matches permute_input");
    }
}

// Every column of every row: against the FP64 oracle within a bound of the dot's magnitude, and
// byte-equal across instruction sets.
void test_row_dots(QType qtype) {
    const int n = 128, k = 256;
    qw::PatternedWeightOptions options;
    if (qtype == QType::NVFP4) {
        options.weight_scale_divisor = 3.0F;
        options.input_scale_divisor  = 1.0F;
    }
    const auto packed    = qw::make_patterned_weight(qtype, n, k, 7, options);
    const HostPlane plane = host_plane(packed.weight);
    const int columns     = 11;
    std::vector<std::vector<float>> x, xp;
    for (int c = 0; c < columns; ++c) {
        x.push_back(random_inputs(k, 100 + c));
        xp.emplace_back(static_cast<std::size_t>(k));
        permute_input(x.back().data(), k, xp.back().data());
    }
    std::vector<const float*> in;
    for (auto& v : xp) { in.push_back(v.data()); }
    for (int row = 0; row < n; ++row) {
        std::vector<float> reference(static_cast<std::size_t>(columns));
        bool first = true;
        for (const Isa isa : isas()) {
            const RowKernels kernels = row_kernels(isa);
            std::vector<float> out(static_cast<std::size_t>(columns));
            const std::uint8_t* codes = plane.codes + static_cast<std::ptrdiff_t>(row) * (k / 2);
            if (qtype == QType::MXFP4) {
                kernels.mxfp4(codes, plane.scales + row * (k / 32), k / 32, in.data(), columns, out.data());
            } else {
                kernels.nvfp4(codes, nvfp4_scale_row(plane.scales, row, k), k / 32, in.data(), columns,
                              out.data());
                for (auto& v : out) { v /= plane.divisor; }
            }
            if (first) {
                reference = out;
                first     = false;
                for (int c = 0; c < columns; ++c) {
                    const double exact = qw::dot_fp64(packed, row, x[static_cast<std::size_t>(c)].data(), k);
                    double magnitude   = 0.0;
                    for (int j = 0; j < k; ++j) {
                        magnitude += std::fabs(qw::logical_weight_fp64(packed, row, j) *
                                               x[static_cast<std::size_t>(c)][static_cast<std::size_t>(j)]);
                    }
                    if (std::fabs(out[static_cast<std::size_t>(c)] - exact) > 1e-5 * magnitude + 1e-30) {
                        std::printf("  row %d column %d: %.9g vs %.9g (magnitude %.3g)\n", row, c,
                                    out[static_cast<std::size_t>(c)], exact, magnitude);
                        check(false, qtype == QType::MXFP4 ? "MXFP4 row dot vs FP64" : "NVFP4 row dot vs FP64");
                    }
                }
            } else {
                check(std::memcmp(out.data(), reference.data(), out.size() * sizeof(float)) == 0,
                      qtype == QType::MXFP4 ? "MXFP4 row dot bytes across ISAs"
                                            : "NVFP4 row dot bytes across ISAs");
            }
        }
    }
}

void test_pool() {
    HostPool pool(0);
    check(pool.size() >= 1, "pool has the caller");
    std::vector<int> hits(1000);
    auto body = [&](int task) { hits[static_cast<std::size_t>(task)] += 1; };
    for (int round = 0; round < 2000; ++round) { pool.run(static_cast<int>(hits.size()), body); }
    bool all = true;
    for (int h : hits) { all = all && h == 2000; }
    check(all, "pool runs every task once per run");
    // Past the spin: the workers sleep and must wake for the next run.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::atomic<int> sum{0};
    auto add = [&](int task) { sum.fetch_add(task); };
    pool.run(100, add);
    check(sum.load() == 4950, "pool wakes sleeping workers");
    HostPool one(1);
    check(one.size() == 1, "a one-thread pool is the caller alone");
    one.run(10, add);
    check(sum.load() == 4950 + 45, "a one-thread pool runs on the caller");
}

// FP64 formula of the generic layer: mid and y rounded to BF16 where the GPU route rounds.
struct GenericFixture {
    int E = 8, H = 64, F = 64, K = 2;
    qw::PackedWeight gate_up, down;
    std::vector<float> gate_up_bias, down_bias;
    GenericLayer layer;
    GenericFixture(QType qtype, bool interleaved, MoeActivation act) {
        qw::PatternedWeightOptions o;
        if (qtype == QType::NVFP4) {
            o.weight_scale_divisor = 2.0F;
            o.input_scale_divisor  = 1.0F;
            o.upper_half_seed      = 5;
        }
        gate_up      = qw::make_patterned_weight(qtype, E * 2 * F, H, 11, o);
        down         = qw::make_patterned_weight(qtype, E * H, F, 13, o);
        gate_up_bias = random_inputs(E * 2 * F, 21, 0.1F);
        down_bias    = random_inputs(E * H, 22, 0.1F);
        layer.gate_up = host_plane(gate_up.weight);
        layer.down    = host_plane(down.weight);
        layer.gate_up_bias = gate_up_bias.data();
        layer.down_bias    = down_bias.data();
        layer.experts = E;
        layer.hidden  = H;
        layer.inter   = F;
        layer.top_k   = K;
        layer.activation  = act;
        layer.interleaved = interleaved;
        layer.limit = 7.0F;
        layer.alpha = 1.702F;
    }
    double act(double g, double u) const {
        switch (layer.activation) {
        case MoeActivation::ClampedSwiGlu: {
            const double gc = std::fmin(g, layer.limit), uc = std::fmin(std::fmax(u, -layer.limit), layer.limit);
            return (uc + 1.0) * gc / (1.0 + std::exp(-layer.alpha * gc));
        }
        case MoeActivation::GeGluTanh:
            return 0.5 * g * (1.0 + std::tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))) * u;
        case MoeActivation::SwiGlu:
            break;
        }
        return g / (1.0 + std::exp(-g)) * u;
    }
    std::vector<float> expected(int tokens, const std::vector<std::uint16_t>& x, const std::vector<std::int32_t>& ids) const {
        std::vector<float> y(static_cast<std::size_t>(tokens) * K * H);
        std::vector<float> xf(static_cast<std::size_t>(H)), mid(static_cast<std::size_t>(F));
        for (int a = 0; a < tokens * K; ++a) {
            const int t = a / K, e = ids[static_cast<std::size_t>(a)];
            for (int h = 0; h < H; ++h) { xf[static_cast<std::size_t>(h)] = from_bf16(x[static_cast<std::size_t>(t) * H + h]); }
            for (int j = 0; j < F; ++j) {
                const int gr = e * 2 * F + (layer.interleaved ? 2 * j : j);
                const int ur = e * 2 * F + (layer.interleaved ? 2 * j + 1 : F + j);
                const double g = qw::dot_fp64(gate_up, gr, xf.data(), H) + gate_up_bias[static_cast<std::size_t>(gr)];
                const double u = qw::dot_fp64(gate_up, ur, xf.data(), H) + gate_up_bias[static_cast<std::size_t>(ur)];
                mid[static_cast<std::size_t>(j)] = from_bf16(to_bf16(static_cast<float>(act(g, u))));
            }
            for (int h = 0; h < H; ++h) {
                const int r = e * H + h;
                y[static_cast<std::size_t>(a) * H + h] =
                    static_cast<float>(qw::dot_fp64(down, r, mid.data(), F) + down_bias[static_cast<std::size_t>(r)]);
            }
        }
        return y;
    }
};

void inputs(int tokens, int H, int E, int K, std::uint32_t seed, std::vector<std::uint16_t>& x,
            std::vector<std::int32_t>& ids) {
    const auto xf = random_inputs(tokens * H, seed);
    x.resize(xf.size());
    for (std::size_t i = 0; i < xf.size(); ++i) { x[i] = to_bf16(xf[i]); }
    std::mt19937 rng(seed + 1);
    ids.clear();
    for (int t = 0; t < tokens; ++t) {
        std::vector<int> pool(static_cast<std::size_t>(E));
        for (int e = 0; e < E; ++e) { pool[static_cast<std::size_t>(e)] = e; }
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int k = 0; k < K; ++k) { ids.push_back(pool[static_cast<std::size_t>(k)]); }
    }
}

// y within a few BF16 steps of the formula (mid is rounded on both sides, so a near-tie may round
// differently), and byte-equal across instruction sets and thread counts.
void test_generic(QType qtype, bool interleaved, MoeActivation act, const char* name) {
    GenericFixture f(qtype, interleaved, act);
    for (int tokens : {1, 3, 16}) {
        std::vector<std::uint16_t> x;
        std::vector<std::int32_t> ids;
        inputs(tokens, f.H, f.E, f.K, 40 + tokens, x, ids);
        const auto expected = f.expected(tokens, x, ids);
        std::vector<std::uint16_t> reference;
        for (const Isa isa : isas()) {
            for (int threads : {1, 3, 0}) {
                HostMoe moe(threads, isa);
                std::vector<std::uint16_t> y(expected.size());
                moe.generic(f.layer, tokens, x.data(), ids.data(), y.data());
                if (reference.empty()) {
                    reference = y;
                    double worst = 0.0;
                    for (std::size_t i = 0; i < y.size(); ++i) {
                        const double err = std::fabs(from_bf16(y[i]) - expected[i]) /
                                           (std::fabs(expected[i]) + 1e-2);
                        worst = std::max(worst, err);
                    }
                    if (worst > 0.05) {
                        std::printf("  %s T=%d worst relative error %.4f\n", name, tokens, worst);
                        check(false, name);
                    }
                } else {
                    check(y == reference, "generic bytes across ISAs and thread counts");
                }
            }
        }
    }
}

void test_gemma4() {
    const int E = 8, H = 128, F = 64, K = 4;
    qw::PatternedWeightOptions o;
    o.weight_scale_divisor = 2.5F;
    o.input_scale_divisor  = 1.0F;
    o.upper_half_seed      = 9;
    const auto gate_up = qw::make_patterned_weight(QType::NVFP4, E * 2 * F, H, 31, o);
    const auto down    = qw::make_patterned_weight(QType::NVFP4, E * H, F, 33, o);
    Gemma4Layer layer;
    layer.gate_up = host_plane(gate_up.weight);
    layer.down    = host_plane(down.weight);
    layer.experts = E;
    layer.hidden  = H;
    layer.inter   = F;
    layer.top_k   = K;
    for (int tokens : {1, 5, 16}) {
        std::vector<std::uint16_t> x;
        std::vector<std::int32_t> ids;
        inputs(tokens, H, E, K, 70 + tokens, x, ids);
        const auto w = random_inputs(tokens * K, 90 + tokens, 0.3F);
        std::vector<float> expected(static_cast<std::size_t>(tokens) * H, 0.0F);
        for (int t = 0; t < tokens; ++t) {
            std::vector<float> xf(static_cast<std::size_t>(H));
            for (int h = 0; h < H; ++h) { xf[static_cast<std::size_t>(h)] = from_bf16(x[static_cast<std::size_t>(t) * H + h]); }
            std::vector<double> acc(static_cast<std::size_t>(H), 0.0);
            for (int k = 0; k < K; ++k) {
                const int e = ids[static_cast<std::size_t>(t * K + k)];
                std::vector<float> mid(static_cast<std::size_t>(F));
                for (int j = 0; j < F; ++j) {
                    const double g = qw::dot_fp64(gate_up, e * 2 * F + j, xf.data(), H);
                    const double u = qw::dot_fp64(gate_up, e * 2 * F + F + j, xf.data(), H);
                    mid[static_cast<std::size_t>(j)] = static_cast<float>(
                        0.5 * g * (1.0 + std::tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))) * u);
                }
                for (int h = 0; h < H; ++h) {
                    acc[static_cast<std::size_t>(h)] +=
                        w[static_cast<std::size_t>(t * K + k)] * qw::dot_fp64(down, e * H + h, mid.data(), F);
                }
            }
            for (int h = 0; h < H; ++h) { expected[static_cast<std::size_t>(t) * H + h] = static_cast<float>(acc[static_cast<std::size_t>(h)]); }
        }
        std::vector<std::uint16_t> reference;
        for (const Isa isa : isas()) {
            for (int threads : {1, 2, 0}) {
                HostMoe moe(threads, isa);
                std::vector<std::uint16_t> out(expected.size());
                moe.gemma4(layer, tokens, x.data(), ids.data(), w.data(), out.data());
                if (reference.empty()) {
                    reference = out;
                    double worst = 0.0;
                    for (std::size_t i = 0; i < out.size(); ++i) {
                        worst = std::max(worst, std::fabs(from_bf16(out[i]) - expected[i]) /
                                                    (std::fabs(expected[i]) + 1e-2));
                    }
                    if (worst > 0.02) {
                        std::printf("  gemma4 T=%d worst relative error %.4f\n", tokens, worst);
                        check(false, "gemma4 layer vs FP64");
                    }
                } else {
                    check(out == reference, "gemma4 bytes across ISAs and thread counts");
                }
            }
        }
    }
}

void test_refusals() {
    const auto q8 = qw::make_patterned_weight(QType::Q8_G32_FP16, 128, 256, 3);
    std::string why;
    check(!host_readable(q8.weight, &why) && !why.empty(), "a Q8 bank is refused with a reason");
    bool threw = false;
    try {
        HostPool too_many(100000);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "more threads than physical cores is refused");
}

} // namespace

int main() {
    std::printf("host experts: ISA %s, %zu physical cores\n", isa_name(detect_isa()),
                physical_core_cpus().size());
    test_tables();
    test_row_dots(QType::MXFP4);
    test_row_dots(QType::NVFP4);
    test_pool();
    test_generic(QType::MXFP4, true, MoeActivation::ClampedSwiGlu, "generic MXFP4 interleaved clamped");
    test_generic(QType::NVFP4, false, MoeActivation::SwiGlu, "generic NVFP4 concatenated SwiGlu");
    test_generic(QType::MXFP4, false, MoeActivation::GeGluTanh, "generic MXFP4 GeGLU");
    test_gemma4();
    test_refusals();
    if (failures != 0) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("host experts: all checks passed\n");
    return 0;
}
