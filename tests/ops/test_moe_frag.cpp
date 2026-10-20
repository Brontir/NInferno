// ggml_frag_v1 routed-expert banks (core/ggml_frag.h) in the grouped MoE kernel: for every format
// with a map, a layer whose gate/up and down banks hold their code bytes in fragment order gives
// the same bytes as the published layer at every T from 1 to 64, under the prefill and invariant
// profiles, on every compiled schedule pair, and (IQ2_S, IQ3_XXS, Q4_K, Q4_0) on the A8 route's
// R8 kernel at decode widths. A control reads the fragment-order bytes as published and must differ
// (the permutation is not the identity on this data). A projection (ops::linear) refuses a frag
// weight.
#include "core/arena.h"
#include "core/ggml_frag.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/moe/moe_internal.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninferno;
namespace qw   = ninferno::test::quantized_weight;
namespace dmoe = ninferno::ops::detail::moe;

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    }
}

std::uint16_t to_bf16(float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, 4);
    return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

// One bank on the device, published, and its fragment-order copy.
struct Bank {
    qw::PackedWeight packed;
    DeviceBuffer published, frag;
    Weight published_weight{}, frag_weight{};
};

std::unique_ptr<Bank> make_bank(QType format, std::int32_t n, std::int32_t k, std::uint32_t seed) {
    auto bank    = std::make_unique<Bank>();
    bank->packed = qw::make_patterned_weight(format, n, k, seed);
    const std::uint64_t shape[] = {static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k)};
    const auto g = weight_geometry(format, bank->packed.weight.layout, shape);
    if (g.code_bytes_per_row * n != bank->packed.code_plane_bytes) {
        throw std::runtime_error("the fixture's code plane is not the geometry's");
    }
    auto bytes = bank->packed.payload;
    if (ggml_frag_format(format)) {
        for (std::int32_t r = 0; r < n; ++r) {
            const auto at = static_cast<std::size_t>(r) * g.code_bytes_per_row;
            ggml_frag_row(format, static_cast<std::uint64_t>(k), bank->packed.payload.data() + at,
                          bytes.data() + at, true);
        }
    }
    bank->published = DeviceBuffer(bank->packed.payload.size());
    bank->published.copy_from_host(bank->packed.payload.data(), bank->published.bytes);
    bank->frag = DeviceBuffer(bytes.size());
    bank->frag.copy_from_host(bytes.data(), bank->frag.bytes);
    bank->published_weight = bank->packed.device_weight(bank->published.p);
    bank->frag_weight      = bank->packed.device_weight(bank->frag.p);
    if (ggml_frag_format(format)) { bank->frag_weight.layout = QuantLayout::GgmlFrag; }
    return bank;
}

std::unique_ptr<Bank> make_router(std::int32_t experts, std::int32_t hidden, std::uint32_t seed) {
    auto bank = std::make_unique<Bank>();
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.08F);
    std::vector<std::uint16_t> words(static_cast<std::size_t>(experts) * hidden);
    for (auto& w : words) { w = to_bf16(normal(rng)); }
    bank->published = DeviceBuffer(words.size() * 2);
    bank->published.copy_from_host(words.data(), bank->published.bytes);
    Weight w{};
    w.qtype  = QType::BF16;
    w.layout = QuantLayout::Contiguous;
    w.qdata = w.payload = bank->published.p;
    w.payload_bytes     = bank->published.bytes;
    w.n = w.shape[0] = w.padded_shape[0] = experts;
    w.k = w.shape[1] = w.padded_shape[1] = hidden;
    w.ndim                 = 2;
    bank->published_weight = bank->frag_weight = w;
    return bank;
}

struct Case {
    std::string label;
    QType gate_up, down;
    std::int32_t experts, top_k, hidden, inter;
    ops::MoeActivation activation;
};

struct Layer {
    std::unique_ptr<Bank> router, gate_up, down;
    ops::MoeWeights published, frag, control;
    ops::MoeConfig config;
};

Layer make_layer(const Case& c, std::uint32_t seed) {
    Layer l;
    const int sets = c.activation == ops::MoeActivation::Relu2 ? 1 : 2;
    l.router  = make_router(c.experts, c.hidden, seed);
    l.gate_up = make_bank(c.gate_up, c.experts * sets * c.inter, c.hidden, seed + 1);
    l.down    = make_bank(c.down, c.experts * c.hidden, c.inter, seed + 2);
    l.published.router = l.frag.router = l.control.router = l.router->published_weight;
    l.published.gate_up = l.gate_up->published_weight;
    l.published.down    = l.down->published_weight;
    l.frag.gate_up      = l.gate_up->frag_weight;
    l.frag.down         = l.down->frag_weight;
    // The fragment-order bytes read as published.
    l.control.gate_up         = l.gate_up->published_weight;
    l.control.gate_up.qdata   = l.gate_up->frag_weight.qdata;
    l.control.gate_up.payload = l.gate_up->frag_weight.payload;
    l.control.down            = l.down->published_weight;
    l.control.down.qdata      = l.down->frag_weight.qdata;
    l.control.down.payload    = l.down->frag_weight.payload;
    l.config.top_k            = c.top_k;
    l.config.activation       = c.activation;
    if (c.activation == ops::MoeActivation::Relu2) {
        l.config.scoring      = ops::MoeScoring::Sigmoid;
        l.config.routed_scale = 2.5F;
    }
    return l;
}

std::vector<std::uint16_t> run(const Case& c, const Layer& l, const ops::MoeWeights& w,
                               const std::vector<std::uint16_t>& x, std::int32_t t,
                               const dmoe::MoePlan* plan = nullptr) {
    DeviceBuffer dx(static_cast<std::size_t>(t) * c.hidden * 2);
    dx.copy_from_host(x.data(), dx.bytes);
    DeviceBuffer dout(dx.bytes);
    check(cudaMemset(dout.p, 0xff, dout.bytes), "poison");
    const auto capacity = plan ? dmoe::plan_workspace_bytes(w, l.config, t, *plan)
                               : ops::moe_workspace_bytes(w, l.config, t, t);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    Tensor tx(dx.p, DType::BF16, {c.hidden, t});
    Tensor tout(dout.p, DType::BF16, {c.hidden, t});
    if (plan) {
        dmoe::moe_with_plan(tx, w, l.config, ops::MoeEpilogue::Store, tout, workspace, nullptr,
                            *plan);
    } else {
        ops::moe(tx, w, l.config, ops::MoeEpilogue::Store, tout, workspace, nullptr);
    }
    check(cudaDeviceSynchronize(), "moe");
    std::vector<std::uint16_t> out(static_cast<std::size_t>(t) * c.hidden);
    check(cudaMemcpy(out.data(), dout.p, dout.bytes, cudaMemcpyDeviceToHost), "download");
    return out;
}

std::vector<std::uint16_t> random_bf16(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    std::vector<std::uint16_t> out(n);
    for (auto& v : out) { v = to_bf16(u(rng)); }
    return out;
}

int compare(const std::string& label, const std::vector<std::uint16_t>& a,
            const std::vector<std::uint16_t>& b) {
    if (a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 2) == 0) { return 0; }
    std::size_t first = 0;
    while (first < std::min(a.size(), b.size()) && a[first] == b[first]) { ++first; }
    std::cerr << label << ": frag differs from published (first at " << first << ")\n";
    return 1;
}

int check_case(const Case& c, std::uint32_t seed) {
    const Layer l = make_layer(c, seed);
    const auto x  = random_bf16(static_cast<std::size_t>(300) * c.hidden, seed + 7);
    int failures  = 0;
    int compared  = 0;
    // Default routes: decode, streaming and wide widths.
    for (std::int32_t t = 1; t <= 64; ++t, ++compared) {
        failures += compare(c.label + " T=" + std::to_string(t), run(c, l, l.published, x, t),
                            run(c, l, l.frag, x, t));
    }
    // The prefill profile (the wide schedule from 17 tokens) and the invariant profile.
    for (const std::int32_t t : {17, 33, 64, 130, 300}) {
        const ops::ScopedPrefillRoutes profile(true);
        failures += compare(c.label + " prefill T=" + std::to_string(t),
                            run(c, l, l.published, x, t), run(c, l, l.frag, x, t));
        ++compared;
    }
    for (const std::int32_t t : {1, 8, 40, 300}) {
        const ops::ScopedInvariantRoutes invariant(true);
        failures += compare(c.label + " invariant T=" + std::to_string(t),
                            run(c, l, l.published, x, t), run(c, l, l.frag, x, t));
        ++compared;
    }
    // Every compiled schedule pair.
    int plans = 0;
    for (int gu = 0; gu < dmoe::schedule_count(); ++gu) {
        for (int dn = 0; dn < dmoe::schedule_count(); ++dn) {
            dmoe::MoePlan p;
            p.router_schedule  = dmoe::select_plan(l.published, l.config, 1, false).router_schedule;
            p.gate_up_schedule = gu;
            p.down_schedule    = dn;
            p.job_columns      = std::max(dmoe::schedule_info(gu).tokens_per_cta(),
                                          dmoe::schedule_info(dn).tokens_per_cta());
            if (!dmoe::plan_compiled(l.published, p)) { continue; }
            ++plans;
            for (const std::int32_t t : {1, 9, 40}) {
                failures += compare(c.label + " plan " + dmoe::plan_name(p) + " T=" +
                                        std::to_string(t),
                                    run(c, l, l.published, x, t, &p),
                                    run(c, l, l.frag, x, t, &p));
                ++compared;
            }
        }
    }
    // Control: the permuted bytes under the published layout must give other results.
    const bool permuted = ggml_frag_format(c.gate_up) || ggml_frag_format(c.down);
    if (permuted && run(c, l, l.published, x, 8) == run(c, l, l.control, x, 8)) {
        std::cerr << c.label << ": the fragment-order bytes read as published gave the same "
                     "output; the comparison would not see a wrong map\n";
        ++failures;
    }
    if (failures == 0) {
        std::cout << c.label << ": " << compared << " calls byte-identical (" << plans
                  << " schedule pairs)\n";
    }
    return failures;
}

// The A8 route (R8, moe_r8.cuh) on a frag bank: under an A8 expert policy an IQ2_S, IQ3_XXS,
// Q4_K or Q4_0 frag bank gives the published bank's bytes at every decode width, on the default
// route and on every R8 schedule of either stage the bank admits (Q4_0 at any K of whole blocks,
// its natural tail too); the fragment-order bytes read as published differ.
int check_a8(const Case& c, std::uint32_t seed) {
    if (c.gate_up != QType::GGML_IQ2_S && c.gate_up != QType::GGML_IQ3_XXS &&
        c.gate_up != QType::GGML_Q4_K && c.gate_up != QType::GGML_Q4_0) {
        return 0;
    }
    Layer l = make_layer(c, seed);
    for (auto* w : {&l.published, &l.frag, &l.control}) {
        w->expert_policy = ops::LinearPolicy::AllowA8;
    }
    const bool down_r8 = dmoe::r8_expert_supports(l.frag.down);
    const auto x = random_bf16(static_cast<std::size_t>(dmoe::kR8Tokens) * c.hidden, seed + 7);
    int failures = 0, compared = 0;
    // The default route takes the class's frag schedule on the frag bank (A16 without a route).
    const auto route = dmoe::a8_default_route();
    for (std::int32_t t = 1; t <= dmoe::kR8Tokens; ++t, ++compared) {
        const int expected = route ? route->gate_up_at(t, true) : -1;
        const auto p       = dmoe::select_plan(l.frag, l.config, t, false);
        if (p.gate_up_r8 != expected) {
            std::cerr << c.label << " T=" << t << ": the frag bank took " << dmoe::plan_name(p)
                      << ", not the class's frag schedule " << expected << '\n';
            ++failures;
        }
        failures += compare(c.label + " A8 default T=" + std::to_string(t),
                            run(c, l, l.published, x, t), run(c, l, l.frag, x, t));
    }
    for (int s = 0; s < dmoe::r8_schedule_count(); ++s) {
        for (const std::int32_t t : {1, 3, dmoe::kR8Tokens}) {
            auto p       = dmoe::select_plan(l.published, l.config, t, false);
            p.gate_up_r8 = s;
            p.down_r8    = down_r8 ? s : -1;
            failures += compare(c.label + " A8 " + dmoe::plan_name(p) + " T=" + std::to_string(t),
                                run(c, l, l.published, x, t, &p), run(c, l, l.frag, x, t, &p));
            ++compared;
        }
    }
    auto p       = dmoe::select_plan(l.published, l.config, 2, false);
    p.gate_up_r8 = 0;
    p.down_r8    = down_r8 ? 0 : -1;
    if (run(c, l, l.published, x, 2, &p) == run(c, l, l.control, x, 2, &p)) {
        std::cerr << c.label << ": the fragment-order bytes read as published by R8 gave the "
                     "same output\n";
        ++failures;
    }
    if (failures == 0) { std::cout << c.label << ": " << compared << " A8 calls byte-identical\n"; }
    return failures;
}

// The grouped W4A8 (MoePlan::w4a8, moe_w4a8_frag.cu) on a frag bank: under an A8 expert policy
// the prefill plan with both stages on the int8 tile gives the published bank's bytes at widths
// across a job and with partial tiles; the fragment-order bytes read as published differ.
int check_w4a8(const Case& c, std::uint32_t seed) {
    Layer l = make_layer(c, seed);
    for (auto* w : {&l.published, &l.frag, &l.control}) {
        w->expert_policy = ops::LinearPolicy::AllowA8;
    }
    const auto x = random_bf16(static_cast<std::size_t>(130) * c.hidden, seed + 9);
    // A fragment-ordered i-quant bank does not take the grouped W4A8 (its fragment loaders already
    // match it): a forced plan is refused before a launch.
    const auto iquant = [](QType q) { return q == QType::GGML_IQ2_S || q == QType::GGML_IQ3_XXS; };
    if (iquant(c.gate_up) || iquant(c.down)) {
        auto p = dmoe::select_plan(l.published, l.config, 17, true);
        p.w4a8 = true;
        try {
            (void)run(c, l, l.frag, x, 17, &p);
        } catch (const std::invalid_argument&) {
            std::cout << c.label << ": W4A8 refused on the fragment-order bank\n";
            return 0;
        }
        std::cerr << c.label << ": the fragment-order i-quant bank took the grouped W4A8\n";
        return 1;
    }
    int failures = 0, compared = 0;
    for (const std::int32_t t : {17, 64, 130}) {
        auto p = dmoe::select_plan(l.published, l.config, t, true);
        p.w4a8 = true;
        failures += compare(c.label + " W4A8 T=" + std::to_string(t),
                            run(c, l, l.published, x, t, &p), run(c, l, l.frag, x, t, &p));
        ++compared;
    }
    auto p = dmoe::select_plan(l.published, l.config, 17, true);
    p.w4a8 = true;
    if (run(c, l, l.published, x, 17, &p) == run(c, l, l.control, x, 17, &p)) {
        std::cerr << c.label
                  << ": the fragment-order bytes read as published by W4A8 gave the "
                     "same output\n";
        ++failures;
    }
    if (failures == 0) {
        std::cout << c.label << ": " << compared << " W4A8 calls byte-identical\n";
    }
    return failures;
}

int check_linear_refuses(std::uint32_t seed) {
    const auto bank = make_bank(QType::GGML_Q4_0, 64, 2048, seed);
    DeviceBuffer dx(2048 * 2), dout(64 * 2);
    Tensor x(dx.p, DType::BF16, {2048, 1});
    Tensor out(dout.p, DType::BF16, {64, 1});
    try {
        ops::linear(x, bank->frag_weight, out, nullptr);
    } catch (const std::exception& error) {
        if (std::string(error.what()).find("ggml_frag_v1") != std::string::npos) {
            std::cout << "linear refuses a frag weight: " << error.what() << '\n';
            return 0;
        }
        std::cerr << "linear refused a frag weight with another message: " << error.what()
                  << '\n';
        return 1;
    }
    std::cerr << "linear accepted a ggml_frag_v1 weight\n";
    return 1;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "CUDA unavailable; skipping moe frag test\n";
        return 77;
    }
    using A = ops::MoeActivation;
    // Each adopted format; K with and without a Q4_0 tail (K/32 % 4 blocks left natural); the
    // shapes of the shipped layers at reduced E (Nemotron Q4_0 2688 / 1856, Gemma 4 26B Q4_0
    // 2816 / 704, Qwen3.6 Q4_K 2048 with a Q6_K down, Qwen3.6 IQ2_S / IQ3_XXS 2048 / 512).
    const Case cases[] = {
        {"q40 H1024 F224", QType::GGML_Q4_0, QType::GGML_Q4_0, 8, 2, 1024, 224, A::SwiGlu},
        {"nemotron-like q40 H2688 F1856", QType::GGML_Q4_0, QType::GGML_Q4_0, 8, 6, 2688, 1856,
         A::Relu2},
        {"gemma-like q40 H2816 F704", QType::GGML_Q4_0, QType::GGML_Q4_0, 8, 4, 2816, 704,
         A::GeGluTanh},
        {"q4k H512 F256", QType::GGML_Q4_K, QType::GGML_Q4_K, 8, 2, 512, 256, A::SwiGlu},
        {"qwen-like q4k/q6k H2048 F512", QType::GGML_Q4_K, QType::GGML_Q6_K, 8, 4, 2048, 512,
         A::SwiGlu},
        {"iq3xxs H512 F512", QType::GGML_IQ3_XXS, QType::GGML_IQ3_XXS, 8, 2, 512, 512, A::SwiGlu},
        {"iq2s H512 F256", QType::GGML_IQ2_S, QType::GGML_IQ2_S, 8, 2, 512, 256, A::SwiGlu},
        {"qwen-like iq2s/iq3xxs H2048 F512", QType::GGML_IQ2_S, QType::GGML_IQ3_XXS, 8, 4, 2048,
         512, A::SwiGlu},
    };
    int failures = 0;
    try {
        std::uint32_t seed = 20260927U;
        for (const auto& c : cases) {
            failures += check_case(c, seed += 100);
            failures += check_a8(c, seed + 50);
            failures += check_w4a8(c, seed + 70);
        }
        failures += check_linear_refuses(seed + 1);
    } catch (const std::exception& e) {
        std::cerr << "moe frag test: unexpected exception: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "moe frag: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "moe frag: all cases passed\n";
    return 0;
}
