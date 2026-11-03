// The activation each generic route prepares and which preparations serve one another (#308,
// ops/generic/generic_gemm.h prepared_rule): one rule, one row length and variant share bytes;
// the same nominal int8 activation does not. Weights are described, never read.
#include "core/weight_view.h"
#include "ops/generic/generic_gemm.h"

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>

namespace generic = ninferno::ops::detail::generic;
using generic::Family;
using generic::PreparedRepresentation;
using generic::PreparedRule;
using generic::ResolvedRoute;
using generic::Route;
using ninferno::QType;
using ninferno::Weight;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// A weight as the converter lays it out, at an aligned stand-in address.
struct Described {
    ninferno::WeightParent parent;
    ninferno::WeightView view;
    Weight weight;
    Described(QType qtype, std::int32_t n, std::int32_t k) {
        parent.data = reinterpret_cast<const std::byte*>(std::uintptr_t{1} << 32);
        const std::uint64_t shape[2] = {static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k)};
        parent.geometry = ninferno::weight_geometry(qtype, ninferno::default_weight_layout(qtype), shape);
        view.shape      = {shape[0], shape[1]};
        view.parts      = {ninferno::WeightRegion{&parent, 0, shape[0] * shape[1]}};
        weight          = ninferno::native_weight(view, 0.0F);
    }
    Described(const Described&)            = delete;
    Described& operator=(const Described&) = delete;
};

ResolvedRoute on(Family family, bool r8_prequantized = false) {
    const Route route{family, 0};
    return ResolvedRoute{route, route, r8_prequantized};
}

std::optional<PreparedRule> rule(Family family, const Weight& weight, bool pre = false) {
    return generic::prepared_rule(on(family, pre), weight);
}

bool match(const std::optional<PreparedRule>& a, const std::optional<PreparedRule>& b) {
    return a && b && generic::prepared_rules_match(*a, *b);
}

} // namespace

int main() {
    const Described q4k(QType::GGML_Q4_K, 4096, 4096), q6k(QType::GGML_Q6_K, 4096, 4096);
    const Described q8(QType::Q8_G32_FP16, 4096, 4096), q4g64(QType::Q4_G64_FP16, 4096, 4096);
    const Described q8_odd(QType::Q8_G32_FP16, 4096, 2080), nv(QType::NVFP4, 4096, 4096);
    const Described bf16(QType::BF16, 4096, 4096);

    // The A16 families read x as BF16; R8 at decode quantizes inside the kernel.
    for (const Family f : {Family::RegisterMma, Family::Scalar, Family::TiledMma, Family::StagedMma,
                           Family::Nvfp4TiledMma}) {
        check(!rule(f, bf16.weight), "an A16 family prepares nothing");
    }
    check(!rule(Family::R8Mma, q8.weight), "R8 at decode prepares nothing");

    // ggml's vector kernel: its q8_1 serves another vector call on the same x.
    check(match(rule(Family::GgmlVec, q4k.weight), rule(Family::GgmlVec, q6k.weight)),
          "two vector calls share ggml's q8_1");
    // The W4A8 tiles and the R8 verify widths' pre-quantized form: one quantizer
    // (launch_w4a8_quantize), the same bytes where K fills whole 128-value tiles.
    check(match(rule(Family::W4a8Mma, q8.weight), rule(Family::R8Mma, q8.weight, true)),
          "W4A8 and pre-quantized R8 share the blocks at K = 4096");
    check(!match(rule(Family::W4a8Mma, q8_odd.weight), rule(Family::R8Mma, q8_odd.weight, true)),
          "K = 2080: W4A8 pads its rows to 2176, R8 reads 2080: no match");
    // The same nominal int8 activation, another rule: no match.
    check(!match(rule(Family::GgmlVec, q4k.weight), rule(Family::GgmlMmq, q4k.weight)),
          "ggml's vector q8_1 and MMQ's q8_1_mmq tiles");
    check(!match(rule(Family::W4a8Mma, q8.weight), rule(Family::I8Mma, q4g64.weight)),
          "W4A8 blocks of 32 and I8 groups of 64");
    check(!match(rule(Family::GgmlVec, q8.weight), rule(Family::W4a8Mma, q8.weight)),
          "ggml's q8_1 and W4A8 blocks");
    // MMQ: the scale/sum layout follows the weight format; formats are counted apart.
    check(match(rule(Family::GgmlMmq, q4k.weight), rule(Family::GgmlMmq, q4k.weight)) &&
              !match(rule(Family::GgmlMmq, q4k.weight), rule(Family::GgmlMmq, q6k.weight)),
          "MMQ shares within its format only");
    // W4A4: the codes depend on the weight's input divisor.
    check(!match(rule(Family::W4a4Mma, nv.weight), rule(Family::W4a4Mma, nv.weight)),
          "W4A4 serves no other weight");

    if (failures != 0) {
        std::cerr << "prepared rules: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "prepared rules: one rule, row length and variant share bytes\n";
    return 0;
}
