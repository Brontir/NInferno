// The codec traits the kernels export (generic_w4a8.cuh codec_traits<C>, tabled by
// generic_codec_traits.cu) against the literals r8_fits, w4a8_fits and mmq_plane_bytes carried
// before (#615 F3): for every QType value in a byte's range, the old formula (kept below in
// `legacy`, verbatim but for comments) and the new read of the traits agree, and so do the two
// public fit rules on every schedule, K and T. Which codec struct has which constant is a
// static_assert in generic_codec_traits.cu. Host only.
#include "ops/common/arch_limits.h"
#include "ops/generic/codec_traits.h"
#include "ops/generic/generic_gemm.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace generic = ninferno::ops::detail::generic;
using generic::CodecTraits;
using ninferno::QType;
using ninferno::ops::kMaxSharedBytesPerBlock;
using ninferno::ops::kStaticSharedBytesPerBlock;

namespace ninferno::ops::detail::generic {
// generic_r8.cuh's host-side declaration: the device's answer, which the engine's r8_fits reads.
[[nodiscard]] bool r8_stage_grid();
} // namespace ninferno::ops::detail::generic

namespace legacy {

// generic_gemm.cpp mmq_plane_bytes: the whole-block cases before the ggml planes.
int mmq_whole_block_bytes(QType q) {
    if (q == QType::GGML_Q4_K) return 144;
    if (q == QType::GGML_Q5_K) return 176;
    return 0;
}

// generic_gemm.cpp ggml_layout_admits: the formats read as planes.
bool ggml_planes(QType q) { return q != QType::GGML_Q4_K && q != QType::GGML_Q5_K; }

// generic_gemm.cpp w4a8_fits.
bool w4a8_fits(int schedule, QType qtype) {
    const auto s = generic::w4a8_schedule_info(schedule);
    if (s.producers > 0) return kMaxSharedBytesPerBlock >= 99 * 1024;
    const int half_min = qtype == QType::GGML_Q2_K ? s.block_n * 32 : 0;
    return 2 * ((s.block_m + s.block_n) * (128 + 32) + half_min) <=
           static_cast<int>(kMaxSharedBytesPerBlock);
}

// generic_gemm.cpp r8_fits: the sums list and the grid switch.
bool r8_sums(QType qtype) {
    return qtype == QType::GGML_Q4_1 || qtype == QType::GGML_Q5_1 || qtype == QType::GGML_Q4_K ||
           qtype == QType::GGML_Q5_K;
}

std::int64_t r8_grid_bytes(QType qtype) {
    std::int64_t grid = 0;
    switch (qtype) {
    case QType::GGML_IQ3_XXS:
        grid = 1024;
        break;
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ2_XXS:
        grid = 2048;
        break;
    case QType::GGML_IQ2_XS:
        grid = 4096;
        break;
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
        grid = 8192;
        break;
    default:
        break;
    }
    return grid;
}

bool r8_fits(QType qtype, std::int32_t k, std::int32_t tokens, bool stage_grid) {
    if (tokens <= 0 || tokens > generic::kR8MaxTokens || k <= 0) return false;
    std::int64_t slots = 1;
    while (slots < tokens) slots *= 2;
    const bool sums   = r8_sums(qtype);
    std::int64_t grid = stage_grid ? r8_grid_bytes(qtype) : 0;
    if (grid > 0) grid += 4;
    return slots * (static_cast<std::int64_t>(k) / 32) * (sums ? 40 : 36) + grid <=
           kStaticSharedBytesPerBlock;
}

} // namespace legacy

namespace {

int failures    = 0;
int comparisons = 0;

template <class T>
void agree(const T& old_value, const T& new_value, const char* what, int q, int a = -1,
           int b = -1) {
    ++comparisons;
    if (old_value == new_value) return;
    ++failures;
    std::cerr << what << " disagrees for QType " << q;
    if (a >= 0) std::cerr << " (" << a << ", " << b << ")";
    std::cerr << ": old " << old_value << ", new " << new_value << "\n";
}

} // namespace

int main() {
    const bool stage_grid = generic::r8_stage_grid();
    for (int v = 0; v < 256; ++v) {
        const auto q         = static_cast<QType>(v);
        const CodecTraits t  = generic::codec_traits(q);
        const bool has_codec = ninferno::format_facts(q).codec.has_value();

        // mmq_plane_bytes and ggml_layout_admits.
        agree(legacy::mmq_whole_block_bytes(q), t.block_bytes, "block_bytes", v);
        agree(legacy::ggml_planes(q), t.block_bytes == 0, "ggml planes", v);
        // r8_fits's two literal lists.
        agree(legacy::r8_sums(q), t.min, "min (block sums)", v);
        agree(legacy::r8_grid_bytes(q),
              static_cast<std::int64_t>(sizeof(std::uint32_t)) * t.grid_words, "grid bytes", v);
        // w4a8_fits's half-min plane.
        agree(q == QType::GGML_Q2_K, t.half_min, "half_min", v);
        // A format with no codec reads the default row; one with a codec spans at least a block.
        if (!has_codec) agree(true, t == CodecTraits{}, "default row", v);
        agree(true, t.span >= 1, "span", v);

        // The engine's rules, on every schedule, and on every K step and T of decode.
        for (int s = 0; s < generic::w4a8_schedule_count(); ++s) {
            agree(legacy::w4a8_fits(s, q), generic::w4a8_fits(s, q), "w4a8_fits", v, s);
        }
        for (std::int32_t k : {0, 32, 96, 2112, 2880, 4096, 8192, 12288, 16384, 24576, 65536}) {
            for (std::int32_t tokens = -1; tokens <= generic::kR8MaxTokens + 1; ++tokens) {
                agree(legacy::r8_fits(q, k, tokens, stage_grid), generic::r8_fits(q, k, tokens),
                      "r8_fits", v, k, tokens);
            }
        }
    }
    if (failures != 0) {
        std::cerr << failures << " disagreements in " << comparisons << " comparisons\n";
        return 1;
    }
    std::cout << "codec traits: " << comparisons
              << " comparisons agree (grid staged: " << (stage_grid ? "yes" : "no") << ")\n";
    return 0;
}
