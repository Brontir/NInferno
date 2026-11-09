// The format traits table (ops/generic/format_traits.h) against the switches it replaced: for
// every QType value, the old predicate (kept below, verbatim, in `legacy`, until a follow-up
// deletes it) and the new read of the table agree; where the predicate is public in
// generic_gemm.h, the engine's function agrees as well. Values the enum does not name read as
// "no capability" in both.
#include "ops/common/native_nvfp4.h"
#include "ops/generic/format_traits.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_select_internal.h"

#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>

namespace generic = ninferno::ops::detail::generic;
using ninferno::QType;
using ninferno::ops::detail::kNativeNvfp4;

namespace legacy {

bool i8_format(QType qtype) {
    return qtype == QType::Q4_G64_FP16 || qtype == QType::Q5_G64_FP16 ||
           qtype == QType::Q6_G64_FP16;
}

bool qx_format(QType qtype) {
    return qtype == QType::Q4_G64_FP16 || qtype == QType::Q5_G64_FP16 ||
           qtype == QType::Q6_G64_FP16;
}

bool mmq_verify_format(QType q) {
    switch (q) {
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ4_NL:
        return true;
    default:
        return false;
    }
}

bool light_ggml_step(QType q) {
    switch (q) {
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ2_S:
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
        return true;
    default:
        return false;
    }
}

bool tall_tile_format(QType q) { return q == QType::GGML_Q4_K; }

bool w4a8_ws_extra_format(QType q) {
    return q == QType::GGML_IQ4_XS || q == QType::GGML_IQ2_XS || q == QType::GGML_IQ2_XXS ||
           q == QType::GGML_IQ3_S;
}

bool w4a8_ws_supports(QType q) {
    return q == QType::GGML_IQ3_XXS || q == QType::GGML_IQ2_S || q == QType::GGML_Q4_K ||
           q == QType::GGML_Q6_K || q == QType::GGML_Q4_0 || q == QType::GGML_IQ4_XS ||
           q == QType::GGML_IQ2_XS || q == QType::GGML_IQ2_XXS || q == QType::GGML_IQ3_S;
}

bool r8_supports(QType qtype) {
    switch (qtype) {
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::MXFP4:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ4_XS:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
    case QType::Q8_G32_FP16:
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
    case QType::Q3_G128_FP16:
    case QType::Q3_G64_FP16:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
        return true;
    case QType::NVFP4:
        return !kNativeNvfp4;
    default:
        return false;
    }
}

std::int32_t r8_k_values(QType qtype) {
    switch (qtype) {
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
        return 64;
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
        return 128;
    default:
        return 32;
    }
}

bool r8_verify_supports(QType qtype) {
    switch (qtype) {
    case QType::Q8_G32_FP16:
    case QType::GGML_Q4_0:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
        return true;
    default:
        return false;
    }
}

bool r8_verify_compiled(QType qtype) {
    return r8_supports(qtype) && qtype != QType::GGML_Q3_K && qtype != QType::GGML_Q2_K;
}

bool w4a8_k_admits(QType qtype, std::int32_t k) {
    if (k <= 0) return false;
    if (k % 128 == 0) return true;
    switch (qtype) {
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
    case QType::MXFP4:
    case QType::Q8_G32_FP16:
    case QType::U4_G32_ZP_FP16:
        return k % 32 == 0;
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::U4_G64_ZP_FP16:
        return k % 64 == 0;
    default:
        return false;
    }
}

bool w4a8_supports(QType qtype) {
    switch (qtype) {
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::MXFP4:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ4_XS:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::Q3_G128_FP16:
    case QType::Q3_G64_FP16:
    case QType::Q8_G32_FP16:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
    case QType::NVFP4:
        return true;
    default:
        return false;
    }
}

} // namespace legacy

namespace {

int failures     = 0;
long comparisons = 0;

void agree(bool old_value, bool new_value, const char* what, int q, long k = -1) {
    ++comparisons;
    if (old_value == new_value) return;
    ++failures;
    std::cerr << "FAIL: " << what << " disagrees for QType " << q;
    if (k >= 0) std::cerr << " K " << k;
    std::cerr << ": old " << old_value << ", table " << new_value << '\n';
}

// Every enumerator of QType: each must have a row (a missing row would read as "nothing").
constexpr QType kNamed[] = {
    QType::Q4_G64_FP16,
    QType::Q5_G64_FP16,
    QType::Q6_G64_FP16,
    QType::Q8_G32_FP16,
    QType::BF16,
    QType::FP32,
    QType::INT32,
    QType::NVFP4,
    QType::FP8_E4M3FN_ROW_BF16,
    QType::MXFP4,
    QType::FP8_E4M3FN_ROW_F32,
    QType::FP8_E4M3FN_B128_F32,
    QType::U4_G128_ZP_FP16,
    QType::U4_G32_ZP_FP16,
    QType::U4_G64_ZP_FP16,
    QType::Q3_G128_FP16,
    QType::Q3_G64_FP16,
    QType::GGML_Q4_K,
    QType::GGML_Q5_K,
    QType::GGML_Q6_K,
    QType::GGML_Q3_K,
    QType::GGML_Q2_K,
    QType::GGML_IQ4_XS,
    QType::GGML_IQ3_XXS,
    QType::GGML_IQ3_S,
    QType::GGML_Q4_0,
    QType::GGML_Q4_1,
    QType::GGML_Q5_0,
    QType::GGML_Q5_1,
    QType::GGML_IQ4_NL,
    QType::GGML_IQ2_XXS,
    QType::GGML_IQ2_XS,
    QType::GGML_IQ2_S,
    QType::GGML_IQ1_S,
    QType::GGML_IQ1_M,
    QType::GGML_TQ1_0,
    QType::GGML_TQ2_0,
    QType::GGML_Q1_0,
    QType::GGML_Q2_0,
};

} // namespace

int main() {
    using generic::admits;
    using generic::format_traits;

    for (const QType q : kNamed) {
        if (!generic::has_format_row(static_cast<QType>(static_cast<std::uint16_t>(q)))) {
            ++failures;
            std::cerr << "FAIL: no traits row for QType " << static_cast<int>(q) << '\n';
        }
    }
    // The table names each format once.
    for (std::size_t i = 0; i < generic::kFormatTraits.size(); ++i) {
        for (std::size_t j = i + 1; j < generic::kFormatTraits.size(); ++j) {
            if (generic::kFormatTraits[i].q == generic::kFormatTraits[j].q) {
                ++failures;
                std::cerr << "FAIL: two traits rows for QType "
                          << static_cast<int>(generic::kFormatTraits[i].q) << '\n';
            }
        }
    }
    if (generic::kFormatTraits.size() != std::size(kNamed)) {
        ++failures;
        std::cerr << "FAIL: " << generic::kFormatTraits.size() << " rows for " << std::size(kNamed)
                  << " named formats\n";
    }

    // Every value a QType holds in a byte's range, named or not.
    for (int v = 0; v < 256; ++v) {
        const QType q     = static_cast<QType>(v);
        const auto& t     = format_traits(q);
        const bool native = kNativeNvfp4;
        agree(legacy::i8_format(q), t.qx_g64, "i8_format", v);
        agree(legacy::qx_format(q), t.qx_g64, "qx_format", v);
        agree(legacy::mmq_verify_format(q), t.mmq_verify, "mmq_verify_format", v);
        agree(legacy::light_ggml_step(q), t.light_step, "light_ggml_step", v);
        agree(legacy::tall_tile_format(q), t.tall_tile, "tall_tile_format", v);
        agree(legacy::w4a8_ws_extra_format(q), t.w4a8_ws_tuned, "w4a8_ws_extra format", v);
        agree(legacy::w4a8_ws_supports(q), t.w4a8_ws, "w4a8_ws_supports", v);
        agree(legacy::w4a8_ws_supports(q), generic::w4a8_ws_supports(q), "engine w4a8_ws_supports",
              v);
        agree(legacy::r8_supports(q), admits(t.r8, native), "r8_supports", v);
        agree(legacy::r8_supports(q), generic::r8_supports(q), "engine r8_supports", v);
        agree(legacy::r8_verify_supports(q), t.r8_verify, "r8_verify_supports", v);
        agree(legacy::r8_verify_supports(q), generic::r8_verify_supports(q),
              "engine r8_verify_supports", v);
        agree(legacy::r8_verify_compiled(q), admits(t.r8, native) && t.r8_prequant,
              "r8_verify_compiled", v);
        agree(legacy::r8_verify_compiled(q), generic::r8_verify_compiled(q),
              "engine r8_verify_compiled", v);
        agree(legacy::w4a8_supports(q), admits(t.w4a8, native), "w4a8_supports", v);
        agree(legacy::w4a8_supports(q), generic::w4a8_supports(q), "engine w4a8_supports", v);
        agree(legacy::r8_k_values(q) == t.r8_k_values, true, "r8_k_values", v);
        agree(legacy::r8_k_values(q) == generic::r8_k_values(q), true, "engine r8_k_values", v);
        // K around every group boundary the rule looks at, and the edge values.
        for (std::int32_t k : {-128, -1,  0,   1,   16,   31,   32,   33,   48,   63,    64,   96,
                               127,  128, 160, 192, 2112, 2880, 4096, 4128, 4160, 12288, 12320}) {
            const std::int32_t tail = t.w4a8_tail_k;
            const bool table        = k > 0 && (k % 128 == 0 || (tail != 0 && k % tail == 0));
            agree(legacy::w4a8_k_admits(q, k), table, "w4a8_k_admits", v, k);
            agree(legacy::w4a8_k_admits(q, k), generic::w4a8_k_admits(q, k), "engine w4a8_k_admits",
                  v, k);
        }
    }

    if (failures != 0) {
        std::cerr << failures << " disagreements in " << comparisons << " comparisons\n";
        return 1;
    }
    std::cout << "format traits: " << comparisons << " comparisons agree\n";
    return 0;
}
