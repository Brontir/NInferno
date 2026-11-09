// The register MMA's min-term MMA factor (ops/generic/rmma_min_term.h, #501): the tensor work over
// the weight MMAs, (Sets x MT + 1) / (Sets x MT) for the decoders with a min term where the tile
// does not add the activation sums in FP32, 1 otherwise. The ratios are ncu's exact HMMA counts of
// Q4_K 4096x3840 at T = 24 (m1.n1 2.0x, m2.n1 and m2.n2 1.5x, m1.n4 and up 1.0x). No GPU.

#include "ops/generic/rmma_min_term.h"

#include <cmath>
#include <cstdio>

namespace {
using ninferno::QType;
namespace generic = ninferno::ops::detail::generic;

int failures = 0;

void expect(const char* what, double got, double want) {
    if (std::fabs(got - want) < 1e-12) return;
    std::fprintf(stderr, "%s: %g, want %g\n", what, got, want);
    ++failures;
}
} // namespace

int main() {
    for (const QType q : {QType::GGML_Q4_K, QType::GGML_Q5_K, QType::GGML_Q2_K, QType::GGML_Q4_1,
                          QType::GGML_Q5_1}) {
        expect("min-term format, MT 1", generic::rmma_min_term_factor(q, 1, 1, false), 2.0);
        expect("min-term format, MT 2", generic::rmma_min_term_factor(q, 1, 2, false), 1.5);
        expect("min-term format, MT 4", generic::rmma_min_term_factor(q, 1, 4, false), 1.25);
        expect("SwiGLU, MT 1 (two sets)", generic::rmma_min_term_factor(q, 2, 1, false), 1.5);
        expect("SwiGLU, MT 2", generic::rmma_min_term_factor(q, 2, 2, false), 1.25);
        expect("FP32 adds", generic::rmma_min_term_factor(q, 1, 1, true), 1.0);
    }
    for (const QType q : {QType::GGML_Q6_K, QType::GGML_Q3_K, QType::GGML_Q4_0, QType::GGML_Q8_0,
                          QType::GGML_IQ4_XS}) {
        expect("format without a min term", generic::rmma_min_term_factor(q, 1, 1, false), 1.0);
    }
    std::printf("rmma min term: %s\n", failures == 0 ? "ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
