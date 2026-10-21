// The formats ggml's vector kernel takes by rule at T = 2..4 (ops/generic/vec_verify_rule.h), no
// GPU: Q6_K leaves it for the register MMA where tensor mma.sync runs; the other formats it reads
// keep it; without tensor mma.sync every format keeps it.
#include "ops/generic/vec_verify_rule.h"

#include <iostream>
#include <string>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    static_assert(!generic::vec_verify_format(QType::GGML_Q6_K, true));
    expect(!generic::vec_verify_format(QType::GGML_Q6_K, true),
           "Q6_K with tensor mma.sync: the register MMA");
    for (const QType q : {QType::GGML_Q4_K, QType::GGML_Q5_K, QType::GGML_Q3_K, QType::GGML_Q2_K,
                          QType::GGML_Q4_0, QType::GGML_IQ4_XS, QType::GGML_IQ3_S}) {
        expect(generic::vec_verify_format(q, true),
               "qtype " + std::to_string(static_cast<int>(q)) + " keeps the vector kernel");
    }
    expect(generic::vec_verify_format(QType::GGML_Q6_K, false),
           "Q6_K without tensor mma.sync keeps the vector kernel");
    std::cout << (failures == 0 ? "PASS" : "FAILED") << '\n';
    return failures == 0 ? 0 : 1;
}
