// The route cost model's constants (ops/generic/route_cost_constants.h): every one is listed once
// with its provenance and a value in range. `--list` prints them, the fitted ones first: the
// numbers the model still takes from route_tune timings instead of deriving them. No GPU.

#include "ops/generic/route_cost_constants.h"

#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <string_view>

namespace rc = ninferno::ops::detail::generic::route_cost;

namespace {

const char* provenance_name(rc::Provenance p) {
    switch (p) {
    case rc::Provenance::Fact:
        return "fact";
    case rc::Provenance::Counted:
        return "counted";
    case rc::Provenance::Fitted:
        return "fitted";
    }
    return "?";
}

template <std::size_t N>
int check_fraction(const rc::Constant<N>& c) {
    int failures = 0;
    for (double v : c.values) {
        if (!(v > 0.0 && v <= 1.0)) {
            std::fprintf(stderr, "%.*s: %g is not a fraction of a peak\n",
                         static_cast<int>(c.name.size()), c.name.data(), v);
            ++failures;
        }
    }
    return failures;
}

template <std::size_t N>
int check_positive(const rc::Constant<N>& c) {
    int failures = 0;
    for (double v : c.values) {
        if (!(std::isfinite(v) && v > 0.0)) {
            std::fprintf(stderr, "%.*s: %g is not a positive value\n",
                         static_cast<int>(c.name.size()), c.name.data(), v);
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    int failures = 0;
    std::set<std::string_view> names;
    for (const auto& c : rc::kRouteCostConstants) {
        if (c.name.empty() || !names.insert(c.name).second) {
            std::fprintf(stderr, "constant name '%.*s' empty or listed twice\n",
                         static_cast<int>(c.name.size()), c.name.data());
            ++failures;
        }
    }
    failures += check_fraction(rc::kW4a8Efficiency) + check_fraction(rc::kI8Efficiency) +
                check_fraction(rc::kMmqEfficiency) + check_fraction(rc::kStagedTensorFraction) +
                check_fraction(rc::kTiledTensorFraction) + check_fraction(rc::kMinWarpFill);
    failures += check_positive(rc::kRmmaStepCycles) + check_positive(rc::kPipelinedStageCycles) +
                check_positive(rc::kW4a8StepCycles) + check_positive(rc::kW4a8ProducerStepCycles) +
                check_positive(rc::kI8StageCycles) + check_positive(rc::kMmqChainCycles) +
                check_positive(rc::kMmqChainK) + check_positive(rc::kQuantLaunchSeconds) +
                check_positive(rc::kQuantBytesPerValue) + check_positive(rc::kMmqSliceSeconds) +
                check_positive(rc::kClusterReduceCycles) + check_positive(rc::kWarpReduceCycles);
    // Counted K steps match the kernels' tiles (generic_gemm.cuh: 16 K per register-MMA warp step,
    // 64 K a pipelined or I8 stage, 128 K a W4A8 step).
    if (rc::kRmmaStepK.value() != 16.0 || rc::kPipelinedStageK.value() != 64.0 ||
        rc::kI8StageK.value() != 64.0 || rc::kW4a8StepK.value() != 128.0) {
        std::fprintf(stderr, "a counted K step no longer matches its kernel\n");
        ++failures;
    }
    if (argc > 1 && std::string(argv[1]) == "--list") {
        for (auto want : {rc::Provenance::Fitted, rc::Provenance::Counted, rc::Provenance::Fact}) {
            for (const auto& c : rc::kRouteCostConstants) {
                if (c.from != want) continue;
                std::printf("%-8s %-32.*s %zu value%s\n", provenance_name(c.from),
                            static_cast<int>(c.name.size()), c.name.data(), c.values,
                            c.values == 1 ? "" : "s");
            }
        }
    }
    std::size_t fitted = 0;
    for (const auto& c : rc::kRouteCostConstants) fitted += c.from == rc::Provenance::Fitted;
    std::printf("route cost constants: %zu listed, %zu fitted\n",
                std::size(rc::kRouteCostConstants), fitted);
    return failures == 0 ? 0 : 1;
}
