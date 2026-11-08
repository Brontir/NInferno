// The W4A8 tile choice of Blackwell's Q4_K and Q6_K prefill (DECISIONS 75, core/w4a8_tiles.h):
// heuristic_w4a8 takes the 64 x 64 tile exactly where the shared function says its tiles cost less
// (or the grid is under half the SMs, the rule that predates it), over 129..512 tokens on a 70-SM
// and a 170-SM board and the shapes of Qwen3.5 9B's linears; and the product's estimate
// (product/choice prefill_linears_seconds) prices the tile the kernel runs: its cost ratio
// between T tokens and 128 is the kernel's tile work ratio. Host only (substituted DeviceClass).
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/units.h"
#include "core/w4a8_tiles.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_select_internal.h"
#include "product/choice/choice.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;
namespace ch      = ninferno::product::choice;
using ops::LinearPolicy;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

constexpr int k64x64 = 4; // generic_schedules.h: W4A8 schedule 4

struct Board {
    const char* name;
    std::uint32_t pci;
    int sms;
    double clock_hz, dram;
};

constexpr Board kBoards[] = {{"RTX 5070 Ti", 0x2C05, 70, 2.452e9, 896e9},
                             {"RTX 5090", 0x2B85, 170, 2.407e9, 1792e9}};

struct Shape {
    generic::Epilogue epilogue;
    int n, k; // the kernel's problem: SwiGLU's n is the gate rows
};

constexpr Shape kShapes[] = {{generic::Epilogue::Store, 1024, 4096},
                             {generic::Epilogue::Store, 4096, 4096},
                             {generic::Epilogue::Store, 10240, 4096},
                             {generic::Epilogue::AddResidual, 4096, 12288},
                             {generic::Epilogue::SwiGlu, 12288, 4096}};

} // namespace

int main() {
    constexpr std::uint64_t kMiB = 1024 * 1024;
    void* const payload          = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    long narrow_calls = 0, calls = 0;
    for (const Board& b : kBoards) {
        DeviceClass d = synthetic_device_class(b.pci, 12, 0, b.sms, 64 * kMiB, 16 * 1024 * kMiB);
        const ScopedDeviceClass scoped(d);
        const w4a8_tiles::Residency res = w4a8_tiles::residency(d.smem_per_sm);
        check(w4a8_tiles::priced(res),
              std::string(b.name) + ": sm_120 residency is the priced one");
        const ch::CardFacts card{.major            = 12,
                                 .minor            = 0,
                                 .sms              = b.sms,
                                 .clock_hz         = b.clock_hz,
                                 .dram_bytes_per_s = b.dram};
        for (const QType qtype : {QType::GGML_Q4_K, QType::GGML_Q6_K}) {
            const w4a8_tiles::NarrowCost& cost =
                qtype == QType::GGML_Q4_K ? w4a8_tiles::kQ4k : w4a8_tiles::kQ6k;
            ch::QuantOption quant;
            quant.label   = qtype == QType::GGML_Q4_K ? "Q4_K_M" : "Q6_K";
            quant.weights = ch::WeightClass::KQuant;
            for (const Shape& s : kShapes) {
                const auto host = quantized_weight::make_patterned_weight(qtype, s.n, s.k, 7U, {});
                const Weight w  = host.device_weight(payload);
                const generic::Problem problem{qtype, LinearPolicy::AllowA8, s.n, s.k, s.epilogue};
                const bool gated        = s.epilogue == generic::Epilogue::SwiGlu;
                const std::int64_t rows = gated ? 32 : 64;
                const std::int64_t rt   = (s.n + rows - 1) / rows;
                // The estimate's linear: a gated pair is one linear of 2 x the gate rows.
                ch::ModelFacts model;
                model.linears = {{double(gated ? 2 * s.n : s.n), double(s.k), 1, false, gated}};
                const auto kernel_work = [&](int t) {
                    const int pick = generic::heuristic_w4a8(problem, w, t);
                    check(pick == k64x64 || pick == 5,
                          std::string(b.name) + ": a W4A8 tile of 64 rows");
                    return pick == k64x64 ? w4a8_tiles::narrow_work(rt, t, b.sms, cost)
                                          : w4a8_tiles::wide_work(rt, t, b.sms, res);
                };
                const double base_work = kernel_work(128);
                const double base_cost = ch::prefill_linears_seconds(model, card, quant, 128);
                check(base_cost > 0, std::string(b.name) + ": the estimate prices 128 tokens");
                for (int t = 129; t <= 512; ++t) {
                    const int pick               = generic::heuristic_w4a8(problem, w, t);
                    const std::int64_t wide_ctas = rt * ((t + 127) / 128);
                    // The rule that predates the comparison: a grid under half the SMs.
                    const bool expect_narrow =
                        2 * wide_ctas < b.sms || w4a8_tiles::narrow_wins(rt, t, b.sms, res, cost);
                    check((pick == k64x64) == expect_narrow,
                          std::string(b.name) + " " + std::to_string(s.n) + "x" +
                              std::to_string(s.k) + " T=" + std::to_string(t) +
                              ": heuristic_w4a8 and narrow_wins agree");
                    ++calls;
                    narrow_calls += pick == k64x64;
                    const double ratio =
                        ch::prefill_linears_seconds(model, card, quant, t) / base_cost;
                    check(std::abs(ratio - kernel_work(t) / base_work) < 1e-9,
                          std::string(b.name) + " " + std::to_string(s.n) + "x" +
                              std::to_string(s.k) + " T=" + std::to_string(t) +
                              ": the estimate prices the tile the kernel runs");
                }
            }
        }
    }
    // The tile is taken somewhere, and not everywhere.
    check(narrow_calls > 0 && narrow_calls < calls / 2, "the 64 x 64 tile is the minority choice");
    std::cout << "w4a8 tile choice: " << narrow_calls << " of " << calls << " calls on 64 x 64, "
              << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
