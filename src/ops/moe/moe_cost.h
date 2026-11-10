#pragma once

// What a grouped MoE stage costs by the facts alone (#616 step 4): the bytes law and the roofline.
// Every term is a fact of the chip or of the stored format; where the law has no such term the
// price says so (`compute_priced` false, `compute_note`) and the DRAM floor stands alone. The
// generic MoE op has no cost model of its own (select_plan is a cascade of authorities, moe.cpp);
// this prices a stage of a candidate kernel kind so that the question "which kernel would the facts
// pick" has an answer that can be compared with the plan the cascade gives, and with the sweeps on
// disk (tests/ops/test_moe_cost.cpp: the replica). It changes no route.
//
// One stage (GateUp or Down) of T tokens reads
//   - the weights of the experts the T x k assignments touch, once from DRAM: of E experts, k per
//     token, uniform routing, an expected E (1 - (1 - k/E)^T) of them (docs/maintainer/decisions.md
//     65 prices concurrent rows so); each a whole bank row span (`bank_bytes` / E);
//   - its activations (BF16, or int8 with a block scale, and a block sum for a format with a min,
//     on the W4A8 and R8 tiers): a gate and up stage reads the T rows of x once (the k assignments
//     of a token share its row, the CTAs' re-reads hit the L2), a down stage the T x k rows of the
//     activation; and writes the stage's output rows once. This is a DRAM floor for a call that
//     reads x once; in the tuner's regime (the same x every repetition) the L2 serves re-reads, so
//     a measured time can sit below it;
// and computes 2 x assignments x rows x cols operations on the tensor cores the kind uses (the
// generation's `bf16` rate for the register MMA and Volta's HMMA, `s8` for the W4A8 tier; both
// ops per clock per SM, core/generation.h) at the SM count and the reference clock.
// seconds = max(DRAM seconds, compute seconds): the bytes and the unit rates bound the time from
// below, so a measured time is expected to be at least this (the replica verifies that against a
// tolerance, tests/ops/test_moe_cost.cpp). Where the law has no term it says so instead of
// guessing, and the price is its DRAM floor alone (`compute_priced` false):
//   - R8: dp4a issue work (a SIMT instruction count, the law the campaign refuted five times);
//   - the block-scaled MXFP8 MMA of two MXFP4 banks (route "w4a8.mx8"): the generation row has no
//     rate for it, and the int8 rate is not its rate;
//   - the BF16 MMA where the generation has no native one (Volta, Turing: DeviceFeatures::
//     mma_bf16_native false): the row's `bf16` there is a fitted aggregate of an emulated path
//     (core/generation.h), not a fact of the chip, and the same figure would price Volta's native
//     HMMA (m8n8k4) like the emulated register MMA. The facts of those two generations need their
//     own rate rows (#668 measures Volta's) before the law can price them.
// A candidate list never ranks a kind against another by a price that has no compute term (status
// Unpriced), and two kinds of equal price are Tied, not one dominated. Launch times, tile
// efficiencies and wave effects are fitted quantities elsewhere (route_cost_constants.h) and are
// absent here.

#include "core/device_class.h"
#include "core/weight.h"
#include "ninferno/ops/moe.h"
#include "ops/moe/moe_internal.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::ops::detail::moe {

// The facts of a board a stage's price reads (DeviceClass::facts and the generation table); nullopt
// where the board has no roofline (a clock, a bus width or a memory clock unknown).
struct MoeBoardFacts {
    double dram_bytes_per_s      = 0;
    int sms                      = 0;
    double clock_hz              = 0;
    double bf16_ops_per_clock_sm = 0; // dense MMA rate: the register MMA, Volta's HMMA
    double s8_ops_per_clock_sm   = 0; // the int8 tier; 0 where the generation has none (Volta)
    // The generation's BF16 MMA is native (sm_80 on): only then is `bf16_ops_per_clock_sm` a fact
    // of the chip and not a fitted aggregate of an emulated path.
    bool bf16_native = true;
};

[[nodiscard]] std::optional<MoeBoardFacts> moe_board_facts(const DeviceClass& device);

// One grouped stage's problem: the bank's bytes as stored (Weight::payload_bytes), the experts, the
// top-k, the rows each expert's span has and its output rows (half of them under a gate and up),
// the columns (K) of the stage, the bank's format (the int8 activation carries a block sum for a
// format with a min), whether the plan's W4A8 stage is the block-scaled MXFP8 one.
struct MoeStageShape {
    Stage stage           = Stage::GateUp;
    QType qtype           = QType::BF16;
    bool mx8              = false;
    double bank_bytes     = 0;
    std::int32_t experts  = 0;
    std::int32_t top_k    = 0;
    std::int32_t rows     = 0; // per expert
    std::int32_t out_rows = 0; // per assignment
    std::int32_t cols     = 0;
};

struct MoeStagePrice {
    double bytes           = 0;
    double dram_seconds    = 0;
    double compute_seconds = 0;
    double seconds         = 0; // max of the two
    bool compute_priced    = true;
    // Why `compute_priced` is false (empty where it is true).
    std::string compute_note;
};

// E (1 - (1 - k/E)^T): the experts T tokens of top-k routing touch on average, uniform routing; at
// most E, and k at T = 1. The same law as touched_expert_fraction_q32 (runtime/engine/
// interleave_cost.h, fixed point; that header pulls the runtime library in, so this is a copy
// bound to it by a test on a grid of values).
[[nodiscard]] double expected_experts_touched(std::int32_t experts, std::int32_t top_k,
                                              std::int32_t tokens);

// The price of a stage of `shape` at `tokens` on `kind`; nullopt where the board has no rate the
// kind needs (the int8 tier on Volta, a BF16 rate of zero) or the shape is empty.
[[nodiscard]] std::optional<MoeStagePrice> price_moe_stage(const MoeStageShape& shape,
                                                           MoeStageKernel kind, std::int32_t tokens,
                                                           const MoeBoardFacts& board);

// The shape of a layer's GateUp or Down stage from its weights (the router gives E).
[[nodiscard]] std::optional<MoeStageShape> moe_stage_shape(const MoeWeights& weights,
                                                           const MoeConfig& config, Stage stage);

// The kinds a stage may run and what the facts say of each, as linear_candidates does for the dense
// path:
//   Selected: the kind the plan runs;  Inadmissible: the bank or call refuses it (`why`);
//   Dominated / PricedBelow: priced dearer / cheaper than the selected, by the same law (a
//   PricedBelow kind is one the cascade did not take though the facts rank it first);
//   Tied: the same price as the selected (the law has no term to order them);
//   Unpriced: admitted but not comparable by the law (a price without its compute term, see
//   above, or the board has no roofline), `why` says which.
// `seconds` is the stage's price where there is one.
struct MoeCandidate {
    enum class Status { Selected, Inadmissible, Dominated, PricedBelow, Tied, Unpriced };
    MoeStageKernel kernel = MoeStageKernel::RegisterMma;
    Status status         = Status::Unpriced;
    std::string why;
    std::optional<double> seconds;
};

[[nodiscard]] const char* moe_candidate_status_name(MoeCandidate::Status status);
[[nodiscard]] const char* moe_kernel_name(MoeStageKernel kind);

// How a kind of price `price` stands against the selected kind's `selected` (never Selected or
// Inadmissible): Unpriced where either has no price or lacks its compute term, Tied at equal
// seconds, else PricedBelow / Dominated. `why` says the reason of an Unpriced or Tied.
[[nodiscard]] MoeCandidate::Status
rank_against_selected(const std::optional<MoeStagePrice>& price,
                      const std::optional<MoeStagePrice>& selected, std::string& why);

// The candidates of `stage` (GateUp or Down) of a call of `tokens` under select_plan's plan.
[[nodiscard]] std::vector<MoeCandidate> moe_candidates(const MoeWeights& weights,
                                                       const MoeConfig& config, std::int32_t tokens,
                                                       bool prefill, Stage stage);

// The layer's two grouped stages on `plan` (each stage's kernel from stage_routes), seconds; the
// router and combine launches, whose bytes the law prices as the router's weights, are added.
// nullopt where a stage has no price.
[[nodiscard]] std::optional<double> moe_layer_seconds(const MoeWeights& weights,
                                                      const MoeConfig& config, const MoePlan& plan,
                                                      std::int32_t tokens,
                                                      const MoeBoardFacts& board);

} // namespace ninferno::ops::detail::moe
