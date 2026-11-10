#pragma once

// What a grouped MoE stage costs by the facts alone (#616 step 4): the bytes law and the roofline,
// no fitted constant. The generic MoE op has no cost model of its own (select_plan is a cascade of
// authorities, moe.cpp); this prices a stage of a candidate kernel kind so that the question "which
// kernel would the facts pick" has an answer that can be compared with the plan the cascade gives,
// and with the sweeps on disk (tests/ops/test_moe_cost.cpp: the replica). It changes no route.
//
// One stage (GateUp or Down) of T tokens reads
//   - the weights of the experts the T x k assignments touch, once from DRAM: of E experts, k per
//     token, uniform routing, an expected E (1 - (1 - k/E)^T) of them (docs/maintainer/decisions.md
//     65 prices concurrent rows so); each a whole bank row span (`bank_bytes` / E);
//   - the activations of its assignments (BF16, or int8 with a block scale on the W4A8 and R8
//     tiers) and writes the stage's output rows, once: a DRAM floor, since the CTAs' re-reads hit
//     the L2;
// and computes 2 x assignments x rows x cols operations on the tensor cores the kind uses (the
// generation's `bf16` rate for the register MMA and Volta's HMMA, `s8` for the W4A8 tier; both
// ops per clock per SM, core/generation.h) at the SM count and the reference clock.
// seconds = max(DRAM seconds, compute seconds): a floor, the stage cannot run faster, so a measured
// time is at least this (the replica holds that). Where the law has no term it says so instead of
// guessing: R8's dp4a work (a SIMT instruction count, the law the campaign refuted five times) is
// not priced, so an R8 price is its DRAM floor alone (`compute_priced` false), and a candidate list
// never ranks R8 against another kind by it (status Unpriced). Launch times, tile efficiencies and
// wave effects are fitted quantities elsewhere (route_cost_constants.h) and are absent here.

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
};

[[nodiscard]] std::optional<MoeBoardFacts> moe_board_facts(const DeviceClass& device);

// One grouped stage's problem: the bank's bytes as stored (Weight::payload_bytes), the experts, the
// top-k, the rows each expert's span has and its output rows (half of them under a gate and up),
// the columns (K) of the stage.
struct MoeStageShape {
    Stage stage           = Stage::GateUp;
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
};

// E (1 - (1 - k/E)^T): the experts T tokens of top-k routing touch on average, uniform routing; at
// most E, and k at T = 1.
[[nodiscard]] double expected_experts_touched(std::int32_t experts, std::int32_t top_k,
                                              std::int32_t tokens);

// The price of a stage of `shape` at `tokens` on `kind`; nullopt where the board has no rate the
// kind needs (the int8 tier on Volta) or the shape is empty.
[[nodiscard]] std::optional<MoeStagePrice> price_moe_stage(const MoeStageShape& shape,
                                                           MoeStageKernel kind, std::int32_t tokens,
                                                           const MoeBoardFacts& board);

// The shape of a layer's GateUp or Down stage from its weights (the router gives E).
[[nodiscard]] std::optional<MoeStageShape> moe_stage_shape(const MoeWeights& weights,
                                                           const MoeConfig& config, Stage stage);

// The kinds a stage may run and what the facts say of each, as linear_candidates does for the dense
// path:
//   Selected: the kind the plan runs;  Inadmissible: the bank or call refuses it (`why`);
//   Dominated / PricedBelow: priced no cheaper / cheaper than the selected, by the same law (a
//   PricedBelow kind is one the cascade did not take though the facts rank it first);
//   Unpriced: admitted but not comparable by the law (R8, or the board has no roofline).
// `seconds` is the stage's price where there is one.
struct MoeCandidate {
    enum class Status { Selected, Inadmissible, Dominated, PricedBelow, Unpriced };
    MoeStageKernel kernel = MoeStageKernel::RegisterMma;
    Status status         = Status::Unpriced;
    std::string why;
    std::optional<double> seconds;
};

[[nodiscard]] const char* moe_candidate_status_name(MoeCandidate::Status status);
[[nodiscard]] const char* moe_kernel_name(MoeStageKernel kind);

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
