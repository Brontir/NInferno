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
//     65 prices concurrent rows so); each a whole bank row span (`bank_bytes` / E). The bytes are
//     those of the logical K the kernel walks, not of the K the bank is stored at (a grouped
//     format stores K padded to a multiple of 128; the kernel never reads the tail);
//   - its activations: BF16 on the register MMA, on Volta's HMMA and on an R8 down stage (which
//     reads the BF16 intermediate and quantizes it itself, and on an R8 gate/up of a padded x,
//     which quantizes in the kernel); int8 with a block scale, and a block sum for a format with a
//     min, on an R8 gate/up of an unpadded x (the router's CTAs wrote the codes) and on the W4A8
//     tier, whose stage first runs a quantize pass that reads the BF16 activation and writes the
//     codes, scales and sums (40 bytes a 32-value block) of a plane at K rounded up to 128, which
//     the kernel then reads whole (both priced). A gate and
//     up stage reads the T rows of x once (the k assignments of a token share its row), a down
//     stage the T x k rows of the activation; and writes the stage's output rows once. These are
//     the expected DRAM bytes of a call that reads each byte once from DRAM, under uniform routing:
//     a lower bound of the call at T = 1 only (k experts are read whatever the router picks). At T
//     > 1 a skewed router touches fewer experts than the uniform expectation, so the figure can
//     exceed a measured time (moe_layer_dram_floor_seconds is the bound that holds for any
//     routing). The tuner flushes the L2 before a sample where the device has room for it
//     (moe_tune.cpp, tune_timer.h), so the L2 does not lower a measured time below it there;
// and computes 2 x assignments x rows x cols operations on the tensor cores the kind uses (the
// generation's `bf16` rate for the register MMA and Volta's HMMA, `s8` for the W4A8 tier; both
// ops per clock per SM, core/generation.h) at the SM count and the reference clock.
// seconds = max(DRAM seconds, compute seconds): under uniform routing the bytes and the unit rates
// bound the time from below at T = 1, and a measured time is expected to be at least this there
// (the replica verifies that at T = 1 and against the strict bound at every T,
// tests/ops/test_moe_cost.cpp). It is a lower bound of the time (measured 1.05x to 4.4x above it),
// not yet a cost to select by: the absent terms below are larger than the margins between kinds.
// Where the law has no term it says so instead of guessing, and the price is its DRAM figure alone
// (`compute_priced` false):
//   - R8: dp4a issue work (a SIMT instruction count that no law here holds yet);
//   - the block-scaled MXFP8 MMA of two MXFP4 banks (route "w4a8.mx8"): the generation row has no
//     rate for it, and the int8 rate is not its rate;
//   - any generation without a rate row of its own (MoeBoardFacts::generation_known false: the
//     lent row's rates are not a fact of that chip);
//   - the BF16 MMA where the generation has no native one (Volta, Turing: DeviceFeatures::
//     mma_bf16_native false): the row's `bf16` there is a fitted aggregate of an emulated path
//     (core/generation.h), not a fact of the chip, and the same figure would price Volta's native
//     HMMA (m8n8k4) like the emulated register MMA. The facts of those two generations need their
//     own rate rows (#668 measures Volta's) before the law can price them.
// Also absent from every price: launch and fixed overheads, the kernel's DRAM efficiency below
// the peak bandwidth, tile and wave effects (fitted quantities elsewhere, route_cost_constants.h),
// the R8 gate/up's code write by the router's CTAs, Volta's x conversion pass, the zero-padded x
// copy of a padded gate/up (moe.cpp), the shared expert, and the combine fused into an R8 down
// stage (a separate one is priced by moe_layer_price). A candidate list never ranks a kind against
// another by a price that has no compute term (status Unpriced), and two kinds of equal price are
// Tied, not one dominated.

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
    // The table has a row of its own for the board's generation (generation_known): where it has
    // not, generation_row lends another's rates, and they are not a fact of this chip.
    bool generation_known = true;
};

[[nodiscard]] std::optional<MoeBoardFacts> moe_board_facts(const DeviceClass& device);

// One grouped stage's problem: the bank's bytes the kernel reads (Weight::payload_bytes scaled from
// the stored K, padded_shape[1], to the logical K), the experts, the top-k, the rows each expert's
// span has and its output rows (half of them under a gate and up), the columns (K) of the stage,
// the bank's format (the int8 activation carries a block sum for a format with a min), whether the
// plan's W4A8 stage is the block-scaled MXFP8 one.
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
    // The live values of an activation row, as the router's x holds them (0: `cols`): the router's
    // width for a gate/up, whose bank K is larger where x is padded. The W4A8 quantizer reads this
    // many; every BF16 kernel reads the row at `cols` (the zero tail included).
    std::int32_t x_cols = 0;
    // The router's CTAs quantize x for an R8 gate/up (the bank's K is the router's: x is not
    // padded); only then an R8 gate/up reads int8 codes, else it reads BF16 x and quantizes itself.
    bool router_quantizes_x = false;
};

struct MoeStagePrice {
    double bytes = 0;
    double quantize_bytes =
        0; // of `bytes`: the W4A8 stage's quantize pass (read BF16, write codes)
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

// A lower bound of the DRAM seconds of a whole layer's call of `tokens`, true for any routing:
// the weights of at least top-k experts of each bank (the fewest any T tokens of top-k routing
// touch; k at T = 1 and at T > 1 when every token picks the same k), the router's weights and x
// read once. Nothing written is counted (a write into a clean L2 may not reach DRAM), nor any
// expert past k, nor the activations in between. `moe_layer_price` is the expectation under uniform
// routing; this is what no routing can go below.
[[nodiscard]] std::optional<double> moe_layer_dram_floor_seconds(const MoeWeights& weights,
                                                                 const MoeConfig& config,
                                                                 std::int32_t tokens,
                                                                 const MoeBoardFacts& board);

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
//   PricedBelow kind is one the cascade did not take though the law's terms rank it first; the
//   price is a lower bound (measured 1.05x to 4.4x above it), not a cost to select by, and the
//   margins are smaller than the absent terms); `why` gives the margin;
//   Tied: the same price as the selected (the law has no term to order them);
//   Unpriced: admitted but not comparable by the law (a price without its compute term, see
//   above, or the board has no roofline), `why` says which.
// `seconds` is the stage's price only where it is a full one (`compute_priced`); a kind without its
// compute term carries `dram_seconds` alone, a floor that is not a price to sort by.
struct MoeCandidate {
    enum class Status { Selected, Inadmissible, Dominated, PricedBelow, Tied, Unpriced };
    MoeStageKernel kernel = MoeStageKernel::RegisterMma;
    Status status         = Status::Unpriced;
    std::string why;
    bool compute_priced = false; // `seconds` is a full price
    std::optional<double> seconds;
    std::optional<double> dram_seconds;
};

[[nodiscard]] const char* moe_candidate_status_name(MoeCandidate::Status status);
[[nodiscard]] const char* moe_kernel_name(MoeStageKernel kind);

// How a kind of price `price` stands against the selected kind's `selected` (never Selected or
// Inadmissible): Unpriced where either has no price or lacks its compute term, Tied at equal
// seconds, else PricedBelow / Dominated. `why` says the reason of an Unpriced or Tied, and the
// margin (microseconds and percent of the selected kind's price) of the other two.
[[nodiscard]] MoeCandidate::Status
rank_against_selected(const std::optional<MoeStagePrice>& price,
                      const std::optional<MoeStagePrice>& selected, std::string& why);

// The candidates of `stage` (GateUp or Down) of a call of `tokens` under select_plan's plan.
[[nodiscard]] std::vector<MoeCandidate> moe_candidates(const MoeWeights& weights,
                                                       const MoeConfig& config, std::int32_t tokens,
                                                       bool prefill, Stage stage);

// The layer's price on a plan: its two grouped stages (each stage's kernel from stage_routes), the
// router's bytes (its weights, the activation, the logits) and, where the down stage does not fuse
// the combine (an R8 down stage's last CTAs do, unless the layer has a shared expert or the row
// grid exceeds the ticket array: moe_r8_down_fuses_combine), the separate combine (reads T x k x H
// BF16, writes T x H). A stage
// without its compute term stands at its DRAM floor: `compute_priced` false says the number is a
// floor in that case, not a full price.
struct MoeLayerPrice {
    double seconds      = 0;     // stages' max(DRAM, compute) + router + combine
    double dram_seconds = 0;     // the same with every stage at its DRAM time
    bool compute_priced = true;  // every stage has its compute term
    bool compute_binds  = false; // some stage's compute time exceeds its DRAM time
    std::string compute_note;    // the first unpriced stage's reason
};

// nullopt where a stage has no price.
[[nodiscard]] std::optional<MoeLayerPrice> moe_layer_price(const MoeWeights& weights,
                                                           const MoeConfig& config,
                                                           const MoePlan& plan, std::int32_t tokens,
                                                           const MoeBoardFacts& board);

} // namespace ninferno::ops::detail::moe
