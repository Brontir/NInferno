// --concurrent-prefill shares the GPU by predicted cost, never by timing: a prefill unit and a
// decode round are priced from their shapes (interleave_cost.h), and the decode owed per unit is
// a pure function of those prices. On a MoE shape at concurrency 8 beside a 24K-token prompt, the
// rounds per chunk stay bounded and the decode time owed is the equal share, within one round.

#include "runtime/engine/interleave_cost.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using namespace ninferno::runtime;

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

} // namespace

int main() {
    constexpr std::uint64_t q32 = 1ULL << 32U;
    // A 35B-A3B-like MoE at 4 bits on a large card: about 6 000 prompt tokens a second.
    ContextPrefillCost prefill{.chunk_ns              = 200'000,
                               .token_ns_q32          = 166'000ULL * q32,
                               .attention_pair_ns_q32 = q32 / 2,
                               .vision_item_ns        = 2'000'000,
                               .vision_patch_ns_q32   = 20'000ULL * q32};
    ContextMachineCostModel model{.prefill = prefill};
    const DecodeRoundShape moe{.weight_bytes        = 19'000'000'000ULL,
                               .routed_expert_bytes = 17'500'000'000ULL,
                               .experts             = 256,
                               .experts_per_token   = 8};
    const DecodeRoundShape dense{.weight_bytes = 19'000'000'000ULL};

    // Distinct experts: one token touches k/E of a layer, eight about 22 %.
    const std::uint64_t one   = touched_expert_fraction_q32(256, 8, 1);
    const std::uint64_t eight = touched_expert_fraction_q32(256, 8, 8);
    check(one == q32 / 32 && eight > q32 / 5 && eight < q32 / 4,
          "touched expert fraction is not 1 - (1 - k/E)^L");
    check(touched_expert_fraction_q32(4, 8, 3) == q32 && touched_expert_fraction_q32(0, 8, 3) == 0,
          "degenerate expert shapes are not bounded");

    const std::array<std::uint64_t, 1> solo{4096};
    std::array<std::uint64_t, 8> lanes{};
    lanes.fill(4096);
    const std::uint64_t moe_1   = decode_round_ns(prefill, moe, solo);
    const std::uint64_t moe_8   = decode_round_ns(prefill, moe, lanes);
    const std::uint64_t dense_1 = decode_round_ns(prefill, dense, solo);
    const std::uint64_t dense_8 = decode_round_ns(prefill, dense, lanes);
    // MoE lanes touch distinct experts: eight lanes cost far more than eight tokens of one round,
    // while a dense round's weight read is shared by its lanes.
    check(moe_8 > 2 * moe_1, "eight MoE lanes were priced like one");
    check(dense_8 < 2 * dense_1, "eight dense lanes were priced as eight weight reads");
    // Attention grows with the lanes' contexts.
    std::array<std::uint64_t, 8> long_lanes{};
    long_lanes.fill(65'536);
    check(decode_round_ns(prefill, moe, long_lanes) > moe_8, "context did not cost attention");

    // A 24K-token prompt in 2048-token chunks beside seven decoding lanes (C = 8). The chunk cost
    // grows with its start position (attention pairs).
    constexpr std::uint32_t kD   = 7;
    std::uint64_t prefill_total  = 0;
    std::uint64_t decode_total   = 0;
    std::uint64_t previous_chunk = 0;
    std::uint32_t worst_rounds   = 0;
    std::uint64_t context        = 4096;
    for (std::uint32_t start = 0; start < 24'576; start += 2048) {
        const PrefillWork work       = make_prefill_work(start, 2048, 0, 0, 2048);
        const std::uint64_t chunk_ns = prefill_unit_ns(model, work);
        check(chunk_ns > previous_chunk, "a later chunk was not priced higher");
        previous_chunk = chunk_ns;
        prefill_total += chunk_ns;
        std::uint64_t credit = fair_decode_credit(chunk_ns, kD, 1);
        check(credit == chunk_ns * kD, "decode owed is not the decoding lanes' share");
        std::uint32_t rounds = 0;
        while (credit > 0) {
            std::array<std::uint64_t, kD> contexts{};
            contexts.fill(context++);
            const std::uint64_t round = decode_round_ns(prefill, moe, contexts);
            credit -= std::min(credit, round);
            decode_total += round;
            ++rounds;
        }
        worst_rounds = std::max(worst_rounds, rounds);
    }
    // The decode time beside the prompt is its equal share (seven lanes to one), within one round
    // per chunk: the prompt's predicted time to first token is (1 + 7) x its prefill time, as with
    // the former timed share, however many lanes' experts a round touches.
    check(decode_total >= kD * prefill_total &&
              decode_total <= kD * prefill_total + 12 * decode_round_ns(prefill, moe, long_lanes),
          "decode time beside the prompt is not the equal share");
    check(worst_rounds > 0 && worst_rounds <= 256, "decode rounds per chunk are not bounded");
    std::cout << "rounds per chunk (worst) " << worst_rounds << ", MoE round 1/8 lanes " << moe_1
              << "/" << moe_8 << " ns, TTFT share "
              << (prefill_total + decode_total) / prefill_total << "x\n";

    // Images are priced from their own work.
    PrefillWork vision = make_prefill_work(0, 1024, 1, 4096, 2048);
    check(prefill_unit_ns(model, vision) >
              prefill_unit_ns(model, make_prefill_work(0, 1024, 0, 0, 2048)),
          "an image chunk earned no more decode than text");

    // Purity: the same shapes give the same prices.
    check(decode_round_ns(prefill, moe, lanes) == moe_8 &&
              fair_decode_credit(previous_chunk, kD, 2) ==
                  fair_decode_credit(previous_chunk, kD, 2),
          "prices are not pure functions of their inputs");
    check(fair_decode_credit(1'000, 3, 0) == 0 && fair_decode_credit(1'000, 0, 1) == 0,
          "no decoders or no prefill owed decode");

    std::cout << (failures ? "FAIL" : "OK") << " interleave_cost\n";
    return failures ? 1 : 0;
}
